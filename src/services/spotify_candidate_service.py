"""Generate and rank real Spotify track candidates from recent listening."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
import re
from typing import Any

import pandas as pd

from app.demo_data import DemoUserProfile
from app.demo_service import DemoRecommendationExplanation, DemoViewState
from config.settings import ProjectSettings
from data.preprocessor import Preprocessor
from data.spotify_client import SpotifyAPIClient, SpotifyAPIClientError
from features.feature_builder import FeatureBuilder
from models.content_recommender import ContentRecommender
from models.hybrid_recommender import HybridRecommendation, HybridRecommender
from models.playlist_generator import PlaylistGenerator
from services.advanced_hybrid_ranking_service import AdvancedHybridRankingService, AdvancedHybridWeights
from services.diversity_reranking_service import DiversityRerankingService
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter
from services.user_profile_service import ListeningHistorySnapshot


LOGGER = logging.getLogger(__name__)
SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")


def top_k_overlap_percent(list_a: list[str], list_b: list[str], k: int) -> float:
    """Return overlap percentage between two top-k ranked ID lists."""

    if k <= 0:
        return 0.0
    top_a = set(list_a[:k])
    top_b = set(list_b[:k])
    denominator = min(k, len(top_a), len(top_b))
    if denominator <= 0:
        return 0.0
    return len(top_a.intersection(top_b)) / denominator


@dataclass(slots=True)
class SpotifyCandidateTrack:
    """Lightweight representation for one real Spotify recommendation candidate."""

    spotify_track_id: str
    track_name: str
    artist_name: str
    spotify_url: str | None = None
    album_image_url: str | None = None
    popularity: float | None = None
    source: str = "spotify"


@dataclass(slots=True)
class SpotifyCandidateSet:
    """Bundle real Spotify candidates and metadata used by ranking/explanations."""

    candidates: list[SpotifyCandidateTrack]
    track_catalog: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    source_labels_by_track_id: dict[str, list[str]] = field(default_factory=dict)
    debug_summary: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateDeduplicationState:
    """Track duplicate accounting across Spotify candidate sources."""

    track_ids_seen: set[str] = field(default_factory=set)
    track_ids_by_name_artist: dict[tuple[str, str], str] = field(default_factory=dict)
    raw_candidate_count: int = 0
    duplicate_candidates_removed: int = 0


@dataclass(slots=True)
class RecommendationBucket:
    """UI-ready group of Spotify recommendations for one product intent."""

    bucket_name: str
    bucket_label: str
    description: str
    recommendations: list[DemoRecommendationExplanation]


@dataclass(slots=True)
class SpotifyRealRecommendationResult:
    """Store a UI-ready real Spotify recommendation view."""

    view_state: DemoViewState
    candidate_set: SpotifyCandidateSet
    source_message: str
    recommendation_buckets: list[RecommendationBucket] = field(default_factory=list)
    bucketed_explanations: dict[str, list[DemoRecommendationExplanation]] = field(default_factory=dict)


@dataclass(slots=True)
class SpotifyCandidateService:
    """Build and rank real Spotify candidates while keeping Streamlit thin."""

    client: SpotifyAPIClient
    preprocessor: Preprocessor
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    playlist_generator: PlaylistGenerator = field(default_factory=PlaylistGenerator)
    recommendation_adapter: SpotifyRecommendationAdapter = field(default_factory=SpotifyRecommendationAdapter)
    diversity_reranker: DiversityRerankingService = field(default_factory=DiversityRerankingService)
    max_recent_artists: int = 5
    top_tracks_per_artist: int = 6
    search_tracks_per_artist: int = 4
    search_tracks_per_recent_track: int = 2
    genre_search_limit: int = 2
    max_recent_track_searches: int = 4
    max_candidates: int = 80
    default_market: str = "US"

    @classmethod
    def from_settings(cls, settings: ProjectSettings) -> "SpotifyCandidateService":
        """Build the candidate service from project settings."""

        return cls(
            client=SpotifyAPIClient.from_settings(settings),
            preprocessor=Preprocessor(settings=settings),
            default_market=settings.spotify_default_market,
        )

    def build_real_spotify_view(
        self,
        access_token: str,
        listening_history_snapshot: ListeningHistorySnapshot,
        exploration_level: float,
        recommendation_count: int,
        mood_label: str,
        playlist_length: int,
        ranking_focus: str = "Balanced",
    ) -> SpotifyRealRecommendationResult | None:
        """Generate real Spotify candidates, rank them, and return UI-ready output."""

        candidate_set = self.build_candidate_set(
            access_token=access_token,
            listening_history_snapshot=listening_history_snapshot,
        )
        if candidate_set.track_catalog.empty:
            return None

        profile = DemoUserProfile(
            user_id=f"spotify_real::{listening_history_snapshot.user_id}",
            display_name=f"{listening_history_snapshot.display_name} | Spotify Real Tracks",
            summary="Real Spotify tracks generated from recent artists, top tracks, and Spotify search.",
            seed_track_ids=listening_history_snapshot.seed_track_ids,
            preferred_mood=mood_label,
        )
        recommendations = self._rank_candidates(
            candidate_catalog=candidate_set.track_catalog,
            listening_history_snapshot=listening_history_snapshot,
            profile=profile,
            exploration_level=exploration_level,
            recommendation_count=recommendation_count,
            mood_label=mood_label,
            ranking_focus=ranking_focus,
            debug_summary=candidate_set.debug_summary,
        )
        if not recommendations:
            return None

        recommendation_table = self._build_recommendation_table(recommendations, candidate_set.track_catalog)
        explanations = self._build_explanations(
            recommendations=recommendations,
            listening_history_snapshot=listening_history_snapshot,
            candidate_set=candidate_set,
            mood_label=mood_label,
            bucket_label="Balanced",
        )
        recommendation_buckets = self._build_recommendation_buckets(
            candidate_catalog=candidate_set.track_catalog,
            listening_history_snapshot=listening_history_snapshot,
            candidate_set=candidate_set,
            exploration_level=exploration_level,
            mood_label=mood_label,
            recommendation_count=recommendation_count,
        )
        bucketed_explanations = {
            bucket.bucket_label: bucket.recommendations
            for bucket in recommendation_buckets
        }
        playlist = self.playlist_generator.generate_playlist(
            candidate_tracks=self._build_playlist_candidate_frame(recommendations, candidate_set.track_catalog),
            mood_label=mood_label,
            max_items=playlist_length,
        )
        view_state = DemoViewState(
            profile=profile,
            hybrid_weights=self._build_hybrid_weights(exploration_level),
            recommendations=recommendations,
            recommendation_table=recommendation_table,
            explanations=explanations,
            playlist=playlist,
            taste_clusters=None,
        )
        return SpotifyRealRecommendationResult(
            view_state=view_state,
            candidate_set=candidate_set,
            source_message="Spotify real-track recommendations generated from your recent listening.",
            recommendation_buckets=recommendation_buckets,
            bucketed_explanations=bucketed_explanations,
        )

    def build_candidate_set(
        self,
        access_token: str,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> SpotifyCandidateSet:
        """Build a deduplicated pool of real Spotify tracks from recent history."""

        candidate_payloads: list[dict[str, Any]] = []
        source_labels_by_track_id: dict[str, list[str]] = {}
        dedup_state = CandidateDeduplicationState()
        warnings: list[str] = []
        recent_artist_ids = self._extract_recent_artist_ids(listening_history_snapshot)
        recent_artist_names = self._extract_recent_artist_names(listening_history_snapshot)
        recent_track_search_terms = self._extract_recent_track_search_terms(listening_history_snapshot)
        genre_search_terms = self._extract_recent_genre_terms(listening_history_snapshot)
        valid_recent_artist_ids = [
            artist_id
            for artist_id in recent_artist_ids
            if self._looks_like_spotify_id(artist_id)
        ][: self.max_recent_artists]
        top_track_requests_attempted = 0
        top_track_requests_failed = 0
        top_track_candidates_found = 0
        search_candidates_found = 0
        search_requests_failed = 0

        for artist_id in valid_recent_artist_ids:
            top_track_requests_attempted += 1
            try:
                payload = self.client.get_artist_top_tracks(
                    artist_id,
                    access_token=access_token,
                    market=self.default_market,
                )
            except SpotifyAPIClientError as error:
                top_track_requests_failed += 1
                LOGGER.info("Skipping Spotify artist top tracks for %s: %s", artist_id, error)
                continue
            for track_payload in payload.get("tracks", [])[: self.top_tracks_per_artist]:
                top_track_candidates_found += 1
                self._add_candidate_payload(
                    candidate_payloads,
                    source_labels_by_track_id,
                    track_payload,
                    source_label="recent artist top track",
                    dedup_state=dedup_state,
                )

        search_queries = self._build_candidate_search_queries(
            recent_artist_names=recent_artist_names,
            recent_track_search_terms=recent_track_search_terms,
            genre_search_terms=genre_search_terms,
        )
        for search_query, source_label, search_limit in search_queries:
            try:
                payload = self.client.search_tracks(
                    search_query,
                    access_token=access_token,
                    limit=search_limit,
                    market=self.default_market,
                )
            except SpotifyAPIClientError as error:
                search_requests_failed += 1
                LOGGER.info("Skipping Spotify track search for %s: %s", search_query, error)
                continue
            for track_payload in payload.get("tracks", {}).get("items", []):
                search_candidates_found += 1
                self._add_candidate_payload(
                    candidate_payloads,
                    source_labels_by_track_id,
                    track_payload,
                    source_label=source_label,
                    dedup_state=dedup_state,
                )

        candidate_payloads = candidate_payloads[: self.max_candidates]
        artist_counts = self._build_artist_candidate_counts(candidate_payloads)
        debug_summary = {
            "candidate_count": len(candidate_payloads),
            "top_track_candidate_count": top_track_candidates_found,
            "search_candidate_count": search_candidates_found,
            "skipped_artist_expansion_count": top_track_requests_failed,
            "raw_candidate_count": dedup_state.raw_candidate_count,
            "unique_candidate_count": len(candidate_payloads),
            "duplicate_candidates_removed": dedup_state.duplicate_candidates_removed,
            "candidate_artist_count": len(artist_counts),
            "max_candidates_from_single_artist": max(artist_counts.values(), default=0),
            "search_queries_used": len(search_queries),
            "candidate_source_breakdown": self._build_source_breakdown(source_labels_by_track_id),
            "recent_artists_found": len(recent_artist_names),
            "valid_artist_ids_used": len(valid_recent_artist_ids),
            "top_track_requests_attempted": top_track_requests_attempted,
            "top_track_requests_failed": top_track_requests_failed,
            "top_track_candidates_found": top_track_candidates_found,
            "search_requests_failed": search_requests_failed,
            "search_candidates_found": search_candidates_found,
            "final_candidate_count": len(candidate_payloads),
            "ranking_mode": "metadata-only",
        }
        LOGGER.info("Spotify candidate generation summary: %s", debug_summary)
        if top_track_requests_failed:
            warnings.append(
                "Some recent artists could not be expanded into top tracks, so search-based candidates were used instead."
            )
        if search_requests_failed:
            warnings.append("Some Spotify searches could not be completed, so available candidates were used.")
        if not candidate_payloads:
            return SpotifyCandidateSet(
                candidates=[],
                track_catalog=pd.DataFrame(),
                warnings=warnings,
                debug_summary=debug_summary,
            )

        candidate_ids = [str(track.get("id", "")).strip() for track in candidate_payloads if track.get("id")]
        audio_features_payload: dict[str, Any] = {"audio_features": []}
        try:
            audio_features_payload = self.client.get_audio_features(candidate_ids, access_token=access_token)
        except SpotifyAPIClientError as error:
            LOGGER.info("Spotify candidate audio features unavailable: %s", error)
            warnings.append(
                "Spotify audio features were unavailable for candidates, so ranking used metadata-only signals."
            )
        if audio_features_payload.get("audio_features"):
            debug_summary["ranking_mode"] = "audio-feature-based"

        artist_ids = self._extract_artist_ids_from_track_payloads(candidate_payloads)
        artist_metadata_payload = self._safe_get_artists(artist_ids, access_token, warnings)
        track_catalog = self._build_candidate_catalog(
            candidate_payloads=candidate_payloads,
            audio_features_payload=audio_features_payload,
            artist_metadata_payload=artist_metadata_payload,
            source_labels_by_track_id=source_labels_by_track_id,
            ranking_mode=str(debug_summary["ranking_mode"]),
        )
        self._add_artist_count_metadata(track_catalog)
        candidates = self._build_candidate_tracks(track_catalog, source_labels_by_track_id)
        return SpotifyCandidateSet(
            candidates=candidates,
            track_catalog=track_catalog,
            warnings=warnings,
            source_labels_by_track_id=source_labels_by_track_id,
            debug_summary=debug_summary,
        )

    def _rank_candidates(
        self,
        candidate_catalog: pd.DataFrame,
        listening_history_snapshot: ListeningHistorySnapshot,
        profile: DemoUserProfile,
        exploration_level: float,
        recommendation_count: int,
        mood_label: str,
        ranking_focus: str,
        debug_summary: dict[str, object],
        als_scores: dict[str, float] | None = None,
        embedding_scores: dict[str, float] | None = None,
    ) -> list[HybridRecommendation]:
        """Rank real Spotify candidates with the existing hybrid scorer."""

        combined_catalog = pd.concat(
            [listening_history_snapshot.track_level_frame.copy(), candidate_catalog.copy()],
            axis=0,
            ignore_index=True,
            sort=False,
        ).drop_duplicates(subset=["track_id"], keep="first")
        feature_artifacts = self.feature_builder.create_model_ready_feature_matrix(combined_catalog)
        content_recommender = ContentRecommender(
            feature_artifacts=feature_artifacts,
            track_catalog=combined_catalog,
        )
        popularity_scores = self._build_popularity_scores(candidate_catalog)
        novelty_scores = {track_id: 1.0 - score for track_id, score in popularity_scores.items()}
        hybrid_recommender = HybridRecommender(
            content_recommender=content_recommender,
            weights=self._build_hybrid_weights(exploration_level),
            popularity_scores=popularity_scores,
            novelty_scores=novelty_scores,
            user_seed_track_ids_by_user={profile.user_id: listening_history_snapshot.seed_track_ids},
            track_catalog=combined_catalog,
        )
        candidate_track_ids = candidate_catalog["track_id"].astype(str).tolist()
        ranked = hybrid_recommender.recommend(
            user_id=profile.user_id,
            candidate_item_ids=candidate_track_ids,
            k=max(len(candidate_track_ids), recommendation_count),
        )
        before_track_ids = [recommendation.track_id for recommendation in ranked[:10]]
        reranked = self._apply_spotify_ranking_adjustments(
            recommendations=ranked,
            candidate_catalog=candidate_catalog,
            listening_history_snapshot=listening_history_snapshot,
            exploration_level=exploration_level,
            mood_label=mood_label,
            ranking_focus=ranking_focus,
        )
        reranked = self._apply_optional_advanced_scores(
            recommendations=reranked,
            als_scores=als_scores,
            embedding_scores=embedding_scores,
        )
        reranked = self.diversity_reranker.rerank(
            recommendations=reranked,
            candidate_catalog=candidate_catalog,
            exploration_level=exploration_level,
            k=len(reranked),
            strength=self._build_diversity_strength(ranking_focus),
        )
        after_track_ids = [recommendation.track_id for recommendation in reranked[:10]]
        mood_debug_summary = self._build_mood_debug_summary(
            recommendations=reranked,
            candidate_catalog=candidate_catalog,
            mood_label=mood_label,
        )
        debug_summary.update(
            {
                "selected_mood": mood_label,
                "exploration_level": f"{float(exploration_level):.2f}",
                "recommendation_count": recommendation_count,
                "ranking_focus": ranking_focus,
                "selected_filters": {
                    "mood": mood_label,
                    "exploration_level": f"{float(exploration_level):.2f}",
                    "ranking_focus": ranking_focus,
                    "recommendation_count": recommendation_count,
                },
                "top_candidate_ids_before_reranking": before_track_ids,
                "top_candidate_ids_after_reranking": after_track_ids,
                "positions_changed_after_reranking": self._count_position_changes(
                    before_track_ids,
                    after_track_ids,
                ),
                "diversity_reranking_active": True,
                "als_score_available": bool(als_scores),
                "embedding_score_available": bool(embedding_scores),
                "advanced_score_used": bool(als_scores or embedding_scores),
                **mood_debug_summary,
            }
        )
        LOGGER.info("Spotify ranking control summary: %s", debug_summary)
        return reranked[:recommendation_count]

    def _apply_spotify_ranking_adjustments(
        self,
        recommendations: list[HybridRecommendation],
        candidate_catalog: pd.DataFrame,
        listening_history_snapshot: ListeningHistorySnapshot,
        exploration_level: float,
        mood_label: str,
        ranking_focus: str = "Balanced",
    ) -> list[HybridRecommendation]:
        """Apply small real-Spotify controls for mood and exploration."""

        if not recommendations:
            return recommendations

        catalog_lookup = candidate_catalog.set_index("track_id").to_dict(orient="index")
        recent_artist_affinity = self._build_recent_artist_affinity(listening_history_snapshot)
        focus_weights = self._build_focus_weights(ranking_focus)
        normalized_base_scores = self._normalize_recommendation_scores(recommendations)
        adjusted: list[HybridRecommendation] = []
        for recommendation in recommendations:
            track_row = catalog_lookup.get(recommendation.track_id, {})
            artist_name = str(track_row.get("artist_name", "")).strip()
            source_labels = str(track_row.get("candidate_sources", ""))
            popularity = float(track_row.get("catalog_popularity", 0.0) or 0.0)
            novelty = float(track_row.get("catalog_novelty", 0.0) or 0.0)
            bounded_exploration = min(max(float(exploration_level), 0.0), 1.0)
            familiarity = 1.0 - bounded_exploration
            is_search_candidate = "search" in source_labels.lower()
            is_top_track_candidate = "top track" in source_labels.lower()
            artist_affinity = recent_artist_affinity.get(artist_name, 0.0)
            mood_score = self._compute_mood_alignment(track_row, mood_label)
            base_score = normalized_base_scores.get(recommendation.track_id, 0.0)
            familiar_source_boost = 2.10 * focus_weights["familiar"] * familiarity if is_top_track_candidate else 0.0
            artist_boost = 2.35 * focus_weights["familiar"] * familiarity * artist_affinity
            popularity_boost = 1.40 * focus_weights["familiar"] * familiarity * popularity
            mood_boost = 3.25 * focus_weights["mood"] * mood_score
            novelty_boost = 2.45 * focus_weights["discovery"] * bounded_exploration * novelty
            search_discovery_boost = 2.05 * focus_weights["discovery"] * bounded_exploration if is_search_candidate else 0.0
            recent_artist_penalty = 1.35 * focus_weights["discovery"] * bounded_exploration * artist_affinity
            overpopular_penalty = 1.05 * focus_weights["discovery"] * bounded_exploration * popularity
            top_track_discovery_penalty = (
                1.15 * focus_weights["discovery"] * bounded_exploration
                if is_top_track_candidate
                else 0.0
            )
            adjusted.append(
                self._replace_recommendation_score(
                    recommendation,
                    (
                        (focus_weights["base"] * base_score)
                        + artist_boost
                        + popularity_boost
                        + familiar_source_boost
                        + mood_boost
                        + novelty_boost
                        + search_discovery_boost
                        - recent_artist_penalty
                        - overpopular_penalty
                        - top_track_discovery_penalty
                    ),
                )
            )
        adjusted.sort(key=lambda item: (-item.final_score, item.item_id))
        return self._apply_artist_diversity_for_discovery(
            recommendations=adjusted,
            candidate_catalog=candidate_catalog,
            exploration_level=exploration_level,
            ranking_focus=ranking_focus,
        )

    def _build_candidate_catalog(
        self,
        candidate_payloads: list[dict[str, Any]],
        audio_features_payload: dict[str, Any],
        artist_metadata_payload: dict[str, Any],
        source_labels_by_track_id: dict[str, list[str]],
        ranking_mode: str,
    ) -> pd.DataFrame:
        """Normalize Spotify payloads into the recommender-compatible schema."""

        track_metadata_frame = self.preprocessor.normalize_track_metadata({"tracks": candidate_payloads})
        audio_features_frame = self.preprocessor.normalize_audio_features(audio_features_payload)
        artist_metadata_frame = self.preprocessor.normalize_artist_metadata(artist_metadata_payload)
        track_catalog = self.preprocessor.create_track_level_table(
            track_metadata_frame=track_metadata_frame,
            audio_features_frame=audio_features_frame,
            playlist_tracks_frame=None,
            artist_metadata_frame=artist_metadata_frame,
        )
        if track_catalog.empty:
            return track_catalog

        image_urls = {
            str(track_payload.get("id")): self._extract_album_image_url(track_payload)
            for track_payload in candidate_payloads
            if track_payload.get("id")
        }
        track_catalog["spotify_track_id"] = track_catalog["track_id"].astype(str)
        track_catalog["spotify_url"] = track_catalog.get("track_url", pd.Series(index=track_catalog.index, dtype=object))
        track_catalog["album_image_url"] = track_catalog["track_id"].astype(str).map(image_urls).fillna("")
        track_catalog["artist_name"] = track_catalog.get("primary_artist_name", track_catalog.get("artist_names", ""))
        track_catalog["catalog_popularity"] = pd.to_numeric(track_catalog.get("popularity", 0), errors="coerce").fillna(0.0) / 100.0
        track_catalog["catalog_novelty"] = 1.0 - track_catalog["catalog_popularity"]
        track_catalog["ranking_mode"] = ranking_mode
        for feature_name, default_value in {
            "danceability": 0.5,
            "energy": 0.5,
            "valence": 0.5,
            "tempo": 120.0,
            "acousticness": 0.5,
            "speechiness": 0.1,
            "instrumentalness": 0.0,
            "loudness": -10.0,
        }.items():
            if feature_name not in track_catalog.columns:
                track_catalog[feature_name] = default_value
        track_catalog["candidate_sources"] = track_catalog["track_id"].astype(str).map(
            lambda track_id: ", ".join(source_labels_by_track_id.get(track_id, ["spotify candidate"]))
        )
        return track_catalog

    def _build_recommendation_table(
        self,
        recommendations: list[HybridRecommendation],
        candidate_catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create a compact table for real Spotify recommendations."""

        metadata = candidate_catalog.set_index("track_id").to_dict(orient="index")
        rows: list[dict[str, object]] = []
        for recommendation in recommendations:
            track_row = metadata.get(recommendation.track_id, {})
            rows.append(
                {
                    "track_id": recommendation.track_id,
                    "track_name": recommendation.track_name,
                    "artist_name": recommendation.artist_name,
                    "spotify_url": track_row.get("spotify_url", ""),
                    "candidate_sources": track_row.get("candidate_sources", ""),
                    "final_score": round(recommendation.final_score, 3),
                    "content": round(recommendation.score_breakdown.content_score, 3),
                    "novelty": round(recommendation.score_breakdown.novelty_score, 3),
                    "discovery": round(recommendation.score_breakdown.discovery_score, 3),
                    "popularity_prior": round(recommendation.score_breakdown.popularity_prior, 3),
                }
            )
        return pd.DataFrame(rows)

    def _build_explanations(
        self,
        recommendations: list[HybridRecommendation],
        listening_history_snapshot: ListeningHistorySnapshot,
        candidate_set: SpotifyCandidateSet,
        mood_label: str,
        bucket_label: str = "Balanced",
    ) -> list[DemoRecommendationExplanation]:
        """Build item-specific real Spotify explanations."""

        catalog_lookup = candidate_set.track_catalog.set_index("track_id").to_dict(orient="index")
        recent_labels = [
            f"{track.track_name} - {track.artist_name}"
            for track in listening_history_snapshot.recent_tracks[:3]
        ]
        explanations: list[DemoRecommendationExplanation] = []
        for recommendation in recommendations:
            track_row = catalog_lookup.get(recommendation.track_id, {})
            sources = candidate_set.source_labels_by_track_id.get(recommendation.track_id, ["Spotify candidate"])
            bucket_rationale = self._build_bucket_rationale(bucket_label, track_row)
            mood_details = self._compute_mood_profile_details(track_row, mood_label)
            mood_score = float(mood_details["mood_score"])
            explanations.append(
                DemoRecommendationExplanation(
                    track_id=recommendation.track_id,
                    track_name=recommendation.track_name,
                    artist_name=recommendation.artist_name,
                    summary_lines=[
                        f"Real Spotify candidate ranked with score {recommendation.final_score:.3f}.",
                        bucket_rationale,
                    ],
                    spotify_rationale_lines=[
                        f"Recommended because it came from {', '.join(sources)}.",
                        f"Balances {mood_label.replace('_', ' ')} mood fit with Spotify listening similarity.",
                        f"Selected because it strongly matches the selected mood profile. Mood score: {mood_score:.2f}.",
                        self._describe_mood_fit(track_row, mood_label),
                    ],
                    spotify_recent_track_labels=recent_labels,
                    spotify_matched_seed_labels=[", ".join(sources)],
                    spotify_inferred_mood=mood_label,
                    spotify_taste_signals=self._build_taste_signal_labels(listening_history_snapshot),
                    spotify_url=str(track_row.get("spotify_url", "") or ""),
                    album_image_url=str(track_row.get("album_image_url", "") or ""),
                    recommendation_source=f"Spotify real-track recommendations | {bucket_label}",
                )
            )
        return explanations

    def _build_bucketed_explanations(
        self,
        candidate_catalog: pd.DataFrame,
        listening_history_snapshot: ListeningHistorySnapshot,
        candidate_set: SpotifyCandidateSet,
        mood_label: str,
        recommendation_count: int,
    ) -> dict[str, list[DemoRecommendationExplanation]]:
        """Return legacy bucket mapping for older UI/tests."""

        buckets = self._build_recommendation_buckets(
            candidate_catalog=candidate_catalog,
            listening_history_snapshot=listening_history_snapshot,
            candidate_set=candidate_set,
            exploration_level=0.5,
            mood_label=mood_label,
            recommendation_count=recommendation_count,
        )
        return {bucket.bucket_label: bucket.recommendations for bucket in buckets}

    def _build_recommendation_buckets(
        self,
        candidate_catalog: pd.DataFrame,
        listening_history_snapshot: ListeningHistorySnapshot,
        candidate_set: SpotifyCandidateSet,
        exploration_level: float,
        mood_label: str,
        recommendation_count: int,
    ) -> list[RecommendationBucket]:
        """Build distinct familiar, discovery, and mood-specific Spotify buckets."""

        if candidate_catalog.empty:
            return []

        per_bucket_count = max(int(recommendation_count), 1)
        bucket_specs = [
            (
                "familiar",
                "Familiar Picks",
                "Close to your recent artists/listening, favoring top-track and popularity signals.",
                self._rank_bucket_candidates(
                    candidate_catalog=candidate_catalog,
                    listening_history_snapshot=listening_history_snapshot,
                    exploration_level=exploration_level,
                    mood_label=mood_label,
                    bucket_label="Familiar",
                    limit=per_bucket_count,
                ),
            ),
            (
                "discovery",
                "Discovery Picks",
                "Adds novelty from search-discovered tracks, less familiar artists, and lower popularity.",
                self._rank_bucket_candidates(
                    candidate_catalog=candidate_catalog,
                    listening_history_snapshot=listening_history_snapshot,
                    exploration_level=exploration_level,
                    mood_label=mood_label,
                    bucket_label="Discovery",
                    limit=per_bucket_count,
                ),
            ),
            (
                "mood_based",
                "Mood-Based Picks",
                "Ranked for the selected mood using audio features when available or metadata-only signals.",
                self._rank_bucket_candidates(
                    candidate_catalog=candidate_catalog,
                    listening_history_snapshot=listening_history_snapshot,
                    exploration_level=exploration_level,
                    mood_label=mood_label,
                    bucket_label="Mood-based",
                    limit=per_bucket_count,
                ),
            ),
        ]
        return [
            RecommendationBucket(
                bucket_name=bucket_name,
                bucket_label=bucket_label,
                description=description,
                recommendations=self._build_explanations(
                    recommendations=recommendations,
                    listening_history_snapshot=listening_history_snapshot,
                    candidate_set=candidate_set,
                    mood_label=mood_label,
                    bucket_label=bucket_label,
                ),
            )
            for bucket_name, bucket_label, description, recommendations in bucket_specs
        ]

    def _rank_bucket_candidates(
        self,
        candidate_catalog: pd.DataFrame,
        listening_history_snapshot: ListeningHistorySnapshot,
        exploration_level: float,
        mood_label: str,
        bucket_label: str,
        limit: int,
    ) -> list[HybridRecommendation]:
        """Rank one explainable Spotify recommendation bucket."""

        bounded_exploration = min(max(float(exploration_level), 0.0), 1.0)
        familiarity_preference = 1.0 - bounded_exploration
        discovery_preference = bounded_exploration
        recent_artist_affinity = self._build_recent_artist_affinity(listening_history_snapshot)
        scored_recommendations: list[HybridRecommendation] = []
        seen_artist_counts: Counter[str] = Counter()
        for row in candidate_catalog.itertuples(index=False):
            track_row = row._asdict()
            artist_name = str(track_row.get("artist_name", "")).strip()
            source_labels = str(track_row.get("candidate_sources", "")).lower()
            popularity = float(track_row.get("catalog_popularity", 0.0) or 0.0)
            novelty = float(track_row.get("catalog_novelty", 0.0) or 0.0)
            is_top_track = "top track" in source_labels
            is_search = "search" in source_labels
            artist_affinity = recent_artist_affinity.get(artist_name, 0.0)
            mood_alignment = self._compute_mood_alignment(track_row, mood_label)
            if bucket_label == "Familiar":
                score = (
                    ((3.4 + (1.8 * familiarity_preference)) * artist_affinity)
                    + ((1.8 + (0.9 * familiarity_preference)) if is_top_track else 0.0)
                    + ((1.7 + (0.8 * familiarity_preference)) * popularity)
                    + (0.20 * mood_alignment)
                    - (0.65 * discovery_preference * novelty)
                )
            elif bucket_label == "Discovery":
                diversity_penalty = 0.75 * seen_artist_counts[artist_name]
                score = (
                    ((2.4 + (2.1 * discovery_preference)) * novelty)
                    + ((1.8 + (1.3 * discovery_preference)) if is_search else 0.0)
                    + (0.35 * mood_alignment)
                    - ((1.2 + (0.9 * discovery_preference)) * artist_affinity)
                    - (0.75 * popularity)
                    - diversity_penalty
                )
            else:
                score = (
                    (8.00 * mood_alignment)
                    + (0.08 * novelty)
                    + (0.03 * popularity)
                    + (0.08 if is_search else 0.0)
                )

            scored_recommendations.append(
                self._build_bucket_recommendation(
                    track_id=str(track_row.get("track_id", "")),
                    track_name=str(track_row.get("track_name", "")),
                    artist_name=artist_name,
                    score=score,
                )
            )
            if artist_name:
                seen_artist_counts[artist_name] += 1

        scored_recommendations.sort(key=lambda recommendation: (-recommendation.final_score, recommendation.track_id))
        if bucket_label == "Discovery":
            scored_recommendations = self.diversity_reranker.rerank(
                recommendations=scored_recommendations,
                candidate_catalog=candidate_catalog,
                exploration_level=max(bounded_exploration, 0.75),
                k=len(scored_recommendations),
                strength=1.0,
            )
        return scored_recommendations[:limit]

    def _build_bucket_recommendation(
        self,
        track_id: str,
        track_name: str,
        artist_name: str,
        score: float,
    ) -> HybridRecommendation:
        """Create a lightweight recommendation object for bucket display."""

        from models.hybrid_recommender import HybridScoreBreakdown

        score_breakdown = HybridScoreBreakdown(
            collaborative_score=0.0,
            content_score=max(score, 0.0),
            novelty_score=0.0,
            popularity_prior=0.0,
            discovery_score=0.0,
            final_score=score,
        )
        return HybridRecommendation(
            item_id=track_id,
            score=score,
            source="spotify_bucket",
            track_name=track_name,
            artist_name=artist_name,
            score_breakdown=score_breakdown,
            used_cold_start_fallback=False,
        )

    def _build_bucket_rationale(self, bucket_label: str, track_row: dict[str, object]) -> str:
        """Return concise bucket-specific recommendation rationale."""

        ranking_mode = str(track_row.get("ranking_mode", "metadata-only")).replace("-", " ")
        if bucket_label == "Familiar Picks":
            return "Similar to your recent artists/listening, with familiar Spotify candidate signals."
        if bucket_label == "Discovery Picks":
            return "Adds novelty from search-discovered candidates and more varied artists."
        if bucket_label == "Mood-Based Picks":
            return f"Ranked for the selected mood using {ranking_mode} signals."
        return "Ranking blends similarity to recent listening, artist affinity, mood fit, novelty, and popularity."

    def _build_playlist_candidate_frame(
        self,
        recommendations: list[HybridRecommendation],
        candidate_catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return candidate rows aligned to the recommendation set."""

        track_ids = [recommendation.track_id for recommendation in recommendations]
        return candidate_catalog.loc[candidate_catalog["track_id"].isin(track_ids)].copy()

    def _build_hybrid_weights(self, exploration_level: float) -> dict[str, float]:
        """Map the UI exploration slider into real-candidate ranking weights."""

        bounded_exploration = min(max(float(exploration_level), 0.0), 1.0)
        familiarity = 1.0 - bounded_exploration
        return {
            "collaborative": 0.0,
            "content": 1.15 + (0.65 * familiarity),
            "novelty": 0.25 + (1.35 * bounded_exploration),
            "popularity_prior": 0.40 * familiarity,
            "discovery": 0.25 + (1.05 * bounded_exploration),
        }

    def _build_focus_weights(self, ranking_focus: str) -> dict[str, float]:
        """Return simple multipliers for the optional ranking-focus control."""

        normalized_focus = ranking_focus.strip().lower()
        if normalized_focus == "familiar":
            return {"base": 0.65, "familiar": 2.35, "discovery": 0.35, "mood": 0.85}
        if normalized_focus == "discovery":
            return {"base": 0.55, "familiar": 0.30, "discovery": 2.60, "mood": 0.85}
        if normalized_focus == "mood-first":
            return {"base": 0.25, "familiar": 0.35, "discovery": 0.45, "mood": 4.20}
        return {"base": 1.50, "familiar": 1.0, "discovery": 1.0, "mood": 0.25}

    def _normalize_recommendation_scores(
        self,
        recommendations: list[HybridRecommendation],
    ) -> dict[str, float]:
        """Normalize base recommender scores so mood profiles can dominate when requested."""

        if not recommendations:
            return {}
        scores = [float(recommendation.final_score) for recommendation in recommendations]
        minimum_score = min(scores)
        maximum_score = max(scores)
        if maximum_score == minimum_score:
            return {recommendation.track_id: 0.5 for recommendation in recommendations}
        return {
            recommendation.track_id: (float(recommendation.final_score) - minimum_score)
            / (maximum_score - minimum_score)
            for recommendation in recommendations
        }

    def _build_diversity_strength(self, ranking_focus: str) -> float:
        """Return how strongly final reranking should enforce diversity."""

        normalized_focus = ranking_focus.strip().lower()
        if normalized_focus == "discovery":
            return 1.0
        if normalized_focus == "familiar":
            return 0.45
        if normalized_focus == "mood-first":
            return 0.35
        return 0.75

    def _apply_artist_diversity_for_discovery(
        self,
        recommendations: list[HybridRecommendation],
        candidate_catalog: pd.DataFrame,
        exploration_level: float,
        ranking_focus: str,
    ) -> list[HybridRecommendation]:
        """Downweight repeated artists when exploration/discovery is requested."""

        if exploration_level < 0.5 and ranking_focus.strip().lower() != "discovery":
            return recommendations

        catalog_lookup = candidate_catalog.set_index("track_id").to_dict(orient="index")
        seen_artist_counts: Counter[str] = Counter()
        diversified: list[HybridRecommendation] = []
        normalized_focus = ranking_focus.strip().lower()
        focus_multiplier = 1.85 if normalized_focus == "discovery" else 1.0
        if normalized_focus == "mood-first":
            focus_multiplier = 0.65
        for recommendation in recommendations:
            artist_name = str(catalog_lookup.get(recommendation.track_id, {}).get("artist_name", "")).strip()
            penalty = 0.55 * focus_multiplier * exploration_level * seen_artist_counts[artist_name]
            diversified.append(
                self._replace_recommendation_score(
                    recommendation,
                    recommendation.final_score - penalty,
                )
            )
            if artist_name:
                seen_artist_counts[artist_name] += 1
        diversified.sort(key=lambda item: (-item.final_score, item.item_id))
        return diversified

    def _count_position_changes(
        self,
        before_track_ids: list[str],
        after_track_ids: list[str],
    ) -> int:
        """Count how many top-list positions changed after control reranking."""

        max_length = max(len(before_track_ids), len(after_track_ids))
        return sum(
            1
            for index in range(max_length)
            if (
                before_track_ids[index] if index < len(before_track_ids) else None
            )
            != (
                after_track_ids[index] if index < len(after_track_ids) else None
            )
        )

    def _build_mood_debug_summary(
        self,
        *,
        recommendations: list[HybridRecommendation],
        candidate_catalog: pd.DataFrame,
        mood_label: str,
    ) -> dict[str, object]:
        """Return compact mood-profile diagnostics for the top ranked candidates."""

        if not recommendations or candidate_catalog.empty:
            return {}
        catalog_lookup = candidate_catalog.set_index("track_id").to_dict(orient="index")
        top_mood_scores: dict[str, float] = {}
        first_details: dict[str, object] = {}
        for recommendation in recommendations[:10]:
            details = self._compute_mood_profile_details(
                catalog_lookup.get(recommendation.track_id, {}),
                mood_label,
            )
            if not first_details:
                first_details = details
            top_mood_scores[recommendation.track_id] = round(float(details["mood_score"]), 3)
        return {
            "mood_score": next(iter(top_mood_scores.values()), 0.0),
            "top_mood_scores_after_reranking": top_mood_scores,
            "mood_profile_used": first_details.get("mood_profile_used", mood_label),
            "mood_matching_mode": first_details.get("mood_matching_mode", "metadata_only"),
            "positive_mood_signals": first_details.get("positive_mood_signals", []),
            "negative_mood_penalties": first_details.get("negative_mood_penalties", []),
        }

    def _build_popularity_scores(self, candidate_catalog: pd.DataFrame) -> dict[str, float]:
        """Return normalized Spotify popularity priors."""

        return {
            str(row.track_id): float(row.catalog_popularity)
            for row in candidate_catalog.itertuples(index=False)
        }

    def _apply_optional_advanced_scores(
        self,
        recommendations: list[HybridRecommendation],
        *,
        als_scores: dict[str, float] | None = None,
        embedding_scores: dict[str, float] | None = None,
    ) -> list[HybridRecommendation]:
        """Blend optional ALS/embedding score maps into Spotify real-track ranking."""

        if not recommendations or not als_scores and not embedding_scores:
            return recommendations

        advanced_ranker = AdvancedHybridRankingService(
            weights=AdvancedHybridWeights(
                hybrid=1.0,
                als=0.35 if als_scores else 0.0,
                embedding=0.35 if embedding_scores else 0.0,
            )
        )
        advanced_recommendations = advanced_ranker.rerank(
            recommendations,
            als_scores=als_scores,
            embedding_scores=embedding_scores,
            k=len(recommendations),
        )
        recommendations_by_id = {recommendation.track_id: recommendation for recommendation in recommendations}
        return [
            self._replace_recommendation_score(
                recommendations_by_id[advanced_recommendation.item_id],
                advanced_recommendation.score,
            )
            for advanced_recommendation in advanced_recommendations
            if advanced_recommendation.item_id in recommendations_by_id
        ]

    def _build_candidate_search_queries(
        self,
        *,
        recent_artist_names: list[str],
        recent_track_search_terms: list[tuple[str, str]],
        genre_search_terms: list[str],
    ) -> list[tuple[str, str, int]]:
        """Build bounded Spotify search queries from recent artists, tracks, and genres."""

        queries: list[tuple[str, str, int]] = []
        for artist_name in recent_artist_names[: self.max_recent_artists]:
            safe_artist = self._escape_spotify_search_term(artist_name)
            queries.append((f'artist:"{safe_artist}"', "recent artist search match", self.search_tracks_per_artist))

        for track_name, artist_name in recent_track_search_terms[: self.max_recent_track_searches]:
            safe_track = self._escape_spotify_search_term(track_name)
            safe_artist = self._escape_spotify_search_term(artist_name)
            query = f'track:"{safe_track}"'
            if safe_artist:
                query = f'{query} artist:"{safe_artist}"'
            queries.append((query, "recent track search match", self.search_tracks_per_recent_track))

        for genre_name in genre_search_terms[: self.genre_search_limit]:
            safe_genre = self._escape_spotify_search_term(genre_name)
            queries.append((f'genre:"{safe_genre}"', "genre search match", self.genre_search_limit))

        unique_queries: list[tuple[str, str, int]] = []
        seen_queries: set[str] = set()
        for query, source_label, search_limit in queries:
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            unique_queries.append((query, source_label, search_limit))
        return unique_queries

    def _escape_spotify_search_term(self, value: str) -> str:
        """Remove quotes from a term before placing it in a Spotify search query."""

        return value.replace('"', " ").strip()

    def _extract_recent_track_search_terms(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> list[tuple[str, str]]:
        """Return recent track title/artist pairs for bounded search expansion."""

        terms: list[tuple[str, str]] = []
        for track in listening_history_snapshot.recent_tracks:
            track_name = str(track.track_name).strip()
            artist_name = str(track.artist_name).split(",", maxsplit=1)[0].strip()
            if track_name:
                terms.append((track_name, artist_name))
        seen: set[tuple[str, str]] = set()
        unique_terms: list[tuple[str, str]] = []
        for track_name, artist_name in terms:
            key = (track_name.lower(), artist_name.lower())
            if key in seen:
                continue
            seen.add(key)
            unique_terms.append((track_name, artist_name))
        return unique_terms

    def _extract_recent_genre_terms(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> list[str]:
        """Extract lightweight genre-like terms from recent normalized metadata."""

        frame = listening_history_snapshot.track_level_frame
        if frame.empty or "artist_genres" not in frame.columns:
            return []

        genre_terms: list[str] = []
        for raw_value in frame["artist_genres"].dropna().astype(str).tolist():
            cleaned_value = raw_value.replace("[", "").replace("]", "").replace("'", "")
            for genre_name in re.split(r"[,|;/]", cleaned_value):
                normalized_genre = genre_name.strip()
                if normalized_genre:
                    genre_terms.append(normalized_genre)
        return self._deduplicate(genre_terms)

    def _build_name_artist_dedupe_key(self, track_payload: dict[str, Any]) -> tuple[str, str]:
        """Build a normalized fallback duplicate key from title and primary artist."""

        track_id = str(track_payload.get("id", "")).strip()
        track_name = self._normalize_candidate_text(str(track_payload.get("name", "")))
        artist_name = ""
        artists = track_payload.get("artists", [])
        if artists:
            artist_name = self._normalize_candidate_text(str(artists[0].get("name", "")))
        if not track_name and not artist_name:
            return (track_id, "")
        return (track_name, artist_name)

    def _normalize_candidate_text(self, value: str) -> str:
        """Normalize titles/artists for duplicate checks across versions/remixes."""

        normalized_value = value.lower()
        normalized_value = re.sub(r"\([^)]*(live|remix|edit|version|mono|stereo)[^)]*\)", " ", normalized_value)
        normalized_value = re.sub(r"\[[^]]*(live|remix|edit|version|mono|stereo)[^]]*\]", " ", normalized_value)
        normalized_value = re.sub(r"\b(live|remix|radio edit|edit|version|mono|stereo|remastered)\b", " ", normalized_value)
        normalized_value = re.sub(r"[^a-z0-9]+", " ", normalized_value)
        return " ".join(normalized_value.split())

    def _build_artist_candidate_counts(self, candidate_payloads: list[dict[str, Any]]) -> Counter[str]:
        """Count how many raw candidates each primary artist contributes."""

        counts: Counter[str] = Counter()
        for track_payload in candidate_payloads:
            artists = track_payload.get("artists", [])
            artist_name = ""
            if artists:
                artist_name = str(artists[0].get("name", "")).strip()
            if artist_name:
                counts[artist_name] += 1
        return counts

    def _build_source_breakdown(self, source_labels_by_track_id: dict[str, list[str]]) -> dict[str, int]:
        """Count unique candidates by source label for debug summaries."""

        source_counts: Counter[str] = Counter()
        for source_labels in source_labels_by_track_id.values():
            for source_label in source_labels or ["spotify candidate"]:
                source_counts[source_label] += 1
        return dict(sorted(source_counts.items()))

    def _add_artist_count_metadata(self, track_catalog: pd.DataFrame) -> None:
        """Annotate candidate catalog rows with artist pool concentration metadata."""

        if track_catalog.empty or "artist_name" not in track_catalog.columns:
            return
        artist_counts = track_catalog["artist_name"].fillna("").astype(str).value_counts()
        track_catalog["candidate_artist_pool_count"] = track_catalog["artist_name"].fillna("").astype(str).map(artist_counts).fillna(0).astype(int)
        total_candidates = max(len(track_catalog), 1)
        track_catalog["candidate_artist_pool_share"] = track_catalog["candidate_artist_pool_count"] / total_candidates

    def _add_candidate_payload(
        self,
        candidate_payloads: list[dict[str, Any]],
        source_labels_by_track_id: dict[str, list[str]],
        track_payload: dict[str, Any],
        source_label: str,
        dedup_state: CandidateDeduplicationState | None = None,
    ) -> None:
        """Add one candidate payload while preserving first-seen order."""

        track_id = str(track_payload.get("id", "")).strip()
        if not track_id:
            return
        if dedup_state is not None:
            dedup_state.raw_candidate_count += 1
            name_artist_key = self._build_name_artist_dedupe_key(track_payload)
            if track_id in dedup_state.track_ids_seen:
                dedup_state.duplicate_candidates_removed += 1
            elif name_artist_key in dedup_state.track_ids_by_name_artist:
                canonical_track_id = dedup_state.track_ids_by_name_artist[name_artist_key]
                dedup_state.duplicate_candidates_removed += 1
                if source_label not in source_labels_by_track_id.get(canonical_track_id, []):
                    source_labels_by_track_id.setdefault(canonical_track_id, []).append(source_label)
                return
            else:
                dedup_state.track_ids_seen.add(track_id)
                dedup_state.track_ids_by_name_artist[name_artist_key] = track_id
        if track_id not in source_labels_by_track_id:
            candidate_payloads.append(track_payload)
            source_labels_by_track_id[track_id] = []
        if source_label not in source_labels_by_track_id[track_id]:
            source_labels_by_track_id[track_id].append(source_label)

    def _safe_get_artists(
        self,
        artist_ids: list[str],
        access_token: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        """Fetch artist metadata without making candidates depend on it."""

        try:
            return self.client.get_artists(artist_ids, access_token=access_token)
        except SpotifyAPIClientError as error:
            LOGGER.info("Spotify artist metadata unavailable: %s", error)
            warnings.append("Spotify artist metadata was unavailable, so ranking used track metadata only.")
            return {"artists": []}

    def _build_candidate_tracks(
        self,
        track_catalog: pd.DataFrame,
        source_labels_by_track_id: dict[str, list[str]],
    ) -> list[SpotifyCandidateTrack]:
        """Return UI-friendly candidate representations."""

        candidates: list[SpotifyCandidateTrack] = []
        for row in track_catalog.itertuples(index=False):
            track_id = str(row.track_id)
            candidates.append(
                SpotifyCandidateTrack(
                    spotify_track_id=track_id,
                    track_name=str(row.track_name),
                    artist_name=str(row.artist_name),
                    spotify_url=str(getattr(row, "spotify_url", "") or ""),
                    album_image_url=str(getattr(row, "album_image_url", "") or ""),
                    popularity=self._safe_float(getattr(row, "popularity", None)),
                    source=", ".join(source_labels_by_track_id.get(track_id, ["spotify candidate"])),
                )
            )
        return candidates

    def _extract_recent_artist_ids(self, listening_history_snapshot: ListeningHistorySnapshot) -> list[str]:
        """Extract recent artist IDs from normalized Spotify history."""

        if "primary_artist_id" not in listening_history_snapshot.track_level_frame.columns:
            return []
        return self._deduplicate(
            listening_history_snapshot.track_level_frame["primary_artist_id"].dropna().astype(str).tolist()
        )

    def _looks_like_spotify_id(self, value: object) -> bool:
        """Return whether a value looks like a Spotify base62 identifier."""

        return bool(SPOTIFY_ID_PATTERN.fullmatch(str(value).strip()))

    def _extract_recent_artist_names(self, listening_history_snapshot: ListeningHistorySnapshot) -> list[str]:
        """Extract recent artist names from display summaries."""

        return self._deduplicate(
            [
                artist_name.strip()
                for track in listening_history_snapshot.recent_tracks
                for artist_name in str(track.artist_name).split(",")
                if artist_name.strip()
            ]
        )

    def _extract_artist_ids_from_track_payloads(self, track_payloads: list[dict[str, Any]]) -> list[str]:
        """Return unique artist IDs from candidate track payloads."""

        return self._deduplicate(
            [
                str(artist.get("id", "")).strip()
                for track in track_payloads
                for artist in track.get("artists", [])
                if str(artist.get("id", "")).strip()
            ]
        )

    def _build_recent_artist_affinity(self, listening_history_snapshot: ListeningHistorySnapshot) -> dict[str, float]:
        """Build recency-weighted artist affinity from recent plays."""

        counts: Counter[str] = Counter()
        total = 0.0
        for index, track in enumerate(listening_history_snapshot.recent_tracks):
            weight = max(len(listening_history_snapshot.recent_tracks) - index, 1)
            for artist_name in str(track.artist_name).split(","):
                normalized_artist = artist_name.strip()
                if not normalized_artist:
                    continue
                counts[normalized_artist] += weight
                total += weight
        if total <= 0.0:
            return {}
        return {artist_name: count / total for artist_name, count in counts.items()}

    def _compute_mood_alignment(self, track_row: dict[str, object], mood_label: str) -> float:
        """Return a calibrated mood-fit score for real Spotify tracks."""

        return float(self._compute_mood_profile_details(track_row, mood_label)["mood_score"])

    def _compute_mood_profile_details(
        self,
        track_row: dict[str, object],
        mood_label: str,
    ) -> dict[str, object]:
        """Return score and diagnostics for the selected mood ranking profile."""

        mood = mood_label.strip().lower()
        if str(track_row.get("ranking_mode", "")).strip() == "metadata-only":
            return self._compute_metadata_mood_profile_details(track_row, mood)

        energy = self._safe_float_or(track_row.get("energy"), 0.5)
        danceability = self._safe_float_or(track_row.get("danceability"), 0.5)
        valence = self._safe_float_or(track_row.get("valence"), 0.5)
        acousticness = self._safe_float_or(track_row.get("acousticness"), 0.5)
        instrumentalness = self._safe_float_or(track_row.get("instrumentalness"), 0.0)
        tempo = self._safe_float_or(track_row.get("tempo"), 120.0)
        tempo_fast = min(max((tempo - 90.0) / 70.0, 0.0), 1.0)
        tempo_slow = 1.0 - min(max((tempo - 70.0) / 80.0, 0.0), 1.0)
        profile = self._build_audio_mood_profile(
            energy=energy,
            danceability=danceability,
            valence=valence,
            acousticness=acousticness,
            instrumentalness=instrumentalness,
            tempo_fast=tempo_fast,
            tempo_slow=tempo_slow,
            popularity=self._safe_float_or(track_row.get("catalog_popularity"), 0.0),
        )
        if mood == "workout":
            return self._build_audio_mood_details(
                mood,
                profile,
                positive_weights={"energy": 0.38, "danceability": 0.30, "tempo_fast": 0.22},
                penalty_weights={"acousticness": 0.25, "instrumentalness": 0.20, "low_energy": 0.35, "tempo_slow": 0.20},
            )
        if mood == "happy":
            return self._build_audio_mood_details(
                mood,
                profile,
                positive_weights={"valence": 0.48, "energy": 0.25, "danceability": 0.20, "tempo_fast": 0.07},
                penalty_weights={"low_valence": 0.65, "low_energy": 0.35},
            )
        if mood == "melancholic":
            return self._build_audio_mood_details(
                mood,
                profile,
                positive_weights={"low_valence": 0.38, "acousticness": 0.22, "medium_low_energy": 0.25, "tempo_slow": 0.15},
                penalty_weights={"valence": 0.35, "danceability": 0.30, "energy": 0.20, "tempo_fast": 0.15},
            )
        if mood == "party":
            return self._build_audio_mood_details(
                mood,
                profile,
                positive_weights={"energy": 0.32, "danceability": 0.32, "tempo_fast": 0.20, "popularity": 0.16},
                penalty_weights={"low_energy": 0.40, "acousticness": 0.30, "tempo_slow": 0.30},
            )
        if mood == "study":
            return self._build_audio_mood_details(
                mood,
                profile,
                positive_weights={"instrumentalness": 0.35, "acousticness": 0.25, "low_energy": 0.22, "tempo_slow": 0.18},
                penalty_weights={"danceability": 0.30, "energy": 0.35, "tempo_fast": 0.35},
            )
        return self._build_audio_mood_details(
            "calm",
            profile,
            positive_weights={"acousticness": 0.30, "instrumentalness": 0.25, "low_energy": 0.25, "tempo_slow": 0.20},
            penalty_weights={"energy": 0.40, "tempo_fast": 0.25, "danceability": 0.20, "not_acoustic": 0.15},
        )

    def _build_audio_mood_details(
        self,
        mood: str,
        profile: dict[str, float],
        *,
        positive_weights: dict[str, float],
        penalty_weights: dict[str, float],
    ) -> dict[str, object]:
        """Build calibrated audio mood score diagnostics."""

        positive_score = self._weighted_average(profile, positive_weights)
        penalty_score = self._weighted_average(profile, penalty_weights)
        mood_score = self._calibrate_mood_score(positive_score, penalty_score)
        return {
            "mood_score": mood_score,
            "mood_profile_used": mood,
            "mood_matching_mode": "audio_features",
            "positive_mood_signals": self._top_weighted_signal_labels(profile, positive_weights),
            "negative_mood_penalties": self._top_weighted_signal_labels(profile, penalty_weights),
        }

    def _calibrate_mood_score(self, positive_score: float, penalty_score: float) -> float:
        """Convert positive fit and negative penalties into one dominant profile score."""

        score = (0.72 * positive_score) + (0.28 * (1.0 - penalty_score))
        return min(max(float(score), 0.0), 1.0)

    def _top_weighted_signal_labels(
        self,
        values: dict[str, float],
        weights: dict[str, float],
    ) -> list[str]:
        """Return compact mood signal labels ordered by contribution."""

        contributions = [
            (signal_name, values.get(signal_name, 0.0) * weight)
            for signal_name, weight in weights.items()
        ]
        contributions.sort(key=lambda item: (-item[1], item[0]))
        return [
            signal_name.replace("_", " ")
            for signal_name, contribution in contributions[:3]
            if contribution > 0.05
        ]

    def _build_audio_mood_profile(
        self,
        *,
        energy: float,
        danceability: float,
        valence: float,
        acousticness: float,
        instrumentalness: float,
        tempo_fast: float,
        tempo_slow: float,
        popularity: float,
    ) -> dict[str, float]:
        """Return reusable normalized audio mood signals."""

        return {
            "energy": min(max(energy, 0.0), 1.0),
            "danceability": min(max(danceability, 0.0), 1.0),
            "valence": min(max(valence, 0.0), 1.0),
            "acousticness": min(max(acousticness, 0.0), 1.0),
            "instrumentalness": min(max(instrumentalness, 0.0), 1.0),
            "tempo_fast": min(max(tempo_fast, 0.0), 1.0),
            "tempo_slow": min(max(tempo_slow, 0.0), 1.0),
            "popularity": min(max(popularity, 0.0), 1.0),
            "not_acoustic": 1.0 - min(max(acousticness, 0.0), 1.0),
            "not_low_energy": 1.0 if energy >= 0.55 else energy / 0.55,
            "low_energy": 1.0 - min(max(energy, 0.0), 1.0),
            "medium_low_energy": 1.0 - abs(min(max(energy, 0.0), 1.0) - 0.35),
            "low_valence": 1.0 - min(max(valence, 0.0), 1.0),
            "softness": (1.0 - min(max(energy, 0.0), 1.0) + min(max(acousticness, 0.0), 1.0)) / 2.0,
            "not_dance": 1.0 - min(max(danceability, 0.0), 1.0),
        }

    def _weighted_average(self, values: dict[str, float], weights: dict[str, float]) -> float:
        """Compute a bounded weighted average for mood profiles."""

        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            return 0.0
        score = sum(values.get(name, 0.0) * weight for name, weight in weights.items()) / total_weight
        return min(max(float(score), 0.0), 1.0)

    def _compute_metadata_mood_alignment(self, track_row: dict[str, object], mood: str) -> float:
        """Infer mood fit from text/source metadata when audio features are missing."""

        return float(self._compute_metadata_mood_profile_details(track_row, mood)["mood_score"])

    def _compute_metadata_mood_profile_details(
        self,
        track_row: dict[str, object],
        mood: str,
    ) -> dict[str, object]:
        """Infer mood profile details from text/source metadata when audio features are missing."""

        text = " ".join(
            [
                str(track_row.get("track_name", "")),
                str(track_row.get("artist_name", "")),
                str(track_row.get("artist_genres", "")),
                str(track_row.get("candidate_sources", "")),
                str(track_row.get("search_query", "")),
            ]
        ).lower()
        mood_terms = {
            "workout": ("workout", "run", "running", "gym", "power", "hype", "pump", "energy", "beat", "dance"),
            "happy": ("happy", "sunshine", "love", "good", "smile", "joy", "sweet", "dance", "bright"),
            "melancholic": ("sad", "lonely", "blue", "heartbreak", "cry", "rain", "empty", "slow", "acoustic"),
            "party": ("party", "club", "dance", "remix", "beat", "night", "dj", "hype"),
            "study": ("acoustic", "piano", "instrumental", "soft", "calm", "ambient", "chill", "study", "lo-fi", "lofi", "sleep"),
            "calm": ("acoustic", "piano", "instrumental", "soft", "calm", "ambient", "chill", "study", "lo-fi", "lofi", "sleep"),
        }
        anti_terms = {
            "workout": ("sleep", "lullaby", "acoustic", "piano", "slow", "calm", "sad"),
            "happy": ("sad", "lonely", "cry", "heartbreak", "dark"),
            "melancholic": ("party", "hype", "club", "happy", "sunshine", "dance"),
            "party": ("acoustic", "piano", "sleep", "calm", "sad", "slow"),
            "study": ("party", "hype", "club", "dance", "remix", "workout"),
            "calm": ("party", "hype", "club", "dance", "remix", "workout"),
        }
        terms = mood_terms.get(mood, mood_terms["calm"])
        blockers = anti_terms.get(mood, anti_terms["calm"])
        matched_terms = [term for term in terms if term in text]
        matched_blockers = [term for term in blockers if term in text]
        matches = len(matched_terms)
        anti_matches = len(matched_blockers)
        source_bonus = 0.12 if "search" in text else 0.0
        genre_bonus = 0.15 if any(term in text for term in terms[:4]) else 0.0
        positive_score = min(1.0, (0.22 * matches) + source_bonus + genre_bonus)
        penalty_score = min(1.0, 0.28 * anti_matches)
        return {
            "mood_score": self._calibrate_mood_score(positive_score, penalty_score),
            "mood_profile_used": mood,
            "mood_matching_mode": "metadata_only",
            "positive_mood_signals": matched_terms[:4],
            "negative_mood_penalties": matched_blockers[:4],
        }

    def _describe_mood_fit(self, track_row: dict[str, object], mood_label: str) -> str:
        """Return a concise explanation of why a track fits the selected mood."""

        mood = mood_label.strip().lower()
        if str(track_row.get("ranking_mode", "")).strip() == "metadata-only":
            return "Mood fit used metadata-only signals."
        if mood == "workout":
            return "Mood fit used audio features: high energy, danceability, and faster tempo."
        if mood == "happy":
            return "Mood fit used audio features: positive valence with medium/high energy."
        if mood == "melancholic":
            return "Mood fit used audio features: lower valence with softer, slower energy."
        if mood == "party":
            return "Mood fit used audio features: high energy, danceability, and party tempo."
        if mood == "study":
            return "Mood fit used audio features: calmer energy with acoustic or instrumental signals."
        return "Mood fit used audio features: lower energy, slower tempo, and calmer acoustic signals."

    def _build_taste_signal_labels(self, listening_history_snapshot: ListeningHistorySnapshot) -> list[str]:
        """Summarize recent taste signals for explanations."""

        frame = listening_history_snapshot.track_level_frame
        if frame.empty:
            return ["metadata-only recent listening"]
        signals: list[str] = []
        for column in ["energy", "valence", "danceability"]:
            if column in frame.columns:
                value = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not value.empty:
                    signals.append(f"{column} {float(value.mean()):.2f}")
        return signals[:3] or ["metadata-only recent listening"]

    def _replace_recommendation_score(
        self,
        recommendation: HybridRecommendation,
        score: float,
    ) -> HybridRecommendation:
        """Return a recommendation with an adjusted final score."""

        return HybridRecommendation(
            item_id=recommendation.item_id,
            score=score,
            source=recommendation.source,
            track_name=recommendation.track_name,
            artist_name=recommendation.artist_name,
            score_breakdown=recommendation.score_breakdown,
            used_cold_start_fallback=recommendation.used_cold_start_fallback,
        )

    def _extract_album_image_url(self, track_payload: dict[str, Any]) -> str:
        """Return the largest album image URL when Spotify provides one."""

        images = (track_payload.get("album") or {}).get("images", [])
        if not images:
            return ""
        return str(images[0].get("url", ""))

    def _safe_float(self, value: object) -> float | None:
        """Safely coerce a sparse Spotify value into a float."""

        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            return None
        return float(numeric_value)

    def _safe_float_or(self, value: object, default: float) -> float:
        """Coerce a value to float while preserving legitimate zero values."""

        numeric_value = self._safe_float(value)
        if numeric_value is None:
            return default
        return numeric_value

    def _deduplicate(self, values: list[str]) -> list[str]:
        """Deduplicate values while preserving order."""

        seen: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values
