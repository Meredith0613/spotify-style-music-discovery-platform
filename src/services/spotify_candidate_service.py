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
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter
from services.user_profile_service import ListeningHistorySnapshot


LOGGER = logging.getLogger(__name__)
SPOTIFY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{22}$")


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
class SpotifyRealRecommendationResult:
    """Store a UI-ready real Spotify recommendation view."""

    view_state: DemoViewState
    candidate_set: SpotifyCandidateSet
    source_message: str
    bucketed_explanations: dict[str, list[DemoRecommendationExplanation]] = field(default_factory=dict)


@dataclass(slots=True)
class SpotifyCandidateService:
    """Build and rank real Spotify candidates while keeping Streamlit thin."""

    client: SpotifyAPIClient
    preprocessor: Preprocessor
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    playlist_generator: PlaylistGenerator = field(default_factory=PlaylistGenerator)
    recommendation_adapter: SpotifyRecommendationAdapter = field(default_factory=SpotifyRecommendationAdapter)
    max_recent_artists: int = 5
    top_tracks_per_artist: int = 6
    search_tracks_per_artist: int = 4
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
        bucketed_explanations = self._build_bucketed_explanations(
            candidate_catalog=candidate_set.track_catalog,
            listening_history_snapshot=listening_history_snapshot,
            candidate_set=candidate_set,
            mood_label=mood_label,
            recommendation_count=recommendation_count,
        )
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
        warnings: list[str] = []
        recent_artist_ids = self._extract_recent_artist_ids(listening_history_snapshot)
        recent_artist_names = self._extract_recent_artist_names(listening_history_snapshot)
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
                )

        for artist_name in recent_artist_names[: self.max_recent_artists]:
            try:
                payload = self.client.search_tracks(
                    f'artist:"{artist_name}"',
                    access_token=access_token,
                    limit=self.search_tracks_per_artist,
                    market=self.default_market,
                )
            except SpotifyAPIClientError as error:
                search_requests_failed += 1
                LOGGER.info("Skipping Spotify track search for %s: %s", artist_name, error)
                continue
            for track_payload in payload.get("tracks", {}).get("items", []):
                search_candidates_found += 1
                self._add_candidate_payload(
                    candidate_payloads,
                    source_labels_by_track_id,
                    track_payload,
                    source_label="recent artist search match",
                )

        candidate_payloads = candidate_payloads[: self.max_candidates]
        debug_summary = {
            "candidate_count": len(candidate_payloads),
            "top_track_candidate_count": top_track_candidates_found,
            "search_candidate_count": search_candidates_found,
            "skipped_artist_expansion_count": top_track_requests_failed,
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
        after_track_ids = [recommendation.track_id for recommendation in reranked[:10]]
        debug_summary.update(
            {
                "selected_mood": mood_label,
                "exploration_level": f"{float(exploration_level):.2f}",
                "recommendation_count": recommendation_count,
                "ranking_focus": ranking_focus,
                "top_candidate_ids_before_reranking": before_track_ids,
                "top_candidate_ids_after_reranking": after_track_ids,
                "positions_changed_after_reranking": self._count_position_changes(
                    before_track_ids,
                    after_track_ids,
                ),
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
        adjusted: list[HybridRecommendation] = []
        for recommendation in recommendations:
            track_row = catalog_lookup.get(recommendation.track_id, {})
            artist_name = str(track_row.get("artist_name", "")).strip()
            source_labels = str(track_row.get("candidate_sources", ""))
            popularity = float(track_row.get("catalog_popularity", 0.0) or 0.0)
            novelty = float(track_row.get("catalog_novelty", 0.0) or 0.0)
            familiarity = 1.0 - min(max(float(exploration_level), 0.0), 1.0)
            is_search_candidate = "search" in source_labels.lower()
            is_top_track_candidate = "top track" in source_labels.lower()
            familiar_source_boost = 1.15 * focus_weights["familiar"] * familiarity if is_top_track_candidate else 0.0
            artist_boost = 1.35 * focus_weights["familiar"] * familiarity * recent_artist_affinity.get(artist_name, 0.0)
            popularity_boost = 0.90 * focus_weights["familiar"] * familiarity * popularity
            mood_boost = 1.75 * focus_weights["mood"] * self._compute_mood_alignment(track_row, mood_label)
            novelty_boost = 1.55 * focus_weights["discovery"] * exploration_level * novelty
            search_discovery_boost = 1.20 * focus_weights["discovery"] * exploration_level if is_search_candidate else 0.0
            overfamiliar_penalty = 0.70 * focus_weights["discovery"] * exploration_level * popularity
            adjusted.append(
                self._replace_recommendation_score(
                    recommendation,
                    (
                        recommendation.final_score
                        + artist_boost
                        + popularity_boost
                        + familiar_source_boost
                        + mood_boost
                        + novelty_boost
                        + search_discovery_boost
                        - overfamiliar_penalty
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
        """Build distinct familiar, discovery, and mood-specific Spotify buckets."""

        if candidate_catalog.empty:
            return {}

        per_bucket_count = max(int(recommendation_count), 1)
        bucketed_recommendations = {
            "Familiar picks": self._rank_bucket_candidates(
                candidate_catalog=candidate_catalog,
                listening_history_snapshot=listening_history_snapshot,
                mood_label=mood_label,
                bucket_label="Familiar",
                limit=per_bucket_count,
            ),
            "Discovery picks": self._rank_bucket_candidates(
                candidate_catalog=candidate_catalog,
                listening_history_snapshot=listening_history_snapshot,
                mood_label=mood_label,
                bucket_label="Discovery",
                limit=per_bucket_count,
            ),
            "Mood-based picks": self._rank_bucket_candidates(
                candidate_catalog=candidate_catalog,
                listening_history_snapshot=listening_history_snapshot,
                mood_label=mood_label,
                bucket_label="Mood-based",
                limit=per_bucket_count,
            ),
        }
        return {
            label: self._build_explanations(
                recommendations=recommendations,
                listening_history_snapshot=listening_history_snapshot,
                candidate_set=candidate_set,
                mood_label=mood_label,
                bucket_label=label,
            )
            for label, recommendations in bucketed_recommendations.items()
        }

    def _rank_bucket_candidates(
        self,
        candidate_catalog: pd.DataFrame,
        listening_history_snapshot: ListeningHistorySnapshot,
        mood_label: str,
        bucket_label: str,
        limit: int,
    ) -> list[HybridRecommendation]:
        """Rank one explainable Spotify recommendation bucket."""

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
            if bucket_label == "Familiar":
                score = (2.0 * artist_affinity) + (1.25 if is_top_track else 0.0) + (1.15 * popularity)
            elif bucket_label == "Discovery":
                diversity_penalty = 0.45 * seen_artist_counts[artist_name]
                score = (1.8 * novelty) + (1.4 if is_search else 0.0) - (0.65 * popularity) - diversity_penalty
            else:
                score = (2.8 * self._compute_mood_alignment(track_row, mood_label)) + (0.25 * novelty)

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
        if bucket_label == "Familiar picks":
            return "Similar to your recent artists/listening, with familiar Spotify candidate signals."
        if bucket_label == "Discovery picks":
            return "Adds novelty from search-discovered candidates and more varied artists."
        if bucket_label == "Mood-based picks":
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
            return {"familiar": 1.65, "discovery": 0.65, "mood": 0.80}
        if normalized_focus == "discovery":
            return {"familiar": 0.55, "discovery": 1.75, "mood": 0.85}
        if normalized_focus == "mood-first":
            return {"familiar": 0.75, "discovery": 0.85, "mood": 1.95}
        return {"familiar": 1.0, "discovery": 1.0, "mood": 1.0}

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
        focus_multiplier = 1.35 if ranking_focus.strip().lower() == "discovery" else 1.0
        for recommendation in recommendations:
            artist_name = str(catalog_lookup.get(recommendation.track_id, {}).get("artist_name", "")).strip()
            penalty = 0.35 * focus_multiplier * exploration_level * seen_artist_counts[artist_name]
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

    def _build_popularity_scores(self, candidate_catalog: pd.DataFrame) -> dict[str, float]:
        """Return normalized Spotify popularity priors."""

        return {
            str(row.track_id): float(row.catalog_popularity)
            for row in candidate_catalog.itertuples(index=False)
        }

    def _add_candidate_payload(
        self,
        candidate_payloads: list[dict[str, Any]],
        source_labels_by_track_id: dict[str, list[str]],
        track_payload: dict[str, Any],
        source_label: str,
    ) -> None:
        """Add one candidate payload while preserving first-seen order."""

        track_id = str(track_payload.get("id", "")).strip()
        if not track_id:
            return
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
        """Return a simple mood-fit score for real Spotify tracks."""

        mood = mood_label.strip().lower()
        if str(track_row.get("ranking_mode", "")).strip() == "metadata-only":
            return self._compute_metadata_mood_alignment(track_row, mood)

        energy = self._safe_float(track_row.get("energy")) or 0.5
        danceability = self._safe_float(track_row.get("danceability")) or 0.5
        valence = self._safe_float(track_row.get("valence")) or 0.5
        acousticness = self._safe_float(track_row.get("acousticness")) or 0.5
        instrumentalness = self._safe_float(track_row.get("instrumentalness")) or 0.0
        tempo = self._safe_float(track_row.get("tempo")) or 120.0
        tempo_fast = min(max((tempo - 90.0) / 70.0, 0.0), 1.0)
        tempo_slow = 1.0 - min(max((tempo - 70.0) / 80.0, 0.0), 1.0)
        if mood == "workout":
            return (energy + danceability + tempo_fast) / 3.0
        if mood == "happy":
            return (valence + energy) / 2.0
        if mood == "melancholic":
            return (1.0 - valence + (1.0 - energy)) / 2.0
        if mood == "study":
            return (acousticness + instrumentalness + (1.0 - energy) + tempo_slow) / 4.0
        return (acousticness + (1.0 - energy) + tempo_slow) / 3.0

    def _compute_metadata_mood_alignment(self, track_row: dict[str, object], mood: str) -> float:
        """Infer mood fit from text/source metadata when audio features are missing."""

        text = " ".join(
            [
                str(track_row.get("track_name", "")),
                str(track_row.get("artist_name", "")),
                str(track_row.get("artist_genres", "")),
                str(track_row.get("candidate_sources", "")),
            ]
        ).lower()
        mood_terms = {
            "workout": ("run", "dance", "club", "power", "fast", "hype", "energy", "remix"),
            "happy": ("happy", "bright", "sun", "gold", "love", "smile", "summer", "party"),
            "melancholic": ("sad", "blue", "rain", "night", "alone", "dark", "deep", "slow"),
            "study": ("ambient", "focus", "soft", "calm", "piano", "instrumental", "study", "acoustic"),
            "calm": ("calm", "soft", "quiet", "dream", "night", "ambient", "acoustic"),
        }
        terms = mood_terms.get(mood, mood_terms["calm"])
        matches = sum(1 for term in terms if term in text)
        source_bonus = 0.15 if "search" in text and mood in {"study", "happy", "workout"} else 0.0
        return min(1.0, 0.05 + (0.40 * matches) + source_bonus)

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
