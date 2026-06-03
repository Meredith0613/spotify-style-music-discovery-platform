"""Apply lightweight Spotify-aware reranking on top of existing recommendations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import pandas as pd

from models.hybrid_recommender import HybridRecommendation
from services.spotify_recommendation_adapter import SpotifyRecommendationContext
from services.user_profile_service import ListeningHistorySnapshot


@dataclass(slots=True)
class SpotifyRerankingResult:
    """Store reranked recommendations and the metadata needed to explain them."""

    recommendations: list[HybridRecommendation]
    applied: bool
    message: str | None = None
    original_scores_by_track_id: dict[str, float] = field(default_factory=dict)
    score_adjustments_by_track_id: dict[str, float] = field(default_factory=dict)
    reason_labels_by_track_id: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class SpotifyRerankingService:
    """Apply small, explainable ranking adjustments from recent Spotify listening."""

    feature_columns: tuple[str, ...] = ("danceability", "energy", "valence", "acousticness")
    max_artist_boost: float = 0.025
    max_genre_boost: float = 0.025
    max_taste_boost: float = 0.04
    max_mood_boost: float = 0.015
    repeat_artist_penalty: float = 0.01

    def rerank_recommendations(
        self,
        spotify_context: SpotifyRecommendationContext,
        recommendations: list[HybridRecommendation],
        listening_history_snapshot: ListeningHistorySnapshot,
        demo_track_catalog: pd.DataFrame,
    ) -> SpotifyRerankingResult:
        """Return reranked recommendations when recent Spotify signals are available."""

        original_scores_by_track_id = {
            recommendation.track_id: recommendation.final_score
            for recommendation in recommendations
        }
        if not recommendations or demo_track_catalog.empty:
            return SpotifyRerankingResult(
                recommendations=recommendations,
                applied=False,
                original_scores_by_track_id=original_scores_by_track_id,
            )

        catalog_lookup = self._build_catalog_lookup(demo_track_catalog)
        recent_artist_affinity = self._build_recent_artist_affinity(listening_history_snapshot)
        seed_artist_affinity = self._build_seed_artist_affinity(
            spotify_context.matched_seed_track_ids,
            catalog_lookup,
        )
        genre_affinity = self._build_seed_genre_affinity(
            spotify_context.matched_seed_track_ids,
            catalog_lookup,
        )
        recent_taste_profile = self._build_recent_taste_profile(listening_history_snapshot)
        inferred_mood = spotify_context.profile.preferred_mood.strip().lower()

        if not any([recent_artist_affinity, seed_artist_affinity, genre_affinity, recent_taste_profile]):
            return SpotifyRerankingResult(
                recommendations=recommendations,
                applied=False,
                original_scores_by_track_id=original_scores_by_track_id,
            )

        score_adjustments_by_track_id: dict[str, float] = {}
        reason_labels_by_track_id: dict[str, list[str]] = {}
        reranked_recommendations: list[HybridRecommendation] = []

        for recommendation in recommendations:
            adjustment, reason_labels = self._compute_adjustment(
                recommendation=recommendation,
                catalog_lookup=catalog_lookup,
                recent_artist_affinity=recent_artist_affinity,
                seed_artist_affinity=seed_artist_affinity,
                genre_affinity=genre_affinity,
                recent_taste_profile=recent_taste_profile,
                inferred_mood=inferred_mood,
            )
            score_adjustments_by_track_id[recommendation.track_id] = adjustment
            reason_labels_by_track_id[recommendation.track_id] = reason_labels
            reranked_recommendations.append(
                replace(recommendation, score=recommendation.final_score + adjustment)
            )

        reranked_recommendations = self._apply_redundancy_penalty(
            recommendations=reranked_recommendations,
            catalog_lookup=catalog_lookup,
            score_adjustments_by_track_id=score_adjustments_by_track_id,
            reason_labels_by_track_id=reason_labels_by_track_id,
        )
        reranked_recommendations.sort(key=lambda item: (-item.final_score, item.item_id))

        applied = any(
            abs(score_adjustments_by_track_id.get(recommendation.track_id, 0.0)) > 1e-9
            for recommendation in reranked_recommendations
        )
        return SpotifyRerankingResult(
            recommendations=reranked_recommendations,
            applied=applied,
            message=(
                "Spotify-aware reranking applied. Final ranking now includes recent artist, style, and mood signals."
                if applied
                else None
            ),
            original_scores_by_track_id=original_scores_by_track_id,
            score_adjustments_by_track_id=score_adjustments_by_track_id,
            reason_labels_by_track_id=reason_labels_by_track_id,
        )

    def _build_catalog_lookup(
        self,
        demo_track_catalog: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        """Return a track-id keyed metadata lookup for reranking."""

        normalized_frame = demo_track_catalog.copy()
        normalized_frame["track_id"] = normalized_frame["track_id"].astype(str)
        return normalized_frame.set_index("track_id").to_dict(orient="index")

    def _build_recent_artist_affinity(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> dict[str, float]:
        """Build normalized artist affinity weights from recent Spotify plays."""

        if not listening_history_snapshot.recent_tracks:
            return {}

        artist_weights: Counter[str] = Counter()
        total_weight = 0.0
        for index, recent_track in enumerate(listening_history_snapshot.recent_tracks):
            artist_name = str(recent_track.artist_name).strip()
            if not artist_name:
                continue
            weight = float(max(len(listening_history_snapshot.recent_tracks) - index, 1))
            artist_weights[artist_name] += weight
            total_weight += weight

        if total_weight <= 0.0:
            return {}
        return {
            artist_name: weight / total_weight
            for artist_name, weight in artist_weights.items()
        }

    def _build_seed_artist_affinity(
        self,
        matched_seed_track_ids: list[str],
        catalog_lookup: dict[str, dict[str, object]],
    ) -> dict[str, float]:
        """Build normalized demo-space artist affinity from matched demo seeds."""

        artist_weights: Counter[str] = Counter()
        total_weight = 0.0
        for index, track_id in enumerate(matched_seed_track_ids):
            track_row = catalog_lookup.get(str(track_id), {})
            artist_name = str(track_row.get("artist_name", "")).strip()
            if not artist_name:
                continue
            weight = float(max(len(matched_seed_track_ids) - index, 1))
            artist_weights[artist_name] += weight
            total_weight += weight

        if total_weight <= 0.0:
            return {}
        return {
            artist_name: weight / total_weight
            for artist_name, weight in artist_weights.items()
        }

    def _build_seed_genre_affinity(
        self,
        matched_seed_track_ids: list[str],
        catalog_lookup: dict[str, dict[str, object]],
    ) -> dict[str, float]:
        """Build normalized genre affinity from matched demo seeds."""

        genre_weights: Counter[str] = Counter()
        total_weight = 0.0
        for index, track_id in enumerate(matched_seed_track_ids):
            track_row = catalog_lookup.get(str(track_id), {})
            weight = float(max(len(matched_seed_track_ids) - index, 1))
            genres = self._extract_genres(track_row.get("artist_genres", ""))
            for genre_name in genres:
                genre_weights[genre_name] += weight
                total_weight += weight

        if total_weight <= 0.0:
            return {}
        return {
            genre_name: weight / total_weight
            for genre_name, weight in genre_weights.items()
        }

    def _build_recent_taste_profile(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> dict[str, float]:
        """Build an average numeric taste profile from recent Spotify features."""

        track_level_frame = listening_history_snapshot.track_level_frame
        if track_level_frame.empty:
            return {}

        taste_profile: dict[str, float] = {}
        for column_name in self.feature_columns:
            if column_name not in track_level_frame.columns:
                continue
            column_values = pd.to_numeric(track_level_frame[column_name], errors="coerce").dropna()
            if column_values.empty:
                continue
            taste_profile[column_name] = float(column_values.mean())
        return taste_profile

    def _compute_adjustment(
        self,
        recommendation: HybridRecommendation,
        catalog_lookup: dict[str, dict[str, object]],
        recent_artist_affinity: dict[str, float],
        seed_artist_affinity: dict[str, float],
        genre_affinity: dict[str, float],
        recent_taste_profile: dict[str, float],
        inferred_mood: str,
    ) -> tuple[float, list[str]]:
        """Compute a small, explainable reranking adjustment for one track."""

        track_row = catalog_lookup.get(recommendation.track_id, {})
        candidate_artist_name = str(track_row.get("artist_name", "")).strip()
        artist_affinity = max(
            recent_artist_affinity.get(candidate_artist_name, 0.0),
            seed_artist_affinity.get(candidate_artist_name, 0.0),
        )
        artist_boost = self.max_artist_boost * min(artist_affinity * 4.0, 1.0)

        candidate_genres = self._extract_genres(track_row.get("artist_genres", ""))
        genre_alignment = 0.0
        if candidate_genres and genre_affinity:
            genre_alignment = sum(genre_affinity.get(genre_name, 0.0) for genre_name in candidate_genres)
            genre_alignment = min(genre_alignment * 3.0, 1.0)
        genre_boost = self.max_genre_boost * genre_alignment

        taste_alignment = self._compute_taste_alignment(track_row, recent_taste_profile)
        taste_boost = self.max_taste_boost * taste_alignment

        mood_alignment = self._compute_mood_alignment(track_row, inferred_mood)
        mood_boost = self.max_mood_boost * mood_alignment

        reason_labels: list[str] = []
        if artist_boost > 0.0:
            reason_labels.append("recent artist affinity")
        if genre_boost > 0.0:
            reason_labels.append("matched demo-seed style overlap")
        if taste_boost > 0.0:
            reason_labels.append("recent taste-profile alignment")
        if mood_boost > 0.0:
            reason_labels.append("inferred recent mood alignment")

        return artist_boost + genre_boost + taste_boost + mood_boost, reason_labels

    def _compute_taste_alignment(
        self,
        track_row: dict[str, object],
        recent_taste_profile: dict[str, float],
    ) -> float:
        """Return how closely a candidate matches the average recent taste profile."""

        if not recent_taste_profile:
            return 0.0

        differences: list[float] = []
        for column_name, profile_value in recent_taste_profile.items():
            candidate_value = pd.to_numeric(pd.Series([track_row.get(column_name)]), errors="coerce").iloc[0]
            if pd.isna(candidate_value):
                continue
            differences.append(abs(float(candidate_value) - profile_value))
        if not differences:
            return 0.0
        return max(0.0, 1.0 - (sum(differences) / len(differences)))

    def _compute_mood_alignment(
        self,
        track_row: dict[str, object],
        inferred_mood: str,
    ) -> float:
        """Return a small boost when the candidate fits the inferred recent mood."""

        energy = self._safe_float(track_row.get("energy"))
        danceability = self._safe_float(track_row.get("danceability"))
        valence = self._safe_float(track_row.get("valence"))
        acousticness = self._safe_float(track_row.get("acousticness"))
        instrumentalness = self._safe_float(track_row.get("instrumentalness"))

        if inferred_mood == "workout" and energy >= 0.72 and danceability >= 0.68:
            return 1.0
        if inferred_mood == "happy" and valence >= 0.68:
            return 1.0
        if inferred_mood == "melancholic" and valence <= 0.35 and energy <= 0.45:
            return 1.0
        if inferred_mood == "study" and acousticness >= 0.55 and instrumentalness >= 0.08:
            return 1.0
        if inferred_mood == "calm" and energy <= 0.45:
            return 1.0
        return 0.0

    def _apply_redundancy_penalty(
        self,
        recommendations: list[HybridRecommendation],
        catalog_lookup: dict[str, dict[str, object]],
        score_adjustments_by_track_id: dict[str, float],
        reason_labels_by_track_id: dict[str, list[str]],
    ) -> list[HybridRecommendation]:
        """Slightly downweight repeated artists to keep the reranked list varied."""

        ranked_recommendations = sorted(
            recommendations,
            key=lambda item: (-item.final_score, item.item_id),
        )
        seen_artist_counts: Counter[str] = Counter()
        adjusted_recommendations: list[HybridRecommendation] = []

        for recommendation in ranked_recommendations:
            track_row = catalog_lookup.get(recommendation.track_id, {})
            artist_name = str(track_row.get("artist_name", "")).strip()
            penalty = self.repeat_artist_penalty * seen_artist_counts[artist_name] if artist_name else 0.0
            if penalty > 0.0:
                score_adjustments_by_track_id[recommendation.track_id] -= penalty
                reason_labels_by_track_id.setdefault(recommendation.track_id, []).append(
                    "slight artist-repetition penalty"
                )
                adjusted_recommendations.append(
                    replace(recommendation, score=recommendation.final_score - penalty)
                )
            else:
                adjusted_recommendations.append(recommendation)
            if artist_name:
                seen_artist_counts[artist_name] += 1

        return adjusted_recommendations

    def _extract_genres(self, raw_genres: object) -> list[str]:
        """Split a genre string into normalized display labels."""

        normalized_genres: list[str] = []
        for raw_genre in str(raw_genres).split(","):
            normalized_genre = raw_genre.strip().replace("_", " ")
            if not normalized_genre or normalized_genre in normalized_genres:
                continue
            normalized_genres.append(normalized_genre)
        return normalized_genres

    def _safe_float(self, value: object) -> float:
        """Return a float value with a neutral default for sparse metadata."""

        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            return 0.0
        return float(numeric_value)
