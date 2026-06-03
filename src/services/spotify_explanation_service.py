"""Add Spotify-specific rationale on top of existing recommendation explanations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from app.demo_service import DemoRecommendationExplanation
from features.feature_builder import FeatureBuilder
from services.spotify_recommendation_adapter import SpotifyRecommendationContext
from services.user_profile_service import ListeningHistorySnapshot


@dataclass(slots=True)
class SpotifyExplanationService:
    """Enrich existing recommendation explanations with Spotify listening context."""

    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    max_recent_tracks: int = 3
    max_seed_tracks: int = 3
    max_artists: int = 3
    max_genres: int = 2
    max_item_matches: int = 2

    def enrich_explanations(
        self,
        spotify_context: SpotifyRecommendationContext,
        explanations: list[DemoRecommendationExplanation],
        listening_history_snapshot: ListeningHistorySnapshot,
        demo_track_catalog: pd.DataFrame,
    ) -> list[DemoRecommendationExplanation]:
        """Return copies of explanation cards with Spotify rationale populated."""

        if not explanations:
            return explanations

        catalog_lookup = self._build_catalog_lookup(demo_track_catalog)
        global_recent_track_labels = self._build_recent_track_labels(listening_history_snapshot)
        global_matched_seed_labels = self._build_matched_seed_labels(
            spotify_context.matched_seed_track_ids,
            catalog_lookup,
        )
        recent_artist_names = self._build_recent_artist_names(listening_history_snapshot)
        taste_signals = self._build_taste_signals(
            listening_history_snapshot=listening_history_snapshot,
            matched_seed_track_ids=spotify_context.matched_seed_track_ids,
            catalog_lookup=catalog_lookup,
        )
        inferred_mood = spotify_context.profile.preferred_mood.strip() or None
        similarity_inputs = self._build_similarity_inputs(
            spotify_context=spotify_context,
            listening_history_snapshot=listening_history_snapshot,
            demo_track_catalog=demo_track_catalog,
        )

        return [
            replace(
                explanation,
                spotify_rationale_lines=self._build_rationale_lines(
                    explanation=explanation,
                    item_specific_recent_track_labels=self._build_item_specific_recent_track_labels(
                        explanation=explanation,
                        similarity_inputs=similarity_inputs,
                    ),
                    item_specific_seed_labels=self._build_item_specific_seed_labels(
                        explanation=explanation,
                        similarity_inputs=similarity_inputs,
                        catalog_lookup=catalog_lookup,
                    ),
                    global_recent_track_labels=global_recent_track_labels,
                    global_matched_seed_labels=global_matched_seed_labels,
                    recent_artist_names=recent_artist_names,
                    catalog_lookup=catalog_lookup,
                    taste_signals=taste_signals,
                    inferred_mood=inferred_mood,
                ),
                spotify_recent_track_labels=(
                    self._build_item_specific_recent_track_labels(
                        explanation=explanation,
                        similarity_inputs=similarity_inputs,
                    )
                    or global_recent_track_labels
                ),
                spotify_matched_seed_labels=(
                    self._build_item_specific_seed_labels(
                        explanation=explanation,
                        similarity_inputs=similarity_inputs,
                        catalog_lookup=catalog_lookup,
                    )
                    or global_matched_seed_labels
                ),
                spotify_inferred_mood=inferred_mood,
                spotify_taste_signals=taste_signals,
            )
            for explanation in explanations
        ]

    def _build_catalog_lookup(
        self,
        demo_track_catalog: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        """Return a track-id keyed catalog lookup for explanation formatting."""

        if demo_track_catalog.empty or "track_id" not in demo_track_catalog.columns:
            return {}
        normalized_frame = demo_track_catalog.copy()
        normalized_frame["track_id"] = normalized_frame["track_id"].astype(str)
        return normalized_frame.set_index("track_id").to_dict(orient="index")

    def _build_recent_track_labels(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> list[str]:
        """Return a short, distinct list of recent Spotify tracks for UI display."""

        labels: list[str] = []
        for recent_track in listening_history_snapshot.recent_tracks:
            label = self._format_track_label(recent_track.track_name, recent_track.artist_name)
            if not label or label in labels:
                continue
            labels.append(label)
            if len(labels) >= self.max_recent_tracks:
                break
        return labels

    def _build_matched_seed_labels(
        self,
        matched_seed_track_ids: list[str],
        catalog_lookup: dict[str, dict[str, object]],
    ) -> list[str]:
        """Return short labels for the demo seeds chosen from Spotify listening."""

        labels: list[str] = []
        for track_id in matched_seed_track_ids:
            track_row = catalog_lookup.get(str(track_id), {})
            label = self._format_track_label(
                track_row.get("track_name", track_id),
                track_row.get("artist_name", ""),
            )
            if not label or label in labels:
                continue
            labels.append(label)
            if len(labels) >= self.max_seed_tracks:
                break
        return labels

    def _build_recent_artist_names(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> list[str]:
        """Return a short list of distinct recent artists for narrative explanations."""

        artist_names: list[str] = []
        for recent_track in listening_history_snapshot.recent_tracks:
            artist_name = str(recent_track.artist_name).strip()
            if not artist_name or artist_name in artist_names:
                continue
            artist_names.append(artist_name)
            if len(artist_names) >= self.max_artists:
                break
        return artist_names

    def _build_similarity_inputs(
        self,
        spotify_context: SpotifyRecommendationContext,
        listening_history_snapshot: ListeningHistorySnapshot,
        demo_track_catalog: pd.DataFrame,
    ) -> dict[str, object]:
        """Build shared feature-space inputs for item-specific explanation matching."""

        spotify_frame = listening_history_snapshot.track_level_frame.copy()
        if spotify_frame.empty or demo_track_catalog.empty:
            return {}

        spotify_frame["track_id"] = spotify_frame["track_id"].astype(str).map(lambda value: f"spotify::{value}")
        combined_frame = pd.concat(
            [spotify_frame, demo_track_catalog.copy()],
            axis=0,
            ignore_index=True,
            sort=False,
        )
        feature_artifacts = self.feature_builder.create_model_ready_feature_matrix(combined_frame)
        if feature_artifacts.feature_matrix.size == 0:
            return {}

        row_index_by_track_id = {
            track_id: row_index
            for row_index, track_id in enumerate(feature_artifacts.track_ids)
        }
        recent_track_labels_by_track_id = self._build_recent_track_labels_by_track_id(
            listening_history_snapshot
        )
        recent_track_ids = [
            f"spotify::{track_id}"
            for track_id in listening_history_snapshot.seed_track_ids
            if f"spotify::{track_id}" in row_index_by_track_id
        ]
        matched_seed_track_ids = [
            track_id
            for track_id in spotify_context.matched_seed_track_ids
            if track_id in row_index_by_track_id
        ]

        return {
            "normalized_feature_matrix": self._normalize_matrix_rows(feature_artifacts.feature_matrix),
            "row_index_by_track_id": row_index_by_track_id,
            "recent_track_ids": recent_track_ids,
            "matched_seed_track_ids": matched_seed_track_ids,
            "recent_track_labels_by_track_id": recent_track_labels_by_track_id,
        }

    def _build_recent_track_labels_by_track_id(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> dict[str, str]:
        """Return recent Spotify track labels keyed by prefixed temporary track IDs."""

        labels_by_track_id: dict[str, str] = {}
        for recent_track in listening_history_snapshot.recent_tracks:
            track_id = str(recent_track.track_id).strip()
            if not track_id:
                continue
            labels_by_track_id[f"spotify::{track_id}"] = self._format_track_label(
                recent_track.track_name,
                recent_track.artist_name,
            )
        return labels_by_track_id

    def _build_item_specific_recent_track_labels(
        self,
        explanation: DemoRecommendationExplanation,
        similarity_inputs: dict[str, object],
    ) -> list[str]:
        """Return the most relevant recent Spotify tracks for one recommendation."""

        recent_track_labels_by_track_id = similarity_inputs.get("recent_track_labels_by_track_id", {})
        ranked_recent_track_ids = self._rank_reference_track_ids(
            candidate_track_id=explanation.track_id,
            comparison_track_ids=similarity_inputs.get("recent_track_ids", []),
            similarity_inputs=similarity_inputs,
        )
        return [
            recent_track_labels_by_track_id[track_id]
            for track_id in ranked_recent_track_ids
            if track_id in recent_track_labels_by_track_id
        ]

    def _build_item_specific_seed_labels(
        self,
        explanation: DemoRecommendationExplanation,
        similarity_inputs: dict[str, object],
        catalog_lookup: dict[str, dict[str, object]],
    ) -> list[str]:
        """Return the most relevant matched demo seeds for one recommendation."""

        ranked_seed_track_ids = self._rank_reference_track_ids(
            candidate_track_id=explanation.track_id,
            comparison_track_ids=similarity_inputs.get("matched_seed_track_ids", []),
            similarity_inputs=similarity_inputs,
        )
        return self._build_matched_seed_labels(ranked_seed_track_ids, catalog_lookup)

    def _rank_reference_track_ids(
        self,
        candidate_track_id: str,
        comparison_track_ids: list[str],
        similarity_inputs: dict[str, object],
    ) -> list[str]:
        """Rank reference tracks by feature similarity to one recommended track."""

        normalized_feature_matrix = similarity_inputs.get("normalized_feature_matrix")
        row_index_by_track_id = similarity_inputs.get("row_index_by_track_id", {})
        if not isinstance(normalized_feature_matrix, np.ndarray) or candidate_track_id not in row_index_by_track_id:
            return []

        candidate_row_index = row_index_by_track_id[candidate_track_id]
        candidate_vector = normalized_feature_matrix[candidate_row_index]
        ranked_track_scores: list[tuple[float, int, str]] = []

        for position, comparison_track_id in enumerate(comparison_track_ids):
            if comparison_track_id == candidate_track_id or comparison_track_id not in row_index_by_track_id:
                continue
            comparison_vector = normalized_feature_matrix[row_index_by_track_id[comparison_track_id]]
            similarity_score = float(candidate_vector @ comparison_vector)
            ranked_track_scores.append((similarity_score, -position, comparison_track_id))

        ranked_track_scores.sort(reverse=True)
        return [
            comparison_track_id
            for _, _, comparison_track_id in ranked_track_scores[: self.max_item_matches]
        ]

    def _build_taste_signals(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
        matched_seed_track_ids: list[str],
        catalog_lookup: dict[str, dict[str, object]],
    ) -> list[str]:
        """Return lightweight taste descriptors derived from Spotify and matched seeds."""

        signals: list[str] = []
        track_level_frame = listening_history_snapshot.track_level_frame
        if not track_level_frame.empty:
            average_energy = self._safe_mean(track_level_frame, "energy")
            average_danceability = self._safe_mean(track_level_frame, "danceability")
            average_valence = self._safe_mean(track_level_frame, "valence")
            average_acousticness = self._safe_mean(track_level_frame, "acousticness")

            if average_energy >= 0.72:
                signals.append("high-energy")
            if average_danceability >= 0.68:
                signals.append("dance-forward")
            if average_valence >= 0.68:
                signals.append("feel-good")
            elif average_valence <= 0.35:
                signals.append("moody")
            if average_acousticness >= 0.55:
                signals.append("acoustic-leaning")

        genre_counter: Counter[str] = Counter()
        for track_id in matched_seed_track_ids:
            track_row = catalog_lookup.get(str(track_id), {})
            for genre_name in self._extract_genres(track_row.get("artist_genres", "")):
                genre_counter[genre_name] += 1

        for genre_name, _ in genre_counter.most_common(self.max_genres):
            if genre_name not in signals:
                signals.append(genre_name)

        return signals

    def _build_rationale_lines(
        self,
        explanation: DemoRecommendationExplanation,
        item_specific_recent_track_labels: list[str],
        item_specific_seed_labels: list[str],
        global_recent_track_labels: list[str],
        global_matched_seed_labels: list[str],
        recent_artist_names: list[str],
        catalog_lookup: dict[str, dict[str, object]],
        taste_signals: list[str],
        inferred_mood: str | None,
    ) -> list[str]:
        """Build concise Spotify rationale text for one recommended track."""

        rationale_lines: list[str] = []

        if item_specific_recent_track_labels:
            rationale_lines.append(
                "Recommended because it is especially close to your recent Spotify tracks: "
                + ", ".join(item_specific_recent_track_labels)
                + "."
            )
        elif global_recent_track_labels:
            rationale_lines.append(
                "Recommended because it is similar to tracks from your recent Spotify listening."
            )

        if item_specific_seed_labels:
            rationale_lines.append(
                "Closest matched demo seeds for this pick: "
                + ", ".join(item_specific_seed_labels)
                + "."
            )
        elif global_matched_seed_labels:
            rationale_lines.append(
                "Mapped from your recent listening to demo seeds: "
                + ", ".join(global_matched_seed_labels)
                + "."
            )

        track_row = catalog_lookup.get(explanation.track_id, {})
        recommendation_artist_name = str(track_row.get("artist_name", "")).strip()
        recommendation_genres = self._extract_genres(track_row.get("artist_genres", ""))
        shared_genres = [genre for genre in recommendation_genres if genre in taste_signals]

        if recommendation_artist_name and recommendation_artist_name in recent_artist_names:
            rationale_lines.append(
                f"This pick stays close to your recent preference for {recommendation_artist_name}."
            )
        elif shared_genres:
            rationale_lines.append(
                "This recommendation reflects your recent preference for "
                + ", ".join(shared_genres[: self.max_genres])
                + "."
            )
        elif taste_signals:
            rationale_lines.append(
                "This recommendation reflects your recent preference for "
                + ", ".join(taste_signals[: self.max_genres])
                + "."
            )
        elif inferred_mood:
            rationale_lines.append(
                f"This recommendation leans toward the {inferred_mood} mood inferred from your recent listening."
            )

        if not rationale_lines:
            rationale_lines.append(
                "Recommended because it aligns with your recent Spotify listening and the closest demo-catalog matches."
            )

        return rationale_lines

    def _normalize_matrix_rows(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize matrix rows while avoiding division by zero."""

        if matrix.size == 0:
            return matrix
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        return matrix / row_norms

    def _extract_genres(self, raw_genres: object) -> list[str]:
        """Split a genre string into normalized display labels."""

        normalized_genres: list[str] = []
        for raw_genre in str(raw_genres).split(","):
            normalized_genre = raw_genre.strip().replace("_", " ")
            if not normalized_genre or normalized_genre in normalized_genres:
                continue
            normalized_genres.append(normalized_genre)
        return normalized_genres

    def _format_track_label(self, track_name: object, artist_name: object) -> str:
        """Return a compact human-readable track label."""

        normalized_track_name = str(track_name).strip()
        normalized_artist_name = str(artist_name).strip()
        if normalized_track_name and normalized_artist_name:
            return f"{normalized_track_name} - {normalized_artist_name}"
        return normalized_track_name or normalized_artist_name

    def _safe_mean(self, track_level_frame: pd.DataFrame, column_name: str) -> float:
        """Return a numeric column mean or a neutral default when unavailable."""

        if column_name not in track_level_frame.columns:
            return 0.5
        column_values = pd.to_numeric(track_level_frame[column_name], errors="coerce").dropna()
        if column_values.empty:
            return 0.5
        return float(column_values.mean())
