"""Bridge Spotify recent listening history into the existing demo profile flow."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.demo_data import DemoUserProfile
from features.feature_builder import FeatureBuilder
from services.user_profile_service import ListeningHistorySnapshot


@dataclass(slots=True)
class SpotifyRecommendationContext:
    """Store a demo-compatible recommendation context derived from Spotify history.

    Attributes:
        profile: Temporary profile compatible with the existing demo pipeline.
        source_message: UI-friendly note explaining that Spotify history is active.
        matched_seed_track_ids: Demo catalog seed tracks chosen from recent Spotify listening.
    """

    profile: DemoUserProfile
    source_message: str
    matched_seed_track_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SpotifyRecommendationAdapter:
    """Adapt Spotify recent listening history into demo-profile inputs."""

    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    max_seed_tracks: int = 5

    def build_context(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
        demo_track_catalog: pd.DataFrame,
    ) -> SpotifyRecommendationContext | None:
        """Return a demo-compatible recommendation context when Spotify history is usable."""

        if listening_history_snapshot.track_level_frame.empty or demo_track_catalog.empty:
            return None

        matched_seed_track_ids = self._match_seed_track_ids(
            listening_history_snapshot=listening_history_snapshot,
            demo_track_catalog=demo_track_catalog,
        )
        if not matched_seed_track_ids:
            return None

        profile = DemoUserProfile(
            user_id=f"spotify_recent::{listening_history_snapshot.user_id}",
            display_name=f"{listening_history_snapshot.display_name} | Spotify Recent",
            summary=self._build_profile_summary(listening_history_snapshot),
            seed_track_ids=matched_seed_track_ids,
            preferred_mood=self._infer_preferred_mood(listening_history_snapshot),
        )
        return SpotifyRecommendationContext(
            profile=profile,
            source_message="Recommendations powered by your recent Spotify listening.",
            matched_seed_track_ids=matched_seed_track_ids,
        )

    def _match_seed_track_ids(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
        demo_track_catalog: pd.DataFrame,
    ) -> list[str]:
        """Map recent Spotify tracks onto the closest demo-catalog seed tracks."""

        spotify_frame = listening_history_snapshot.track_level_frame.copy()
        if spotify_frame.empty:
            return []

        # Prefixing the temporary Spotify IDs keeps the combined feature matrix
        # unambiguous while we compare Spotify history rows against demo tracks.
        spotify_frame["track_id"] = spotify_frame["track_id"].astype(str).map(lambda value: f"spotify::{value}")
        combined_frame = pd.concat(
            [spotify_frame, demo_track_catalog.copy()],
            axis=0,
            ignore_index=True,
            sort=False,
        )
        feature_artifacts = self.feature_builder.create_model_ready_feature_matrix(combined_frame)
        if feature_artifacts.feature_matrix.size == 0:
            return []

        spotify_track_ids_in_order = [
            f"spotify::{track_id}"
            for track_id in listening_history_snapshot.seed_track_ids
            if f"spotify::{track_id}" in feature_artifacts.track_ids
        ]
        demo_track_id_set = set(demo_track_catalog["track_id"].astype(str))
        demo_track_ids = [
            track_id
            for track_id in feature_artifacts.track_ids
            if track_id in demo_track_id_set
        ]
        if not spotify_track_ids_in_order or not demo_track_ids:
            return []

        spotify_matrix = self._select_feature_rows(
            feature_artifacts.feature_matrix,
            feature_artifacts.track_ids,
            spotify_track_ids_in_order,
        )
        demo_matrix = self._select_feature_rows(
            feature_artifacts.feature_matrix,
            feature_artifacts.track_ids,
            demo_track_ids,
        )
        if spotify_matrix.size == 0 or demo_matrix.size == 0:
            return []

        similarity_matrix = self._cosine_similarity_matrix(spotify_matrix, demo_matrix)
        recent_track_weights = self._build_recent_track_weights(spotify_track_ids_in_order)
        aggregated_similarity_scores = recent_track_weights @ similarity_matrix

        ranked_demo_track_ids = [
            demo_track_ids[index]
            for index in np.argsort(-aggregated_similarity_scores)
        ]
        deduplicated_seed_track_ids: list[str] = []
        for track_id in ranked_demo_track_ids:
            if track_id in deduplicated_seed_track_ids:
                continue
            deduplicated_seed_track_ids.append(track_id)
            if len(deduplicated_seed_track_ids) >= self.max_seed_tracks:
                break
        return deduplicated_seed_track_ids

    def _select_feature_rows(
        self,
        feature_matrix: np.ndarray,
        track_ids: list[str],
        requested_track_ids: list[str],
    ) -> np.ndarray:
        """Return matrix rows aligned to a requested track-id order."""

        row_index_by_track_id = {
            track_id: row_index
            for row_index, track_id in enumerate(track_ids)
        }
        selected_row_indexes = [
            row_index_by_track_id[track_id]
            for track_id in requested_track_ids
            if track_id in row_index_by_track_id
        ]
        if not selected_row_indexes:
            return np.empty((0, feature_matrix.shape[1] if feature_matrix.ndim == 2 else 0))
        return feature_matrix[selected_row_indexes, :]

    def _build_recent_track_weights(self, spotify_track_ids_in_order: list[str]) -> np.ndarray:
        """Weight more recent Spotify seeds more heavily during catalog matching."""

        if not spotify_track_ids_in_order:
            return np.empty((0,), dtype=float)
        descending_weights = np.arange(len(spotify_track_ids_in_order), 0, -1, dtype=float)
        return descending_weights / descending_weights.sum()

    def _cosine_similarity_matrix(
        self,
        source_matrix: np.ndarray,
        target_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between source and target feature rows."""

        normalized_source = self._normalize_matrix_rows(source_matrix)
        normalized_target = self._normalize_matrix_rows(target_matrix)
        return normalized_source @ normalized_target.T

    def _normalize_matrix_rows(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize matrix rows while avoiding division by zero."""

        if matrix.size == 0:
            return matrix
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        return matrix / row_norms

    def _build_profile_summary(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> str:
        """Summarize recent Spotify listening in the same style as demo profiles."""

        recent_artist_names: list[str] = []
        for recent_track in listening_history_snapshot.recent_tracks:
            artist_name = recent_track.artist_name.strip()
            if not artist_name or artist_name in recent_artist_names:
                continue
            recent_artist_names.append(artist_name)
            if len(recent_artist_names) >= 3:
                break

        if recent_artist_names:
            joined_artist_names = ", ".join(recent_artist_names)
            return (
                "Recent Spotify listening points toward "
                f"{joined_artist_names}, adapted into the demo catalog for recommendations."
            )
        return "Recent Spotify listening adapted into the demo catalog for recommendations."

    def _infer_preferred_mood(
        self,
        listening_history_snapshot: ListeningHistorySnapshot,
    ) -> str:
        """Infer the closest existing playlist mood from recent Spotify features."""

        track_level_frame = listening_history_snapshot.track_level_frame
        if track_level_frame.empty:
            return "calm"

        average_energy = self._safe_mean(track_level_frame, "energy", default_value=0.5)
        average_valence = self._safe_mean(track_level_frame, "valence", default_value=0.5)
        average_danceability = self._safe_mean(track_level_frame, "danceability", default_value=0.5)
        average_acousticness = self._safe_mean(track_level_frame, "acousticness", default_value=0.5)
        average_instrumentalness = self._safe_mean(track_level_frame, "instrumentalness", default_value=0.0)

        if average_energy >= 0.72 and average_danceability >= 0.68:
            return "workout"
        if average_valence >= 0.68:
            return "happy"
        if average_valence <= 0.35 and average_energy <= 0.45:
            return "melancholic"
        if average_acousticness >= 0.55 and average_instrumentalness >= 0.08:
            return "study"
        return "calm"

    def _safe_mean(
        self,
        track_level_frame: pd.DataFrame,
        column_name: str,
        default_value: float,
    ) -> float:
        """Return a numeric column mean with a stable default for sparse history rows."""

        if column_name not in track_level_frame.columns:
            return default_value
        column_values = pd.to_numeric(track_level_frame[column_name], errors="coerce").dropna()
        if column_values.empty:
            return default_value
        return float(column_values.mean())
