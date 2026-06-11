"""Build lightweight taste profile summaries for Spotify real-track sessions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.embedding_builder import ProjectionBuilder
from features.feature_builder import FeatureBuilder
from services.spotify_candidate_service import SpotifyRealRecommendationResult
from services.user_profile_service import ListeningHistorySnapshot


@dataclass(slots=True)
class TasteProfileTrackPoint:
    """Represent one track point on the taste-map projection."""

    track_id: str
    track_name: str
    artist_name: str
    x: float
    y: float
    cluster_id: int
    is_recent: bool


@dataclass(slots=True)
class TasteProfileSummary:
    """Store a UI-ready taste profile summary."""

    cluster_id: int | None
    cluster_label: str
    top_artists: list[str]
    top_genres: list[str]
    plot_points: list[TasteProfileTrackPoint]
    explanation: str
    warning: str | None = None


@dataclass(slots=True)
class TasteProfileService:
    """Create a compact listener taste map from recent history and candidates."""

    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    projection_builder: ProjectionBuilder = field(default_factory=ProjectionBuilder)
    min_tracks_for_map: int = 3
    max_clusters: int = 4
    random_state: int = 42

    def build_taste_profile(
        self,
        *,
        listening_history_snapshot: ListeningHistorySnapshot,
        spotify_real_recommendation_result: SpotifyRealRecommendationResult,
    ) -> TasteProfileSummary:
        """Build a deterministic, fallback-safe taste profile summary."""

        profile_frame = self._build_profile_frame(
            listening_history_snapshot=listening_history_snapshot,
            spotify_real_recommendation_result=spotify_real_recommendation_result,
        )
        top_artists = self._extract_top_artists(profile_frame)
        top_genres = self._extract_top_genres(profile_frame)
        base_explanation = (
            "Your taste profile is estimated from recent Spotify listening and candidate tracks "
            "using feature embeddings, dimensionality reduction, and clustering."
        )
        if len(profile_frame) < self.min_tracks_for_map:
            return TasteProfileSummary(
                cluster_id=None,
                cluster_label="Not enough data",
                top_artists=top_artists,
                top_genres=top_genres,
                plot_points=[],
                explanation=base_explanation,
                warning="Not enough recent listening data yet to build a stable taste map.",
            )

        feature_artifacts = self.feature_builder.create_model_ready_feature_matrix(profile_frame)
        feature_matrix = feature_artifacts.feature_matrix
        coordinates, projection_warning = self._project_features(feature_matrix)
        cluster_ids = self._cluster_features(feature_matrix)
        recent_track_ids = set(listening_history_snapshot.seed_track_ids)
        dominant_cluster_id = self._dominant_recent_cluster(
            track_ids=feature_artifacts.track_ids,
            cluster_ids=cluster_ids,
            recent_track_ids=recent_track_ids,
        )
        cluster_label = self._label_cluster(profile_frame, feature_artifacts.track_ids, cluster_ids, dominant_cluster_id)
        plot_points = self._build_plot_points(
            profile_frame=profile_frame,
            track_ids=feature_artifacts.track_ids,
            coordinates=coordinates,
            cluster_ids=cluster_ids,
            recent_track_ids=recent_track_ids,
        )
        explanation = base_explanation
        if projection_warning:
            explanation = f"{explanation} {projection_warning}"
        return TasteProfileSummary(
            cluster_id=dominant_cluster_id,
            cluster_label=cluster_label,
            top_artists=top_artists,
            top_genres=top_genres,
            plot_points=plot_points,
            explanation=explanation,
            warning=projection_warning,
        )

    def _build_profile_frame(
        self,
        *,
        listening_history_snapshot: ListeningHistorySnapshot,
        spotify_real_recommendation_result: SpotifyRealRecommendationResult,
    ) -> pd.DataFrame:
        """Combine recent tracks with real Spotify candidates for visualization."""

        frames: list[pd.DataFrame] = []
        if not listening_history_snapshot.track_level_frame.empty:
            recent_frame = listening_history_snapshot.track_level_frame.copy()
            recent_frame["is_recent"] = True
            frames.append(recent_frame)
        candidate_frame = spotify_real_recommendation_result.candidate_set.track_catalog
        if not candidate_frame.empty:
            candidate_copy = candidate_frame.copy()
            candidate_copy["is_recent"] = candidate_copy["track_id"].astype(str).isin(
                set(listening_history_snapshot.seed_track_ids)
            )
            frames.append(candidate_copy)
        if not frames:
            return pd.DataFrame()

        profile_frame = pd.concat(frames, ignore_index=True, sort=False)
        profile_frame["track_id"] = profile_frame["track_id"].astype(str)
        profile_frame = profile_frame.drop_duplicates(subset=["track_id"], keep="first").reset_index(drop=True)
        if "track_name" not in profile_frame.columns:
            profile_frame["track_name"] = profile_frame["track_id"]
        if "artist_name" not in profile_frame.columns:
            profile_frame["artist_name"] = profile_frame.get("primary_artist_name", "")
        if "artist_genres" not in profile_frame.columns:
            profile_frame["artist_genres"] = ""
        return profile_frame

    def _project_features(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, str | None]:
        """Project features into 2D with UMAP when available, otherwise SVD."""

        if feature_matrix.size == 0:
            return np.zeros((0, 2)), None
        if feature_matrix.shape[0] >= 5:
            try:
                coordinates = self.projection_builder.project_umap(
                    feature_matrix,
                    n_components=2,
                    random_state=self.random_state,
                )
                return self._ensure_two_columns(coordinates), None
            except Exception:
                pass
        return (
            self._project_with_svd(feature_matrix),
            "UMAP was unavailable, so the app used a PCA/SVD fallback.",
        )

    def _project_with_svd(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Project standardized features to two dimensions with NumPy SVD."""

        if feature_matrix.size == 0:
            return np.zeros((0, 2))
        centered_matrix = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
        try:
            _, _, right_vectors = np.linalg.svd(centered_matrix, full_matrices=False)
            coordinates = centered_matrix @ right_vectors[:2].T
        except np.linalg.LinAlgError:
            coordinates = centered_matrix[:, :2]
        return self._ensure_two_columns(coordinates)

    def _cluster_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Cluster features using existing K-Means when available, with NumPy fallback."""

        if feature_matrix.size == 0:
            return np.empty((0,), dtype=int)
        n_clusters = min(self.max_clusters, max(1, int(np.sqrt(feature_matrix.shape[0]))), feature_matrix.shape[0])
        if n_clusters <= 1:
            return np.zeros((feature_matrix.shape[0],), dtype=int)
        try:
            return self.projection_builder.cluster_kmeans(
                feature_matrix,
                n_clusters=n_clusters,
                random_state=self.random_state,
            ).astype(int)
        except Exception:
            return self._cluster_with_numpy(feature_matrix, n_clusters=n_clusters)

    def _cluster_with_numpy(self, feature_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Run a small deterministic K-Means fallback."""

        if n_clusters <= 1:
            return np.zeros((feature_matrix.shape[0],), dtype=int)
        initial_indices = np.linspace(0, feature_matrix.shape[0] - 1, n_clusters, dtype=int)
        centroids = feature_matrix[initial_indices].astype(float).copy()
        labels = np.zeros((feature_matrix.shape[0],), dtype=int)
        for _ in range(20):
            distances = np.linalg.norm(feature_matrix[:, None, :] - centroids[None, :, :], axis=2)
            next_labels = distances.argmin(axis=1).astype(int)
            if np.array_equal(labels, next_labels):
                break
            labels = next_labels
            for cluster_id in range(n_clusters):
                cluster_rows = feature_matrix[labels == cluster_id]
                if cluster_rows.size:
                    centroids[cluster_id] = cluster_rows.mean(axis=0)
        return labels

    def _dominant_recent_cluster(
        self,
        *,
        track_ids: list[str],
        cluster_ids: np.ndarray,
        recent_track_ids: set[str],
    ) -> int:
        """Return the most common cluster among recent tracks."""

        recent_clusters = [
            int(cluster_id)
            for track_id, cluster_id in zip(track_ids, cluster_ids)
            if track_id in recent_track_ids
        ]
        if not recent_clusters:
            recent_clusters = [int(cluster_id) for cluster_id in cluster_ids]
        return Counter(recent_clusters).most_common(1)[0][0]

    def _label_cluster(
        self,
        profile_frame: pd.DataFrame,
        track_ids: list[str],
        cluster_ids: np.ndarray,
        dominant_cluster_id: int,
    ) -> str:
        """Assign a simple explainable label to the dominant taste cluster."""

        cluster_track_ids = [
            track_id
            for track_id, cluster_id in zip(track_ids, cluster_ids)
            if int(cluster_id) == dominant_cluster_id
        ]
        cluster_frame = profile_frame.loc[profile_frame["track_id"].astype(str).isin(cluster_track_ids)]
        genre_text = " ".join(self._extract_top_genres(cluster_frame, limit=5)).lower()
        energy = self._column_mean(cluster_frame, "energy", default=0.5)
        danceability = self._column_mean(cluster_frame, "danceability", default=0.5)
        acousticness = self._column_mean(cluster_frame, "acousticness", default=0.5)
        valence = self._column_mean(cluster_frame, "valence", default=0.5)
        if energy >= 0.72 and danceability >= 0.62:
            return "Dance / Workout Mix"
        if acousticness >= 0.62 or (energy <= 0.42 and valence <= 0.55):
            return "Chill Acoustic Listening"
        if any(token in genre_text for token in ["indie", "alternative", "bedroom"]):
            return "Indie / Alternative Discovery"
        if "pop" in genre_text:
            return "Pop Familiarity"
        if energy >= 0.65:
            return "High-Energy Discovery"
        return "Balanced Listener"

    def _build_plot_points(
        self,
        *,
        profile_frame: pd.DataFrame,
        track_ids: list[str],
        coordinates: np.ndarray,
        cluster_ids: np.ndarray,
        recent_track_ids: set[str],
    ) -> list[TasteProfileTrackPoint]:
        """Build UI-ready taste-map points."""

        lookup = profile_frame.set_index("track_id").to_dict(orient="index")
        points: list[TasteProfileTrackPoint] = []
        for index, track_id in enumerate(track_ids):
            row = lookup.get(track_id, {})
            points.append(
                TasteProfileTrackPoint(
                    track_id=track_id,
                    track_name=str(row.get("track_name", track_id)),
                    artist_name=str(row.get("artist_name") or row.get("primary_artist_name") or ""),
                    x=float(coordinates[index, 0]),
                    y=float(coordinates[index, 1]),
                    cluster_id=int(cluster_ids[index]),
                    is_recent=track_id in recent_track_ids,
                )
            )
        return points

    def _extract_top_artists(self, profile_frame: pd.DataFrame, limit: int = 5) -> list[str]:
        """Return top artist names from recent rows when possible."""

        if profile_frame.empty:
            return []
        source_frame = profile_frame.loc[profile_frame.get("is_recent", False) == True]  # noqa: E712
        if source_frame.empty:
            source_frame = profile_frame
        artist_values = source_frame.get("artist_name", source_frame.get("primary_artist_name", pd.Series(dtype=str)))
        counter: Counter[str] = Counter()
        for raw_value in artist_values.fillna("").astype(str):
            for artist_name in raw_value.split(","):
                normalized = artist_name.strip()
                if normalized:
                    counter[normalized] += 1
        return [artist_name for artist_name, _ in counter.most_common(limit)]

    def _extract_top_genres(self, profile_frame: pd.DataFrame, limit: int = 5) -> list[str]:
        """Return top genre labels when metadata is available."""

        if profile_frame.empty or "artist_genres" not in profile_frame.columns:
            return []
        counter: Counter[str] = Counter()
        for raw_value in profile_frame["artist_genres"].fillna("").astype(str):
            for genre in raw_value.replace("|", ",").split(","):
                normalized = genre.strip().replace("_", " ")
                if normalized:
                    counter[normalized.title()] += 1
        return [genre for genre, _ in counter.most_common(limit)]

    def _column_mean(self, frame: pd.DataFrame, column_name: str, default: float) -> float:
        """Return a safe numeric column mean."""

        if column_name not in frame.columns:
            return default
        values = pd.to_numeric(frame[column_name], errors="coerce").dropna()
        if values.empty:
            return default
        return float(values.mean())

    def _ensure_two_columns(self, coordinates: np.ndarray) -> np.ndarray:
        """Ensure projection output always has two numeric columns."""

        if coordinates.size == 0:
            return np.zeros((0, 2))
        coordinate_frame = np.asarray(coordinates, dtype=float)
        if coordinate_frame.ndim == 1:
            coordinate_frame = coordinate_frame.reshape(-1, 1)
        if coordinate_frame.shape[1] >= 2:
            return coordinate_frame[:, :2]
        return np.column_stack([coordinate_frame[:, 0], np.zeros(coordinate_frame.shape[0])])
