"""Collaborative filtering recommender built on implicit user-track interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from utils.io_utils import read_json, write_json

from .base_recommender import BaseRecommender, RecommendationResult
from .collaborative_model import ImplicitSimilarityModel
from .interaction_matrix import InteractionMatrixArtifacts


@dataclass(slots=True)
class CollaborativeArtifactPaths:
    """Track file paths created when collaborative artifacts are saved."""

    interaction_matrix_path: Path
    item_similarity_path: Path
    metadata_path: Path


@dataclass(slots=True)
class CollaborativeRecommender(BaseRecommender):
    """Recommend tracks from implicit collaborative filtering signals.

    Attributes:
        model: Trainable implicit-feedback similarity model.
        interaction_artifacts: Sparse matrix artifacts aligned to user and track IDs.
    """

    model: ImplicitSimilarityModel = field(default_factory=ImplicitSimilarityModel)
    interaction_artifacts: InteractionMatrixArtifacts | None = None
    name: str = "collaborative"

    def fit(self, interaction_artifacts: InteractionMatrixArtifacts) -> "CollaborativeRecommender":
        """Train the collaborative model from sparse interaction artifacts.

        Args:
            interaction_artifacts: Sparse user-track interaction artifacts.

        Returns:
            The fitted recommender instance.
        """

        self.interaction_artifacts = interaction_artifacts
        self.model.fit(interaction_artifacts.interaction_matrix)
        return self

    def recommend(
        self,
        user_id: str,
        candidate_item_ids: list[str],
        k: int,
    ) -> list[RecommendationResult]:
        """Return top-k collaborative recommendations for a user.

        This shared interface treats `candidate_item_ids` as an optional filter
        over the full item catalog stored in the interaction matrix.
        """

        return self.recommend_for_user(
            user_id=user_id,
            k=k,
            candidate_track_ids=candidate_item_ids,
        )

    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
        candidate_track_ids: list[str] | None = None,
    ) -> list[RecommendationResult]:
        """Generate collaborative recommendations for a user.

        Args:
            user_id: User identifier to recommend for.
            k: Maximum number of recommendations to return.
            candidate_track_ids: Optional subset of track IDs to consider.

        Returns:
            Ranked recommendation results aligned to track IDs.
        """

        self._validate_fitted_state()
        if k <= 0 or self.interaction_artifacts is None:
            return []

        user_interaction_vector = self._get_user_interaction_vector(user_id)
        if user_interaction_vector is None:
            return []

        candidate_track_set = self._resolve_candidate_track_set(candidate_track_ids)
        seen_track_ids = self._get_seen_track_ids(user_interaction_vector)
        raw_scores = self.model.score_user(user_interaction_vector)
        scored_tracks = self._build_candidate_scores(
            raw_scores=raw_scores,
            seen_track_ids=seen_track_ids,
            candidate_track_set=candidate_track_set,
        )
        return self.rank_scored_items(item_scores=scored_tracks, k=k)

    def save_artifacts(self, output_dir: Path, output_prefix: str = "collaborative") -> CollaborativeArtifactPaths:
        """Save the trained collaborative artifacts for reuse."""

        self._validate_fitted_state()
        if self.interaction_artifacts is None:
            raise ValueError("Interaction artifacts must exist before saving.")

        output_dir.mkdir(parents=True, exist_ok=True)
        interaction_matrix_path = output_dir / f"{output_prefix}_interaction_matrix.npz"
        item_similarity_path = output_dir / f"{output_prefix}_item_similarity.npz"
        metadata_path = output_dir / f"{output_prefix}_index_metadata.json"

        save_npz(interaction_matrix_path, self.interaction_artifacts.interaction_matrix)
        self.model.save(item_similarity_path)
        write_json(
            metadata_path,
            {
                "user_ids": self.interaction_artifacts.user_ids,
                "track_ids": self.interaction_artifacts.track_ids,
                "neighborhood_size": self.model.neighborhood_size,
            },
        )

        return CollaborativeArtifactPaths(
            interaction_matrix_path=interaction_matrix_path,
            item_similarity_path=item_similarity_path,
            metadata_path=metadata_path,
        )

    @classmethod
    def load_artifacts(
        cls,
        artifact_dir: Path,
        output_prefix: str = "collaborative",
    ) -> "CollaborativeRecommender":
        """Load a previously saved collaborative recommender."""

        interaction_matrix_path = artifact_dir / f"{output_prefix}_interaction_matrix.npz"
        item_similarity_path = artifact_dir / f"{output_prefix}_item_similarity.npz"
        metadata_path = artifact_dir / f"{output_prefix}_index_metadata.json"

        metadata = read_json(metadata_path)
        user_ids = [str(user_id) for user_id in metadata.get("user_ids", [])]
        track_ids = [str(track_id) for track_id in metadata.get("track_ids", [])]
        interaction_artifacts = InteractionMatrixArtifacts(
            user_ids=user_ids,
            track_ids=track_ids,
            user_index_by_id={user_id: index for index, user_id in enumerate(user_ids)},
            track_index_by_id={track_id: index for index, track_id in enumerate(track_ids)},
            interaction_matrix=load_npz(interaction_matrix_path).tocsr(),
        )
        model = ImplicitSimilarityModel.load(
            input_path=item_similarity_path,
            neighborhood_size=int(metadata.get("neighborhood_size", 100)),
        )
        return cls(model=model, interaction_artifacts=interaction_artifacts)

    def _validate_fitted_state(self) -> None:
        """Ensure the recommender has both a fitted model and matrix artifacts."""

        if self.interaction_artifacts is None or self.model.item_similarity_matrix is None:
            raise ValueError("CollaborativeRecommender must be fitted before recommendation or saving.")

    def _get_user_interaction_vector(self, user_id: str) -> csr_matrix | None:
        """Return the sparse interaction row for one known user.

        Args:
            user_id: User identifier to look up.

        Returns:
            The one-row sparse interaction vector, or `None` for unknown users.
        """

        if self.interaction_artifacts is None:
            return None
        if user_id not in self.interaction_artifacts.user_index_by_id:
            return None

        user_index = self.interaction_artifacts.user_index_by_id[user_id]
        return self.interaction_artifacts.interaction_matrix[user_index]

    def _resolve_candidate_track_set(self, candidate_track_ids: list[str] | None) -> set[str] | None:
        """Resolve an optional candidate track filter into a known-track set."""

        if candidate_track_ids is None or self.interaction_artifacts is None:
            return None
        known_tracks = set(self.interaction_artifacts.track_ids)
        return {track_id for track_id in candidate_track_ids if track_id in known_tracks}

    def _get_seen_track_ids(self, user_interaction_vector: csr_matrix) -> set[str]:
        """Return track IDs the user has already interacted with."""

        if self.interaction_artifacts is None:
            return set()
        return {
            self.interaction_artifacts.track_ids[item_index]
            for item_index in user_interaction_vector.indices
        }

    def _build_candidate_scores(
        self,
        raw_scores: np.ndarray,
        seen_track_ids: set[str],
        candidate_track_set: set[str] | None,
    ) -> dict[str, float]:
        """Convert model scores into a filtered track-score mapping."""

        if self.interaction_artifacts is None:
            return {}

        scored_tracks: dict[str, float] = {}

        # Recommendation serving filters out seen tracks and optionally restricts
        # scoring to a supplied candidate set before ranking the remaining items.
        for item_index, score in enumerate(raw_scores):
            track_id = self.interaction_artifacts.track_ids[item_index]
            if track_id in seen_track_ids:
                continue
            if candidate_track_set is not None and track_id not in candidate_track_set:
                continue
            if float(score) <= 0.0:
                continue
            scored_tracks[track_id] = float(score)

        return scored_tracks
