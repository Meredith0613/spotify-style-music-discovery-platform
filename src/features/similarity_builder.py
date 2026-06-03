"""Similarity computation for content-based recommendation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import ProjectSettings

from .feature_builder import FeatureMatrixArtifacts


@dataclass(slots=True)
class TrackSimilarityArtifacts:
    """Store track-to-track similarity outputs aligned to track IDs.

    Attributes:
        track_ids: Track identifiers aligned to the matrix rows and columns.
        similarity_matrix: Dense cosine similarity matrix across tracks.
    """

    track_ids: list[str]
    similarity_matrix: np.ndarray


@dataclass(slots=True)
class SimilarityArtifactPaths:
    """Store file paths for persisted similarity artifacts.

    Attributes:
        matrix_path: Path to the saved similarity matrix array.
        index_path: Path to the saved track ID index file.
    """

    matrix_path: Path
    index_path: Path


@dataclass(slots=True)
class SimilarityBuilder:
    """Compute cosine similarity across model-ready track feature matrices.

    Attributes:
        settings: Project settings used to resolve artifact output paths.
    """

    settings: ProjectSettings

    def __post_init__(self) -> None:
        """Ensure artifact directories exist before saving similarity outputs."""

        self.settings.ensure_project_directories()

    def compute_cosine_similarity(
        self,
        feature_artifacts: FeatureMatrixArtifacts,
    ) -> TrackSimilarityArtifacts:
        """Compute cosine similarity across all tracks in a feature matrix.

        Args:
            feature_artifacts: Standardized features aligned to track IDs.

        Returns:
            Similarity artifacts containing track IDs and cosine similarity scores.
        """

        feature_matrix = feature_artifacts.feature_matrix
        if feature_matrix.size == 0:
            return TrackSimilarityArtifacts(track_ids=[], similarity_matrix=np.empty((0, 0)))

        # Row-wise normalization ensures cosine similarity reflects direction
        # in feature space rather than raw magnitude after standardization.
        row_norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        safe_row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        normalized_matrix = feature_matrix / safe_row_norms
        similarity_matrix = normalized_matrix @ normalized_matrix.T

        return TrackSimilarityArtifacts(
            track_ids=feature_artifacts.track_ids,
            similarity_matrix=similarity_matrix,
        )

    def get_top_k_similar_tracks(
        self,
        similarity_artifacts: TrackSimilarityArtifacts,
        track_id: str,
        k: int = 10,
        include_self: bool = False,
    ) -> list[tuple[str, float]]:
        """Return the top-k most similar tracks for a given track ID.

        Args:
            similarity_artifacts: Precomputed cosine similarity artifacts.
            track_id: Query track identifier.
            k: Maximum number of similar tracks to return.
            include_self: Whether to include the query track in the results.

        Returns:
            List of `(track_id, similarity_score)` tuples sorted by similarity.
        """

        if track_id not in similarity_artifacts.track_ids or k <= 0:
            return []

        track_index = similarity_artifacts.track_ids.index(track_id)
        similarity_scores = similarity_artifacts.similarity_matrix[track_index]

        ranked_pairs = [
            (candidate_track_id, float(score))
            for candidate_track_id, score in zip(similarity_artifacts.track_ids, similarity_scores)
            if include_self or candidate_track_id != track_id
        ]
        ranked_pairs.sort(key=lambda item: (-item[1], item[0]))
        return ranked_pairs[:k]

    def save_similarity_artifacts(
        self,
        similarity_artifacts: TrackSimilarityArtifacts,
        output_prefix: str,
    ) -> SimilarityArtifactPaths:
        """Persist similarity artifacts for later reuse by recommenders.

        Args:
            similarity_artifacts: Track similarity outputs to save.
            output_prefix: Prefix used when naming saved artifact files.

        Returns:
            Paths to the saved matrix and track index artifacts.
        """

        matrix_path = self.settings.artifacts_dir / f"{output_prefix}_track_similarity.npy"
        index_path = self.settings.artifacts_dir / f"{output_prefix}_track_similarity_index.csv"

        # Saving both the dense matrix and the aligned track index makes the
        # artifact straightforward to reload in recommenders or notebooks.
        np.save(matrix_path, similarity_artifacts.similarity_matrix)
        pd.DataFrame({"track_id": similarity_artifacts.track_ids}).to_csv(index_path, index=False)

        return SimilarityArtifactPaths(
            matrix_path=matrix_path,
            index_path=index_path,
        )
