"""Implicit-feedback model training for collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


@dataclass(slots=True)
class ItemSimilarityModelArtifacts:
    """Store learned item-item similarity artifacts.

    Attributes:
        item_similarity_matrix: Sparse item-item similarity matrix.
        neighborhood_size: Maximum number of neighbors retained per item.
    """

    item_similarity_matrix: csr_matrix
    neighborhood_size: int


@dataclass(slots=True)
class ImplicitSimilarityModel:
    """Train an item-item collaborative model from implicit interactions.

    Attributes:
        neighborhood_size: Maximum number of neighbors to retain per item.
        item_similarity_matrix: Learned sparse item-item similarity matrix.
    """

    neighborhood_size: int = 100
    item_similarity_matrix: csr_matrix | None = None

    def fit(self, interaction_matrix: csr_matrix) -> "ImplicitSimilarityModel":
        """Train item-item similarity from a user-track interaction matrix.

        Args:
            interaction_matrix: Sparse user-track matrix in CSR format.

        Returns:
            The fitted model instance.
        """

        # Item-item collaborative filtering compares track interaction patterns
        # across users, so we transpose to get one sparse vector per track.
        item_user_matrix = interaction_matrix.T.tocsr()
        normalized_item_matrix = self._row_normalize(item_user_matrix)
        raw_similarity_matrix = (normalized_item_matrix @ normalized_item_matrix.T).tocsr()

        # Self-similarity is removed because recommenders should surface neighbors,
        # not the query item itself.
        raw_similarity_matrix.setdiag(0.0)
        raw_similarity_matrix.eliminate_zeros()

        self.item_similarity_matrix = self._keep_top_k_per_row(
            similarity_matrix=raw_similarity_matrix,
            k=self.neighborhood_size,
        )
        return self

    def score_user(self, user_interaction_vector: csr_matrix) -> np.ndarray:
        """Score candidate tracks for one user from the learned model.

        Args:
            user_interaction_vector: One-row sparse user interaction vector.

        Returns:
            Dense score array aligned to the track index.
        """

        if self.item_similarity_matrix is None:
            raise ValueError("The collaborative model must be fitted before scoring users.")

        # Multiplying the user's history by the item-item similarity matrix
        # propagates preference signal from seen tracks to similar unseen tracks.
        score_vector = user_interaction_vector @ self.item_similarity_matrix
        return np.asarray(score_vector.toarray()).ravel()

    def save(self, output_path: Path) -> Path:
        """Save the learned similarity matrix as a sparse artifact."""

        if self.item_similarity_matrix is None:
            raise ValueError("The collaborative model must be fitted before saving.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(output_path, self.item_similarity_matrix)
        return output_path

    @classmethod
    def load(
        cls,
        input_path: Path,
        neighborhood_size: int,
    ) -> "ImplicitSimilarityModel":
        """Load a saved similarity matrix into a collaborative model."""

        model = cls(neighborhood_size=neighborhood_size)
        model.item_similarity_matrix = load_npz(input_path).tocsr()
        return model

    def _row_normalize(self, matrix: csr_matrix) -> csr_matrix:
        """Normalize each sparse row to unit length for cosine similarity."""

        row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        safe_row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        inverse_row_norms = 1.0 / safe_row_norms
        diagonal_scaling = csr_matrix(
            (
                inverse_row_norms,
                (np.arange(len(inverse_row_norms)), np.arange(len(inverse_row_norms))),
            ),
            shape=(len(inverse_row_norms), len(inverse_row_norms)),
        )
        return diagonal_scaling @ matrix

    def _keep_top_k_per_row(
        self,
        similarity_matrix: csr_matrix,
        k: int,
    ) -> csr_matrix:
        """Keep only the top-k similarity values per row."""

        if k <= 0:
            return csr_matrix(similarity_matrix.shape, dtype=float)

        truncated_matrix = similarity_matrix.tolil(copy=True)

        # Truncating to the strongest neighbors keeps the artifact compact and
        # makes recommendation scoring easier to explain in interviews.
        for row_index in range(truncated_matrix.shape[0]):
            row_values = truncated_matrix.data[row_index]
            row_columns = truncated_matrix.rows[row_index]
            if len(row_values) <= k:
                continue

            ranked_positions = sorted(
                range(len(row_values)),
                key=lambda position: row_values[position],
                reverse=True,
            )[:k]
            truncated_matrix.data[row_index] = [row_values[position] for position in ranked_positions]
            truncated_matrix.rows[row_index] = [row_columns[position] for position in ranked_positions]

        return truncated_matrix.tocsr()
