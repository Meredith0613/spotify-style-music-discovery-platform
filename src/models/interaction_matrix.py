"""Sparse interaction-matrix construction for collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(slots=True)
class InteractionMatrixArtifacts:
    """Store sparse user-track interaction artifacts for collaborative models.

    Attributes:
        user_ids: User identifiers aligned to matrix rows.
        track_ids: Track identifiers aligned to matrix columns.
        user_index_by_id: Mapping from user ID to row index.
        track_index_by_id: Mapping from track ID to column index.
        interaction_matrix: Sparse user-track matrix in CSR format.
    """

    user_ids: list[str]
    track_ids: list[str]
    user_index_by_id: dict[str, int]
    track_index_by_id: dict[str, int]
    interaction_matrix: csr_matrix


@dataclass(slots=True)
class InteractionMatrixStats:
    """Summarize sparsity characteristics of an interaction matrix.

    Attributes:
        num_users: Number of distinct users.
        num_tracks: Number of distinct tracks.
        num_interactions: Number of non-zero user-track entries.
        density: Share of non-zero entries in the full matrix.
        sparsity: One minus density.
    """

    num_users: int
    num_tracks: int
    num_interactions: int
    density: float
    sparsity: float


@dataclass(slots=True)
class InteractionMatrixBuilder:
    """Build sparse user-track matrices from implicit interaction tables."""

    def build(
        self,
        interactions: pd.DataFrame,
        user_column: str = "user_id",
        track_column: str = "track_id",
        value_column: str = "interaction_strength",
    ) -> InteractionMatrixArtifacts:
        """Build a sparse user-track interaction matrix.

        Args:
            interactions: Interaction DataFrame with user and track identifiers.
            user_column: Column containing user identifiers.
            track_column: Column containing track identifiers.
            value_column: Column containing implicit interaction strength.

        Returns:
            Sparse interaction artifacts aligned to stable user and track indexes.
        """

        aggregated_interactions = self._aggregate_interactions(
            interactions=interactions,
            user_column=user_column,
            track_column=track_column,
            value_column=value_column,
        )

        user_ids = aggregated_interactions[user_column].drop_duplicates().astype(str).tolist()
        track_ids = aggregated_interactions[track_column].drop_duplicates().astype(str).tolist()
        user_index_by_id = {user_id: index for index, user_id in enumerate(user_ids)}
        track_index_by_id = {track_id: index for index, track_id in enumerate(track_ids)}

        # Sparse matrix construction happens in coordinate form first:
        # every aggregated interaction becomes one `(row, column, value)` entry.
        row_indices = aggregated_interactions[user_column].map(user_index_by_id).to_numpy()
        column_indices = aggregated_interactions[track_column].map(track_index_by_id).to_numpy()
        interaction_values = aggregated_interactions[value_column].astype(float).to_numpy()

        # COO is convenient for construction from triplets, then CSR is used
        # for efficient row slicing during user-based recommendation serving.
        sparse_matrix = coo_matrix(
            (interaction_values, (row_indices, column_indices)),
            shape=(len(user_ids), len(track_ids)),
            dtype=float,
        ).tocsr()

        return InteractionMatrixArtifacts(
            user_ids=user_ids,
            track_ids=track_ids,
            user_index_by_id=user_index_by_id,
            track_index_by_id=track_index_by_id,
            interaction_matrix=sparse_matrix,
        )

    def summarize(self, artifacts: InteractionMatrixArtifacts) -> InteractionMatrixStats:
        """Summarize sparsity statistics for an interaction matrix.

        Args:
            artifacts: Interaction artifacts returned by `build`.

        Returns:
            High-level matrix sparsity metrics.
        """

        num_users, num_tracks = artifacts.interaction_matrix.shape
        num_interactions = int(artifacts.interaction_matrix.nnz)
        total_possible_entries = max(num_users * num_tracks, 1)
        density = num_interactions / total_possible_entries

        return InteractionMatrixStats(
            num_users=num_users,
            num_tracks=num_tracks,
            num_interactions=num_interactions,
            density=density,
            sparsity=1.0 - density,
        )

    def _aggregate_interactions(
        self,
        interactions: pd.DataFrame,
        user_column: str,
        track_column: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Aggregate repeated user-track events into one implicit strength value."""

        if interactions.empty:
            return pd.DataFrame(columns=[user_column, track_column, value_column])

        prepared_interactions = interactions.copy()

        # If no explicit strength column exists, every event counts as one
        # implicit positive signal for the user-track pair.
        if value_column not in prepared_interactions.columns:
            prepared_interactions[value_column] = 1.0

        prepared_interactions[user_column] = prepared_interactions[user_column].astype(str)
        prepared_interactions[track_column] = prepared_interactions[track_column].astype(str)
        prepared_interactions[value_column] = pd.to_numeric(
            prepared_interactions[value_column],
            errors="coerce",
        ).fillna(0.0)

        aggregated_interactions = (
            prepared_interactions.groupby([user_column, track_column], as_index=False)[value_column]
            .sum()
            .sort_values([user_column, track_column], kind="stable")
            .reset_index(drop=True)
        )
        return aggregated_interactions
