"""Tests for sparse interaction-matrix construction."""

from __future__ import annotations

import pandas as pd

from models.interaction_matrix import InteractionMatrixBuilder


def build_interactions() -> pd.DataFrame:
    """Create a small implicit-feedback interaction table for tests."""

    return pd.DataFrame(
        [
            {"user_id": "u1", "track_id": "t1", "interaction_strength": 2.0},
            {"user_id": "u1", "track_id": "t1", "interaction_strength": 1.0},
            {"user_id": "u1", "track_id": "t2", "interaction_strength": 1.0},
            {"user_id": "u2", "track_id": "t2", "interaction_strength": 3.0},
            {"user_id": "u2", "track_id": "t3", "interaction_strength": 1.0},
        ]
    )


def test_interaction_matrix_builder_constructs_sparse_matrix() -> None:
    """The builder should aggregate repeated user-track interactions."""

    artifacts = InteractionMatrixBuilder().build(build_interactions())

    assert artifacts.interaction_matrix.shape == (2, 3)
    assert artifacts.interaction_matrix.nnz == 4
    assert float(artifacts.interaction_matrix[0, 0]) == 3.0


def test_interaction_matrix_builder_summarizes_sparsity() -> None:
    """The builder should summarize density and sparsity statistics."""

    builder = InteractionMatrixBuilder()
    artifacts = builder.build(build_interactions())
    stats = builder.summarize(artifacts)

    assert stats.num_users == 2
    assert stats.num_tracks == 3
    assert stats.num_interactions == 4
    assert stats.sparsity > 0.0
