"""Tests for the lightweight ALS recommender."""

from __future__ import annotations

import pandas as pd

from models.als_recommender import ALSRecommender
from models.interaction_matrix import InteractionMatrixBuilder


def _build_tiny_interaction_artifacts():
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "track_id": "t1", "interaction_strength": 3.0},
            {"user_id": "u1", "track_id": "t2", "interaction_strength": 1.0},
            {"user_id": "u2", "track_id": "t2", "interaction_strength": 2.0},
            {"user_id": "u2", "track_id": "t3", "interaction_strength": 3.0},
            {"user_id": "u3", "track_id": "t3", "interaction_strength": 2.0},
            {"user_id": "u3", "track_id": "t4", "interaction_strength": 1.0},
        ]
    )
    return InteractionMatrixBuilder().build(interactions)


def test_als_recommender_trains_on_tiny_matrix() -> None:
    """ALS should produce aligned user and item factor matrices."""

    artifacts = _build_tiny_interaction_artifacts()
    recommender = ALSRecommender(n_factors=3, n_iterations=3, random_state=7).fit(artifacts)

    assert recommender.factor_artifacts is not None
    assert recommender.factor_artifacts.user_factors.shape == (3, 3)
    assert recommender.factor_artifacts.item_factors.shape == (4, 3)


def test_als_recommender_returns_expected_number_of_items() -> None:
    """Known users should receive top-k unseen recommendations."""

    artifacts = _build_tiny_interaction_artifacts()
    recommender = ALSRecommender(n_factors=3, n_iterations=3, random_state=7).fit(artifacts)

    recommendations = recommender.recommend_for_user("u1", k=2)

    assert len(recommendations) == 2
    assert all(recommendation.item_id not in {"t1", "t2"} for recommendation in recommendations)


def test_als_recommender_handles_unseen_user_and_item_gracefully() -> None:
    """Unknown IDs should not crash scoring or recommendation."""

    artifacts = _build_tiny_interaction_artifacts()
    recommender = ALSRecommender(n_factors=3, n_iterations=3, random_state=7).fit(artifacts)

    assert recommender.recommend_for_user("missing_user", k=2) == []
    assert recommender.score_candidates("u1", ["t3", "missing_track"])["missing_track"] == 0.0


def test_als_recommender_is_deterministic_with_fixed_seed() -> None:
    """Fixed seeds should produce stable ranking scores."""

    artifacts = _build_tiny_interaction_artifacts()
    left = ALSRecommender(n_factors=3, n_iterations=3, random_state=11).fit(artifacts)
    right = ALSRecommender(n_factors=3, n_iterations=3, random_state=11).fit(artifacts)

    assert left.score_candidates("u1", ["t3", "t4"]) == right.score_candidates("u1", ["t3", "t4"])
