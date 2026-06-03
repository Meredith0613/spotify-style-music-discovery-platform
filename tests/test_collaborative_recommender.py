"""Tests for the collaborative filtering recommender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.collaborative_recommender import CollaborativeRecommender
from models.interaction_matrix import InteractionMatrixBuilder


def build_interactions() -> pd.DataFrame:
    """Create a small interaction table with overlapping user preferences."""

    return pd.DataFrame(
        [
            {"user_id": "u1", "track_id": "t1", "interaction_strength": 3.0},
            {"user_id": "u1", "track_id": "t2", "interaction_strength": 2.0},
            {"user_id": "u2", "track_id": "t2", "interaction_strength": 3.0},
            {"user_id": "u2", "track_id": "t3", "interaction_strength": 2.0},
            {"user_id": "u3", "track_id": "t2", "interaction_strength": 2.0},
            {"user_id": "u3", "track_id": "t3", "interaction_strength": 3.0},
        ]
    )


def build_fitted_recommender() -> CollaborativeRecommender:
    """Create a fitted collaborative recommender for tests."""

    interaction_artifacts = InteractionMatrixBuilder().build(build_interactions())
    return CollaborativeRecommender().fit(interaction_artifacts)


def test_collaborative_recommender_generates_user_recommendations() -> None:
    """The recommender should suggest unseen tracks for a known user."""

    recommender = build_fitted_recommender()
    results = recommender.recommend_for_user(user_id="u1", k=2)

    assert [item.item_id for item in results] == ["t3"]


def test_collaborative_recommender_filters_seen_tracks() -> None:
    """Seen tracks should not appear in collaborative recommendations."""

    recommender = build_fitted_recommender()
    results = recommender.recommend_for_user(user_id="u2", k=3)

    assert "t2" not in [item.item_id for item in results]
    assert "t3" not in [item.item_id for item in results]


def test_collaborative_recommender_saves_and_loads_artifacts(tmp_path: Path) -> None:
    """The recommender should persist and reload collaborative artifacts."""

    recommender = build_fitted_recommender()
    recommender.save_artifacts(tmp_path, output_prefix="demo_cf")

    loaded_recommender = CollaborativeRecommender.load_artifacts(tmp_path, output_prefix="demo_cf")
    results = loaded_recommender.recommend_for_user(user_id="u1", k=2)

    assert [item.item_id for item in results] == ["t3"]
