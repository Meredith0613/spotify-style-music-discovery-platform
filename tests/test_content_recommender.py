"""Tests for the content-based recommender module."""

from __future__ import annotations

import pandas as pd

from features.feature_builder import FeatureBuilder
from models.content_recommender import ContentRecommender


def build_track_level_frame() -> pd.DataFrame:
    """Create a small curated track table for recommender tests."""

    return pd.DataFrame(
        [
            {
                "track_id": "seed_1",
                "track_name": "Seed One",
                "primary_artist_name": "Artist A",
                "danceability": 0.82,
                "energy": 0.76,
                "valence": 0.68,
                "tempo": 121.0,
                "acousticness": 0.14,
                "speechiness": 0.06,
                "instrumentalness": 0.0,
                "loudness": -4.9,
                "artist_genres": "pop, dance pop",
            },
            {
                "track_id": "seed_2",
                "track_name": "Seed Two",
                "primary_artist_name": "Artist B",
                "danceability": 0.79,
                "energy": 0.73,
                "valence": 0.64,
                "tempo": 118.0,
                "acousticness": 0.18,
                "speechiness": 0.05,
                "instrumentalness": 0.0,
                "loudness": -5.3,
                "artist_genres": "pop",
            },
            {
                "track_id": "close_match",
                "track_name": "Close Match",
                "primary_artist_name": "Artist C",
                "danceability": 0.8,
                "energy": 0.74,
                "valence": 0.66,
                "tempo": 119.0,
                "acousticness": 0.17,
                "speechiness": 0.05,
                "instrumentalness": 0.0,
                "loudness": -5.1,
                "artist_genres": "pop, dance pop",
            },
            {
                "track_id": "far_match",
                "track_name": "Far Match",
                "primary_artist_name": "Artist D",
                "danceability": 0.2,
                "energy": 0.22,
                "valence": 0.15,
                "tempo": 82.0,
                "acousticness": 0.88,
                "speechiness": 0.02,
                "instrumentalness": 0.47,
                "loudness": -14.5,
                "artist_genres": "ambient",
            },
        ]
    )


def build_recommender() -> ContentRecommender:
    """Create a content recommender with standardized feature artifacts."""

    track_level_frame = build_track_level_frame()
    feature_artifacts = FeatureBuilder(max_genre_features=4).create_model_ready_feature_matrix(
        track_level_frame
    )
    return ContentRecommender(
        feature_artifacts=feature_artifacts,
        track_catalog=track_level_frame,
    )


def test_content_recommender_returns_ranked_track_recommendations() -> None:
    """The recommender should return ranked metadata-rich track results."""

    recommender = build_recommender()
    results = recommender.recommend_from_seed_tracks(
        seed_track_ids=["seed_1", "seed_2"],
        seen_track_ids=["seed_1", "seed_2"],
        k=2,
    )

    assert [item.track_id for item in results] == ["close_match"]
    assert results[0].track_name == "Close Match"
    assert results[0].artist_name == "Artist C"


def test_content_recommender_filters_seen_tracks() -> None:
    """Seen tracks should be excluded from the returned recommendations."""

    recommender = build_recommender()
    results = recommender.recommend_from_seed_tracks(
        seed_track_ids=["seed_1"],
        seen_track_ids=["seed_1", "close_match"],
        k=3,
    )

    assert "seed_1" not in [item.track_id for item in results]
    assert "close_match" not in [item.track_id for item in results]


def test_content_recommender_excludes_negative_similarity_tracks() -> None:
    """Tracks that are clearly dissimilar should not be recommended."""

    recommender = build_recommender()
    results = recommender.recommend_from_seed_tracks(
        seed_track_ids=["seed_1"],
        seen_track_ids=["seed_1"],
        k=5,
    )

    assert "far_match" not in [item.track_id for item in results]


def test_content_recommender_explains_feature_contributions() -> None:
    """The recommender should explain the top feature contributions."""

    recommender = build_recommender()
    explanation = recommender.explain_recommendation(
        seed_track_ids=["seed_1", "seed_2"],
        candidate_track_id="close_match",
        top_n_features=3,
    )

    assert explanation.candidate_track_id == "close_match"
    assert len(explanation.top_feature_contributions) == 3
    assert explanation.top_feature_contributions[0].feature_name != ""
