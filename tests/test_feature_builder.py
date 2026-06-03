"""Tests for the content feature engineering layer."""

from __future__ import annotations

import pandas as pd

from features.feature_builder import FeatureBuilder


def build_track_level_frame() -> pd.DataFrame:
    """Create a small curated track-level table for feature tests."""

    return pd.DataFrame(
        [
            {
                "track_id": "t1",
                "danceability": 0.8,
                "energy": 0.7,
                "valence": 0.6,
                "tempo": 120.0,
                "acousticness": 0.2,
                "speechiness": 0.1,
                "instrumentalness": 0.0,
                "loudness": -5.0,
                "artist_genres": "pop, dance pop",
            },
            {
                "track_id": "t2",
                "danceability": 0.4,
                "energy": 0.5,
                "valence": 0.3,
                "tempo": 95.0,
                "acousticness": 0.7,
                "speechiness": 0.05,
                "instrumentalness": 0.2,
                "loudness": -11.0,
                "artist_genres": "indie pop",
            },
        ]
    )


def test_feature_builder_creates_genre_and_audio_features() -> None:
    """The feature builder should create audio and genre feature columns."""

    builder = FeatureBuilder(max_genre_features=3)
    feature_table = builder.build_content_feature_table(build_track_level_frame())

    assert "danceability" in feature_table.columns
    assert "genre_pop" in feature_table.columns


def test_feature_builder_creates_model_ready_matrix() -> None:
    """The feature builder should return aligned feature artifacts."""

    builder = FeatureBuilder(max_genre_features=3)
    artifacts = builder.create_model_ready_feature_matrix(build_track_level_frame())

    assert artifacts.track_ids == ["t1", "t2"]
    assert artifacts.feature_matrix.shape[0] == 2
    assert len(artifacts.feature_names) == artifacts.feature_matrix.shape[1]
