"""Tests for cosine similarity construction."""

from __future__ import annotations

from pathlib import Path

from config.settings import ProjectSettings
from features.feature_builder import FeatureBuilder
from features.similarity_builder import SimilarityBuilder
import pandas as pd
import pytest


def build_settings(tmp_path: Path) -> ProjectSettings:
    """Create local project settings for similarity tests."""

    return ProjectSettings(
        project_root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        interim_data_dir=tmp_path / "data" / "interim",
        processed_data_dir=tmp_path / "data" / "processed",
        artifacts_dir=tmp_path / "artifacts",
        spotify_client_id="client-id",
        spotify_client_secret="client-secret",
        spotify_redirect_uri="",
        spotify_api_base_url="https://api.spotify.com/v1",
        spotify_accounts_base_url="https://accounts.spotify.com",
        spotify_request_timeout_seconds=30,
        spotify_default_market="US",
    )


def build_track_level_frame() -> pd.DataFrame:
    """Create a simple track-level frame for similarity tests."""

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
                "artist_genres": "pop",
            },
            {
                "track_id": "t2",
                "danceability": 0.79,
                "energy": 0.71,
                "valence": 0.58,
                "tempo": 119.0,
                "acousticness": 0.22,
                "speechiness": 0.11,
                "instrumentalness": 0.0,
                "loudness": -5.1,
                "artist_genres": "pop",
            },
            {
                "track_id": "t3",
                "danceability": 0.2,
                "energy": 0.3,
                "valence": 0.1,
                "tempo": 80.0,
                "acousticness": 0.9,
                "speechiness": 0.02,
                "instrumentalness": 0.5,
                "loudness": -15.0,
                "artist_genres": "ambient",
            },
        ]
    )


def test_similarity_builder_computes_cosine_similarity(tmp_path: Path) -> None:
    """The similarity builder should compute a square similarity matrix."""

    feature_artifacts = FeatureBuilder(max_genre_features=3).create_model_ready_feature_matrix(
        build_track_level_frame()
    )
    builder = SimilarityBuilder(settings=build_settings(tmp_path))
    similarity_artifacts = builder.compute_cosine_similarity(feature_artifacts)

    assert similarity_artifacts.similarity_matrix.shape == (3, 3)
    assert float(similarity_artifacts.similarity_matrix[0, 0]) == pytest.approx(1.0)


def test_similarity_builder_returns_top_k_neighbors(tmp_path: Path) -> None:
    """The similarity builder should rank the nearest neighbors first."""

    feature_artifacts = FeatureBuilder(max_genre_features=3).create_model_ready_feature_matrix(
        build_track_level_frame()
    )
    builder = SimilarityBuilder(settings=build_settings(tmp_path))
    similarity_artifacts = builder.compute_cosine_similarity(feature_artifacts)
    neighbors = builder.get_top_k_similar_tracks(similarity_artifacts, track_id="t1", k=2)

    assert neighbors[0][0] == "t2"
