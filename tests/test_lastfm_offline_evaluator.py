"""Tests for the Last.fm offline evaluation wrapper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.lastfm_catalog_builder import LastfmCatalogBuilder
from data.lastfm_preprocessor import LastfmPreprocessor
from evaluation.lastfm_offline_evaluator import (
    LastfmOfflineEvaluator,
    compute_lastfm_coverage_statistics,
    run_lastfm_offline_evaluation,
)


def test_lastfm_offline_evaluator_uses_latest_timestamp_holdout(tmp_path: Path) -> None:
    """The split should hold out the most recent user-track interaction when timestamps exist."""

    interactions = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "track_id": "track_a",
                "interaction_strength": 3,
                "artist_name": "Artist A",
                "track_name": "Song A",
                "last_timestamp": "2024-01-01T00:00:00+00:00",
            },
            {
                "user_id": "u1",
                "track_id": "track_b",
                "interaction_strength": 1,
                "artist_name": "Artist B",
                "track_name": "Song B",
                "last_timestamp": "2024-01-05T00:00:00+00:00",
            },
            {
                "user_id": "u2",
                "track_id": "track_c",
                "interaction_strength": 2,
                "artist_name": "Artist C",
                "track_name": "Song C",
                "last_timestamp": "2024-01-02T00:00:00+00:00",
            },
            {
                "user_id": "u2",
                "track_id": "track_d",
                "interaction_strength": 1,
                "artist_name": "Artist D",
                "track_name": "Song D",
                "last_timestamp": "2024-01-06T00:00:00+00:00",
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {"track_id": "track_a", "track_name": "Song A", "artist_name": "Artist A", "popularity": 90.0},
            {"track_id": "track_b", "track_name": "Song B", "artist_name": "Artist B", "popularity": 70.0},
            {"track_id": "track_c", "track_name": "Song C", "artist_name": "Artist C", "popularity": 60.0},
            {"track_id": "track_d", "track_name": "Song D", "artist_name": "Artist D", "popularity": 50.0},
        ]
    )

    evaluator = LastfmOfflineEvaluator(
        track_catalog=catalog,
        interactions=interactions,
        hybrid_weights={"collaborative": 1.0, "content": 1.0},
        min_interactions_per_user=2,
        holdout_count=1,
        k=2,
    )

    split = evaluator.build_interaction_split()

    assert split.test_track_ids_by_user["u1"] == {"track_b"}
    assert split.test_track_ids_by_user["u2"] == {"track_d"}


def test_lastfm_offline_evaluator_runs_on_processed_fixture(tmp_path: Path) -> None:
    """The Last.fm evaluation wrapper should run end to end on a tiny processed dataset."""

    raw_frame = pd.DataFrame(
        [
            {"user_id": "u1", "artist": "Artist A", "track": "Song One", "timestamp": "2024-01-01T00:00:00Z"},
            {"user_id": "u1", "artist": "Artist A", "track": "Song One", "timestamp": "2024-01-02T00:00:00Z"},
            {"user_id": "u1", "artist": "Artist B", "track": "Song Two", "timestamp": "2024-01-03T00:00:00Z"},
            {"user_id": "u1", "artist": "Artist C", "track": "Song Three", "timestamp": "2024-01-04T00:00:00Z"},
            {"user_id": "u2", "artist": "Artist A", "track": "Song One", "timestamp": "2024-01-01T00:00:00Z"},
            {"user_id": "u2", "artist": "Artist D", "track": "Song Four", "timestamp": "2024-01-02T00:00:00Z"},
            {"user_id": "u2", "artist": "Artist E", "track": "Song Five", "timestamp": "2024-01-03T00:00:00Z"},
            {"user_id": "u3", "artist": "Artist A", "track": "Song One", "timestamp": "2024-01-01T00:00:00Z"},
            {"user_id": "u3", "artist": "Artist B", "track": "Song Two", "timestamp": "2024-01-02T00:00:00Z"},
            {"user_id": "u3", "artist": "Artist F", "track": "Song Six", "timestamp": "2024-01-03T00:00:00Z"},
        ]
    )

    interactions_path = tmp_path / "lastfm_interactions.csv"
    catalog_path = tmp_path / "lastfm_catalog.csv"

    preprocessor = LastfmPreprocessor(output_path=interactions_path)
    interactions = preprocessor.preprocess_frame(raw_frame)
    preprocessor.save_interactions(interactions, interactions_path)

    catalog_builder = LastfmCatalogBuilder(output_path=catalog_path)
    catalog = catalog_builder.build_catalog(interactions)
    catalog_builder.save_catalog(catalog, catalog_path)

    result = run_lastfm_offline_evaluation(
        interactions_path=str(interactions_path),
        catalog_path=str(catalog_path),
        k=2,
        min_user_interactions=2,
        holdout_count=1,
    )

    assert not result.comparison_table.empty
    assert set(result.comparison_table["model"]) == {
        "content_only",
        "collaborative_only",
        "hybrid",
        "ALS_only",
        "Word2Vec_similarity_only",
        "hybrid_plus_ALS",
        "hybrid_plus_ALS_Word2Vec",
    }
    assert "precision@2" in result.comparison_table.columns
    assert "recall@2" in result.comparison_table.columns
    assert "ndcg@2" in result.comparison_table.columns
    assert result.comparison_table["evaluated_users"].min() > 0


def test_compute_lastfm_coverage_statistics() -> None:
    """Coverage statistics should summarize density and listener overlap."""

    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "track_id": "t1"},
            {"user_id": "u1", "track_id": "t2"},
            {"user_id": "u2", "track_id": "t1"},
            {"user_id": "u2", "track_id": "t3"},
            {"user_id": "u3", "track_id": "t3"},
        ]
    )
    catalog = pd.DataFrame(
        [
            {"track_id": "t1"},
            {"track_id": "t2"},
            {"track_id": "t3"},
            {"track_id": "t4"},
        ]
    )

    statistics = compute_lastfm_coverage_statistics(
        interactions=interactions,
        catalog=catalog,
        min_user_interactions=2,
    )

    assert statistics["interaction_count"] == 5
    assert statistics["user_count"] == 3
    assert statistics["unique_tracks_in_interactions"] == 3
    assert statistics["unique_tracks_in_catalog"] == 4
    assert statistics["average_interactions_per_user"] == 5 / 3
    assert statistics["average_users_per_track"] == 5 / 3
    assert statistics["tracks_with_more_than_one_listener_pct"] == (2 / 3) * 100
    assert statistics["possible_interactions"] == 9
    assert statistics["observed_interactions"] == 5
    assert statistics["matrix_density"] == 5 / 9
    assert statistics["matrix_sparsity"] == 4 / 9
    assert statistics["evaluated_users"] == 2


def test_lastfm_catalog_builder_handles_missing_track_names() -> None:
    """Missing track titles should not crash lightweight catalog feature generation."""

    interactions = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "track_id": "track_a",
                "interaction_strength": 2.0,
                "artist_name": "Artist A",
                "track_name": pd.NA,
            },
            {
                "user_id": "u2",
                "track_id": "track_b",
                "interaction_strength": 1.0,
                "artist_name": None,
                "track_name": "Song Two",
            },
        ]
    )

    catalog = LastfmCatalogBuilder().build_catalog(interactions)

    missing_title_row = catalog.loc[catalog["track_id"] == "track_a"].iloc[0]
    missing_artist_row = catalog.loc[catalog["track_id"] == "track_b"].iloc[0]
    assert missing_title_row["track_name"] == ""
    assert missing_title_row["title_token_count"] == 0.0
    assert missing_artist_row["artist_genres"] == "artist_unknown"
