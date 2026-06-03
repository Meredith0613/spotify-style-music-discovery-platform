"""Tests for Word2Vec-inspired track context embeddings."""

from __future__ import annotations

import pandas as pd

from models.track_embedding_model import TrackEmbeddingModel


def test_track_embedding_model_builds_sequences_from_timestamps() -> None:
    """User histories should be grouped and sorted into track sequences."""

    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "track_id": "t2", "timestamp": "2024-01-02T00:00:00Z"},
            {"user_id": "u1", "track_id": "t1", "timestamp": "2024-01-01T00:00:00Z"},
            {"user_id": "u2", "track_id": "t3", "timestamp": "2024-01-01T00:00:00Z"},
        ]
    )

    sequences = TrackEmbeddingModel().build_sequences(interactions, timestamp_column="timestamp")

    assert sequences == [["t1", "t2"], ["t3"]]


def test_track_embedding_model_returns_similar_tracks() -> None:
    """Tracks sharing listening contexts should have retrievable neighbors."""

    model = TrackEmbeddingModel(embedding_dim=3, window_size=1, random_state=3).fit_sequences(
        [["t1", "t2", "t3"], ["t1", "t2", "t4"]]
    )

    similar_tracks = model.similar_tracks("t1", k=2)

    assert len(similar_tracks) == 2
    assert similar_tracks[0][0] == "t2"


def test_track_embedding_model_handles_unseen_track_fallback() -> None:
    """Unknown tracks should return None or zero scores instead of raising."""

    model = TrackEmbeddingModel(embedding_dim=3).fit_sequences([["t1", "t2"]])

    assert model.get_track_vector("missing") is None
    assert model.similar_tracks("missing", k=2) == []
    assert model.score_candidate_similarity(["missing"], ["t1", "missing"]) == {
        "t1": 0.0,
        "missing": 0.0,
    }


def test_track_embedding_model_scores_candidate_similarity() -> None:
    """Candidate scoring should prefer tracks close to the seed context."""

    model = TrackEmbeddingModel(embedding_dim=3, window_size=1, random_state=3).fit_sequences(
        [["seed", "near", "other"], ["seed", "near", "far"]]
    )

    scores = model.score_candidate_similarity(["seed"], ["near", "far", "missing"])

    assert scores["near"] > scores["far"]
    assert scores["missing"] == 0.0
