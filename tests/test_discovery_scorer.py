"""Tests for the discovery scoring helper."""

from __future__ import annotations

from models.discovery_scorer import DiscoveryScorer


def test_discovery_scorer_rewards_balanced_familiarity_and_novelty() -> None:
    """Balanced tracks should receive the strongest discovery score."""

    scorer = DiscoveryScorer(collaborative_familiarity_weight=0.5)
    discovery_scores = scorer.score_candidates(
        candidate_track_ids=["track_a", "track_b", "track_c"],
        collaborative_scores={"track_a": 1.0, "track_b": 0.6, "track_c": 0.1},
        content_scores={"track_a": 0.9, "track_b": 0.7, "track_c": 0.2},
        novelty_scores={"track_a": 0.1, "track_b": 0.7, "track_c": 1.0},
    )

    assert discovery_scores["track_b"] > discovery_scores["track_c"]
    assert discovery_scores["track_c"] > discovery_scores["track_a"]


def test_discovery_scorer_returns_zero_without_familiarity_or_novelty() -> None:
    """One-sided candidates should not receive discovery credit."""

    scorer = DiscoveryScorer()
    discovery_scores = scorer.score_candidates(
        candidate_track_ids=["track_a", "track_b"],
        collaborative_scores={"track_a": 0.0, "track_b": 0.8},
        content_scores={"track_a": 0.0, "track_b": 0.0},
        novelty_scores={"track_a": 0.9, "track_b": 0.0},
    )

    assert discovery_scores["track_a"] == 0.0
    assert discovery_scores["track_b"] == 0.0
