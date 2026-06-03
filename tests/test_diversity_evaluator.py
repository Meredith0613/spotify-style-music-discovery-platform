"""Tests for beyond-accuracy evaluation metrics."""

from __future__ import annotations

from evaluation.diversity_evaluator import DiversityEvaluator
from models.base_recommender import RecommendationResult


def build_recommendations() -> list[RecommendationResult]:
    """Create a compact recommendation list for diversity metric tests."""

    return [
        RecommendationResult(item_id="t1", score=0.9, source="hybrid"),
        RecommendationResult(item_id="t2", score=0.8, source="hybrid"),
        RecommendationResult(item_id="t3", score=0.7, source="hybrid"),
    ]


def test_diversity_evaluator_diversity() -> None:
    """Diversity should increase as average pairwise similarity decreases."""

    evaluator = DiversityEvaluator()
    diversity_score = evaluator.diversity(
        recommendations=build_recommendations(),
        pairwise_similarity_by_item_ids={
            ("t1", "t2"): 0.2,
            ("t1", "t3"): 0.4,
            ("t2", "t3"): 0.6,
        },
    )

    assert round(diversity_score, 4) == 0.6


def test_diversity_evaluator_novelty() -> None:
    """Novelty should reward lower-popularity recommendations."""

    evaluator = DiversityEvaluator()
    novelty_score = evaluator.novelty(
        recommendations=build_recommendations(),
        popularity_by_item_id={"t1": 0.5, "t2": 0.25, "t3": 0.125},
    )

    expected_score = (1.0 + 2.0 + 3.0) / 3.0
    assert novelty_score == expected_score


def test_diversity_evaluator_coverage() -> None:
    """Coverage should measure unique recommended catalog share across users."""

    evaluator = DiversityEvaluator()
    recommendation_lists = [
        [
            RecommendationResult(item_id="t1", score=1.0, source="content"),
            RecommendationResult(item_id="t2", score=0.9, source="content"),
        ],
        [
            RecommendationResult(item_id="t2", score=1.0, source="hybrid"),
            RecommendationResult(item_id="t4", score=0.8, source="hybrid"),
        ],
    ]

    assert evaluator.coverage(recommendation_lists, ["t1", "t2", "t3", "t4", "t5"]) == 3 / 5


def test_diversity_evaluator_popularity_bias() -> None:
    """Popularity bias should reflect average recommended popularity."""

    evaluator = DiversityEvaluator()
    popularity_bias_score = evaluator.popularity_bias(
        recommendations=build_recommendations(),
        popularity_by_item_id={"t1": 0.9, "t2": 0.5, "t3": 0.1},
    )

    assert popularity_bias_score == (0.9 + 0.5 + 0.1) / 3.0


def test_diversity_evaluator_handles_missing_pairwise_keys() -> None:
    """Missing pairwise similarity entries should default to zero similarity."""

    evaluator = DiversityEvaluator()
    diversity_score = evaluator.diversity(
        recommendations=build_recommendations(),
        pairwise_similarity_by_item_ids={("t1", "t2"): 0.5},
    )

    assert round(diversity_score, 4) == round(1.0 - ((0.5 + 0.0 + 0.0) / 3.0), 4)
