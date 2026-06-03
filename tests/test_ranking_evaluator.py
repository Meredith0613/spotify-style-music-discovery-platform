"""Tests for ranking-oriented evaluation metrics."""

from __future__ import annotations

from evaluation.ranking_evaluator import RankingEvaluator
from models.base_recommender import RecommendationResult


def build_ranked_recommendations() -> list[RecommendationResult]:
    """Create a small ranked list for metric tests."""

    return [
        RecommendationResult(item_id="t1", score=0.95, source="hybrid"),
        RecommendationResult(item_id="t2", score=0.90, source="hybrid"),
        RecommendationResult(item_id="t3", score=0.80, source="hybrid"),
        RecommendationResult(item_id="t4", score=0.70, source="hybrid"),
    ]


def test_ranking_evaluator_precision_at_k() -> None:
    """Precision@K should reflect hit share in the top-K window."""

    evaluator = RankingEvaluator()

    assert evaluator.precision_at_k(build_ranked_recommendations(), {"t1", "t3"}, 3) == 2 / 3


def test_ranking_evaluator_recall_at_k() -> None:
    """Recall@K should reflect recovered share of relevant items."""

    evaluator = RankingEvaluator()

    assert evaluator.recall_at_k(build_ranked_recommendations(), {"t1", "t3", "t5"}, 3) == 2 / 3


def test_ranking_evaluator_ndcg_at_k() -> None:
    """NDCG@K should reward early placement of relevant items."""

    evaluator = RankingEvaluator()
    ndcg_score = evaluator.ndcg_at_k(build_ranked_recommendations(), {"t1", "t3"}, 3)

    assert round(ndcg_score, 4) == 0.9197


def test_ranking_evaluator_map_at_k() -> None:
    """MAP@K should reward repeated early hits across the list."""

    evaluator = RankingEvaluator()
    map_score = evaluator.map_at_k(build_ranked_recommendations(), {"t1", "t3"}, 4)

    assert round(map_score, 4) == 0.8333


def test_ranking_evaluator_handles_empty_relevance_set() -> None:
    """Ranking metrics should return zero cleanly for empty relevance labels."""

    evaluator = RankingEvaluator()

    assert evaluator.recall_at_k(build_ranked_recommendations(), set(), 3) == 0.0
    assert evaluator.ndcg_at_k(build_ranked_recommendations(), set(), 3) == 0.0
    assert evaluator.map_at_k(build_ranked_recommendations(), set(), 3) == 0.0
