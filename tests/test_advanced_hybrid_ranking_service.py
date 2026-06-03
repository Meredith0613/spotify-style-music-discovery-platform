"""Tests for optional ALS/embedding hybrid reranking."""

from __future__ import annotations

from models.base_recommender import RecommendationResult
from services.advanced_hybrid_ranking_service import AdvancedHybridRankingService, AdvancedHybridWeights


def test_advanced_hybrid_ranking_preserves_order_without_optional_signals() -> None:
    """Default weights should keep the existing hybrid ranking behavior."""

    base_recommendations = [
        RecommendationResult(item_id="a", score=0.9, source="hybrid"),
        RecommendationResult(item_id="b", score=0.8, source="hybrid"),
    ]

    reranked = AdvancedHybridRankingService().rerank(base_recommendations)

    assert [recommendation.item_id for recommendation in reranked] == ["a", "b"]
    assert [recommendation.score for recommendation in reranked] == [0.9, 0.8]


def test_advanced_hybrid_ranking_changes_order_when_signals_contribute() -> None:
    """Positive ALS and embedding weights should be able to change ranking."""

    base_recommendations = [
        RecommendationResult(item_id="a", score=0.9, source="hybrid"),
        RecommendationResult(item_id="b", score=0.8, source="hybrid"),
    ]
    service = AdvancedHybridRankingService(
        weights=AdvancedHybridWeights(hybrid=0.1, als=1.0, embedding=1.0)
    )

    reranked = service.rerank(
        base_recommendations,
        als_scores={"a": 0.1, "b": 0.9},
        embedding_scores={"a": 0.2, "b": 1.0},
    )

    assert [recommendation.item_id for recommendation in reranked] == ["b", "a"]
    assert reranked[0].explanation_lines
