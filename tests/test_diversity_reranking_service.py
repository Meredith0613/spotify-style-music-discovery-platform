"""Tests for Spotify diversity-aware reranking."""

from __future__ import annotations

import pandas as pd

from models.hybrid_recommender import HybridRecommendation, HybridScoreBreakdown
from services.diversity_reranking_service import DiversityRerankingService


def test_diversity_reranking_limits_single_artist_at_high_exploration() -> None:
    """High exploration should avoid letting one artist dominate the top list."""

    service = DiversityRerankingService()
    recommendations = [
        build_recommendation("same_1", "Same 1", "Artist A", 1.00),
        build_recommendation("same_2", "Same 2", "Artist A", 0.99),
        build_recommendation("same_3", "Same 3", "Artist A", 0.98),
        build_recommendation("other_1", "Other 1", "Artist B", 0.75),
        build_recommendation("other_2", "Other 2", "Artist C", 0.74),
    ]
    candidate_catalog = build_catalog(recommendations)

    reranked = service.rerank(
        recommendations,
        candidate_catalog,
        exploration_level=1.0,
        k=5,
    )

    top_artists = [recommendation.artist_name for recommendation in reranked[:4]]
    assert top_artists.count("Artist A") <= 2


def test_high_exploration_is_more_artist_diverse_than_low_exploration() -> None:
    """Exploration level should visibly change artist variety."""

    service = DiversityRerankingService()
    recommendations = [
        build_recommendation("same_1", "Same 1", "Artist A", 1.00),
        build_recommendation("same_2", "Same 2", "Artist A", 0.99),
        build_recommendation("same_3", "Same 3", "Artist A", 0.98),
        build_recommendation("other_1", "Other 1", "Artist B", 0.70),
    ]
    candidate_catalog = build_catalog(recommendations)

    low = service.rerank(recommendations, candidate_catalog, exploration_level=0.0, k=3)
    high = service.rerank(recommendations, candidate_catalog, exploration_level=1.0, k=3)

    assert count_unique_artists(high) > count_unique_artists(low)


def test_diversity_reranking_is_deterministic_and_handles_missing_metadata() -> None:
    """Sparse catalog metadata should not crash the reranker."""

    service = DiversityRerankingService()
    recommendations = [
        build_recommendation("a", "Track A", "Artist A", 1.0),
        build_recommendation("b", "Track B", "Artist B", 1.0),
    ]
    sparse_catalog = pd.DataFrame([{"track_id": "a"}, {"track_id": "b"}])

    first = service.rerank(recommendations, sparse_catalog, exploration_level=0.8, k=2)
    second = service.rerank(recommendations, sparse_catalog, exploration_level=0.8, k=2)

    assert [recommendation.track_id for recommendation in first] == [
        recommendation.track_id for recommendation in second
    ]
    assert len(first) == 2


def build_catalog(recommendations: list[HybridRecommendation]) -> pd.DataFrame:
    """Build a minimal candidate catalog for diversity tests."""

    return pd.DataFrame(
        [
            {
                "track_id": recommendation.track_id,
                "artist_name": recommendation.artist_name,
                "candidate_sources": "recent artist top track",
                "catalog_novelty": 0.5,
            }
            for recommendation in recommendations
        ]
    )


def count_unique_artists(recommendations: list[HybridRecommendation]) -> int:
    """Count unique artists in a recommendation list."""

    return len({recommendation.artist_name for recommendation in recommendations})


def build_recommendation(
    track_id: str,
    track_name: str,
    artist_name: str,
    score: float,
) -> HybridRecommendation:
    """Build a recommendation for reranking tests."""

    score_breakdown = HybridScoreBreakdown(
        collaborative_score=0.0,
        content_score=score,
        novelty_score=0.0,
        popularity_prior=0.0,
        discovery_score=0.0,
        final_score=score,
    )
    return HybridRecommendation(
        item_id=track_id,
        score=score,
        source="hybrid",
        track_name=track_name,
        artist_name=artist_name,
        score_breakdown=score_breakdown,
        used_cold_start_fallback=False,
    )
