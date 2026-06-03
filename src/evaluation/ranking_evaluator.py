"""Ranking-oriented evaluation metrics for recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from models.base_recommender import RecommendationResult


@dataclass(slots=True)
class RankingEvaluator:
    """Compute top-k ranking metrics for recommendation lists.

    These metrics focus on relevance quality near the top of the list, where
    users pay the most attention in a recommendation product.
    """

    def precision_at_k(
        self,
        recommendations: list[RecommendationResult],
        relevant_item_ids: set[str],
        k: int,
    ) -> float:
        """Compute Precision@K.

        Precision@K measures how many of the top-k recommendations are relevant.
        It is useful when the product goal is to make the first few results feel
        immediately strong.
        """

        top_k_recommendations = self._truncate_at_k(recommendations, k)
        if not top_k_recommendations:
            return 0.0

        hit_count = self._count_hits(top_k_recommendations, relevant_item_ids)
        return hit_count / len(top_k_recommendations)

    def recall_at_k(
        self,
        recommendations: list[RecommendationResult],
        relevant_item_ids: set[str],
        k: int,
    ) -> float:
        """Compute Recall@K.

        Recall@K measures how much of the relevant set appears within the
        recommendation list. It is useful when the goal is broader retrieval,
        not just top-position precision.
        """

        if not relevant_item_ids:
            return 0.0

        top_k_recommendations = self._truncate_at_k(recommendations, k)
        hit_count = self._count_hits(top_k_recommendations, relevant_item_ids)
        return hit_count / len(relevant_item_ids)

    def ndcg_at_k(
        self,
        recommendations: list[RecommendationResult],
        relevant_item_ids: set[str],
        k: int,
    ) -> float:
        """Compute normalized discounted cumulative gain at K.

        NDCG@K rewards ranking relevant items earlier in the list, which makes
        it especially useful for product surfaces where users rarely scroll far.
        """

        if k <= 0 or not relevant_item_ids:
            return 0.0

        top_k_recommendations = self._truncate_at_k(recommendations, k)
        discounted_gain = 0.0

        # Discount later hits because recommendation quality matters most near
        # the top of the screen or playlist handoff.
        for rank_index, recommendation in enumerate(top_k_recommendations, start=1):
            if recommendation.item_id in relevant_item_ids:
                discounted_gain += 1.0 / log2(rank_index + 1)

        ideal_hit_count = min(k, len(relevant_item_ids))
        ideal_discounted_gain = sum(
            1.0 / log2(rank_index + 1)
            for rank_index in range(1, ideal_hit_count + 1)
        )
        if ideal_discounted_gain == 0.0:
            return 0.0
        return discounted_gain / ideal_discounted_gain

    def map_at_k(
        self,
        recommendations: list[RecommendationResult],
        relevant_item_ids: set[str],
        k: int,
    ) -> float:
        """Compute mean average precision at K for one recommendation list.

        MAP@K rewards systems that retrieve multiple relevant items early and
        consistently. This is useful when the goal is not just one good hit,
        but a well-ordered set of relevant recommendations.
        """

        if k <= 0 or not relevant_item_ids:
            return 0.0

        top_k_recommendations = self._truncate_at_k(recommendations, k)
        cumulative_precision = 0.0
        hit_count = 0

        # Each time we hit a relevant item, we measure how precise the list is
        # up to that position, then average those precision values.
        for rank_index, recommendation in enumerate(top_k_recommendations, start=1):
            if recommendation.item_id not in relevant_item_ids:
                continue
            hit_count += 1
            cumulative_precision += hit_count / rank_index

        normalization = min(len(relevant_item_ids), len(top_k_recommendations))
        if normalization == 0:
            return 0.0
        return cumulative_precision / normalization

    def _truncate_at_k(
        self,
        recommendations: list[RecommendationResult],
        k: int,
    ) -> list[RecommendationResult]:
        """Return the first K recommendations, guarding against invalid K."""

        if k <= 0:
            return []
        return recommendations[:k]

    def _count_hits(
        self,
        recommendations: list[RecommendationResult],
        relevant_item_ids: set[str],
    ) -> int:
        """Count relevant recommendations within a truncated list."""

        return sum(
            1
            for recommendation in recommendations
            if recommendation.item_id in relevant_item_ids
        )
