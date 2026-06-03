"""Beyond-accuracy evaluation metrics for recommendation quality."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from models.base_recommender import RecommendationResult


@dataclass(slots=True)
class DiversityEvaluator:
    """Compute diversity, novelty, coverage, and popularity-bias metrics.

    These metrics capture product qualities that ranking accuracy alone misses,
    such as recommendation variety, discovery potential, catalog breadth, and
    over-reliance on already popular content.
    """

    def diversity(
        self,
        recommendations: list[RecommendationResult],
        pairwise_similarity_by_item_ids: dict[tuple[str, str], float],
    ) -> float:
        """Compute intra-list diversity from pairwise item similarity.

        Diversity is defined here as one minus average pairwise similarity.
        Higher values indicate a more varied recommendation list.
        """

        if len(recommendations) < 2:
            return 0.0

        pairwise_similarities: list[float] = []
        recommendation_ids = [recommendation.item_id for recommendation in recommendations]

        # We inspect every item pair so we can measure whether the list is
        # tightly clustered around one sound or spans a wider musical space.
        for left_index, left_item_id in enumerate(recommendation_ids[:-1]):
            for right_item_id in recommendation_ids[left_index + 1 :]:
                pairwise_similarities.append(
                    self._lookup_pairwise_similarity(
                        left_item_id=left_item_id,
                        right_item_id=right_item_id,
                        pairwise_similarity_by_item_ids=pairwise_similarity_by_item_ids,
                    )
                )

        if not pairwise_similarities:
            return 0.0

        average_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
        return max(0.0, 1.0 - average_similarity)

    def novelty(
        self,
        recommendations: list[RecommendationResult],
        popularity_by_item_id: dict[str, float],
    ) -> float:
        """Compute novelty from item popularity.

        Lower-popularity tracks are treated as more novel. We use self-
        information `-log2(popularity)` so rare tracks contribute more strongly
        to the metric than just taking `1 - popularity`.
        """

        if not recommendations:
            return 0.0

        novelty_values: list[float] = []
        for recommendation in recommendations:
            popularity = self._clip_probability(popularity_by_item_id.get(recommendation.item_id, 1.0))
            novelty_values.append(-log2(popularity))

        return sum(novelty_values) / len(novelty_values)

    def coverage(
        self,
        recommendation_lists: list[list[RecommendationResult]],
        catalog_item_ids: list[str],
    ) -> float:
        """Compute catalog coverage across many recommendation lists.

        Coverage measures how much of the catalog the recommender actually uses.
        Higher coverage indicates the system is not surfacing only a tiny subset
        of the available music.
        """

        if not catalog_item_ids:
            return 0.0

        unique_recommended_ids = {
            recommendation.item_id
            for recommendation_list in recommendation_lists
            for recommendation in recommendation_list
        }
        catalog_id_set = set(catalog_item_ids)
        return len(unique_recommended_ids.intersection(catalog_id_set)) / len(catalog_id_set)

    def popularity_bias(
        self,
        recommendations: list[RecommendationResult],
        popularity_by_item_id: dict[str, float],
    ) -> float:
        """Compute average popularity of recommended items.

        Higher values indicate the recommender leans more heavily toward already
        popular tracks, which can be useful but may also amplify filter bubbles.
        """

        if not recommendations:
            return 0.0

        popularity_values = [
            popularity_by_item_id.get(recommendation.item_id, 0.0)
            for recommendation in recommendations
        ]
        return sum(popularity_values) / len(popularity_values)

    def _lookup_pairwise_similarity(
        self,
        left_item_id: str,
        right_item_id: str,
        pairwise_similarity_by_item_ids: dict[tuple[str, str], float],
    ) -> float:
        """Return pairwise similarity for an unordered item pair."""

        if (left_item_id, right_item_id) in pairwise_similarity_by_item_ids:
            return pairwise_similarity_by_item_ids[(left_item_id, right_item_id)]
        if (right_item_id, left_item_id) in pairwise_similarity_by_item_ids:
            return pairwise_similarity_by_item_ids[(right_item_id, left_item_id)]
        return 0.0

    def _clip_probability(self, popularity_value: float) -> float:
        """Clip popularity to a safe probability-like range for log scoring."""

        return min(max(float(popularity_value), 1e-12), 1.0)
