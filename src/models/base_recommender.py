"""Shared abstractions for recommendation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class RecommendationResult:
    """Represent a scored recommendation produced by a recommender.

    Attributes:
        item_id: Identifier of the recommended track or playlist item.
        score: Ranking score assigned by the recommender.
        source: Name of the recommender that produced the score.
    """

    item_id: str
    score: float
    source: str


class BaseRecommender(ABC):
    """Define the interface shared by all recommendation models."""

    name: str

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        candidate_item_ids: list[str],
        k: int,
    ) -> list[RecommendationResult]:
        """Return top-k recommendations for a user over candidate items."""

    def rank_scored_items(
        self,
        item_scores: dict[str, float],
        k: int,
    ) -> list[RecommendationResult]:
        """Convert item-score mappings into ranked recommendation results.

        Args:
            item_scores: Mapping from item identifier to ranking score.
            k: Maximum number of results to return.

        Returns:
            A descending score-ranked list of recommendation results.
        """

        if k <= 0:
            return []

        # Secondary sorting by item ID keeps placeholder behavior deterministic
        # when two candidates receive identical scores.
        ranked_items = sorted(
            item_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RecommendationResult(item_id=item_id, score=score, source=self.name)
            for item_id, score in ranked_items[:k]
        ]
