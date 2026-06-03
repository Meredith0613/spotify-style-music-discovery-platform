"""Optional advanced ranking layer for ALS and track-context embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

from models.base_recommender import RecommendationResult


@dataclass(slots=True)
class AdvancedHybridWeights:
    """Weights used when blending base hybrid, ALS, and embedding scores."""

    hybrid: float = 1.0
    als: float = 0.0
    embedding: float = 0.0


@dataclass(slots=True)
class AdvancedHybridRecommendation(RecommendationResult):
    """Recommendation result with optional advanced-signal explanations."""

    hybrid_score: float = 0.0
    als_score: float = 0.0
    embedding_score: float = 0.0
    explanation_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AdvancedHybridRankingService:
    """Blend existing hybrid recommendations with optional advanced signals."""

    weights: AdvancedHybridWeights = field(default_factory=AdvancedHybridWeights)

    def rerank(
        self,
        base_recommendations: list[RecommendationResult],
        *,
        als_scores: dict[str, float] | None = None,
        embedding_scores: dict[str, float] | None = None,
        k: int | None = None,
    ) -> list[AdvancedHybridRecommendation]:
        """Return recommendations reranked by optional ALS and embedding scores."""

        if not base_recommendations:
            return []

        limit = len(base_recommendations) if k is None else max(int(k), 0)
        normalized_als_scores = self._normalize_scores(als_scores or {})
        normalized_embedding_scores = self._normalize_scores(embedding_scores or {})
        advanced_recommendations: list[AdvancedHybridRecommendation] = []

        for recommendation in base_recommendations:
            item_id = str(recommendation.item_id)
            hybrid_score = float(recommendation.score)
            als_score = normalized_als_scores.get(item_id, 0.0)
            embedding_score = normalized_embedding_scores.get(item_id, 0.0)
            final_score = (
                self.weights.hybrid * hybrid_score
                + self.weights.als * als_score
                + self.weights.embedding * embedding_score
            )
            advanced_recommendations.append(
                AdvancedHybridRecommendation(
                    item_id=item_id,
                    score=float(final_score),
                    source="hybrid_plus_advanced",
                    hybrid_score=hybrid_score,
                    als_score=als_score,
                    embedding_score=embedding_score,
                    explanation_lines=self._build_explanation_lines(als_score, embedding_score),
                )
            )

        advanced_recommendations.sort(key=lambda item: (-item.score, item.item_id))
        return advanced_recommendations[:limit]

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalize a score map while preserving deterministic zeros."""

        if not scores:
            return {}
        finite_scores = {str(item_id): float(score) for item_id, score in scores.items()}
        minimum_score = min(finite_scores.values())
        maximum_score = max(finite_scores.values())
        if maximum_score == minimum_score:
            return {item_id: 0.0 for item_id in finite_scores}
        return {
            item_id: (score - minimum_score) / (maximum_score - minimum_score)
            for item_id, score in finite_scores.items()
        }

    def _build_explanation_lines(self, als_score: float, embedding_score: float) -> list[str]:
        """Return explanation lines for advanced signals that contributed."""

        lines: list[str] = []
        if self.weights.als > 0.0 and als_score > 0.0:
            lines.append("ALS collaborative signal: users with similar listening patterns also preferred this track.")
        if self.weights.embedding > 0.0 and embedding_score > 0.0:
            lines.append("Embedding signal: this track appears in similar listening contexts to your recent history.")
        return lines
