"""Diversity-aware reranking for Spotify candidate recommendation lists."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from models.hybrid_recommender import HybridRecommendation


@dataclass(slots=True)
class DiversityRerankingService:
    """Apply a small deterministic MMR-style artist/source diversity pass.

    The service keeps the original relevance score as the dominant signal, then
    subtracts repeated-artist penalties and adds light novelty/source bonuses as
    exploration increases.
    """

    low_exploration_max_per_artist: int = 4
    high_exploration_max_per_artist: int = 2
    artist_penalty_weight: float = 0.95
    novelty_bonus_weight: float = 0.35
    source_diversity_bonus: float = 0.12

    def rerank(
        self,
        recommendations: list[HybridRecommendation],
        candidate_catalog: pd.DataFrame,
        *,
        exploration_level: float,
        k: int | None = None,
        strength: float = 1.0,
    ) -> list[HybridRecommendation]:
        """Return a relevance-preserving but more artist-diverse ranking."""

        if not recommendations:
            return []

        limit = len(recommendations) if k is None else max(int(k), 0)
        if limit == 0:
            return []

        bounded_exploration = min(max(float(exploration_level), 0.0), 1.0)
        bounded_strength = min(max(float(strength), 0.0), 1.0)
        catalog_lookup = self._build_catalog_lookup(candidate_catalog)
        remaining = sorted(recommendations, key=lambda item: (-item.final_score, item.track_id))
        selected: list[HybridRecommendation] = []
        selected_artist_counts: Counter[str] = Counter()
        selected_source_counts: Counter[str] = Counter()
        max_per_artist = (
            self.high_exploration_max_per_artist
            if bounded_exploration >= 0.65
            else self.low_exploration_max_per_artist
        )

        while remaining and len(selected) < limit:
            best_index = 0
            best_recommendation = remaining[0]
            best_score = float("-inf")
            has_artist_alternative = self._has_artist_alternative(
                remaining=remaining,
                catalog_lookup=catalog_lookup,
                selected_artist_counts=selected_artist_counts,
                max_per_artist=max_per_artist,
            )
            for index, recommendation in enumerate(remaining):
                row = catalog_lookup.get(recommendation.track_id, {})
                artist_name = self._artist_name(recommendation, row)
                if (
                    has_artist_alternative
                    and artist_name
                    and selected_artist_counts[artist_name] >= max_per_artist
                ):
                    continue

                source_label = self._source_label(row)
                novelty = self._safe_float(row.get("catalog_novelty"), 0.0)
                artist_penalty = (
                    self.artist_penalty_weight
                    * bounded_exploration
                    * bounded_strength
                    * selected_artist_counts[artist_name]
                )
                novelty_bonus = self.novelty_bonus_weight * bounded_exploration * bounded_strength * novelty
                source_bonus = (
                    self.source_diversity_bonus * bounded_exploration * bounded_strength
                    if source_label and selected_source_counts[source_label] == 0
                    else 0.0
                )
                adjusted_score = recommendation.final_score + novelty_bonus + source_bonus - artist_penalty
                if adjusted_score > best_score or (
                    adjusted_score == best_score and recommendation.track_id < best_recommendation.track_id
                ):
                    best_index = index
                    best_recommendation = recommendation
                    best_score = adjusted_score

            chosen = remaining.pop(best_index)
            chosen_row = catalog_lookup.get(chosen.track_id, {})
            selected.append(self._replace_score(chosen, best_score))
            artist_name = self._artist_name(chosen, chosen_row)
            source_label = self._source_label(chosen_row)
            if artist_name:
                selected_artist_counts[artist_name] += 1
            if source_label:
                selected_source_counts[source_label] += 1

        return selected

    def _build_catalog_lookup(self, candidate_catalog: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Return metadata keyed by track ID, tolerating sparse test fixtures."""

        if candidate_catalog.empty or "track_id" not in candidate_catalog.columns:
            return {}
        return candidate_catalog.set_index("track_id").to_dict(orient="index")

    def _has_artist_alternative(
        self,
        *,
        remaining: list[HybridRecommendation],
        catalog_lookup: dict[str, dict[str, Any]],
        selected_artist_counts: Counter[str],
        max_per_artist: int,
    ) -> bool:
        """Return whether a not-yet-capped artist is still available."""

        for recommendation in remaining:
            artist_name = self._artist_name(recommendation, catalog_lookup.get(recommendation.track_id, {}))
            if not artist_name or selected_artist_counts[artist_name] < max_per_artist:
                return True
        return False

    def _artist_name(self, recommendation: HybridRecommendation, row: dict[str, Any]) -> str:
        """Extract the best artist label for diversity accounting."""

        return str(row.get("artist_name") or recommendation.artist_name or "").strip().lower()

    def _source_label(self, row: dict[str, Any]) -> str:
        """Extract the primary candidate source label."""

        return str(row.get("candidate_sources", "")).split(",", maxsplit=1)[0].strip().lower()

    def _safe_float(self, value: object, default: float) -> float:
        """Coerce a metadata value into a finite float."""

        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            return default
        return float(numeric_value)

    def _replace_score(self, recommendation: HybridRecommendation, score: float) -> HybridRecommendation:
        """Return the same recommendation object shape with a new ranking score."""

        return HybridRecommendation(
            item_id=recommendation.item_id,
            score=float(score),
            source=recommendation.source,
            track_name=recommendation.track_name,
            artist_name=recommendation.artist_name,
            score_breakdown=recommendation.score_breakdown,
            used_cold_start_fallback=recommendation.used_cold_start_fallback,
        )
