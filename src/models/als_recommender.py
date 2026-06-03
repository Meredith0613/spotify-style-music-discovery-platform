"""Lightweight implicit-feedback ALS recommender.

This module intentionally avoids extra dependencies. It implements a compact
Hu-Koren-Volinsky-style ALS baseline with confidence weighting, regularization,
and deterministic initialization. The implementation is designed for portfolio
experiments and offline comparisons rather than production-scale training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from .base_recommender import BaseRecommender, RecommendationResult
from .interaction_matrix import InteractionMatrixArtifacts


@dataclass(slots=True)
class ALSFactorArtifacts:
    """Store learned ALS factors and their aligned identifiers."""

    user_ids: list[str]
    track_ids: list[str]
    user_factors: np.ndarray
    item_factors: np.ndarray


@dataclass(slots=True)
class ALSRecommender(BaseRecommender):
    """Train a small implicit-feedback ALS model from sparse interactions."""

    n_factors: int = 16
    n_iterations: int = 8
    regularization: float = 0.1
    confidence_alpha: float = 20.0
    random_state: int = 42
    name: str = "als"
    interaction_artifacts: InteractionMatrixArtifacts | None = None
    factor_artifacts: ALSFactorArtifacts | None = None
    _seen_track_ids_by_user: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)

    def fit(self, interaction_artifacts: InteractionMatrixArtifacts) -> "ALSRecommender":
        """Fit ALS user/item factors from sparse implicit feedback."""

        self.interaction_artifacts = interaction_artifacts
        interaction_matrix = interaction_artifacts.interaction_matrix.tocsr()
        user_count, item_count = interaction_matrix.shape
        factor_count = max(1, min(self.n_factors, max(user_count, item_count, 1)))
        rng = np.random.default_rng(self.random_state)
        user_factors = rng.normal(scale=0.01, size=(user_count, factor_count))
        item_factors = rng.normal(scale=0.01, size=(item_count, factor_count))

        item_user_matrix = interaction_matrix.T.tocsr()
        identity = np.eye(factor_count)
        for _ in range(max(int(self.n_iterations), 0)):
            self._solve_side(
                interaction_matrix=interaction_matrix,
                fixed_factors=item_factors,
                target_factors=user_factors,
                regularization_identity=identity * self.regularization,
            )
            self._solve_side(
                interaction_matrix=item_user_matrix,
                fixed_factors=user_factors,
                target_factors=item_factors,
                regularization_identity=identity * self.regularization,
            )

        self.factor_artifacts = ALSFactorArtifacts(
            user_ids=interaction_artifacts.user_ids,
            track_ids=interaction_artifacts.track_ids,
            user_factors=user_factors,
            item_factors=item_factors,
        )
        self._seen_track_ids_by_user = self._build_seen_track_lookup(interaction_matrix)
        return self

    def recommend(
        self,
        user_id: str,
        candidate_item_ids: list[str],
        k: int,
    ) -> list[RecommendationResult]:
        """Return top-k ALS recommendations over the supplied candidates."""

        return self.recommend_for_user(user_id=user_id, k=k, candidate_track_ids=candidate_item_ids)

    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
        candidate_track_ids: list[str] | None = None,
    ) -> list[RecommendationResult]:
        """Recommend unseen tracks for one known user."""

        if k <= 0 or self.factor_artifacts is None:
            return []
        if user_id not in self.factor_artifacts.user_ids:
            return []

        candidate_ids = candidate_track_ids or self.factor_artifacts.track_ids
        seen_track_ids = self._seen_track_ids_by_user.get(user_id, set())
        candidate_ids = [track_id for track_id in candidate_ids if track_id not in seen_track_ids]
        scores = self.score_candidates(user_id=user_id, candidate_track_ids=candidate_ids)
        return self.rank_scored_items(scores, k=k)

    def score_candidates(
        self,
        user_id: str,
        candidate_track_ids: list[str],
    ) -> dict[str, float]:
        """Score candidate tracks for a user, returning 0.0 for unseen items."""

        if self.factor_artifacts is None or user_id not in self.factor_artifacts.user_ids:
            return {str(track_id): 0.0 for track_id in candidate_track_ids}

        user_index = self.factor_artifacts.user_ids.index(user_id)
        track_index_by_id = {
            track_id: index
            for index, track_id in enumerate(self.factor_artifacts.track_ids)
        }
        user_vector = self.factor_artifacts.user_factors[user_index]
        normalized_candidate_ids = [str(track_id) for track_id in candidate_track_ids]
        candidate_indices = [track_index_by_id.get(track_id) for track_id in normalized_candidate_ids]
        known_positions = [
            position
            for position, track_index in enumerate(candidate_indices)
            if track_index is not None
        ]
        scores = {track_id: 0.0 for track_id in normalized_candidate_ids}
        if not known_positions:
            return scores

        known_indices = [candidate_indices[position] for position in known_positions]
        known_track_ids = [normalized_candidate_ids[position] for position in known_positions]
        raw_scores = self.factor_artifacts.item_factors[known_indices] @ user_vector
        scores.update(
            {
                track_id: float(score)
                for track_id, score in zip(known_track_ids, raw_scores)
            }
        )
        return scores

    def _solve_side(
        self,
        interaction_matrix: csr_matrix,
        fixed_factors: np.ndarray,
        target_factors: np.ndarray,
        regularization_identity: np.ndarray,
    ) -> None:
        """Update one side of the factorization while holding the other fixed."""

        gram_matrix = fixed_factors.T @ fixed_factors
        for row_index in range(interaction_matrix.shape[0]):
            row = interaction_matrix.getrow(row_index)
            if row.nnz == 0:
                continue

            observed_indices = row.indices
            confidence = 1.0 + (self.confidence_alpha * row.data.astype(float))
            observed_factors = fixed_factors[observed_indices]
            confidence_delta = confidence - 1.0
            weighted_gram = (observed_factors.T * confidence_delta) @ observed_factors
            left_hand_side = gram_matrix + weighted_gram + regularization_identity
            right_hand_side = observed_factors.T @ confidence
            target_factors[row_index] = self._safe_solve(left_hand_side, right_hand_side)

    def _safe_solve(self, matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Solve a small linear system, falling back to least squares if needed."""

        try:
            return np.linalg.solve(matrix, values)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, values, rcond=None)[0]

    def _build_seen_track_lookup(self, interaction_matrix: csr_matrix) -> dict[str, set[str]]:
        """Build a user-to-seen-track lookup used during recommendation."""

        if self.interaction_artifacts is None:
            return {}
        lookup: dict[str, set[str]] = {}
        for user_index, user_id in enumerate(self.interaction_artifacts.user_ids):
            row = interaction_matrix.getrow(user_index)
            lookup[user_id] = {
                self.interaction_artifacts.track_ids[item_index]
                for item_index in row.indices
            }
        return lookup
