"""Discovery scoring helpers for balancing familiarity and novelty."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DiscoveryScorer:
    """Compute discovery-oriented scores for hybrid ranking.

    The discovery score is designed to reward a useful middle ground:
    recommendations should feel connected to the user's taste, but they should
    also expand the user's listening rather than only repeat familiar items.

    We define discovery as the harmonic mean of:
    - a familiarity score built from collaborative and content signals
    - a novelty score that rewards less obvious candidates

    Using the harmonic mean is intentional. It penalizes extreme imbalance, so
    a track only gets a strong discovery score when it is both relevant and new.

    Attributes:
        collaborative_familiarity_weight: Share of familiarity assigned to the
            collaborative component. The remainder is assigned to content.
    """

    collaborative_familiarity_weight: float = 0.5

    def score_candidates(
        self,
        candidate_track_ids: list[str],
        collaborative_scores: dict[str, float],
        content_scores: dict[str, float],
        novelty_scores: dict[str, float],
    ) -> dict[str, float]:
        """Compute discovery scores for candidate tracks.

        Args:
            candidate_track_ids: Candidate tracks to score.
            collaborative_scores: Normalized collaborative scores.
            content_scores: Normalized content-based scores.
            novelty_scores: Normalized novelty scores.

        Returns:
            Mapping from track ID to discovery score on a 0-to-1 scale.
        """

        discovery_scores: dict[str, float] = {}

        # Discovery should increase when a track is both relevant and novel.
        # We therefore compute familiarity first, then combine it with novelty
        # using a harmonic mean to penalize one-sided recommendations.
        for track_id in candidate_track_ids:
            familiarity_score = self._compute_familiarity_score(
                collaborative_score=collaborative_scores.get(track_id, 0.0),
                content_score=content_scores.get(track_id, 0.0),
            )
            novelty_score = novelty_scores.get(track_id, 0.0)
            discovery_scores[track_id] = self._harmonic_mean(
                familiarity_score,
                novelty_score,
            )

        return discovery_scores

    def _compute_familiarity_score(
        self,
        collaborative_score: float,
        content_score: float,
    ) -> float:
        """Blend collaborative and content signals into familiarity.

        Args:
            collaborative_score: Behavior-based similarity signal.
            content_score: Feature-space similarity signal.

        Returns:
            Blended familiarity score in the same normalized scale.
        """

        content_weight = 1.0 - self.collaborative_familiarity_weight
        return (
            self.collaborative_familiarity_weight * collaborative_score
            + content_weight * content_score
        )

    def _harmonic_mean(self, left_score: float, right_score: float) -> float:
        """Return the harmonic mean of two non-negative scores.

        Args:
            left_score: First normalized score.
            right_score: Second normalized score.

        Returns:
            Harmonic mean, or zero when either side is zero.
        """

        if left_score <= 0.0 or right_score <= 0.0:
            return 0.0
        return (2.0 * left_score * right_score) / (left_score + right_score)
