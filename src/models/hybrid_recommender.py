"""Hybrid recommendation model that fuses multiple ranking signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base_recommender import BaseRecommender, RecommendationResult
from .collaborative_recommender import CollaborativeRecommender
from .content_recommender import ContentRecommender
from .discovery_scorer import DiscoveryScorer


@dataclass(slots=True)
class HybridScoreWeights:
    """Store configurable weights for each hybrid scoring component.

    The final ranking score follows the weighted linear blend:

    `FinalScore = w1 * collaborative_score + w2 * content_score
                  + w3 * novelty_score + w4 * popularity_prior
                  + w5 * discovery_score`

    Attributes:
        collaborative: Weight applied to collaborative filtering scores.
        content: Weight applied to content-based similarity scores.
        novelty: Weight applied to novelty scores.
        popularity_prior: Weight applied to the popularity prior.
        discovery: Weight applied to the discovery sweet-spot score.
    """

    collaborative: float = 1.0
    content: float = 1.0
    novelty: float = 0.0
    popularity_prior: float = 0.0
    discovery: float = 0.0

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> "HybridScoreWeights":
        """Build weights from a plain dictionary.

        Args:
            values: Mapping from component name to numeric weight.

        Returns:
            A typed weight configuration with missing values filled in.
        """

        return cls(
            collaborative=float(values.get("collaborative", 1.0)),
            content=float(values.get("content", 1.0)),
            novelty=float(values.get("novelty", 0.0)),
            popularity_prior=float(values.get("popularity_prior", 0.0)),
            discovery=float(values.get("discovery", 0.0)),
        )


@dataclass(slots=True)
class HybridScoreBreakdown:
    """Store the component-level scores that make up one final rank score.

    Attributes:
        collaborative_score: Behavior-based score from collaborative filtering.
        content_score: Audio and metadata similarity score from content features.
        novelty_score: Novelty score that promotes less obvious items.
        popularity_prior: Prior score that stabilizes ranking for sparse users.
        discovery_score: Score that rewards the balance of familiarity and novelty.
        final_score: Weighted sum used for final ranking.
    """

    collaborative_score: float
    content_score: float
    novelty_score: float
    popularity_prior: float
    discovery_score: float
    final_score: float


@dataclass(slots=True)
class HybridComponentScores:
    """Bundle normalized score components used during hybrid ranking.

    Attributes:
        collaborative: Collaborative filtering scores by track ID.
        content: Content-based similarity scores by track ID.
        novelty: Novelty scores by track ID.
        popularity_prior: Popularity prior scores by track ID.
        discovery: Discovery scores by track ID.
    """

    collaborative: dict[str, float]
    content: dict[str, float]
    novelty: dict[str, float]
    popularity_prior: dict[str, float]
    discovery: dict[str, float]


@dataclass(slots=True)
class HybridRecommendation(RecommendationResult):
    """Represent one ranked hybrid recommendation with score explanations.

    Attributes:
        track_name: Human-readable track name for display.
        artist_name: Human-readable artist name for display.
        score_breakdown: Component-level score explanation for the final rank.
        used_cold_start_fallback: Whether the rank relied only on fallback priors.
    """

    track_name: str
    artist_name: str
    score_breakdown: HybridScoreBreakdown
    used_cold_start_fallback: bool

    @property
    def final_score(self) -> float:
        """Return the weighted final score used for ranking."""

        return self.score

    @property
    def track_id(self) -> str:
        """Return the recommended track identifier."""

        return self.item_id


@dataclass(slots=True)
class TrackMetadata:
    """Store human-readable metadata for hybrid recommendation outputs."""

    track_name: str
    artist_name: str


@dataclass(slots=True)
class HybridRecommender(BaseRecommender):
    """Blend collaborative, content, novelty, popularity, and discovery signals.

    This class keeps each scoring component modular and testable:
    collaborative filtering captures user-behavior overlap, content filtering
    captures musical similarity, novelty nudges the ranking toward discovery,
    the popularity prior provides a stable fallback in sparse settings, and
    the discovery score rewards tracks that are both relevant and exploratory.

    Attributes:
        collaborative_recommender: Fitted collaborative recommender, if available.
        content_recommender: Fitted content recommender, if available.
        weights: Configurable blend weights for each score component.
        popularity_scores: Optional track-level popularity prior scores.
        novelty_scores: Optional track-level novelty scores.
        discovery_scorer: Helper that balances familiarity and novelty.
        user_seed_track_ids_by_user: Optional external user history for content seeds.
        track_catalog: Optional track metadata table for display fields.
    """

    collaborative_recommender: CollaborativeRecommender | None = None
    content_recommender: ContentRecommender | None = None
    weights: HybridScoreWeights | dict[str, float] = field(default_factory=HybridScoreWeights)
    popularity_scores: dict[str, float] = field(default_factory=dict)
    novelty_scores: dict[str, float] = field(default_factory=dict)
    discovery_scorer: DiscoveryScorer = field(default_factory=DiscoveryScorer)
    user_seed_track_ids_by_user: dict[str, list[str]] = field(default_factory=dict)
    track_catalog: pd.DataFrame | None = None
    name: str = "hybrid"
    _metadata_by_track_id: dict[str, TrackMetadata] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize configuration and build metadata lookups."""

        if isinstance(self.weights, dict):
            self.weights = HybridScoreWeights.from_mapping(self.weights)

        metadata_source = self.track_catalog
        if metadata_source is None and self.content_recommender is not None:
            metadata_source = self.content_recommender.track_catalog
        self._metadata_by_track_id = self._build_metadata_lookup(metadata_source)

    def recommend(
        self,
        user_id: str,
        candidate_item_ids: list[str] | None = None,
        k: int = 10,
    ) -> list[HybridRecommendation]:
        """Return top-k hybrid recommendations for a user.

        Args:
            user_id: User identifier to rank tracks for.
            candidate_item_ids: Optional candidate set to rank. When omitted,
                the recommender builds a catalog from the available components.
            k: Maximum number of recommendations to return.

        Returns:
            Ranked hybrid recommendations with component score breakdowns.
        """

        if k <= 0:
            return []

        candidate_track_ids = self._resolve_candidate_track_ids(candidate_item_ids)
        if not candidate_track_ids:
            return []

        user_history_track_ids = self._get_user_history_track_ids(user_id)
        seen_track_ids = set(user_history_track_ids)
        unseen_candidate_track_ids = [
            track_id
            for track_id in candidate_track_ids
            if track_id not in seen_track_ids
        ]
        if not unseen_candidate_track_ids:
            return []

        collaborative_scores = self._normalize_component_scores(
            self._get_collaborative_scores(
                user_id=user_id,
                candidate_track_ids=unseen_candidate_track_ids,
            )
        )
        content_scores = self._normalize_component_scores(
            self._get_content_scores(
                user_history_track_ids=user_history_track_ids,
                candidate_track_ids=unseen_candidate_track_ids,
            )
        )
        popularity_prior_scores = self._normalize_component_scores(
            self._get_popularity_prior_scores(unseen_candidate_track_ids)
        )
        novelty_scores = self._normalize_component_scores(
            self._get_novelty_scores(unseen_candidate_track_ids)
        )
        discovery_scores = self._get_discovery_scores(
            candidate_track_ids=unseen_candidate_track_ids,
            collaborative_scores=collaborative_scores,
            content_scores=content_scores,
            novelty_scores=novelty_scores,
        )
        component_scores = HybridComponentScores(
            collaborative=collaborative_scores,
            content=content_scores,
            novelty=novelty_scores,
            popularity_prior=popularity_prior_scores,
            discovery=discovery_scores,
        )
        use_cold_start_fallback = self._should_use_cold_start_fallback(
            collaborative_scores=collaborative_scores,
            content_scores=content_scores,
        )

        ranked_recommendations = self._build_ranked_recommendations(
            candidate_track_ids=unseen_candidate_track_ids,
            component_scores=component_scores,
            use_cold_start_fallback=use_cold_start_fallback,
        )
        return ranked_recommendations[:k]

    def _resolve_candidate_track_ids(self, candidate_item_ids: list[str] | None) -> list[str]:
        """Resolve the list of candidate tracks to rank."""

        if candidate_item_ids is not None:
            return self._deduplicate_preserve_order(candidate_item_ids)

        candidate_track_ids: list[str] = []
        if self.collaborative_recommender is not None:
            artifacts = self.collaborative_recommender.interaction_artifacts
            if artifacts is not None:
                candidate_track_ids.extend(artifacts.track_ids)
        if self.content_recommender is not None:
            candidate_track_ids.extend(self.content_recommender.feature_artifacts.track_ids)
        candidate_track_ids.extend(self.popularity_scores.keys())
        candidate_track_ids.extend(self.novelty_scores.keys())
        return self._deduplicate_preserve_order(candidate_track_ids)

    def _get_user_history_track_ids(self, user_id: str) -> list[str]:
        """Collect user-history track IDs from external or collaborative data."""

        history_track_ids = list(self.user_seed_track_ids_by_user.get(user_id, []))
        history_track_ids.extend(self._get_collaborative_history_track_ids(user_id))
        return self._deduplicate_preserve_order(history_track_ids)

    def _get_collaborative_history_track_ids(self, user_id: str) -> list[str]:
        """Read seen tracks for a user from the collaborative interaction matrix."""

        if self.collaborative_recommender is None:
            return []

        artifacts = self.collaborative_recommender.interaction_artifacts
        if artifacts is None or user_id not in artifacts.user_index_by_id:
            return []

        user_index = artifacts.user_index_by_id[user_id]
        interaction_row = artifacts.interaction_matrix[user_index]
        return [artifacts.track_ids[item_index] for item_index in interaction_row.indices]

    def _get_collaborative_scores(
        self,
        user_id: str,
        candidate_track_ids: list[str],
    ) -> dict[str, float]:
        """Fetch collaborative scores for the supplied user and candidates."""

        if self.collaborative_recommender is None:
            return {}

        recommendations = self.collaborative_recommender.recommend_for_user(
            user_id=user_id,
            k=len(candidate_track_ids),
            candidate_track_ids=candidate_track_ids,
        )
        return {result.item_id: float(result.score) for result in recommendations}

    def _get_content_scores(
        self,
        user_history_track_ids: list[str],
        candidate_track_ids: list[str],
    ) -> dict[str, float]:
        """Fetch content-based scores using the user's history as seed tracks."""

        if self.content_recommender is None or not user_history_track_ids:
            return {}

        if hasattr(self.content_recommender, "score_candidates_from_seed_tracks"):
            return self.content_recommender.score_candidates_from_seed_tracks(
                seed_track_ids=user_history_track_ids,
                seen_track_ids=user_history_track_ids,
                candidate_track_ids=candidate_track_ids,
            )

        candidate_track_set = set(candidate_track_ids)
        recommendations = self.content_recommender.recommend_from_seed_tracks(
            seed_track_ids=user_history_track_ids,
            seen_track_ids=user_history_track_ids,
            k=len(self.content_recommender.feature_artifacts.track_ids),
        )
        return {
            result.track_id: float(result.similarity_score)
            for result in recommendations
            if result.track_id in candidate_track_set
        }

    def _get_popularity_prior_scores(self, candidate_track_ids: list[str]) -> dict[str, float]:
        """Return track-level popularity prior scores for candidate tracks.

        The popularity prior is a stabilizing signal. It helps the system return
        reasonable tracks when personalized evidence is sparse or unavailable.
        """

        return {
            track_id: float(self.popularity_scores[track_id])
            for track_id in candidate_track_ids
            if track_id in self.popularity_scores
        }

    def _get_novelty_scores(self, candidate_track_ids: list[str]) -> dict[str, float]:
        """Return novelty scores for candidate tracks.

        Novelty encourages discovery. If explicit novelty scores are not
        provided, the recommender derives a simple proxy from inverse
        popularity so less-popular tracks can still surface.
        """

        if self.novelty_scores:
            return {
                track_id: float(self.novelty_scores[track_id])
                for track_id in candidate_track_ids
                if track_id in self.novelty_scores
            }

        if not self.popularity_scores:
            return {}

        return {
            track_id: -float(self.popularity_scores[track_id])
            for track_id in candidate_track_ids
            if track_id in self.popularity_scores
        }

    def _normalize_component_scores(self, component_scores: dict[str, float]) -> dict[str, float]:
        """Normalize one score component into a comparable 0-to-1 range.

        Normalizing each component keeps the blend weights interpretable:
        `w1` controls the collaborative contribution, `w2` controls the content
        contribution, and so on, without one raw scale dominating the others.
        """

        if not component_scores:
            return {}

        score_values = list(component_scores.values())
        min_score = min(score_values)
        max_score = max(score_values)

        if max_score == min_score:
            if max_score <= 0.0:
                return {}
            return {track_id: 1.0 for track_id in component_scores}

        score_range = max_score - min_score
        return {
            track_id: (float(score) - min_score) / score_range
            for track_id, score in component_scores.items()
        }

    def _get_discovery_scores(
        self,
        candidate_track_ids: list[str],
        collaborative_scores: dict[str, float],
        content_scores: dict[str, float],
        novelty_scores: dict[str, float],
    ) -> dict[str, float]:
        """Compute discovery scores from familiarity and novelty components.

        Discovery is not the same as raw novelty. Instead, it rewards tracks
        that sit in the sweet spot between familiarity and exploration.
        """

        return self.discovery_scorer.score_candidates(
            candidate_track_ids=candidate_track_ids,
            collaborative_scores=collaborative_scores,
            content_scores=content_scores,
            novelty_scores=novelty_scores,
        )

    def _should_use_cold_start_fallback(
        self,
        collaborative_scores: dict[str, float],
        content_scores: dict[str, float],
    ) -> bool:
        """Return whether the ranking must fall back to non-personalized priors."""

        return not collaborative_scores and not content_scores

    def _build_ranked_recommendations(
        self,
        candidate_track_ids: list[str],
        component_scores: HybridComponentScores,
        use_cold_start_fallback: bool,
    ) -> list[HybridRecommendation]:
        """Rank candidate tracks from the weighted hybrid score."""

        ranked_recommendations: list[HybridRecommendation] = []

        for track_id in candidate_track_ids:
            score_breakdown = self._build_score_breakdown(
                track_id=track_id,
                component_scores=component_scores,
            )
            if score_breakdown.final_score <= 0.0:
                continue

            ranked_recommendations.append(
                self._build_hybrid_recommendation(
                    track_id=track_id,
                    score_breakdown=score_breakdown,
                    use_cold_start_fallback=use_cold_start_fallback,
                )
            )

        ranked_recommendations.sort(key=lambda item: (-item.final_score, item.item_id))
        return ranked_recommendations

    def _build_score_breakdown(
        self,
        track_id: str,
        component_scores: HybridComponentScores,
    ) -> HybridScoreBreakdown:
        """Build the component-level score explanation for one candidate.

        Each component has a clear interview-friendly role:
        collaborative filtering captures shared user behavior, content captures
        musical similarity, novelty rewards exploration, popularity provides a
        robust prior for sparse settings, and discovery rewards the overlap
        between familiarity and novelty.
        """

        collaborative_score = component_scores.collaborative.get(track_id, 0.0)
        content_score = component_scores.content.get(track_id, 0.0)
        novelty_score = component_scores.novelty.get(track_id, 0.0)
        popularity_prior = component_scores.popularity_prior.get(track_id, 0.0)
        discovery_score = component_scores.discovery.get(track_id, 0.0)
        final_score = self._compute_final_score(
            collaborative_score=collaborative_score,
            content_score=content_score,
            novelty_score=novelty_score,
            popularity_prior=popularity_prior,
            discovery_score=discovery_score,
        )
        return HybridScoreBreakdown(
            collaborative_score=collaborative_score,
            content_score=content_score,
            novelty_score=novelty_score,
            popularity_prior=popularity_prior,
            discovery_score=discovery_score,
            final_score=final_score,
        )

    def _compute_final_score(
        self,
        collaborative_score: float,
        content_score: float,
        novelty_score: float,
        popularity_prior: float,
        discovery_score: float,
    ) -> float:
        """Compute the final weighted ranking score for one candidate."""

        return (
            self.weights.collaborative * collaborative_score
            + self.weights.content * content_score
            + self.weights.novelty * novelty_score
            + self.weights.popularity_prior * popularity_prior
            + self.weights.discovery * discovery_score
        )

    def _build_hybrid_recommendation(
        self,
        track_id: str,
        score_breakdown: HybridScoreBreakdown,
        use_cold_start_fallback: bool,
    ) -> HybridRecommendation:
        """Build a metadata-rich hybrid recommendation object."""

        metadata = self._metadata_by_track_id.get(
            track_id,
            TrackMetadata(track_name=track_id, artist_name=""),
        )
        return HybridRecommendation(
            item_id=track_id,
            score=score_breakdown.final_score,
            source=self.name,
            track_name=metadata.track_name,
            artist_name=metadata.artist_name,
            score_breakdown=score_breakdown,
            used_cold_start_fallback=use_cold_start_fallback,
        )

    def _build_metadata_lookup(
        self,
        track_catalog: pd.DataFrame | None,
    ) -> dict[str, TrackMetadata]:
        """Build a track metadata lookup for recommendation display fields."""

        if track_catalog is None or track_catalog.empty:
            return {}

        required_columns = {"track_id", "track_name", "artist_name"}
        if not required_columns.issubset(track_catalog.columns):
            return {}

        metadata_lookup: dict[str, TrackMetadata] = {}
        for row in track_catalog[["track_id", "track_name", "artist_name"]].drop_duplicates(
            subset=["track_id"]
        ).itertuples(index=False):
            metadata_lookup[str(row.track_id)] = TrackMetadata(
                track_name=str(row.track_name),
                artist_name=str(row.artist_name),
            )
        return metadata_lookup

    def _deduplicate_preserve_order(self, values: list[str]) -> list[str]:
        """Deduplicate strings while preserving their original order."""

        seen_values: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            if value in seen_values:
                continue
            seen_values.add(value)
            unique_values.append(value)
        return unique_values
