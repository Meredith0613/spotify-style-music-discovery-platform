"""Content-based recommendation model built from track feature vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from features.feature_builder import FeatureMatrixArtifacts

from .base_recommender import BaseRecommender, RecommendationResult


@dataclass(slots=True)
class ContentRecommendation(RecommendationResult):
    """Represent one ranked content-based track recommendation.

    Attributes:
        item_id: Recommended track identifier used by the shared recommender layer.
        score: Similarity score used by the shared recommender layer.
        source: Recommender name that produced the result.
        track_name: Human-readable track name for the recommendation.
        artist_name: Human-readable artist name for the recommendation.
    """

    track_name: str
    artist_name: str

    @property
    def track_id(self) -> str:
        """Return the recommended track identifier."""

        return self.item_id

    @property
    def similarity_score(self) -> float:
        """Return the recommendation similarity score."""

        return self.score


@dataclass(slots=True)
class TrackDisplayMetadata:
    """Store human-readable metadata used in recommendation outputs.

    Attributes:
        track_name: Display name for the track.
        artist_name: Display name for the primary artist.
    """

    track_name: str
    artist_name: str


@dataclass(slots=True)
class FeatureContribution:
    """Represent one feature's contribution to a similarity explanation.

    Attributes:
        feature_name: Name of the feature contributing to similarity.
        seed_value: Seed-profile feature value in standardized space.
        candidate_value: Candidate-track feature value in standardized space.
        contribution_score: Signed contribution to the cosine similarity score.
    """

    feature_name: str
    seed_value: float
    candidate_value: float
    contribution_score: float


@dataclass(slots=True)
class RecommendationExplanation:
    """Represent a feature-level explanation for one recommendation.

    Attributes:
        seed_track_ids: Seed tracks used to build the query taste profile.
        candidate_track_id: Recommended track being explained.
        overall_similarity_score: Overall cosine similarity between the seed profile and candidate.
        top_feature_contributions: Highest-impact feature contributions to similarity.
    """

    seed_track_ids: list[str]
    candidate_track_id: str
    overall_similarity_score: float
    top_feature_contributions: list[FeatureContribution]


@dataclass(slots=True)
class ContentRecommender(BaseRecommender):
    """Recommend tracks using cosine similarity in content feature space.

    Attributes:
        feature_artifacts: Standardized track feature artifacts aligned to track IDs.
        track_catalog: Track-level table containing recommendation metadata.
    """

    feature_artifacts: FeatureMatrixArtifacts
    track_catalog: pd.DataFrame
    name: str = "content"
    minimum_similarity_score: float = 0.0
    _track_index_by_id: dict[str, int] = field(init=False, repr=False)
    _metadata_by_track_id: dict[str, TrackDisplayMetadata] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build index structures used during recommendation and explanation."""

        self._track_index_by_id = {
            track_id: index
            for index, track_id in enumerate(self.feature_artifacts.track_ids)
        }
        self._metadata_by_track_id = self._build_metadata_lookup(self.track_catalog)

    def recommend(
        self,
        user_id: str,
        candidate_item_ids: list[str],
        k: int,
    ) -> list[RecommendationResult]:
        """Adapt the shared recommender interface to a seed-track workflow.

        This adapter treats `candidate_item_ids` as seed tracks for compatibility
        with the shared base interface. The richer content-specific workflow lives
        in `recommend_from_seed_tracks`, which should be preferred by callers.
        """

        recommendations = self.recommend_from_seed_tracks(
            seed_track_ids=candidate_item_ids,
            seen_track_ids=candidate_item_ids,
            k=k,
        )
        return [
            RecommendationResult(
                item_id=item.track_id,
                score=item.similarity_score,
                source=self.name,
            )
            for item in recommendations
        ]

    def recommend_from_seed_tracks(
        self,
        seed_track_ids: list[str],
        seen_track_ids: list[str] | None = None,
        k: int = 10,
    ) -> list[ContentRecommendation]:
        """Return top-k similar tracks for one or more seed tracks.

        Args:
            seed_track_ids: Seed track IDs used to define the user's taste profile.
            seen_track_ids: Track IDs that should be excluded from recommendations.
            k: Maximum number of recommendations to return.

        Returns:
            Ranked content recommendations with track metadata and similarity scores.
        """

        valid_seed_track_ids = self._filter_known_track_ids(seed_track_ids)
        if not valid_seed_track_ids or k <= 0:
            return []

        seed_profile = self._build_seed_profile(valid_seed_track_ids)
        excluded_track_ids = self._build_excluded_track_ids(
            seed_track_ids=valid_seed_track_ids,
            seen_track_ids=seen_track_ids,
        )

        ranked_recommendations: list[ContentRecommendation] = []

        # The recommendation loop is intentionally simple: skip seen tracks,
        # score the rest in feature space, then attach display metadata.
        for candidate_track_id in self.feature_artifacts.track_ids:
            if candidate_track_id in excluded_track_ids:
                continue

            similarity_score = self._score_candidate_track(
                seed_profile=seed_profile,
                candidate_track_id=candidate_track_id,
            )
            if not self._is_recommendable_similarity(similarity_score):
                continue

            ranked_recommendations.append(
                self._build_content_recommendation(
                    track_id=candidate_track_id,
                    similarity_score=similarity_score,
                )
            )

        ranked_recommendations.sort(key=lambda item: (-item.similarity_score, item.track_id))
        return ranked_recommendations[:k]

    def score_candidates_from_seed_tracks(
        self,
        seed_track_ids: list[str],
        candidate_track_ids: list[str],
        seen_track_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Score only the supplied candidate tracks against a seed profile."""

        valid_seed_track_ids = self._filter_known_track_ids(seed_track_ids)
        if not valid_seed_track_ids:
            return {}

        seed_profile = self._build_seed_profile(valid_seed_track_ids)
        excluded_track_ids = self._build_excluded_track_ids(
            seed_track_ids=valid_seed_track_ids,
            seen_track_ids=seen_track_ids,
        )
        scores: dict[str, float] = {}
        for raw_candidate_track_id in candidate_track_ids:
            candidate_track_id = str(raw_candidate_track_id)
            if candidate_track_id in excluded_track_ids or candidate_track_id not in self._track_index_by_id:
                continue
            similarity_score = self._score_candidate_track(
                seed_profile=seed_profile,
                candidate_track_id=candidate_track_id,
            )
            if self._is_recommendable_similarity(similarity_score):
                scores[candidate_track_id] = float(similarity_score)
        return scores

    def explain_recommendation(
        self,
        seed_track_ids: list[str],
        candidate_track_id: str,
        top_n_features: int = 5,
    ) -> RecommendationExplanation:
        """Explain which features contributed most to one recommendation.

        Args:
            seed_track_ids: Seed tracks used to form the query profile.
            candidate_track_id: Recommended track to explain.
            top_n_features: Maximum number of feature contributions to return.

        Returns:
            A structured explanation containing the top feature contributions.
        """

        valid_seed_track_ids = self._validate_explanation_inputs(
            seed_track_ids=seed_track_ids,
            candidate_track_id=candidate_track_id,
        )

        seed_profile = self._build_seed_profile(valid_seed_track_ids)
        candidate_vector = self._get_feature_vector(candidate_track_id)
        overall_similarity_score = self._cosine_similarity(seed_profile, candidate_vector)
        feature_contributions = self._build_feature_contributions(
            seed_profile=seed_profile,
            candidate_vector=candidate_vector,
        )
        return RecommendationExplanation(
            seed_track_ids=valid_seed_track_ids,
            candidate_track_id=candidate_track_id,
            overall_similarity_score=overall_similarity_score,
            top_feature_contributions=feature_contributions[:top_n_features],
        )

    def _build_seed_profile(self, seed_track_ids: list[str]) -> np.ndarray:
        """Average seed-track vectors into one content preference profile.

        Args:
            seed_track_ids: Valid seed track identifiers.

        Returns:
            Mean standardized feature vector for the seed tracks.
        """

        seed_vectors = [
            self._get_feature_vector(track_id)
            for track_id in seed_track_ids
        ]
        return np.mean(seed_vectors, axis=0)

    def _filter_known_track_ids(self, track_ids: list[str]) -> list[str]:
        """Keep only track IDs that exist in the feature matrix."""

        return [track_id for track_id in track_ids if track_id in self._track_index_by_id]

    def _build_excluded_track_ids(
        self,
        seed_track_ids: list[str],
        seen_track_ids: list[str] | None,
    ) -> set[str]:
        """Build the set of tracks that should not be recommended."""

        excluded_track_ids = set(seen_track_ids or [])
        excluded_track_ids.update(seed_track_ids)
        return excluded_track_ids

    def _score_candidate_track(
        self,
        seed_profile: np.ndarray,
        candidate_track_id: str,
    ) -> float:
        """Score one candidate track against the seed taste profile."""

        candidate_vector = self._get_feature_vector(candidate_track_id)
        return self._cosine_similarity(seed_profile, candidate_vector)

    def _is_recommendable_similarity(self, similarity_score: float) -> bool:
        """Return whether a similarity score is high enough to recommend."""

        return similarity_score > self.minimum_similarity_score

    def _build_content_recommendation(
        self,
        track_id: str,
        similarity_score: float,
    ) -> ContentRecommendation:
        """Build a metadata-rich recommendation result for one track."""

        metadata = self._metadata_by_track_id.get(
            track_id,
            TrackDisplayMetadata(track_name=track_id, artist_name=""),
        )
        return ContentRecommendation(
            item_id=track_id,
            score=similarity_score,
            source=self.name,
            track_name=metadata.track_name,
            artist_name=metadata.artist_name,
        )

    def _validate_explanation_inputs(
        self,
        seed_track_ids: list[str],
        candidate_track_id: str,
    ) -> list[str]:
        """Validate explanation inputs and return the usable seed tracks."""

        valid_seed_track_ids = self._filter_known_track_ids(seed_track_ids)
        if not valid_seed_track_ids:
            raise ValueError("At least one valid seed track ID is required for explanation.")
        if candidate_track_id not in self._track_index_by_id:
            raise ValueError(f"Unknown candidate track ID: {candidate_track_id}")
        return valid_seed_track_ids

    def _build_feature_contributions(
        self,
        seed_profile: np.ndarray,
        candidate_vector: np.ndarray,
    ) -> list[FeatureContribution]:
        """Build sorted per-feature contribution explanations."""

        contribution_scale = self._safe_norm(seed_profile) * self._safe_norm(candidate_vector)
        contributions: list[FeatureContribution] = []

        # Each feature gets a small signed contribution score so we can explain
        # similarity in concrete musical terms instead of only one final number.
        for feature_name, seed_value, candidate_value in zip(
            self.feature_artifacts.feature_names,
            seed_profile,
            candidate_vector,
        ):
            if self._skip_feature_in_explanation(feature_name, seed_value, candidate_value):
                continue

            contributions.append(
                FeatureContribution(
                    feature_name=feature_name,
                    seed_value=float(seed_value),
                    candidate_value=float(candidate_value),
                    contribution_score=self._calculate_feature_contribution(
                        seed_value=seed_value,
                        candidate_value=candidate_value,
                        contribution_scale=contribution_scale,
                    ),
                )
            )

        contributions.sort(key=lambda item: (-item.contribution_score, item.feature_name))
        return contributions

    def _skip_feature_in_explanation(
        self,
        feature_name: str,
        seed_value: float,
        candidate_value: float,
    ) -> bool:
        """Hide explanation features that are mathematically valid but unhelpful."""

        return (
            feature_name.startswith("genre_")
            and seed_value <= 0.0
            and candidate_value <= 0.0
        )

    def _calculate_feature_contribution(
        self,
        seed_value: float,
        candidate_value: float,
        contribution_scale: float,
    ) -> float:
        """Calculate one feature's signed cosine-similarity contribution."""

        if contribution_scale <= 0.0:
            return 0.0
        return float((seed_value * candidate_value) / contribution_scale)

    def _get_feature_vector(self, track_id: str) -> np.ndarray:
        """Return the feature vector for one track ID.

        Args:
            track_id: Track identifier aligned to the feature artifacts.

        Returns:
            Standardized numeric feature vector for the track.
        """

        track_index = self._track_index_by_id[track_id]
        return self.feature_artifacts.feature_matrix[track_index]

    def _cosine_similarity(
        self,
        left_vector: np.ndarray,
        right_vector: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two feature vectors.

        Args:
            left_vector: Left-hand vector in standardized content space.
            right_vector: Right-hand vector in standardized content space.

        Returns:
            Cosine similarity score bounded approximately between -1 and 1.
        """

        left_norm = self._safe_norm(left_vector)
        right_norm = self._safe_norm(right_vector)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))

    def _safe_norm(self, vector: np.ndarray) -> float:
        """Return a stable norm for a feature vector.

        Args:
            vector: Numeric feature vector.

        Returns:
            Euclidean norm of the vector.
        """

        return float(np.linalg.norm(vector))

    def _build_metadata_lookup(self, track_catalog: pd.DataFrame) -> dict[str, TrackDisplayMetadata]:
        """Build a metadata lookup used to enrich recommendation outputs.

        Args:
            track_catalog: Track-level DataFrame containing names and artists.

        Returns:
            Mapping from track ID to recommendation display metadata.
        """

        metadata_lookup: dict[str, TrackDisplayMetadata] = {}
        if track_catalog.empty or "track_id" not in track_catalog.columns:
            return metadata_lookup

        for record in track_catalog.to_dict(orient="records"):
            track_id = str(record.get("track_id", ""))
            if not track_id:
                continue

            # Metadata fallbacks keep the recommender resilient to slightly
            # different curated table schemas during early development.
            track_name = self._first_nonempty(
                record.get("track_name"),
                record.get("name"),
                track_id,
            )
            artist_name = self._first_nonempty(
                record.get("primary_artist_name"),
                record.get("artist_name"),
                record.get("artist_names"),
                "",
            )
            metadata_lookup[track_id] = TrackDisplayMetadata(
                track_name=track_name,
                artist_name=artist_name,
            )

        return metadata_lookup

    def _first_nonempty(self, *values: Any) -> str:
        """Return the first non-empty string-like value.

        Args:
            *values: Candidate values to inspect in priority order.

        Returns:
            The first non-empty string representation found.
        """

        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
