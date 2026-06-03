"""Tests for the hybrid recommendation module."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from models.base_recommender import RecommendationResult
from models.content_recommender import ContentRecommendation
from models.hybrid_recommender import HybridRecommender


class StubCollaborativeRecommender:
    """Return deterministic collaborative scores for one user."""

    def __init__(self, scores_by_user: dict[str, dict[str, float]]) -> None:
        """Store user-specific collaborative score mappings."""

        self.scores_by_user = scores_by_user
        self.interaction_artifacts = None

    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
        candidate_track_ids: list[str] | None = None,
    ) -> list[RecommendationResult]:
        """Return deterministic collaborative recommendation results."""

        candidate_track_set = set(candidate_track_ids or [])
        user_scores = self.scores_by_user.get(user_id, {})
        ranked_items = sorted(user_scores.items(), key=lambda item: (-item[1], item[0]))
        results: list[RecommendationResult] = []
        for track_id, score in ranked_items:
            if candidate_track_ids is not None and track_id not in candidate_track_set:
                continue
            results.append(
                RecommendationResult(
                    item_id=track_id,
                    score=score,
                    source="collaborative",
                )
            )
        return results[:k]


class StubContentRecommender:
    """Return deterministic content-based scores for seed-track workflows."""

    def __init__(
        self,
        item_scores: dict[str, float],
        track_catalog: pd.DataFrame,
    ) -> None:
        """Store a deterministic content ranking and track catalog."""

        self.item_scores = item_scores
        self.track_catalog = track_catalog
        self.feature_artifacts = SimpleNamespace(track_ids=list(item_scores.keys()))

    def recommend_from_seed_tracks(
        self,
        seed_track_ids: list[str],
        seen_track_ids: list[str] | None = None,
        k: int = 10,
    ) -> list[ContentRecommendation]:
        """Return deterministic content recommendations for known seed tracks."""

        if not seed_track_ids:
            return []

        excluded_track_ids = set(seen_track_ids or [])
        ranked_items = sorted(self.item_scores.items(), key=lambda item: (-item[1], item[0]))
        recommendations: list[ContentRecommendation] = []

        for track_id, score in ranked_items:
            if track_id in excluded_track_ids:
                continue

            track_row = self.track_catalog.loc[self.track_catalog["track_id"] == track_id].iloc[0]
            recommendations.append(
                ContentRecommendation(
                    item_id=track_id,
                    score=score,
                    source="content",
                    track_name=str(track_row["track_name"]),
                    artist_name=str(track_row["artist_name"]),
                )
            )

        return recommendations[:k]


def build_track_catalog() -> pd.DataFrame:
    """Create a small metadata table for hybrid recommendation tests."""

    return pd.DataFrame(
        [
            {"track_id": "track_a", "track_name": "Track A", "artist_name": "Artist A"},
            {"track_id": "track_b", "track_name": "Track B", "artist_name": "Artist B"},
            {"track_id": "track_c", "track_name": "Track C", "artist_name": "Artist C"},
        ]
    )


def test_hybrid_recommender_combines_weighted_component_scores() -> None:
    """The hybrid ranker should blend collaborative and content scores."""

    track_catalog = build_track_catalog()
    recommender = HybridRecommender(
        collaborative_recommender=StubCollaborativeRecommender(
            scores_by_user={"user_1": {"track_b": 0.9, "track_c": 0.2}}
        ),
        content_recommender=StubContentRecommender(
            item_scores={"track_a": 0.8, "track_b": 0.4, "track_c": 0.1},
            track_catalog=track_catalog,
        ),
        weights={"collaborative": 1.0, "content": 0.5},
        user_seed_track_ids_by_user={"user_1": ["seed_track"]},
        track_catalog=track_catalog,
    )

    results = recommender.recommend(
        user_id="user_1",
        candidate_item_ids=["track_a", "track_b", "track_c"],
        k=3,
    )

    assert [item.item_id for item in results] == ["track_b", "track_a"]
    assert results[0].track_name == "Track B"
    assert results[0].score_breakdown.collaborative_score == 1.0
    assert results[0].score_breakdown.content_score == 0.4285714285714286


def test_hybrid_recommender_respects_configurable_weights() -> None:
    """Weight changes should shift the final ranking outcome."""

    track_catalog = build_track_catalog()
    recommender = HybridRecommender(
        collaborative_recommender=StubCollaborativeRecommender(
            scores_by_user={"user_1": {"track_b": 0.9, "track_c": 0.2}}
        ),
        content_recommender=StubContentRecommender(
            item_scores={"track_a": 0.8, "track_b": 0.4, "track_c": 0.1},
            track_catalog=track_catalog,
        ),
        weights={"collaborative": 0.2, "content": 2.0},
        user_seed_track_ids_by_user={"user_1": ["seed_track"]},
        track_catalog=track_catalog,
    )

    results = recommender.recommend(
        user_id="user_1",
        candidate_item_ids=["track_a", "track_b", "track_c"],
        k=3,
    )

    assert results[0].item_id == "track_a"
    assert results[0].score_breakdown.final_score > results[1].score_breakdown.final_score


def test_hybrid_recommender_uses_cold_start_fallback() -> None:
    """Cold-start users should still receive popularity-plus-novelty results."""

    track_catalog = build_track_catalog()
    recommender = HybridRecommender(
        collaborative_recommender=StubCollaborativeRecommender(scores_by_user={}),
        content_recommender=StubContentRecommender(
            item_scores={"track_a": 0.8, "track_b": 0.4, "track_c": 0.1},
            track_catalog=track_catalog,
        ),
        weights={"novelty": 0.8, "popularity_prior": 0.2},
        popularity_scores={"track_a": 0.9, "track_b": 0.5, "track_c": 0.2},
        novelty_scores={"track_a": 0.1, "track_b": 0.2, "track_c": 1.0},
        track_catalog=track_catalog,
    )

    results = recommender.recommend(user_id="new_user", k=3)

    assert results[0].item_id == "track_c"
    assert results[0].used_cold_start_fallback is True
    assert all(item.score_breakdown.collaborative_score == 0.0 for item in results)
    assert all(item.score_breakdown.content_score == 0.0 for item in results)


def test_hybrid_recommender_filters_seen_tracks() -> None:
    """Seen tracks from user history should not be recommended back."""

    track_catalog = build_track_catalog()
    recommender = HybridRecommender(
        collaborative_recommender=StubCollaborativeRecommender(
            scores_by_user={"user_1": {"track_a": 0.9, "track_b": 0.8}}
        ),
        content_recommender=StubContentRecommender(
            item_scores={"track_a": 0.9, "track_b": 0.4, "track_c": 0.3},
            track_catalog=track_catalog,
        ),
        weights={"collaborative": 1.0, "content": 1.0},
        user_seed_track_ids_by_user={"user_1": ["track_a"]},
        track_catalog=track_catalog,
    )

    results = recommender.recommend(
        user_id="user_1",
        candidate_item_ids=["track_a", "track_b", "track_c"],
        k=3,
    )

    assert "track_a" not in [item.item_id for item in results]


def test_hybrid_recommender_ranking_changes_when_discovery_weight_changes() -> None:
    """Increasing discovery weight should promote balanced exploratory tracks."""

    track_catalog = build_track_catalog()
    collaborative = StubCollaborativeRecommender(
        scores_by_user={"user_1": {"track_a": 0.9, "track_b": 0.55, "track_c": 0.1}}
    )
    content = StubContentRecommender(
        item_scores={"track_a": 0.9, "track_b": 0.6, "track_c": 0.2},
        track_catalog=track_catalog,
    )

    baseline_recommender = HybridRecommender(
        collaborative_recommender=collaborative,
        content_recommender=content,
        weights={"collaborative": 1.0, "content": 1.0, "novelty": 0.5, "discovery": 0.0},
        novelty_scores={"track_a": 0.1, "track_b": 0.8, "track_c": 1.0},
        user_seed_track_ids_by_user={"user_1": ["seed_track"]},
        track_catalog=track_catalog,
    )
    exploratory_recommender = HybridRecommender(
        collaborative_recommender=collaborative,
        content_recommender=content,
        weights={"collaborative": 1.0, "content": 1.0, "novelty": 0.5, "discovery": 2.5},
        novelty_scores={"track_a": 0.1, "track_b": 0.8, "track_c": 1.0},
        user_seed_track_ids_by_user={"user_1": ["seed_track"]},
        track_catalog=track_catalog,
    )

    baseline_results = baseline_recommender.recommend(
        user_id="user_1",
        candidate_item_ids=["track_a", "track_b", "track_c"],
        k=3,
    )
    exploratory_results = exploratory_recommender.recommend(
        user_id="user_1",
        candidate_item_ids=["track_a", "track_b", "track_c"],
        k=3,
    )

    assert baseline_results[0].item_id == "track_a"
    assert exploratory_results[0].item_id == "track_b"
    assert exploratory_results[0].score_breakdown.discovery_score > exploratory_results[1].score_breakdown.discovery_score
