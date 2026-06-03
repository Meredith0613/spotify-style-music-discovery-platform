"""Tests for lightweight Spotify-aware reranking."""

from __future__ import annotations

import pandas as pd

from app.demo_data import DemoUserProfile
from app.demo_service import DemoAppService
from models.hybrid_recommender import HybridRecommendation, HybridScoreBreakdown
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter, SpotifyRecommendationContext
from services.spotify_reranking_service import SpotifyRerankingService
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


def test_spotify_reranking_service_reorders_recommendations() -> None:
    """Recent Spotify taste signals should influence the final ranking order."""

    service = DemoAppService()
    spotify_history_frame = service.track_catalog.loc[
        service.track_catalog["track_id"].isin(["track_1", "track_9"])
    ].copy()
    spotify_history_frame["track_id"] = ["spotify_track_1", "spotify_track_9"]
    snapshot = ListeningHistorySnapshot(
        user_id="spotify_user_1",
        display_name="Casey",
        recent_tracks=[
            RecentTrackSummary(
                track_id="spotify_track_1",
                track_name="Neon Skyline",
                artist_name="City Echo",
                played_at="2024-01-01T00:00:00Z",
            ),
            RecentTrackSummary(
                track_id="spotify_track_9",
                track_name="City Run",
                artist_name="Pulse Unit",
                played_at="2024-01-01T00:05:00Z",
            ),
        ],
        track_level_frame=spotify_history_frame,
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["spotify_track_1", "spotify_track_9"],
    )
    context = SpotifyRecommendationAdapter().build_context(snapshot, service.track_catalog)

    assert context is not None

    recommendations = [
        _build_recommendation("track_4", "Soft Focus", "Velvet Transit", 1.000),
        _build_recommendation("track_12", "Bright Avenue", "Golden Habit", 0.990),
        _build_recommendation("track_6", "Iron Pulse", "Voltage Lane", 0.980),
    ]

    reranking_result = SpotifyRerankingService().rerank_recommendations(
        spotify_context=context,
        recommendations=recommendations,
        listening_history_snapshot=snapshot,
        demo_track_catalog=service.track_catalog,
    )

    assert reranking_result.applied
    assert reranking_result.recommendations[0].track_id != recommendations[0].track_id
    assert reranking_result.score_adjustments_by_track_id["track_12"] > 0.0
    assert reranking_result.score_adjustments_by_track_id["track_6"] > 0.0


def test_spotify_reranking_service_keeps_original_ranking_without_signal() -> None:
    """Sparse Spotify metadata should leave the original ranking unchanged."""

    service = DemoAppService()
    recommendations = [
        _build_recommendation("track_4", "Soft Focus", "Velvet Transit", 1.000),
        _build_recommendation("track_12", "Bright Avenue", "Golden Habit", 0.990),
    ]
    sparse_snapshot = ListeningHistorySnapshot(
        user_id="spotify_user_2",
        display_name="Riley",
        recent_tracks=[],
        track_level_frame=pd.DataFrame(),
        interaction_frame=pd.DataFrame(),
        seed_track_ids=[],
    )
    sparse_context = SpotifyRecommendationContext(
        profile=DemoUserProfile(
            user_id="spotify_recent::spotify_user_2",
            display_name="Riley | Spotify Recent",
            summary="Sparse Spotify history.",
            seed_track_ids=[],
            preferred_mood="calm",
        ),
        source_message="Recommendations powered by your recent Spotify listening.",
        matched_seed_track_ids=[],
    )

    reranking_result = SpotifyRerankingService().rerank_recommendations(
        spotify_context=sparse_context,
        recommendations=recommendations,
        listening_history_snapshot=sparse_snapshot,
        demo_track_catalog=service.track_catalog,
    )

    assert not reranking_result.applied
    assert [recommendation.track_id for recommendation in reranking_result.recommendations] == [
        recommendation.track_id for recommendation in recommendations
    ]


def _build_recommendation(
    track_id: str,
    track_name: str,
    artist_name: str,
    score: float,
) -> HybridRecommendation:
    """Create a minimal hybrid recommendation for reranking tests."""

    return HybridRecommendation(
        item_id=track_id,
        score=score,
        source="hybrid",
        track_name=track_name,
        artist_name=artist_name,
        score_breakdown=HybridScoreBreakdown(
            collaborative_score=0.5,
            content_score=0.5,
            novelty_score=0.2,
            popularity_prior=0.1,
            discovery_score=0.3,
            final_score=score,
        ),
        used_cold_start_fallback=False,
    )
