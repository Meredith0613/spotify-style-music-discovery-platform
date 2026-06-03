"""Tests for the reusable Streamlit demo service."""

from __future__ import annotations

import pandas as pd

from app.demo_service import DemoAppService
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


def test_demo_service_builds_hybrid_view_state() -> None:
    """The demo service should return recommendations, explanations, and a playlist."""

    service = DemoAppService()
    view_state = service.build_demo_view(
        user_id="runner_pop",
        exploration_level=0.4,
        recommendation_count=5,
        mood_label="workout",
        playlist_length=3,
        include_taste_clusters=True,
    )

    assert view_state.recommendations
    assert not view_state.recommendation_table.empty
    assert len(view_state.explanations) == len(view_state.recommendations)
    assert all(not explanation.spotify_rationale_lines for explanation in view_state.explanations)
    assert len(view_state.playlist.tracks) <= 3
    assert view_state.taste_clusters is not None


def test_demo_service_supports_cold_start_profile() -> None:
    """The cold-start demo profile should still receive fallback recommendations."""

    service = DemoAppService()
    view_state = service.build_demo_view(
        user_id="new_listener",
        exploration_level=0.6,
        recommendation_count=4,
        mood_label="calm",
        playlist_length=3,
        include_taste_clusters=False,
    )

    assert view_state.recommendations
    assert any(recommendation.used_cold_start_fallback for recommendation in view_state.recommendations)


def test_demo_service_supports_spotify_derived_profile() -> None:
    """The demo service should build recommendations from an adapter-generated profile."""

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
                track_name="City Run",
                artist_name="Pulse Unit",
                played_at="2024-01-01T00:00:00Z",
            ),
            RecentTrackSummary(
                track_id="spotify_track_9",
                track_name="Neon Skyline",
                artist_name="City Echo",
                played_at="2024-01-01T00:05:00Z",
            ),
        ],
        track_level_frame=spotify_history_frame,
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["spotify_track_1", "spotify_track_9"],
    )
    adapter = SpotifyRecommendationAdapter()
    context = adapter.build_context(snapshot, service.track_catalog)

    assert context is not None

    view_state = service.build_view_for_profile(
        profile=context.profile,
        exploration_level=0.4,
        recommendation_count=5,
        mood_label=context.profile.preferred_mood,
        playlist_length=3,
        include_taste_clusters=False,
    )

    assert view_state.recommendations
    assert view_state.profile.user_id == context.profile.user_id
