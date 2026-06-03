"""Tests for bridging Spotify history into the demo recommendation flow."""

from __future__ import annotations

import pandas as pd

from app.demo_data import build_demo_track_catalog
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


def test_spotify_recommendation_adapter_builds_demo_compatible_profile() -> None:
    """The adapter should map Spotify history onto demo-catalog seed tracks."""

    demo_track_catalog = build_demo_track_catalog()
    spotify_history_frame = demo_track_catalog.loc[
        demo_track_catalog["track_id"].isin(["track_1", "track_3"])
    ].copy()
    spotify_history_frame["track_id"] = ["spotify_track_1", "spotify_track_3"]

    snapshot = ListeningHistorySnapshot(
        user_id="spotify_user_1",
        display_name="Casey",
        recent_tracks=[
            RecentTrackSummary(
                track_id="spotify_track_1",
                track_name="Morning Run",
                artist_name="Casey Artist",
                played_at="2024-01-01T00:00:00Z",
            ),
            RecentTrackSummary(
                track_id="spotify_track_3",
                track_name="Golden Hour",
                artist_name="Casey Artist",
                played_at="2024-01-01T00:05:00Z",
            ),
        ],
        track_level_frame=spotify_history_frame,
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["spotify_track_1", "spotify_track_3"],
    )

    adapter = SpotifyRecommendationAdapter()

    context = adapter.build_context(snapshot, demo_track_catalog)

    assert context is not None
    assert context.profile.user_id == "spotify_recent::spotify_user_1"
    assert context.profile.seed_track_ids
    assert set(context.profile.seed_track_ids).issubset(set(demo_track_catalog["track_id"]))
    assert "Spotify listening" in context.source_message
