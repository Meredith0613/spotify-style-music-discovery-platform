"""Tests for Spotify-specific recommendation explanation enrichment."""

from __future__ import annotations

import pandas as pd

from app.demo_service import DemoAppService, DemoRecommendationExplanation
from services.spotify_explanation_service import SpotifyExplanationService
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


def test_spotify_explanation_service_enriches_existing_explanations() -> None:
    """Spotify-derived recommendations should gain Spotify-specific rationale."""

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

    enriched_explanations = SpotifyExplanationService().enrich_explanations(
        spotify_context=context,
        explanations=view_state.explanations,
        listening_history_snapshot=snapshot,
        demo_track_catalog=service.track_catalog,
    )

    assert len(enriched_explanations) == len(view_state.explanations)
    assert all(explanation.summary_lines for explanation in enriched_explanations)
    assert all(explanation.spotify_rationale_lines for explanation in enriched_explanations)
    assert all(explanation.spotify_recent_track_labels for explanation in enriched_explanations)
    assert all(explanation.spotify_matched_seed_labels for explanation in enriched_explanations)
    assert all(explanation.spotify_inferred_mood for explanation in enriched_explanations)
    assert all(len(explanation.spotify_recent_track_labels) <= 2 for explanation in enriched_explanations)
    assert all(len(explanation.spotify_matched_seed_labels) <= 2 for explanation in enriched_explanations)


def test_spotify_explanation_service_falls_back_when_metadata_is_sparse() -> None:
    """Sparse Spotify metadata should still produce a simple, non-failing rationale."""

    service = DemoAppService()
    spotify_history_frame = service.track_catalog.loc[
        service.track_catalog["track_id"].isin(["track_4", "track_5"])
    ].copy()
    spotify_history_frame["track_id"] = ["spotify_track_4", "spotify_track_5"]
    snapshot = ListeningHistorySnapshot(
        user_id="spotify_user_2",
        display_name="Riley",
        recent_tracks=[],
        track_level_frame=spotify_history_frame,
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["spotify_track_4", "spotify_track_5"],
    )
    adapter = SpotifyRecommendationAdapter()
    context = adapter.build_context(snapshot, service.track_catalog)

    assert context is not None

    view_state = service.build_view_for_profile(
        profile=context.profile,
        exploration_level=0.3,
        recommendation_count=3,
        mood_label=context.profile.preferred_mood,
        playlist_length=3,
        include_taste_clusters=False,
    )

    enriched_explanations = SpotifyExplanationService().enrich_explanations(
        spotify_context=context,
        explanations=view_state.explanations,
        listening_history_snapshot=snapshot,
        demo_track_catalog=service.track_catalog,
    )

    assert enriched_explanations
    assert all(explanation.spotify_rationale_lines for explanation in enriched_explanations)


def test_spotify_explanation_service_varies_item_specific_matches() -> None:
    """Different recommendation cards should surface different top Spotify matches when possible."""

    service = DemoAppService()
    spotify_history_frame = service.track_catalog.loc[
        service.track_catalog["track_id"].isin(["track_1", "track_4", "track_11"])
    ].copy()
    spotify_history_frame["track_id"] = ["spotify_track_1", "spotify_track_4", "spotify_track_11"]
    snapshot = ListeningHistorySnapshot(
        user_id="spotify_user_3",
        display_name="Jordan",
        recent_tracks=[
            RecentTrackSummary(
                track_id="spotify_track_1",
                track_name="Neon Skyline",
                artist_name="City Echo",
                played_at="2024-01-01T00:00:00Z",
            ),
            RecentTrackSummary(
                track_id="spotify_track_4",
                track_name="Soft Focus",
                artist_name="Velvet Transit",
                played_at="2024-01-01T00:05:00Z",
            ),
            RecentTrackSummary(
                track_id="spotify_track_11",
                track_name="Ashes And Echoes",
                artist_name="Grey District",
                played_at="2024-01-01T00:10:00Z",
            ),
        ],
        track_level_frame=spotify_history_frame,
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["spotify_track_1", "spotify_track_4", "spotify_track_11"],
    )
    adapter = SpotifyRecommendationAdapter()
    context = adapter.build_context(snapshot, service.track_catalog)

    assert context is not None

    explanations = [
        DemoRecommendationExplanation(
            track_id="track_12",
            track_name="Bright Avenue",
            artist_name="Golden Habit",
            summary_lines=["Base explanation"],
        ),
        DemoRecommendationExplanation(
            track_id="track_5",
            track_name="Paper Lanterns",
            artist_name="Quiet Avenue",
            summary_lines=["Base explanation"],
        ),
        DemoRecommendationExplanation(
            track_id="track_8",
            track_name="Velvet Morning",
            artist_name="Ash Harbor",
            summary_lines=["Base explanation"],
        ),
    ]

    enriched_explanations = SpotifyExplanationService().enrich_explanations(
        spotify_context=context,
        explanations=explanations,
        listening_history_snapshot=snapshot,
        demo_track_catalog=service.track_catalog,
    )

    matched_seed_sets = {
        tuple(explanation.spotify_matched_seed_labels)
        for explanation in enriched_explanations
    }
    recent_track_sets = {
        tuple(explanation.spotify_recent_track_labels)
        for explanation in enriched_explanations
    }

    assert len(matched_seed_sets) > 1 or len(recent_track_sets) > 1
