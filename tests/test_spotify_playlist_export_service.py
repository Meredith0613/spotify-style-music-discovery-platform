"""Tests for Spotify playlist export service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.demo_data import DemoUserProfile
from app.demo_service import DemoRecommendationExplanation, DemoViewState
from app.streamlit_app import _render_spotify_playlist_export_section
from data.spotify_client import SpotifyAPIClientError
from models.hybrid_recommender import HybridRecommendation, HybridScoreBreakdown
from models.playlist_generator import GeneratedPlaylist
from services.spotify_candidate_service import (
    RecommendationBucket,
    SpotifyCandidateSet,
    SpotifyRealRecommendationResult,
)
from services.spotify_playlist_export_service import (
    SpotifyPlaylistExportResult,
    SpotifyPlaylistExportService,
)


class FakePlaylistClient:
    """Capture playlist API calls without network access."""

    def __init__(self) -> None:
        """Initialize request logs."""

        self.created_playlists: list[dict[str, Any]] = []
        self.added_tracks: list[dict[str, Any]] = []

    def create_playlist(
        self,
        user_token: str,
        user_id: str,
        playlist_name: str,
        description: str,
        public: bool = False,
    ) -> dict[str, Any]:
        """Capture playlist creation arguments."""

        self.created_playlists.append(
            {
                "user_token": user_token,
                "user_id": user_id,
                "playlist_name": playlist_name,
                "description": description,
                "public": public,
            }
        )
        return {
            "playlist_id": "playlist_123",
            "playlist_url": "https://open.spotify.com/playlist/playlist_123",
        }

    def add_tracks_to_playlist(
        self,
        user_token: str,
        playlist_id: str,
        track_uris: list[str] | None = None,
        spotify_track_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Capture playlist track additions."""

        self.added_tracks.append(
            {
                "user_token": user_token,
                "playlist_id": playlist_id,
                "track_uris": track_uris or [],
                "spotify_track_ids": spotify_track_ids or [],
            }
        )
        return {
            "playlist_id": playlist_id,
            "added_track_count": len(spotify_track_ids or track_uris or []),
        }


class FailingPlaylistClient(FakePlaylistClient):
    """Simulate a Spotify playlist API failure."""

    def create_playlist(
        self,
        user_token: str,
        user_id: str,
        playlist_name: str,
        description: str,
        public: bool = False,
    ) -> dict[str, Any]:
        """Raise a controlled client error."""

        raise SpotifyAPIClientError("Spotify POST request failed: 403 forbidden")


class FakeStreamlitModule:
    """Small Streamlit stand-in for export UI no-op tests."""

    def __init__(self) -> None:
        """Initialize captured UI calls."""

        self.session_state: dict[str, Any] = {"spotify_playlist_export_result": "stale"}
        self.warnings: list[str] = []
        self.buttons: list[str] = []

    def warning(self, message: str) -> None:
        """Capture warning messages."""

        self.warnings.append(message)

    def button(self, label: str, **kwargs: Any) -> bool:
        """Capture button labels and return unclicked."""

        self.buttons.append(label)
        return False


def test_playlist_export_service_exports_unique_bucket_tracks() -> None:
    """Bucketed recommendations should export unique Spotify track IDs."""

    client = FakePlaylistClient()
    service = SpotifyPlaylistExportService(client=client)  # type: ignore[arg-type]
    generated_at = datetime(2026, 6, 4, 12, 30, tzinfo=timezone.utc)

    result = service.export_recommendations(
        user_token="user-token",
        user_id="spotify_user",
        spotify_real_recommendation_result=build_real_result_with_buckets(),
        mood_label="happy",
        exploration_level=0.75,
        granted_scopes="user-read-recently-played playlist-modify-private",
        generated_at=generated_at,
    )

    assert result.success is True
    assert result.playlist_id == "playlist_123"
    assert result.playlist_url == "https://open.spotify.com/playlist/playlist_123"
    assert result.track_count == 3
    assert client.created_playlists[0]["playlist_name"] == "Spotify Discovery Mix - 2026-06-04"
    assert client.created_playlists[0]["public"] is False
    assert client.added_tracks[0]["spotify_track_ids"] == ["track_1", "track_2", "track_3"]


def test_playlist_export_service_builds_contextual_description() -> None:
    """Playlist descriptions should include mood, exploration, timestamp, and bucket context."""

    service = SpotifyPlaylistExportService(client=FakePlaylistClient())  # type: ignore[arg-type]
    generated_at = datetime(2026, 6, 4, 12, 30, tzinfo=timezone.utc)

    description = service.build_playlist_description(
        mood_label="calm",
        exploration_level=0.25,
        generated_at=generated_at,
        include_buckets=True,
    )

    assert "Generated by Spotify-Style Music Discovery Platform." in description
    assert "Mood: calm" in description
    assert "Exploration: 0.25" in description
    assert "Generated at: 2026-06-04T12:30:00+00:00" in description
    assert "Source: Spotify recent listening + hybrid ranking." in description
    assert "Buckets: Familiar, Discovery, Mood-Based" in description


def test_playlist_export_service_handles_missing_tracks() -> None:
    """Missing real Spotify track IDs should return a user-friendly failure."""

    service = SpotifyPlaylistExportService(client=FakePlaylistClient())  # type: ignore[arg-type]

    result = service.export_recommendations(
        user_token="user-token",
        user_id="spotify_user",
        spotify_real_recommendation_result=build_real_result_with_buckets(track_ids=[]),
        mood_label="happy",
        exploration_level=0.5,
        granted_scopes="playlist-modify-private",
    )

    assert result.success is False
    assert result.track_count == 0
    assert result.message == "No real Spotify track IDs were available to export."


def test_playlist_export_service_handles_missing_playlist_scope() -> None:
    """Missing playlist scope should produce the exact re-login guidance."""

    service = SpotifyPlaylistExportService(client=FakePlaylistClient())  # type: ignore[arg-type]

    result = service.export_recommendations(
        user_token="user-token",
        user_id="spotify_user",
        spotify_real_recommendation_result=build_real_result_with_buckets(),
        mood_label="happy",
        exploration_level=0.5,
        granted_scopes="user-read-recently-played",
    )

    assert result.success is False
    assert result.message == (
        "Playlist export requires playlist-modify-private scope. Please update your .env and log in again."
    )


def test_playlist_export_service_handles_api_failure_without_crashing() -> None:
    """Spotify API failures should return a non-fatal export result."""

    service = SpotifyPlaylistExportService(client=FailingPlaylistClient())  # type: ignore[arg-type]

    result = service.export_recommendations(
        user_token="user-token",
        user_id="spotify_user",
        spotify_real_recommendation_result=build_real_result_with_buckets(),
        mood_label="happy",
        exploration_level=0.5,
        granted_scopes="playlist-modify-private",
    )

    assert result == SpotifyPlaylistExportResult(
        playlist_id=None,
        playlist_url=None,
        track_count=0,
        success=False,
        message="Spotify playlist export failed. Please try again later.",
    )


def test_playlist_export_ui_noops_in_synthetic_mode() -> None:
    """Synthetic mode should not render or attempt Spotify playlist export."""

    streamlit_module = FakeStreamlitModule()

    _render_spotify_playlist_export_section(
        streamlit_module,
        auth_manager=object(),  # type: ignore[arg-type]
        spotify_playlist_export_service=SpotifyPlaylistExportService(FakePlaylistClient()),  # type: ignore[arg-type]
        spotify_real_recommendation_result=None,
        listening_history_snapshot=None,
        ui_state=object(),  # type: ignore[arg-type]
    )

    assert streamlit_module.buttons == []
    assert streamlit_module.warnings == []
    assert "spotify_playlist_export_result" not in streamlit_module.session_state


def build_real_result_with_buckets(track_ids: list[str] | None = None) -> SpotifyRealRecommendationResult:
    """Build a minimal real Spotify recommendation result for export tests."""

    resolved_track_ids = ["track_1", "track_2", "track_1", "track_3"] if track_ids is None else track_ids
    explanations = [
        DemoRecommendationExplanation(
            track_id=track_id,
            track_name=f"Track {index}",
            artist_name="Artist",
            summary_lines=[],
            spotify_url=f"https://open.spotify.com/track/{track_id}",
            recommendation_source="Spotify real-track recommendations",
        )
        for index, track_id in enumerate(resolved_track_ids, start=1)
    ]
    return SpotifyRealRecommendationResult(
        view_state=DemoViewState(
            profile=DemoUserProfile(
                user_id="spotify_real::spotify_user",
                display_name="Spotify User",
                summary="Real Spotify recommendations.",
                seed_track_ids=[],
                preferred_mood="happy",
            ),
            hybrid_weights={},
            recommendations=[
                build_recommendation(explanation.track_id)
                for explanation in explanations
            ],
            recommendation_table=pd.DataFrame(),
            explanations=explanations,
            playlist=GeneratedPlaylist(name="Playlist", mood="happy", tracks=[]),
            taste_clusters=None,
        ),
        candidate_set=SpotifyCandidateSet(candidates=[], track_catalog=pd.DataFrame()),
        source_message="Spotify real-track recommendations generated from your recent listening.",
        recommendation_buckets=[
            RecommendationBucket(
                bucket_name="familiar",
                bucket_label="Familiar Picks",
                description="",
                recommendations=explanations[:2],
            ),
            RecommendationBucket(
                bucket_name="discovery",
                bucket_label="Discovery Picks",
                description="",
                recommendations=explanations[2:],
            ),
        ],
    )


def build_recommendation(track_id: str) -> HybridRecommendation:
    """Build a minimal hybrid recommendation."""

    score_breakdown = HybridScoreBreakdown(
        collaborative_score=0.0,
        content_score=1.0,
        novelty_score=0.0,
        popularity_prior=0.0,
        discovery_score=0.0,
        final_score=1.0,
    )
    return HybridRecommendation(
        item_id=track_id,
        score=1.0,
        source="spotify_real",
        track_name=track_id,
        artist_name="Artist",
        score_breakdown=score_breakdown,
        used_cold_start_fallback=False,
    )
