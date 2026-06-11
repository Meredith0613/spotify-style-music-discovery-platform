"""Tests for Spotify taste profile visualization service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.demo_data import DemoUserProfile
from app.demo_service import DemoViewState
from app.streamlit_app import _render_taste_profile_summary
from models.playlist_generator import GeneratedPlaylist
from services.spotify_candidate_service import SpotifyCandidateSet, SpotifyRealRecommendationResult
from services.taste_profile_service import TasteProfileService, TasteProfileSummary
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


class FallbackProjectionBuilder:
    """Force the taste profile service through UMAP/K-Means fallback paths."""

    def project_umap(self, *args: Any, **kwargs: Any) -> None:
        """Simulate missing UMAP."""

        raise ImportError("umap unavailable")

    def cluster_kmeans(self, *args: Any, **kwargs: Any) -> None:
        """Simulate missing scikit-learn K-Means."""

        raise ImportError("sklearn unavailable")


class FakeStreamlitModule:
    """Small Streamlit stand-in for rendering tests."""

    def __init__(self) -> None:
        """Initialize captured UI calls."""

        self.messages: list[str] = []

    def __enter__(self) -> "FakeStreamlitModule":
        """Support column context manager usage."""

        return self

    def __exit__(self, *args: object) -> None:
        """Exit a fake column context."""

    def subheader(self, text: str) -> None:
        self.messages.append(text)

    def info(self, text: str) -> None:
        self.messages.append(text)

    def columns(self, count: int) -> list["FakeStreamlitModule"]:
        return [self for _ in range(count)]

    def metric(self, label: str, value: str) -> None:
        self.messages.append(f"{label}: {value}")

    def markdown(self, text: str) -> None:
        self.messages.append(text)

    def write(self, text: str) -> None:
        self.messages.append(text)

    def caption(self, text: str) -> None:
        self.messages.append(text)

    def dataframe(self, frame: pd.DataFrame, use_container_width: bool = True) -> None:
        self.messages.append(f"dataframe:{len(frame)}")


def build_service() -> TasteProfileService:
    """Build a taste profile service with deterministic fallback projection."""

    return TasteProfileService(projection_builder=FallbackProjectionBuilder())  # type: ignore[arg-type]


def test_taste_profile_service_builds_summary_from_recent_and_candidates() -> None:
    """The service should build a UI-ready taste profile from real Spotify data."""

    summary = build_service().build_taste_profile(
        listening_history_snapshot=build_snapshot(),
        spotify_real_recommendation_result=build_real_result(),
    )

    assert summary.cluster_id is not None
    assert summary.cluster_label
    assert "Aurora Lane" in summary.top_artists
    assert "Indie Pop" in summary.top_genres
    assert len(summary.plot_points) == 5
    assert any(point.is_recent for point in summary.plot_points)
    assert summary.warning == "UMAP was unavailable, so the app used a PCA/SVD fallback."


def test_taste_profile_service_handles_missing_genres() -> None:
    """Sparse genre metadata should not crash profile generation."""

    snapshot = build_snapshot()
    snapshot.track_level_frame = snapshot.track_level_frame.drop(columns=["artist_genres"])
    result = build_real_result()
    result.candidate_set.track_catalog = result.candidate_set.track_catalog.drop(columns=["artist_genres"])

    summary = build_service().build_taste_profile(
        listening_history_snapshot=snapshot,
        spotify_real_recommendation_result=result,
    )

    assert summary.top_genres == []
    assert summary.plot_points


def test_taste_profile_service_handles_metadata_only_features() -> None:
    """Missing audio feature columns should be filled by the existing feature builder."""

    snapshot = build_snapshot()
    snapshot.track_level_frame = snapshot.track_level_frame.loc[:, ["track_id", "track_name", "artist_name"]]
    result = build_real_result()
    result.candidate_set.track_catalog = result.candidate_set.track_catalog.loc[
        :,
        ["track_id", "track_name", "artist_name"],
    ]

    summary = build_service().build_taste_profile(
        listening_history_snapshot=snapshot,
        spotify_real_recommendation_result=result,
    )

    assert summary.cluster_label
    assert len(summary.plot_points) == 5


def test_taste_profile_service_handles_too_few_tracks_gracefully() -> None:
    """Too little listening data should produce a friendly warning and no plot."""

    summary = build_service().build_taste_profile(
        listening_history_snapshot=build_snapshot(track_count=2),
        spotify_real_recommendation_result=build_real_result(candidate_count=0),
    )

    assert summary.cluster_id is None
    assert summary.cluster_label == "Not enough data"
    assert summary.plot_points == []
    assert summary.warning == "Not enough recent listening data yet to build a stable taste map."


def test_taste_profile_service_is_deterministic_with_fallback_projection() -> None:
    """Fallback projection and cluster labeling should be deterministic."""

    service = build_service()

    first_summary = service.build_taste_profile(
        listening_history_snapshot=build_snapshot(),
        spotify_real_recommendation_result=build_real_result(),
    )
    second_summary = service.build_taste_profile(
        listening_history_snapshot=build_snapshot(),
        spotify_real_recommendation_result=build_real_result(),
    )

    assert first_summary.cluster_label == second_summary.cluster_label
    assert [(point.track_id, point.cluster_id) for point in first_summary.plot_points] == [
        (point.track_id, point.cluster_id)
        for point in second_summary.plot_points
    ]


def test_taste_profile_summary_renderer_handles_empty_fallback_summary() -> None:
    """The Streamlit summary renderer should tolerate fallback summaries."""

    streamlit_module = FakeStreamlitModule()

    _render_taste_profile_summary(
        streamlit_module,
        TasteProfileSummary(
            cluster_id=None,
            cluster_label="Not enough data",
            top_artists=[],
            top_genres=[],
            plot_points=[],
            explanation="Taste profile explanation.",
            warning="Not enough recent listening data yet to build a stable taste map.",
        ),
    )

    assert "Your Taste Profile" in streamlit_module.messages
    assert "Cluster: Not enough data" in streamlit_module.messages


def build_snapshot(track_count: int = 2) -> ListeningHistorySnapshot:
    """Build a small recent-listening snapshot."""

    rows = [
        {
            "track_id": "recent_1",
            "track_name": "Bright Run",
            "artist_name": "Aurora Lane",
            "artist_genres": "indie pop, alternative",
            "danceability": 0.82,
            "energy": 0.88,
            "valence": 0.78,
            "tempo": 142.0,
            "acousticness": 0.12,
            "speechiness": 0.04,
            "instrumentalness": 0.0,
            "loudness": -5.0,
        },
        {
            "track_id": "recent_2",
            "track_name": "Neon Smile",
            "artist_name": "Aurora Lane",
            "artist_genres": "indie pop",
            "danceability": 0.76,
            "energy": 0.81,
            "valence": 0.74,
            "tempo": 128.0,
            "acousticness": 0.18,
            "speechiness": 0.05,
            "instrumentalness": 0.0,
            "loudness": -6.0,
        },
    ][:track_count]
    recent_tracks = [
        RecentTrackSummary(
            track_id=row["track_id"],
            track_name=row["track_name"],
            artist_name=row["artist_name"],
            played_at="2026-06-01T00:00:00Z",
        )
        for row in rows
    ]
    return ListeningHistorySnapshot(
        user_id="spotify_user",
        display_name="Spotify User",
        recent_tracks=recent_tracks,
        track_level_frame=pd.DataFrame(rows),
        interaction_frame=pd.DataFrame(),
        seed_track_ids=[str(row["track_id"]) for row in rows],
    )


def build_real_result(candidate_count: int = 3) -> SpotifyRealRecommendationResult:
    """Build a minimal real Spotify result with candidate catalog metadata."""

    candidate_rows = [
        {
            "track_id": "candidate_1",
            "track_name": "Alternative Spark",
            "artist_name": "Nova Field",
            "artist_genres": "alternative, bedroom pop",
            "danceability": 0.62,
            "energy": 0.66,
            "valence": 0.58,
            "tempo": 118.0,
            "acousticness": 0.34,
            "speechiness": 0.04,
            "instrumentalness": 0.0,
            "loudness": -7.5,
        },
        {
            "track_id": "candidate_2",
            "track_name": "Quiet Focus",
            "artist_name": "Calm Artist",
            "artist_genres": "ambient, acoustic",
            "danceability": 0.35,
            "energy": 0.24,
            "valence": 0.42,
            "tempo": 78.0,
            "acousticness": 0.88,
            "speechiness": 0.03,
            "instrumentalness": 0.3,
            "loudness": -13.0,
        },
        {
            "track_id": "candidate_3",
            "track_name": "Pop Current",
            "artist_name": "Pulse City",
            "artist_genres": "pop",
            "danceability": 0.72,
            "energy": 0.73,
            "valence": 0.69,
            "tempo": 122.0,
            "acousticness": 0.2,
            "speechiness": 0.05,
            "instrumentalness": 0.0,
            "loudness": -6.8,
        },
    ][:candidate_count]
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
            recommendations=[],
            recommendation_table=pd.DataFrame(),
            explanations=[],
            playlist=GeneratedPlaylist(name="Playlist", mood="happy", tracks=[]),
            taste_clusters=None,
        ),
        candidate_set=SpotifyCandidateSet(
            candidates=[],
            track_catalog=pd.DataFrame(candidate_rows),
        ),
        source_message="Spotify real-track recommendations generated from recent listening.",
    )
