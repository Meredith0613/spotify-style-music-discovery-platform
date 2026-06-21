"""Focused tests for Streamlit product-polish helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from app.demo_data import build_demo_user_profiles
from app.demo_service import DemoRecommendationExplanation
from app.streamlit_app import (
    DemoUIState,
    _build_playlist_preview_metadata,
    _build_recommendation_card_html,
    _render_auth_sidebar_section,
    _render_taste_profile_section,
)
from services.taste_profile_service import TasteProfileSummary
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


class FakeSidebar:
    """Capture narrow sidebar calls used by authentication rendering."""

    def __init__(self) -> None:
        self.link_buttons: list[str] = []
        self.markdowns: list[str] = []

    def header(self, text: str) -> None:
        pass

    def caption(self, text: str) -> None:
        pass

    def link_button(self, label: str, url: str, **kwargs: Any) -> None:
        self.link_buttons.append(f"{label}:{url}")

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)


class FakeAuthStreamlit:
    """Provide only the state used by the disconnected auth helper."""

    def __init__(self) -> None:
        self.sidebar = FakeSidebar()
        self.session_state: dict[str, object] = {}
        self.query_params: dict[str, str] = {}


class FakeTasteStreamlit:
    """Capture personality panel output without running Streamlit."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __enter__(self) -> "FakeTasteStreamlit":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def subheader(self, text: str) -> None:
        self.messages.append(text)

    def info(self, text: str) -> None:
        self.messages.append(text)

    def columns(self, count: int) -> list["FakeTasteStreamlit"]:
        return [self for _ in range(count)]

    def metric(self, label: str, value: str) -> None:
        self.messages.append(f"{label}: {value}")

    def markdown(self, text: str, **kwargs: object) -> None:
        self.messages.append(text)

    def caption(self, text: str) -> None:
        self.messages.append(text)

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.messages.append(f"dataframe:{len(frame)}")


class FakeAuthManager:
    """Return a deterministic one-click OAuth URL."""

    def get_token(self, session_state: dict[str, object]) -> None:
        return None

    def get_authorization_url(self, session_state: dict[str, object]) -> str:
        return "https://accounts.spotify.com/authorize?state=test"


class FakeTasteProfileService:
    """Return a stable real-Spotify taste profile summary."""

    def build_taste_profile(self, **kwargs: object) -> TasteProfileSummary:
        return TasteProfileSummary(
            cluster_id=1,
            cluster_label="Indie / Alternative Discovery",
            top_artists=["Aurora Lane"],
            top_genres=["Indie Pop"],
            plot_points=[],
            explanation="Built from current Spotify listening.",
        )


def test_login_ui_uses_one_connect_link_without_raw_login_markdown() -> None:
    """Disconnected OAuth UI should expose one named link action only."""

    streamlit_module = FakeAuthStreamlit()
    settings = SimpleNamespace(spotify_oauth_available=lambda: True)

    _render_auth_sidebar_section(
        streamlit_module=streamlit_module,
        settings=settings,  # type: ignore[arg-type]
        auth_manager=FakeAuthManager(),  # type: ignore[arg-type]
        listening_history_snapshot=None,
        history_error=None,
        callback_in_progress=False,
    )

    assert streamlit_module.sidebar.link_buttons == [
        "Connect Spotify:https://accounts.spotify.com/authorize?state=test"
    ]
    assert not streamlit_module.sidebar.markdowns


def test_demo_profile_labels_are_preference_only() -> None:
    """Demo profile display labels should not include tutorial-style fake names."""

    labels = [profile.display_name for profile in build_demo_user_profiles().values()]

    assert labels == [
        "Runner Pop",
        "Study Focus",
        "Late-Night Melancholy",
        "Sunny Mix",
        "New Listener",
    ]


def test_recommendation_card_html_handles_missing_media_and_mood_score() -> None:
    """Compact cards should stay readable without optional Spotify metadata."""

    html = _build_recommendation_card_html(
        explanation=DemoRecommendationExplanation(
            track_id="track_1",
            track_name="Track Name",
            artist_name="Artist Name",
            summary_lines=["Built from current ranking."],
        ),
        bucket_label="Mood-Based Picks",
        mood_label="calm",
    )

    assert "No art" in html
    assert "Open in Spotify" not in html
    assert "Mood fit: Unavailable" in html
    assert "Mood-Based" in html


def test_spotify_taste_profile_renders_without_synthetic_cluster_flag() -> None:
    """Real Spotify taste rendering should not depend on the legacy cluster toggle."""

    streamlit_module = FakeTasteStreamlit()
    snapshot = ListeningHistorySnapshot(
        user_id="spotify_user",
        display_name="Spotify User",
        recent_tracks=[RecentTrackSummary("recent_1", "Recent", "Aurora Lane", "2026-01-01T00:00:00Z")],
        track_level_frame=pd.DataFrame([{"track_id": "recent_1", "energy": 0.8}]),
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["recent_1"],
    )
    result = SimpleNamespace(
        candidate_set=SimpleNamespace(
            track_catalog=pd.DataFrame(
                [{"track_id": "candidate_1", "catalog_novelty": 0.7, "album_release_date": "2021-04-01"}]
            )
        )
    )

    _render_taste_profile_section(
        streamlit_module,
        taste_profile_service=FakeTasteProfileService(),  # type: ignore[arg-type]
        listening_history_snapshot=snapshot,
        spotify_real_recommendation_result=result,  # type: ignore[arg-type]
    )

    assert "Your Music Personality" in streamlit_module.messages
    assert not any("Taste clusters were not requested" in message for message in streamlit_module.messages)


def test_playlist_preview_handles_missing_duration() -> None:
    """Playlist preview should remain useful when track durations are unavailable."""

    result = SimpleNamespace(
        candidate_set=SimpleNamespace(track_catalog=pd.DataFrame([{"track_id": "track_1"}])),
        recommendation_buckets=[],
        bucketed_explanations={},
    )
    ui_state = DemoUIState(
        user_id="runner_pop",
        exploration_level=0.65,
        recommendation_count=5,
        mood_label="workout",
        ranking_focus="Balanced",
        playlist_length=4,
        show_taste_clusters=False,
    )

    preview = _build_playlist_preview_metadata(
        spotify_real_recommendation_result=result,  # type: ignore[arg-type]
        track_ids=["track_1"],
        ui_state=ui_state,
    )

    assert preview["duration"] == "Unavailable"
    assert preview["track_count"] == "1"
    assert preview["mood"] == "🔥 Workout"
