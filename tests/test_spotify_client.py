"""Tests for the Spotify API client."""

from __future__ import annotations

from typing import Any

from config.settings import ProjectSettings
from data.spotify_client import SpotifyAPIClient


class FakeResponse:
    """Simple fake response object for non-network client tests."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        """Store a payload and status code for fake HTTP responses."""

        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        """Return the stored JSON payload."""

        return self._payload

    def raise_for_status(self) -> None:
        """Raise nothing for successful fake responses."""

        if self.status_code >= 400:
            raise RuntimeError("fake http error")


class FakeSession:
    """Simple fake session used to capture authentication and GET requests."""

    def __init__(self) -> None:
        """Initialize response queues and request logs."""

        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def post(self, url: str, data: dict[str, str], auth: tuple[str, str], timeout: int) -> FakeResponse:
        """Record a fake POST request and return a token payload."""

        self.post_calls.append(
            {"url": url, "data": data, "auth": auth, "timeout": timeout}
        )
        return FakeResponse({"access_token": "token-123", "expires_in": 3600})

    def get(
        self,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        """Record a fake GET request and return a stub payload."""

        self.get_calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        if "audio-features" in url:
            return FakeResponse({"audio_features": [{"id": "t1", "danceability": 0.8}]})
        if "artists" in url:
            return FakeResponse({"artists": [{"id": "a1", "name": "Artist 1"}]})
        if "recently-played" in url:
            return FakeResponse(
                {
                    "items": [
                        {
                            "played_at": "2024-01-01T00:00:00Z",
                            "track": {
                                "id": "t1",
                                "name": "Track 1",
                                "artists": [{"id": "a1", "name": "Artist 1"}],
                            },
                        }
                    ]
                }
            )
        if "playlists" in url:
            return FakeResponse({"items": [], "total": 0, "href": url, "next": None})
        return FakeResponse({"tracks": [{"id": "t1", "name": "Track 1"}]})


def build_settings(tmp_path: Any) -> ProjectSettings:
    """Create local project settings for client tests."""

    return ProjectSettings(
        project_root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        interim_data_dir=tmp_path / "data" / "interim",
        processed_data_dir=tmp_path / "data" / "processed",
        artifacts_dir=tmp_path / "artifacts",
        spotify_client_id="client-id",
        spotify_client_secret="client-secret",
        spotify_redirect_uri="",
        spotify_api_base_url="https://api.spotify.com/v1",
        spotify_accounts_base_url="https://accounts.spotify.com",
        spotify_request_timeout_seconds=30,
        spotify_default_market="US",
    )


def test_spotify_client_authenticates_and_reuses_token(tmp_path: Any) -> None:
    """The Spotify client should fetch and cache an access token."""

    fake_session = FakeSession()
    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=fake_session)

    first_token = client.authenticate()
    second_token = client.authenticate()

    assert first_token == "token-123"
    assert second_token == "token-123"
    assert len(fake_session.post_calls) == 1


def test_spotify_client_fetches_tracks_with_bearer_token(tmp_path: Any) -> None:
    """The Spotify client should send authenticated GET requests."""

    fake_session = FakeSession()
    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=fake_session)

    payload = client.get_tracks(["t1"])

    assert payload["tracks"][0]["id"] == "t1"
    assert fake_session.get_calls[0]["headers"]["Authorization"] == "Bearer token-123"


def test_spotify_client_supports_authenticated_recent_tracks(tmp_path: Any) -> None:
    """The Spotify client should use an explicit user token for recent history."""

    fake_session = FakeSession()
    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=fake_session)

    payload = client.get_current_user_recent_tracks(access_token="user-token", limit=5)

    assert payload["items"][0]["track"]["id"] == "t1"
    assert fake_session.get_calls[0]["headers"]["Authorization"] == "Bearer user-token"
    assert fake_session.post_calls == []
