"""Tests for the Spotify API client."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from config.settings import ProjectSettings
from data.spotify_client import SpotifyAPIClient, SpotifyAPIClientError


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
            raise requests.HTTPError("fake http error")


class FakeSession:
    """Simple fake session used to capture authentication and GET requests."""

    def __init__(self) -> None:
        """Initialize response queues and request logs."""

        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        data: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> FakeResponse:
        """Record a fake POST request and return a token or playlist payload."""

        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "auth": auth,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if "/users/" in url and url.endswith("/playlists"):
            return FakeResponse(
                {
                    "id": "playlist_123",
                    "external_urls": {"spotify": "https://open.spotify.com/playlist/playlist_123"},
                }
            )
        if "/playlists/" in url and url.endswith("/tracks"):
            return FakeResponse({"snapshot_id": "snapshot_123"})
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


def test_spotify_client_create_playlist_posts_expected_payload(tmp_path: Any) -> None:
    """The Spotify client should create private playlists with a user token."""

    fake_session = FakeSession()
    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=fake_session)

    payload = client.create_playlist(
        user_token="user-token",
        user_id="spotify_user",
        playlist_name="Spotify Discovery Mix",
        description="Generated playlist",
        public=False,
    )

    post_call = fake_session.post_calls[0]
    assert post_call["url"] == "https://api.spotify.com/v1/users/spotify_user/playlists"
    assert post_call["json"] == {
        "name": "Spotify Discovery Mix",
        "description": "Generated playlist",
        "public": False,
    }
    assert post_call["headers"]["Authorization"] == "Bearer user-token"
    assert payload["playlist_id"] == "playlist_123"
    assert payload["playlist_url"] == "https://open.spotify.com/playlist/playlist_123"


def test_spotify_client_add_tracks_to_playlist_deduplicates_track_uris(tmp_path: Any) -> None:
    """The Spotify client should normalize and deduplicate playlist track URIs."""

    fake_session = FakeSession()
    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=fake_session)

    payload = client.add_tracks_to_playlist(
        user_token="user-token",
        playlist_id="playlist_123",
        spotify_track_ids=[
            "track_1",
            "spotify:track:track_1",
            "https://open.spotify.com/track/track_2?si=abc",
            "track_2",
        ],
    )

    post_call = fake_session.post_calls[0]
    assert post_call["url"] == "https://api.spotify.com/v1/playlists/playlist_123/tracks"
    assert post_call["json"] == {"uris": ["spotify:track:track_1", "spotify:track:track_2"]}
    assert payload["added_track_count"] == 2
    assert payload["snapshot_ids"] == ["snapshot_123"]


def test_spotify_client_add_tracks_to_playlist_handles_empty_track_list(tmp_path: Any) -> None:
    """Empty exports should not call Spotify's add-tracks endpoint."""

    fake_session = FakeSession()
    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=fake_session)

    payload = client.add_tracks_to_playlist(
        user_token="user-token",
        playlist_id="playlist_123",
        spotify_track_ids=[],
    )

    assert payload["added_track_count"] == 0
    assert fake_session.post_calls == []


def test_spotify_client_playlist_api_failure_raises_controlled_error(tmp_path: Any) -> None:
    """Spotify write failures should be wrapped in the project client error."""

    class FailingPlaylistSession(FakeSession):
        def post(
            self,
            url: str,
            data: dict[str, str] | None = None,
            auth: tuple[str, str] | None = None,
            json: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
        ) -> FakeResponse:
            self.post_calls.append(
                {
                    "url": url,
                    "data": data,
                    "auth": auth,
                    "json": json,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            return FakeResponse({"error": "forbidden"}, status_code=403)

    client = SpotifyAPIClient.from_settings(build_settings(tmp_path), session=FailingPlaylistSession())

    with pytest.raises(SpotifyAPIClientError):
        client.create_playlist(
            user_token="user-token",
            user_id="spotify_user",
            playlist_name="Spotify Discovery Mix",
            description="Generated playlist",
        )
