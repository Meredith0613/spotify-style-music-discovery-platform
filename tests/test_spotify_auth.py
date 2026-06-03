"""Tests for Spotify PKCE authentication helpers."""

from __future__ import annotations

from typing import Any

import requests

from auth.spotify_auth import SpotifyAuthManager, SpotifySessionToken, _PENDING_PKCE_LOGINS
from config.settings import ProjectSettings


class FakeResponse:
    """Minimal fake HTTP response for Spotify auth tests."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        """Store a fake JSON payload and status code."""

        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        """Return the stored JSON payload."""

        return self._payload

    def raise_for_status(self) -> None:
        """Raise an HTTP error when the fake response is unsuccessful."""

        if self.status_code >= 400:
            raise requests.HTTPError(self.text)


class FakeSession:
    """Capture Spotify auth requests without performing network I/O."""

    def __init__(self) -> None:
        """Initialize request capture and queued responses."""

        self.post_calls: list[dict[str, Any]] = []
        self.responses: list[FakeResponse] = [
            FakeResponse(
                {
                    "access_token": "access-123",
                    "refresh_token": "refresh-123",
                    "expires_in": 3600,
                    "scope": "user-read-recently-played",
                    "token_type": "Bearer",
                }
            ),
            FakeResponse(
                {
                    "access_token": "access-456",
                    "expires_in": 3600,
                    "scope": "user-read-recently-played",
                    "token_type": "Bearer",
                }
            ),
        ]

    def post(
        self,
        url: str,
        data: dict[str, str],
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        """Record a token request and return the next queued response."""

        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


def build_settings(tmp_path: Any) -> ProjectSettings:
    """Create project settings suitable for Spotify auth tests."""

    return ProjectSettings(
        project_root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        interim_data_dir=tmp_path / "data" / "interim",
        processed_data_dir=tmp_path / "data" / "processed",
        artifacts_dir=tmp_path / "artifacts",
        spotify_client_id="client-id",
        spotify_client_secret="client-secret",
        spotify_redirect_uri="http://localhost:8501",
        spotify_api_base_url="https://api.spotify.com/v1",
        spotify_accounts_base_url="https://accounts.spotify.com",
        spotify_request_timeout_seconds=30,
        spotify_default_market="US",
        spotify_oauth_scopes=("user-read-recently-played",),
    )


def test_spotify_auth_manager_builds_authorization_url(tmp_path: Any) -> None:
    """The auth manager should build and cache a PKCE authorization URL."""

    manager = SpotifyAuthManager.from_settings(build_settings(tmp_path), session=FakeSession())
    session_state: dict[str, Any] = {}

    authorization_url = manager.get_authorization_url(session_state)

    assert "accounts.spotify.com/authorize" in authorization_url
    assert "code_challenge=" in authorization_url
    assert "spotify_auth_state" in session_state
    assert "spotify_auth_code_verifier" in session_state


def test_spotify_auth_manager_exchanges_and_refreshes_tokens(tmp_path: Any) -> None:
    """The auth manager should exchange and later refresh a Spotify user token."""

    fake_session = FakeSession()
    manager = SpotifyAuthManager.from_settings(build_settings(tmp_path), session=fake_session)
    session_state: dict[str, Any] = {}
    manager.get_authorization_url(session_state)

    initial_token = manager.complete_authorization(
        code="auth-code",
        state=str(session_state["spotify_auth_state"]),
        session_state=session_state,
    )
    assert isinstance(initial_token, SpotifySessionToken)
    assert initial_token.access_token == "access-123"

    session_state["spotify_auth_token"]["expires_at"] = 0.0
    refreshed_token = manager.ensure_valid_token(session_state)

    assert refreshed_token is not None
    assert refreshed_token.access_token == "access-456"
    assert fake_session.post_calls[0]["data"]["grant_type"] == "authorization_code"
    assert fake_session.post_calls[1]["data"]["grant_type"] == "refresh_token"


def test_spotify_auth_manager_completes_auth_when_session_loses_pending_state(tmp_path: Any) -> None:
    """The auth manager should fall back to the pending-login store after redirect."""

    fake_session = FakeSession()
    manager = SpotifyAuthManager.from_settings(build_settings(tmp_path), session=fake_session)
    session_state: dict[str, Any] = {}
    manager.get_authorization_url(session_state)
    callback_state = str(session_state["spotify_auth_state"])

    session_state.pop("spotify_auth_state", None)
    session_state.pop("spotify_auth_code_verifier", None)

    token = manager.complete_authorization(
        code="auth-code",
        state=callback_state,
        session_state=session_state,
    )

    assert token.access_token == "access-123"
    assert callback_state not in _PENDING_PKCE_LOGINS
    assert "spotify_auth_token" in session_state


def test_spotify_auth_manager_parses_callback_parameters(tmp_path: Any) -> None:
    """The auth manager should normalize callback query parameters for the app layer."""

    manager = SpotifyAuthManager.from_settings(build_settings(tmp_path), session=FakeSession())

    callback = manager.parse_callback_parameters(
        {
            "code": ["auth-code"],
            "state": ["state-123"],
            "error": "",
            "error_description": "",
        }
    )

    assert callback.has_authorization_code
    assert callback.code == "auth-code"
    assert callback.state == "state-123"
