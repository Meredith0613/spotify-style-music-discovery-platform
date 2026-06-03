"""Spotify Authorization Code with PKCE helpers for the Streamlit app."""

from __future__ import annotations

from base64 import urlsafe_b64encode
from dataclasses import asdict, dataclass, field
import hashlib
from time import time
from typing import Any, MutableMapping
from urllib.parse import urlencode
import secrets

import requests

from config.settings import ProjectSettings

_PENDING_PKCE_LOGIN_TTL_SECONDS = 600
_PENDING_PKCE_LOGINS: dict[str, dict[str, Any]] = {}


class SpotifyOAuthError(RuntimeError):
    """Represent an unrecoverable Spotify OAuth failure."""


@dataclass(slots=True)
class SpotifyAuthCallback:
    """Store parsed Spotify OAuth callback parameters.

    Attributes:
        code: Authorization code returned by Spotify when login succeeds.
        state: Anti-forgery state returned by Spotify during login.
        error: OAuth error code returned by Spotify when login fails.
        error_description: Optional human-readable error description from Spotify.
    """

    code: str = ""
    state: str = ""
    error: str = ""
    error_description: str = ""

    @property
    def has_authorization_code(self) -> bool:
        """Return whether the callback contains a usable authorization code."""

        return bool(self.code)


@dataclass(slots=True)
class SpotifySessionToken:
    """Store an OAuth token bundle in Streamlit session state.

    Attributes:
        access_token: Active bearer token for Spotify Web API requests.
        refresh_token: Token used to refresh the access token when it expires.
        token_type: Token type returned by Spotify, usually ``Bearer``.
        scope: Granted OAuth scope string returned by Spotify.
        expires_at: Unix timestamp after which the token should be refreshed.
    """

    access_token: str
    refresh_token: str
    token_type: str
    scope: str
    expires_at: float

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        fallback_refresh_token: str = "",
    ) -> "SpotifySessionToken":
        """Build a token object from a Spotify token response payload."""

        access_token = str(payload.get("access_token", ""))
        expires_in = max(int(payload.get("expires_in", 0)), 1)
        if not access_token:
            raise SpotifyOAuthError("Spotify token response did not include an access token.")

        return cls(
            access_token=access_token,
            refresh_token=str(payload.get("refresh_token") or fallback_refresh_token),
            token_type=str(payload.get("token_type", "Bearer")),
            scope=str(payload.get("scope", "")),
            expires_at=time() + max(expires_in - 60, 1),
        )

    def is_expired(self) -> bool:
        """Return whether the access token should be refreshed."""

        return time() >= self.expires_at

    def to_session_payload(self) -> dict[str, Any]:
        """Return a JSON-safe mapping for Streamlit session storage."""

        return asdict(self)


@dataclass(slots=True)
class SpotifyAuthManager:
    """Handle Spotify Authorization Code with PKCE for the Streamlit app."""

    client_id: str
    redirect_uri: str
    accounts_base_url: str
    scopes: tuple[str, ...]
    request_timeout_seconds: int = 30
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    _token_session_key: str = field(default="spotify_auth_token", init=False, repr=False)
    _state_session_key: str = field(default="spotify_auth_state", init=False, repr=False)
    _verifier_session_key: str = field(default="spotify_auth_code_verifier", init=False, repr=False)
    _url_session_key: str = field(default="spotify_auth_login_url", init=False, repr=False)

    @classmethod
    def from_settings(
        cls,
        settings: ProjectSettings,
        session: requests.Session | None = None,
    ) -> "SpotifyAuthManager":
        """Build an auth manager from project settings."""

        return cls(
            client_id=settings.spotify_client_id,
            redirect_uri=settings.spotify_redirect_uri,
            accounts_base_url=settings.spotify_accounts_base_url,
            scopes=settings.spotify_oauth_scopes,
            request_timeout_seconds=settings.spotify_request_timeout_seconds,
            session=session or requests.Session(),
        )

    def is_configured(self) -> bool:
        """Return whether the minimum OAuth configuration is available."""

        return bool(self.client_id and self.redirect_uri)

    def get_authorization_url(self, session_state: MutableMapping[str, Any]) -> str:
        """Return a reusable Spotify authorization URL for the current session."""

        self._validate_configuration()
        self._cleanup_stale_pending_logins()

        code_verifier = self._generate_code_verifier()
        state_value = secrets.token_urlsafe(24)
        authorization_url = self._build_authorization_url(
            state_value=state_value,
            code_verifier=code_verifier,
        )
        _PENDING_PKCE_LOGINS[state_value] = {
            "code_verifier": code_verifier,
            "created_at": time(),
        }
        self._mirror_pending_login_to_session(
            session_state=session_state,
            state_value=state_value,
            code_verifier=code_verifier,
            authorization_url=authorization_url,
        )
        return authorization_url

    def parse_callback_parameters(
        self,
        query_params: MutableMapping[str, Any],
    ) -> SpotifyAuthCallback:
        """Parse Spotify OAuth callback query parameters into a typed object."""

        return SpotifyAuthCallback(
            code=self._normalize_query_param(query_params.get("code", "")),
            state=self._normalize_query_param(query_params.get("state", "")),
            error=self._normalize_query_param(query_params.get("error", "")),
            error_description=self._normalize_query_param(
                query_params.get("error_description", "")
            ),
        )

    def complete_authorization(
        self,
        *,
        code: str,
        state: str,
        session_state: MutableMapping[str, Any],
    ) -> SpotifySessionToken:
        """Exchange an authorization code for an access token and store it."""

        self._validate_configuration()
        self._cleanup_stale_pending_logins()
        if not code or not state:
            raise SpotifyOAuthError("Spotify redirect did not include a valid authorization code.")

        expected_state = str(session_state.get(self._state_session_key, "")).strip()
        code_verifier = str(session_state.get(self._verifier_session_key, "")).strip()

        if expected_state and expected_state != state:
            raise SpotifyOAuthError("Spotify login state validation failed.")

        if not code_verifier:
            pending_login = _PENDING_PKCE_LOGINS.get(state)
            if pending_login is None:
                raise SpotifyOAuthError(
                    "Spotify login state was not found or expired before callback completion."
                )
            code_verifier = str(pending_login.get("code_verifier", "")).strip()

        if not code_verifier:
            raise SpotifyOAuthError("Spotify code verifier was not found for the active login attempt.")

        token = self._request_token(
            {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": code_verifier,
            }
        )
        self._store_token(session_state, token)
        _PENDING_PKCE_LOGINS.pop(state, None)
        self._clear_pending_login(session_state)
        return token

    def ensure_valid_token(
        self,
        session_state: MutableMapping[str, Any],
    ) -> SpotifySessionToken | None:
        """Return the active session token, refreshing it when necessary."""

        token = self.get_token(session_state)
        if token is None:
            return None
        if not token.is_expired():
            return token
        if not token.refresh_token:
            self.clear_token(session_state)
            return None

        refreshed_token = self._request_token(
            {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": token.refresh_token,
            },
            fallback_refresh_token=token.refresh_token,
        )
        self._store_token(session_state, refreshed_token)
        return refreshed_token

    def get_token(self, session_state: MutableMapping[str, Any]) -> SpotifySessionToken | None:
        """Return the stored OAuth token bundle from Streamlit session state."""

        raw_token = session_state.get(self._token_session_key)
        if isinstance(raw_token, SpotifySessionToken):
            return raw_token
        if isinstance(raw_token, dict):
            try:
                return SpotifySessionToken(
                    access_token=str(raw_token["access_token"]),
                    refresh_token=str(raw_token.get("refresh_token", "")),
                    token_type=str(raw_token.get("token_type", "Bearer")),
                    scope=str(raw_token.get("scope", "")),
                    expires_at=float(raw_token.get("expires_at", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def clear_token(self, session_state: MutableMapping[str, Any]) -> None:
        """Remove any stored Spotify OAuth state from the active session."""

        session_state.pop(self._token_session_key, None)
        self._clear_pending_login(session_state)

    def _store_token(
        self,
        session_state: MutableMapping[str, Any],
        token: SpotifySessionToken,
    ) -> None:
        """Persist the current token bundle into Streamlit session state."""

        session_state[self._token_session_key] = token.to_session_payload()

    def _clear_pending_login(self, session_state: MutableMapping[str, Any]) -> None:
        """Clear the one-time PKCE login state after completion or logout."""

        pending_state = str(session_state.get(self._state_session_key, "")).strip()
        if pending_state:
            _PENDING_PKCE_LOGINS.pop(pending_state, None)
        session_state.pop(self._state_session_key, None)
        session_state.pop(self._verifier_session_key, None)
        session_state.pop(self._url_session_key, None)

    def _request_token(
        self,
        form_data: dict[str, str],
        *,
        fallback_refresh_token: str = "",
    ) -> SpotifySessionToken:
        """Request an OAuth token from Spotify Accounts."""

        response = self.session.post(
            f"{self.accounts_base_url}/api/token",
            data=form_data,
            headers={"Accept": "application/json"},
            timeout=self.request_timeout_seconds,
        )
        self._raise_for_status(response, "Spotify token exchange failed")
        return SpotifySessionToken.from_payload(
            response.json(),
            fallback_refresh_token=fallback_refresh_token,
        )

    def _validate_configuration(self) -> None:
        """Raise a helpful error if PKCE settings are incomplete."""

        if self.is_configured():
            return
        raise SpotifyOAuthError(
            "Spotify OAuth is not configured. Set SPOTIFY_CLIENT_ID and SPOTIFY_REDIRECT_URI."
        )

    def _generate_code_verifier(self) -> str:
        """Generate a PKCE code verifier within Spotify's allowed character set."""

        return secrets.token_urlsafe(72)

    def _build_code_challenge(self, verifier: str) -> str:
        """Build a SHA-256 PKCE code challenge from a verifier."""

        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        return urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    def _build_authorization_url(self, state_value: str, code_verifier: str) -> str:
        """Build the Spotify authorization URL for one PKCE login attempt."""

        code_challenge = self._build_code_challenge(code_verifier)
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state_value,
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
        }
        return f"{self.accounts_base_url}/authorize?{urlencode(params)}"

    def _mirror_pending_login_to_session(
        self,
        session_state: MutableMapping[str, Any],
        state_value: str,
        code_verifier: str,
        authorization_url: str,
    ) -> None:
        """Write the active PKCE login attempt into Streamlit session state."""

        session_state[self._state_session_key] = state_value
        session_state[self._verifier_session_key] = code_verifier
        session_state[self._url_session_key] = authorization_url
        print("SET PKCE STATE:", state_value)
        print("SET CODE VERIFIER EXISTS:", code_verifier is not None)

    def _normalize_query_param(self, raw_value: Any) -> str:
        """Normalize a Streamlit query-parameter value into a plain string."""

        if isinstance(raw_value, list):
            return str(raw_value[0]) if raw_value else ""
        return str(raw_value or "")

    def _cleanup_stale_pending_logins(self) -> None:
        """Remove expired in-process PKCE login attempts."""

        current_time = time()
        expired_states = [
            state_value
            for state_value, metadata in _PENDING_PKCE_LOGINS.items()
            if current_time - float(metadata.get("created_at", 0.0)) > _PENDING_PKCE_LOGIN_TTL_SECONDS
        ]
        for state_value in expired_states:
            _PENDING_PKCE_LOGINS.pop(state_value, None)

    def _raise_for_status(self, response: requests.Response, message: str) -> None:
        """Raise a descriptive OAuth error when a token request fails."""

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            response_text = getattr(response, "text", "")
            raise SpotifyOAuthError(f"{message}: {response.status_code} {response_text}") from error


# Backward-compatible alias for the more explicit class name used internally.
SpotifyPKCEAuthManager = SpotifyAuthManager
