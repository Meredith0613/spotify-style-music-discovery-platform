"""Spotify Web API client used by the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any

import requests

from config.settings import ProjectSettings


class SpotifyAPIClientError(RuntimeError):
    """Represent an unrecoverable Spotify API client error."""


class SpotifyAuthenticationError(SpotifyAPIClientError):
    """Represent an authentication failure during token retrieval."""


@dataclass(slots=True)
class SpotifyAPIClient:
    """Handle authentication and read operations against the Spotify Web API.

    Attributes:
        client_id: Spotify client ID loaded from the environment.
        client_secret: Spotify client secret loaded from the environment.
        api_base_url: Base URL for Spotify Web API resource endpoints.
        accounts_base_url: Base URL for Spotify authentication endpoints.
        request_timeout_seconds: Timeout used for outbound HTTP requests.
        session: Reusable HTTP session used for token and API requests.
    """

    client_id: str
    client_secret: str
    api_base_url: str
    accounts_base_url: str
    request_timeout_seconds: int = 30
    session: requests.Session = field(default_factory=requests.Session, repr=False)
    _access_token: str | None = field(default=None, init=False, repr=False)
    _access_token_expires_at: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_settings(
        cls,
        settings: ProjectSettings,
        session: requests.Session | None = None,
    ) -> "SpotifyAPIClient":
        """Build a client from project settings.

        Args:
            settings: Repository settings containing Spotify credentials and URLs.
            session: Optional custom session for tests or advanced configuration.

        Returns:
            A configured `SpotifyAPIClient` instance.
        """

        return cls(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            api_base_url=settings.spotify_api_base_url,
            accounts_base_url=settings.spotify_accounts_base_url,
            request_timeout_seconds=settings.spotify_request_timeout_seconds,
            session=session or requests.Session(),
        )

    def is_configured(self) -> bool:
        """Return whether the client has the minimum credentials required."""

        return all([self.client_id, self.client_secret])

    def authenticate(self) -> str:
        """Authenticate with Spotify using the client-credentials flow.

        Returns:
            The active bearer token used for subsequent API calls.

        Raises:
            SpotifyAuthenticationError: If credentials are missing or token retrieval fails.
        """

        # Reusing the cached token avoids unnecessary authentication requests
        # during multi-step collection workflows.
        if self._access_token and time() < self._access_token_expires_at:
            return self._access_token

        if not self.is_configured():
            raise SpotifyAuthenticationError(
                "Spotify client credentials are missing. Set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET in your environment or .env file."
            )

        token_url = f"{self.accounts_base_url}/api/token"
        response = self.session.post(
            token_url,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=self.request_timeout_seconds,
        )
        self._raise_for_status(response, "Spotify token request failed")

        payload = response.json()
        access_token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 0))
        if not access_token or expires_in <= 0:
            raise SpotifyAuthenticationError("Spotify token response did not include a valid token.")

        self._access_token = access_token
        # Refreshing slightly before expiry prevents edge-case failures in
        # longer collection jobs with multiple API calls.
        self._access_token_expires_at = time() + max(expires_in - 60, 1)
        return access_token

    def get_track(
        self,
        track_id: str,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch metadata for a single Spotify track.

        Args:
            track_id: Spotify track identifier.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            The raw JSON response from the Spotify track endpoint.
        """

        return self._get_json(f"/tracks/{track_id}", access_token=access_token)

    def get_tracks(
        self,
        track_ids: list[str],
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch metadata for multiple tracks.

        Args:
            track_ids: List of Spotify track identifiers.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            A dictionary containing the aggregated track payloads.
        """

        tracks = self._get_batched_collection(
            endpoint="/tracks",
            ids=track_ids,
            response_collection_key="tracks",
            batch_size=50,
            access_token=access_token,
        )
        return {"tracks": tracks, "requested_track_ids": track_ids}

    def get_audio_features(
        self,
        track_ids: list[str],
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch audio feature payloads for multiple tracks.

        Args:
            track_ids: List of Spotify track identifiers.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            A dictionary containing the aggregated audio feature payloads.
        """

        audio_features = self._get_batched_collection(
            endpoint="/audio-features",
            ids=track_ids,
            response_collection_key="audio_features",
            batch_size=100,
            access_token=access_token,
        )
        return {"audio_features": audio_features, "requested_track_ids": track_ids}

    def get_artist(
        self,
        artist_id: str,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch metadata for a single Spotify artist.

        Args:
            artist_id: Spotify artist identifier.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            The raw JSON response from the Spotify artist endpoint.
        """

        return self._get_json(f"/artists/{artist_id}", access_token=access_token)

    def get_artist_top_tracks(
        self,
        artist_id: str,
        access_token: str | None = None,
        market: str = "US",
    ) -> dict[str, Any]:
        """Fetch Spotify top tracks for one artist."""

        return self._get_json(
            f"/artists/{artist_id}/top-tracks",
            params={"market": market},
            access_token=access_token,
        )

    def search_tracks(
        self,
        query: str,
        access_token: str | None = None,
        limit: int = 10,
        market: str = "US",
    ) -> dict[str, Any]:
        """Search Spotify tracks using a lightweight query."""

        return self._get_json(
            "/search",
            params={
                "q": query,
                "type": "track",
                "limit": min(max(int(limit), 1), 50),
                "market": market,
            },
            access_token=access_token,
        )

    def get_artists(
        self,
        artist_ids: list[str],
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch metadata for multiple artists.

        Args:
            artist_ids: List of Spotify artist identifiers.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            A dictionary containing the aggregated artist payloads.
        """

        artists = self._get_batched_collection(
            endpoint="/artists",
            ids=artist_ids,
            response_collection_key="artists",
            batch_size=50,
            access_token=access_token,
        )
        return {"artists": artists, "requested_artist_ids": artist_ids}

    def get_playlist_tracks(
        self,
        playlist_id: str,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch all track items contained in a Spotify playlist.

        Args:
            playlist_id: Spotify playlist identifier.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            A dictionary containing the aggregated playlist track payloads.
        """

        params: dict[str, Any] = {
            "limit": 100,
            "offset": 0,
            "additional_types": "track",
        }
        payload = self._get_json(
            f"/playlists/{playlist_id}/tracks",
            params=params,
            access_token=access_token,
        )
        items = list(payload.get("items", []))
        next_url = payload.get("next")

        # Paginating until `next` is null ensures that long playlists are
        # collected completely rather than silently truncated.
        while next_url:
            page_payload = self._get_json(
                next_url,
                absolute_url=True,
                access_token=access_token,
            )
            items.extend(page_payload.get("items", []))
            next_url = page_payload.get("next")

        return {
            "playlist_id": playlist_id,
            "items": items,
            "total": payload.get("total", len(items)),
            "href": payload.get("href"),
        }

    def get_current_user_profile(self, access_token: str) -> dict[str, Any]:
        """Fetch the authenticated user's Spotify profile."""

        return self._get_json("/me", access_token=access_token)

    def create_playlist(
        self,
        user_token: str,
        user_id: str,
        playlist_name: str,
        description: str,
        public: bool = False,
    ) -> dict[str, Any]:
        """Create a Spotify playlist for the authenticated user."""

        payload = self._post_json(
            f"/users/{user_id}/playlists",
            payload={
                "name": playlist_name,
                "description": description,
                "public": bool(public),
            },
            access_token=user_token,
        )
        return {
            "playlist_id": str(payload.get("id", "")).strip(),
            "playlist_url": str((payload.get("external_urls") or {}).get("spotify", "")).strip(),
            "payload": payload,
        }

    def add_tracks_to_playlist(
        self,
        user_token: str,
        playlist_id: str,
        track_uris: list[str] | None = None,
        spotify_track_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add unique Spotify tracks to a playlist, accepting IDs or URIs."""

        normalized_track_uris = self._normalize_track_uris(
            list(track_uris or []) + list(spotify_track_ids or [])
        )
        if not normalized_track_uris:
            return {
                "playlist_id": playlist_id,
                "track_uris": [],
                "added_track_count": 0,
                "snapshot_ids": [],
            }

        snapshot_ids: list[str] = []
        for uri_batch in self._chunk_values(normalized_track_uris, 100):
            payload = self._post_json(
                f"/playlists/{playlist_id}/tracks",
                payload={"uris": uri_batch},
                access_token=user_token,
            )
            snapshot_id = str(payload.get("snapshot_id", "")).strip()
            if snapshot_id:
                snapshot_ids.append(snapshot_id)
        return {
            "playlist_id": playlist_id,
            "track_uris": normalized_track_uris,
            "added_track_count": len(normalized_track_uris),
            "snapshot_ids": snapshot_ids,
        }

    def get_current_user_recent_tracks(
        self,
        access_token: str,
        limit: int = 20,
        after: int | None = None,
        before: int | None = None,
    ) -> dict[str, Any]:
        """Fetch the authenticated user's recently played tracks.

        Args:
            access_token: User bearer token issued by Spotify OAuth.
            limit: Maximum number of recent items to request.
            after: Optional Unix timestamp in milliseconds for pagination.
            before: Optional Unix timestamp in milliseconds for pagination.

        Returns:
            The raw JSON response from Spotify's recently played endpoint.
        """

        params: dict[str, Any] = {"limit": min(max(int(limit), 1), 50)}
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)
        return self._get_json(
            "/me/player/recently-played",
            params=params,
            access_token=access_token,
        )

    def _get_batched_collection(
        self,
        endpoint: str,
        ids: list[str],
        response_collection_key: str,
        batch_size: int,
        access_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and aggregate batched Spotify collection endpoints.

        Args:
            endpoint: Relative Spotify endpoint supporting `ids=...` queries.
            ids: Spotify identifiers to request.
            response_collection_key: JSON field containing the returned records.
            batch_size: Maximum number of IDs allowed per batch.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            Aggregated payload records returned across all request batches.
        """

        if not ids:
            return []

        aggregated_records: list[dict[str, Any]] = []

        # Spotify batch endpoints share the same request shape, so a single
        # helper keeps endpoint-specific methods concise and consistent.
        for identifier_batch in self._chunk_values(ids, batch_size):
            payload = self._get_json(
                endpoint,
                params={"ids": ",".join(identifier_batch)},
                access_token=access_token,
            )
            aggregated_records.extend(
                item for item in payload.get(response_collection_key, []) if item is not None
            )

        return aggregated_records

    def _get_json(
        self,
        endpoint_or_url: str,
        params: dict[str, Any] | None = None,
        *,
        absolute_url: bool = False,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute an authenticated GET request and return parsed JSON.

        Args:
            endpoint_or_url: Relative Spotify API endpoint or absolute URL.
            params: Optional query parameters for the request.
            absolute_url: Whether `endpoint_or_url` is already an absolute URL.
            access_token: Optional user bearer token for authenticated requests.

        Returns:
            The parsed JSON payload returned by Spotify.
        """

        token = access_token or self.authenticate()
        request_url = endpoint_or_url if absolute_url else f"{self.api_base_url}{endpoint_or_url}"
        response = self.session.get(
            request_url,
            params=params,
            headers=self._build_api_headers(token),
            timeout=self.request_timeout_seconds,
        )
        self._raise_for_status(response, f"Spotify GET request failed for {request_url}")
        return response.json()

    def _post_json(
        self,
        endpoint_or_url: str,
        payload: dict[str, Any],
        *,
        access_token: str,
        absolute_url: bool = False,
    ) -> dict[str, Any]:
        """Execute an authenticated POST request and return parsed JSON."""

        request_url = endpoint_or_url if absolute_url else f"{self.api_base_url}{endpoint_or_url}"
        headers = self._build_api_headers(access_token)
        headers["Content-Type"] = "application/json"
        response = self.session.post(
            request_url,
            json=payload,
            headers=headers,
            timeout=self.request_timeout_seconds,
        )
        self._raise_for_status(response, f"Spotify POST request failed for {request_url}")
        return response.json()

    def _build_api_headers(self, token: str) -> dict[str, str]:
        """Build standard headers for authenticated Spotify GET requests.

        Args:
            token: Bearer token produced by the authentication flow.

        Returns:
            Header dictionary for Spotify API requests.
        """

        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _normalize_track_uris(self, track_values: list[str]) -> list[str]:
        """Normalize Spotify track IDs, URLs, and URIs into unique track URIs."""

        normalized_uris: list[str] = []
        seen_uris: set[str] = set()
        for track_value in track_values:
            value = str(track_value).strip()
            if not value:
                continue
            if value.startswith("spotify:track:"):
                uri = value
            elif "open.spotify.com/track/" in value:
                track_id = value.split("open.spotify.com/track/", maxsplit=1)[1].split("?", maxsplit=1)[0]
                uri = f"spotify:track:{track_id}"
            else:
                uri = f"spotify:track:{value}"
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            normalized_uris.append(uri)
        return normalized_uris

    def _raise_for_status(self, response: requests.Response, message: str) -> None:
        """Raise a descriptive client error when an HTTP request fails.

        Args:
            response: HTTP response object returned by the requests session.
            message: Prefix describing the failed operation.

        Raises:
            SpotifyAPIClientError: If the response indicates an HTTP failure.
        """

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            response_text = getattr(response, "text", "")
            raise SpotifyAPIClientError(f"{message}: {response.status_code} {response_text}") from error

    def _chunk_values(self, values: list[str], chunk_size: int) -> list[list[str]]:
        """Split identifiers into Spotify-compatible batch sizes.

        Args:
            values: Identifiers to split into batches.
            chunk_size: Maximum number of values per chunk.

        Returns:
            A list of batched identifier lists.
        """

        return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]
