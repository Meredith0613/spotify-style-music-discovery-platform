"""Tests for authenticated user-profile service helpers."""

from __future__ import annotations

from typing import Any

from config.settings import ProjectSettings
from data.preprocessor import Preprocessor
from data.spotify_client import SpotifyAPIClientError
from services.user_profile_service import UserProfileService


class FakeSpotifyClient:
    """Return deterministic authenticated Spotify payloads for service tests."""

    def get_current_user_profile(self, access_token: str) -> dict[str, Any]:
        """Return a fake Spotify user profile."""

        return {"id": "spotify_user_1", "display_name": "Casey"}

    def get_current_user_recent_tracks(self, access_token: str, limit: int = 20) -> dict[str, Any]:
        """Return a fake recent-play payload with repeated listens."""

        return {
            "items": [
                {
                    "played_at": "2024-01-01T00:00:00Z",
                    "track": {
                        "id": "track_a",
                        "name": "Morning Light",
                        "artists": [{"id": "artist_1", "name": "Aurora Lane"}],
                    },
                },
                {
                    "played_at": "2024-01-01T00:05:00Z",
                    "track": {
                        "id": "track_b",
                        "name": "Afterglow",
                        "artists": [{"id": "artist_2", "name": "Silver Echo"}],
                    },
                },
                {
                    "played_at": "2024-01-01T00:10:00Z",
                    "track": {
                        "id": "track_a",
                        "name": "Morning Light",
                        "artists": [{"id": "artist_1", "name": "Aurora Lane"}],
                    },
                },
            ]
        }

    def get_tracks(self, track_ids: list[str], access_token: str) -> dict[str, Any]:
        """Return fake track metadata for requested recent tracks."""

        return {
            "tracks": [
                {
                    "id": "track_a",
                    "name": "Morning Light",
                    "artists": [{"id": "artist_1", "name": "Aurora Lane"}],
                    "album": {
                        "id": "album_1",
                        "name": "Daybreak",
                        "release_date": "2024-01-01",
                        "total_tracks": 10,
                    },
                    "duration_ms": 180000,
                    "explicit": False,
                    "popularity": 64,
                    "preview_url": "",
                    "track_number": 1,
                    "disc_number": 1,
                    "external_urls": {"spotify": "https://open.spotify.com/track/track_a"},
                },
                {
                    "id": "track_b",
                    "name": "Afterglow",
                    "artists": [{"id": "artist_2", "name": "Silver Echo"}],
                    "album": {
                        "id": "album_2",
                        "name": "Sunset",
                        "release_date": "2024-02-01",
                        "total_tracks": 8,
                    },
                    "duration_ms": 200000,
                    "explicit": False,
                    "popularity": 71,
                    "preview_url": "",
                    "track_number": 2,
                    "disc_number": 1,
                    "external_urls": {"spotify": "https://open.spotify.com/track/track_b"},
                },
            ]
        }

    def get_audio_features(self, track_ids: list[str], access_token: str) -> dict[str, Any]:
        """Return fake audio features for requested recent tracks."""

        return {
            "audio_features": [
                {
                    "id": "track_a",
                    "danceability": 0.72,
                    "energy": 0.68,
                    "key": 5,
                    "loudness": -6.0,
                    "mode": 1,
                    "speechiness": 0.05,
                    "acousticness": 0.22,
                    "instrumentalness": 0.0,
                    "liveness": 0.11,
                    "valence": 0.61,
                    "tempo": 118.0,
                    "duration_ms": 180000,
                    "time_signature": 4,
                },
                {
                    "id": "track_b",
                    "danceability": 0.64,
                    "energy": 0.59,
                    "key": 2,
                    "loudness": -7.1,
                    "mode": 1,
                    "speechiness": 0.04,
                    "acousticness": 0.35,
                    "instrumentalness": 0.01,
                    "liveness": 0.10,
                    "valence": 0.52,
                    "tempo": 112.0,
                    "duration_ms": 200000,
                    "time_signature": 4,
                },
            ]
        }

    def get_artists(self, artist_ids: list[str], access_token: str) -> dict[str, Any]:
        """Return fake artist metadata for requested recent tracks."""

        return {
            "artists": [
                {
                    "id": "artist_1",
                    "name": "Aurora Lane",
                    "genres": ["indie pop"],
                    "popularity": 67,
                    "followers": {"total": 1200},
                    "external_urls": {"spotify": "https://open.spotify.com/artist/artist_1"},
                },
                {
                    "id": "artist_2",
                    "name": "Silver Echo",
                    "genres": ["dream pop"],
                    "popularity": 70,
                    "followers": {"total": 1800},
                    "external_urls": {"spotify": "https://open.spotify.com/artist/artist_2"},
                },
            ]
        }


def build_settings(tmp_path: Any) -> ProjectSettings:
    """Create local project settings for service tests."""

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


def test_user_profile_service_builds_recent_history_snapshot(tmp_path: Any) -> None:
    """The service should normalize recent Spotify listens into app-friendly artifacts."""

    service = UserProfileService(
        client=FakeSpotifyClient(),  # type: ignore[arg-type]
        preprocessor=Preprocessor(settings=build_settings(tmp_path)),
    )

    snapshot = service.build_listening_history(access_token="user-token")

    assert snapshot.user_id == "spotify_user_1"
    assert snapshot.display_name == "Casey"
    assert snapshot.recent_track_count == 3
    assert snapshot.seed_track_ids == ["track_a", "track_b"]
    assert not snapshot.track_level_frame.empty
    assert set(snapshot.interaction_frame["track_id"]) == {"track_a", "track_b"}
    assert (
        snapshot.interaction_frame.loc[
            snapshot.interaction_frame["track_id"] == "track_a",
            "interaction_strength",
        ].iloc[0]
        == 2
    )


def test_user_profile_service_handles_audio_features_403_gracefully(tmp_path: Any) -> None:
    """The service should keep recent history available when audio features are forbidden."""

    class AudioFeaturesForbiddenClient(FakeSpotifyClient):
        """Return a 403 for audio features while preserving other Spotify payloads."""

        def get_audio_features(self, track_ids: list[str], access_token: str) -> dict[str, Any]:
            """Raise the same client error shape used by the real Spotify client."""

            raise SpotifyAPIClientError(
                "Spotify GET request failed for https://api.spotify.com/v1/audio-features: 403 forbidden"
            )

    service = UserProfileService(
        client=AudioFeaturesForbiddenClient(),  # type: ignore[arg-type]
        preprocessor=Preprocessor(settings=build_settings(tmp_path)),
    )

    snapshot = service.build_listening_history(access_token="user-token")

    assert snapshot.recent_track_count == 3
    assert snapshot.warnings
    assert "audio features were unavailable" in snapshot.warnings[0].lower()
    assert not snapshot.track_level_frame.empty
