"""Tests for raw data collection workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import ProjectSettings
from data.data_collector import DataCollector, RawCollectionPaths


class FakeSpotifyAPIClient:
    """Simple fake client used to test collection workflows without network IO."""

    def get_tracks(self, track_ids: list[str]) -> dict[str, Any]:
        """Return placeholder track metadata payloads."""

        return {
            "tracks": [{"id": track_id, "name": f"Track {track_id}"} for track_id in track_ids]
        }

    def get_audio_features(self, track_ids: list[str]) -> dict[str, Any]:
        """Return placeholder audio feature payloads."""

        return {
            "audio_features": [{"id": track_id, "danceability": 0.75} for track_id in track_ids]
        }

    def get_artists(self, artist_ids: list[str]) -> dict[str, Any]:
        """Return placeholder artist metadata payloads."""

        return {
            "artists": [{"id": artist_id, "name": f"Artist {artist_id}"} for artist_id in artist_ids]
        }

    def get_playlist_tracks(self, playlist_id: str) -> dict[str, Any]:
        """Return a playlist payload with two tracks and one repeated artist."""

        return {
            "playlist_id": playlist_id,
            "items": [
                {
                    "track": {
                        "id": "t1",
                        "artists": [{"id": "a1", "name": "Artist 1"}],
                    }
                },
                {
                    "track": {
                        "id": "t2",
                        "artists": [
                            {"id": "a1", "name": "Artist 1"},
                            {"id": "a2", "name": "Artist 2"},
                        ],
                    }
                },
            ],
            "total": 2,
        }


def build_settings(tmp_path: Path) -> ProjectSettings:
    """Create local project settings for collector tests."""

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


def test_data_collector_saves_track_metadata(tmp_path: Path) -> None:
    """The collector should persist raw track metadata JSON files."""

    collector = DataCollector(client=FakeSpotifyAPIClient(), settings=build_settings(tmp_path))

    output_path = collector.collect_track_metadata(["t1", "t2"], output_prefix="sample")

    assert output_path.exists()
    assert "track_metadata" in output_path.name


def test_data_collector_collects_playlist_bundle(tmp_path: Path) -> None:
    """The collector should build a coherent raw playlist bundle."""

    collector = DataCollector(client=FakeSpotifyAPIClient(), settings=build_settings(tmp_path))

    raw_paths = collector.collect_playlist_bundle("playlist-123", output_prefix="bundle")

    assert isinstance(raw_paths, RawCollectionPaths)
    assert raw_paths.playlist_tracks_path is not None
    assert raw_paths.track_metadata_path is not None
    assert raw_paths.audio_features_path is not None
    assert raw_paths.artist_metadata_path is not None
