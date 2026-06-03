"""Tests for the Spotify preprocessing layer."""

from __future__ import annotations

from pathlib import Path

from config.settings import ProjectSettings
from data.data_collector import RawCollectionPaths
from data.preprocessor import Preprocessor
from utils.io_utils import write_json


def build_settings(tmp_path: Path) -> ProjectSettings:
    """Create local project settings for preprocessing tests."""

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


def build_preprocessor(tmp_path: Path) -> Preprocessor:
    """Create a preprocessor configured for a temporary test workspace."""

    return Preprocessor(settings=build_settings(tmp_path))


def test_preprocessor_normalizes_track_metadata(tmp_path: Path) -> None:
    """The preprocessor should flatten track payloads into a DataFrame."""

    preprocessor = build_preprocessor(tmp_path)
    payload = {
        "tracks": [
            {
                "id": "t1",
                "name": " Track 1 ",
                "artists": [{"id": "a1", "name": "Artist 1"}],
                "album": {"id": "al1", "name": "Album 1", "release_date": "2024-01-01"},
                "duration_ms": 123000,
                "explicit": False,
                "popularity": 87,
            }
        ]
    }

    frame = preprocessor.normalize_track_metadata(payload)

    assert list(frame["track_id"]) == ["t1"]
    assert list(frame["name"]) == ["Track 1"]
    assert list(frame["artist_names"]) == ["Artist 1"]


def test_preprocessor_creates_track_and_artist_level_tables(tmp_path: Path) -> None:
    """The preprocessor should create curated track-level and artist-level tables."""

    preprocessor = build_preprocessor(tmp_path)
    track_metadata_frame = preprocessor.normalize_track_metadata(
        {
            "tracks": [
                {
                    "id": "t1",
                    "name": "Track 1",
                    "artists": [{"id": "a1", "name": "Artist 1"}],
                    "album": {"id": "al1", "name": "Album 1", "release_date": "2024-01-01"},
                    "popularity": 85,
                }
            ]
        }
    )
    audio_features_frame = preprocessor.normalize_audio_features(
        {
            "audio_features": [
                {
                    "id": "t1",
                    "danceability": 0.81,
                    "energy": 0.72,
                    "valence": 0.63,
                    "tempo": 120.0,
                    "acousticness": 0.14,
                    "speechiness": 0.05,
                    "instrumentalness": 0.0,
                    "loudness": -5.2,
                }
            ]
        }
    )
    playlist_tracks_frame = preprocessor.normalize_playlist_tracks(
        {
            "playlist_id": "p1",
            "items": [
                {
                    "added_at": "2024-01-01T00:00:00Z",
                    "track": {
                        "id": "t1",
                        "name": "Track 1",
                        "artists": [{"id": "a1", "name": "Artist 1"}],
                    },
                }
            ],
        }
    )
    artist_metadata_frame = preprocessor.normalize_artist_metadata(
        {
            "artists": [
                {
                    "id": "a1",
                    "name": "Artist 1",
                    "genres": ["pop", "dance pop"],
                    "followers": {"total": 1000},
                    "popularity": 90,
                }
            ]
        }
    )

    track_level_frame = preprocessor.create_track_level_table(
        track_metadata_frame=track_metadata_frame,
        audio_features_frame=audio_features_frame,
        playlist_tracks_frame=playlist_tracks_frame,
        artist_metadata_frame=artist_metadata_frame,
    )
    artist_level_frame = preprocessor.create_artist_level_table(
        artist_metadata_frame=artist_metadata_frame,
        track_level_frame=track_level_frame,
    )

    assert list(track_level_frame["track_id"]) == ["t1"]
    assert list(track_level_frame["primary_artist_id"]) == ["a1"]
    assert "danceability" in track_level_frame.columns
    assert list(artist_level_frame["artist_id"]) == ["a1"]
    assert "avg_danceability" in artist_level_frame.columns


def test_preprocessor_saves_processed_bundle_outputs(tmp_path: Path) -> None:
    """The preprocessor should save normalized and curated bundle outputs."""

    settings = build_settings(tmp_path)
    settings.ensure_project_directories()

    playlist_path = settings.raw_data_dir / "playlist.json"
    tracks_path = settings.raw_data_dir / "tracks.json"
    audio_path = settings.raw_data_dir / "audio.json"
    artists_path = settings.raw_data_dir / "artists.json"

    write_json(
        playlist_path,
        {
            "playlist_id": "p1",
            "items": [
                {
                    "added_at": "2024-01-01T00:00:00Z",
                    "track": {
                        "id": "t1",
                        "name": "Track 1",
                        "artists": [{"id": "a1", "name": "Artist 1"}],
                        "album": {"id": "al1", "name": "Album 1", "release_date": "2024-01-01"},
                    },
                }
            ],
        },
    )
    write_json(
        tracks_path,
        {
            "tracks": [
                {
                    "id": "t1",
                    "name": "Track 1",
                    "artists": [{"id": "a1", "name": "Artist 1"}],
                    "album": {"id": "al1", "name": "Album 1", "release_date": "2024-01-01"},
                }
            ]
        },
    )
    write_json(
        audio_path,
        {
            "audio_features": [
                {
                    "id": "t1",
                    "danceability": 0.81,
                    "energy": 0.72,
                    "valence": 0.63,
                    "tempo": 120.0,
                    "acousticness": 0.14,
                    "speechiness": 0.05,
                    "instrumentalness": 0.0,
                    "loudness": -5.2,
                }
            ]
        },
    )
    write_json(
        artists_path,
        {
            "artists": [
                {"id": "a1", "name": "Artist 1", "genres": ["pop"], "followers": {"total": 1000}}
            ]
        },
    )

    preprocessor = Preprocessor(settings=settings)
    processed_paths = preprocessor.preprocess_collection_bundle(
        raw_paths=RawCollectionPaths(
            playlist_tracks_path=playlist_path,
            track_metadata_path=tracks_path,
            audio_features_path=audio_path,
            artist_metadata_path=artists_path,
        ),
        output_prefix="sample",
    )

    assert processed_paths.playlist_tracks_path is not None
    assert processed_paths.track_metadata_path is not None
    assert processed_paths.audio_features_path is not None
    assert processed_paths.artist_metadata_path is not None
    assert processed_paths.track_level_path is not None
    assert processed_paths.artist_level_path is not None
    assert processed_paths.track_level_path.exists()
    assert processed_paths.artist_level_path.exists()
