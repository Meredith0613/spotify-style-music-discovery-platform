"""Data collection workflows for raw Spotify API payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from config.settings import ProjectSettings
from utils.io_utils import read_json, write_json

from .spotify_client import SpotifyAPIClient


@dataclass(slots=True)
class RawCollectionPaths:
    """Track raw JSON files produced during a collection workflow.

    Attributes:
        playlist_tracks_path: Raw playlist-track payload path, when collected.
        track_metadata_path: Raw track metadata payload path, when collected.
        audio_features_path: Raw audio feature payload path, when collected.
        artist_metadata_path: Raw artist metadata payload path, when collected.
    """

    playlist_tracks_path: Path | None = None
    track_metadata_path: Path | None = None
    audio_features_path: Path | None = None
    artist_metadata_path: Path | None = None


@dataclass(slots=True)
class DataCollector:
    """Run reusable Spotify collection workflows and persist raw payloads.

    Attributes:
        client: Authenticated Spotify API client used for remote reads.
        settings: Project settings used to resolve raw-data paths.
    """

    client: SpotifyAPIClient
    settings: ProjectSettings

    def __post_init__(self) -> None:
        """Ensure raw and processed project directories exist before collection."""

        self.settings.ensure_project_directories()

    def collect_track_metadata(
        self,
        track_ids: list[str],
        output_prefix: str = "tracks",
    ) -> Path:
        """Collect and persist track metadata for a list of track IDs.

        Args:
            track_ids: Spotify track identifiers to fetch.
            output_prefix: Prefix used when naming the raw JSON snapshot.

        Returns:
            Path to the saved raw JSON file.
        """

        return self._collect_and_save(
            fetch_operation=lambda: self.client.get_tracks(track_ids),
            dataset_name="track_metadata",
            output_prefix=output_prefix,
        )

    def collect_audio_features(
        self,
        track_ids: list[str],
        output_prefix: str = "audio_features",
    ) -> Path:
        """Collect and persist audio features for a list of track IDs.

        Args:
            track_ids: Spotify track identifiers to fetch audio features for.
            output_prefix: Prefix used when naming the raw JSON snapshot.

        Returns:
            Path to the saved raw JSON file.
        """

        return self._collect_and_save(
            fetch_operation=lambda: self.client.get_audio_features(track_ids),
            dataset_name="audio_features",
            output_prefix=output_prefix,
        )

    def collect_artist_metadata(
        self,
        artist_ids: list[str],
        output_prefix: str = "artists",
    ) -> Path:
        """Collect and persist artist metadata for a list of artist IDs.

        Args:
            artist_ids: Spotify artist identifiers to fetch.
            output_prefix: Prefix used when naming the raw JSON snapshot.

        Returns:
            Path to the saved raw JSON file.
        """

        return self._collect_and_save(
            fetch_operation=lambda: self.client.get_artists(artist_ids),
            dataset_name="artist_metadata",
            output_prefix=output_prefix,
        )

    def collect_playlist_tracks(
        self,
        playlist_id: str,
        output_prefix: str = "playlist",
    ) -> Path:
        """Collect and persist playlist-track items for a playlist.

        Args:
            playlist_id: Spotify playlist identifier.
            output_prefix: Prefix used when naming the raw JSON snapshot.

        Returns:
            Path to the saved raw JSON file.
        """

        return self._collect_and_save(
            fetch_operation=lambda: self.client.get_playlist_tracks(playlist_id),
            dataset_name="playlist_tracks",
            output_prefix=output_prefix,
        )

    def collect_playlist_bundle(
        self,
        playlist_id: str,
        output_prefix: str | None = None,
    ) -> RawCollectionPaths:
        """Collect a playlist-centered raw bundle for downstream preprocessing.

        Args:
            playlist_id: Spotify playlist identifier to collect around.
            output_prefix: Optional snapshot prefix. Defaults to the playlist ID.

        Returns:
            Paths to the saved raw JSON artifacts for the bundle.
        """

        resolved_prefix = output_prefix or playlist_id
        playlist_tracks_path = self.collect_playlist_tracks(
            playlist_id=playlist_id,
            output_prefix=resolved_prefix,
        )
        playlist_payload = read_json(playlist_tracks_path)

        track_ids = self._extract_track_ids_from_playlist(playlist_payload)
        artist_ids = self._extract_artist_ids_from_playlist(playlist_payload)

        # The bundle workflow keeps raw snapshots aligned so later
        # preprocessing steps can work from a coherent ingestion run.
        track_metadata_path = self.collect_track_metadata(track_ids, output_prefix=resolved_prefix)
        audio_features_path = self.collect_audio_features(track_ids, output_prefix=resolved_prefix)
        artist_metadata_path = self.collect_artist_metadata(artist_ids, output_prefix=resolved_prefix)

        return RawCollectionPaths(
            playlist_tracks_path=playlist_tracks_path,
            track_metadata_path=track_metadata_path,
            audio_features_path=audio_features_path,
            artist_metadata_path=artist_metadata_path,
        )

    def _collect_and_save(
        self,
        fetch_operation: Callable[[], dict[str, Any]],
        dataset_name: str,
        output_prefix: str,
    ) -> Path:
        """Execute one collection step and persist its raw payload.

        Args:
            fetch_operation: No-argument callable that returns a JSON-compatible payload.
            dataset_name: Logical dataset label such as `track_metadata`.
            output_prefix: Prefix used to group files from one collection run.

        Returns:
            Path to the saved raw JSON file.
        """

        payload = fetch_operation()
        return self._save_raw_payload(
            payload=payload,
            dataset_name=dataset_name,
            output_prefix=output_prefix,
        )

    def _save_raw_payload(
        self,
        payload: dict[str, Any],
        dataset_name: str,
        output_prefix: str,
    ) -> Path:
        """Save a raw Spotify payload into the raw data directory.

        Args:
            payload: JSON-compatible payload returned by a collection step.
            dataset_name: Logical dataset label such as `track_metadata`.
            output_prefix: Prefix used to group files from one collection run.

        Returns:
            Path to the persisted raw JSON file.
        """

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.settings.raw_data_dir / f"{output_prefix}_{dataset_name}_{timestamp}.json"
        write_json(output_path, payload)
        return output_path

    def _extract_track_ids_from_playlist(self, playlist_payload: dict[str, Any]) -> list[str]:
        """Extract unique track identifiers from playlist-track payloads.

        Args:
            playlist_payload: Raw playlist-track payload returned by the API.

        Returns:
            Unique Spotify track identifiers in first-seen order.
        """

        track_ids: list[str] = []
        seen_track_ids: set[str] = set()

        # Deduplicating at collection time avoids unnecessary follow-up API
        # calls for repeated tracks inside long playlists.
        for item in playlist_payload.get("items", []):
            track_payload = item.get("track") or {}
            track_id = track_payload.get("id")
            if isinstance(track_id, str) and track_id not in seen_track_ids:
                seen_track_ids.add(track_id)
                track_ids.append(track_id)

        return track_ids

    def _extract_artist_ids_from_playlist(self, playlist_payload: dict[str, Any]) -> list[str]:
        """Extract unique artist identifiers from playlist-track payloads.

        Args:
            playlist_payload: Raw playlist-track payload returned by the API.

        Returns:
            Unique Spotify artist identifiers in first-seen order.
        """

        artist_ids: list[str] = []
        seen_artist_ids: set[str] = set()

        # Artist IDs are extracted from nested track payloads so the collector
        # can enrich playlists with artist metadata in the same workflow.
        for item in playlist_payload.get("items", []):
            track_payload = item.get("track") or {}
            for artist_payload in track_payload.get("artists", []):
                artist_id = artist_payload.get("id")
                if isinstance(artist_id, str) and artist_id not in seen_artist_ids:
                    seen_artist_ids.add(artist_id)
                    artist_ids.append(artist_id)

        return artist_ids
