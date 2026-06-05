"""Create Spotify playlists from real-track recommendation results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from config.settings import ProjectSettings
from data.spotify_client import SpotifyAPIClient, SpotifyAPIClientError
from services.spotify_candidate_service import SpotifyRealRecommendationResult


LOGGER = logging.getLogger(__name__)
PLAYLIST_EXPORT_REQUIRED_SCOPE = "playlist-modify-private"


@dataclass(slots=True)
class SpotifyPlaylistExportResult:
    """Store the user-facing outcome of a playlist export attempt."""

    playlist_id: str | None
    playlist_url: str | None
    track_count: int
    success: bool
    message: str


@dataclass(slots=True)
class SpotifyPlaylistExportService:
    """Export real Spotify recommendations without touching ranking logic."""

    client: SpotifyAPIClient

    @classmethod
    def from_settings(cls, settings: ProjectSettings) -> "SpotifyPlaylistExportService":
        """Build the export service from project settings."""

        return cls(client=SpotifyAPIClient.from_settings(settings))

    def export_recommendations(
        self,
        *,
        user_token: str,
        user_id: str,
        spotify_real_recommendation_result: SpotifyRealRecommendationResult,
        mood_label: str,
        exploration_level: float,
        granted_scopes: str | None = None,
        include_buckets: bool = True,
        generated_at: datetime | None = None,
    ) -> SpotifyPlaylistExportResult:
        """Create a private Spotify playlist and add recommended real tracks."""

        if granted_scopes is not None and not self.has_playlist_export_scope(granted_scopes):
            return SpotifyPlaylistExportResult(
                playlist_id=None,
                playlist_url=None,
                track_count=0,
                success=False,
                message=(
                    "Playlist export requires playlist-modify-private scope. "
                    "Please update your .env and log in again."
                ),
            )

        track_ids = self.collect_export_track_ids(
            spotify_real_recommendation_result=spotify_real_recommendation_result,
            include_buckets=include_buckets,
        )
        if not track_ids:
            return SpotifyPlaylistExportResult(
                playlist_id=None,
                playlist_url=None,
                track_count=0,
                success=False,
                message="No real Spotify track IDs were available to export.",
            )

        timestamp = generated_at or datetime.now(timezone.utc)
        playlist_name = f"Spotify Discovery Mix - {timestamp.strftime('%Y-%m-%d')}"
        description = self.build_playlist_description(
            mood_label=mood_label,
            exploration_level=exploration_level,
            generated_at=timestamp,
            include_buckets=include_buckets,
        )
        try:
            playlist_payload = self.client.create_playlist(
                user_token=user_token,
                user_id=user_id,
                playlist_name=playlist_name,
                description=description,
                public=False,
            )
            playlist_id = str(playlist_payload.get("playlist_id", "")).strip()
            playlist_url = str(playlist_payload.get("playlist_url", "")).strip()
            if not playlist_id:
                return SpotifyPlaylistExportResult(
                    playlist_id=None,
                    playlist_url=None,
                    track_count=0,
                    success=False,
                    message="Spotify returned a playlist response without a playlist ID.",
                )
            add_payload = self.client.add_tracks_to_playlist(
                user_token=user_token,
                playlist_id=playlist_id,
                spotify_track_ids=track_ids,
            )
        except SpotifyAPIClientError as error:
            LOGGER.info("Spotify playlist export failed: %s", error)
            return SpotifyPlaylistExportResult(
                playlist_id=None,
                playlist_url=None,
                track_count=0,
                success=False,
                message="Spotify playlist export failed. Please try again later.",
            )

        added_track_count = int(add_payload.get("added_track_count", len(track_ids)))
        return SpotifyPlaylistExportResult(
            playlist_id=playlist_id,
            playlist_url=playlist_url or None,
            track_count=added_track_count,
            success=True,
            message=f"Playlist created successfully with {added_track_count} tracks.",
        )

    def collect_export_track_ids(
        self,
        *,
        spotify_real_recommendation_result: SpotifyRealRecommendationResult,
        include_buckets: bool = True,
    ) -> list[str]:
        """Collect unique real Spotify track IDs from buckets or balanced results."""

        track_ids: list[str] = []
        if include_buckets:
            recommendation_buckets = (
                getattr(spotify_real_recommendation_result, "recommendation_buckets", []) or []
            )
            for bucket in recommendation_buckets:
                for recommendation in bucket.recommendations:
                    track_ids.append(recommendation.track_id)
            if not track_ids:
                bucketed_explanations = (
                    getattr(spotify_real_recommendation_result, "bucketed_explanations", {}) or {}
                )
                for explanations in bucketed_explanations.values():
                    for recommendation in explanations:
                        track_ids.append(recommendation.track_id)

        if not track_ids:
            track_ids.extend(
                recommendation.track_id
                for recommendation in spotify_real_recommendation_result.view_state.recommendations
            )
        return self._deduplicate_track_ids(track_ids)

    def build_playlist_description(
        self,
        *,
        mood_label: str,
        exploration_level: float,
        generated_at: datetime,
        include_buckets: bool,
    ) -> str:
        """Build Spotify playlist metadata that explains the recommendation context."""

        lines = [
            "Generated by Spotify-Style Music Discovery Platform.",
            f"Mood: {mood_label}",
            f"Exploration: {float(exploration_level):.2f}",
            f"Generated at: {generated_at.astimezone(timezone.utc).isoformat()}",
            "Source: Spotify recent listening + hybrid ranking.",
        ]
        if include_buckets:
            lines.append("Buckets: Familiar, Discovery, Mood-Based")
        return "\n".join(lines)

    @staticmethod
    def has_playlist_export_scope(granted_scopes: str) -> bool:
        """Return whether the OAuth token can create private playlists."""

        normalized_scopes = granted_scopes.replace(",", " ").split()
        return PLAYLIST_EXPORT_REQUIRED_SCOPE in set(normalized_scopes)

    def _deduplicate_track_ids(self, track_ids: list[str]) -> list[str]:
        """Return unique non-empty track IDs while preserving first-seen order."""

        unique_track_ids: list[str] = []
        seen_track_ids: set[str] = set()
        for track_id in track_ids:
            normalized_track_id = str(track_id).strip()
            if not normalized_track_id or normalized_track_id in seen_track_ids:
                continue
            seen_track_ids.add(normalized_track_id)
            unique_track_ids.append(normalized_track_id)
        return unique_track_ids
