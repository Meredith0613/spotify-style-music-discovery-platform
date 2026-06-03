"""Services for fetching and shaping authenticated Spotify user history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.settings import ProjectSettings
from data.preprocessor import Preprocessor
from data.spotify_client import SpotifyAPIClient, SpotifyAPIClientError


@dataclass(slots=True)
class RecentTrackSummary:
    """Store a lightweight view of one recently played track."""

    track_id: str
    track_name: str
    artist_name: str
    played_at: str


@dataclass(slots=True)
class ListeningHistorySnapshot:
    """Bundle authenticated user history in app-friendly and model-ready forms.

    Attributes:
        user_id: Spotify user identifier when available.
        display_name: Human-readable Spotify display name when available.
        recent_tracks: Ordered recent-track summaries for the UI.
        track_level_frame: Track-level features compatible with later recommender use.
        interaction_frame: Implicit user-track interactions derived from recent listens.
        seed_track_ids: Ordered recent unique track IDs for later profile seeding.
        warnings: Non-fatal enrichment warnings that the UI can surface.
    """

    user_id: str
    display_name: str
    recent_tracks: list[RecentTrackSummary]
    track_level_frame: pd.DataFrame
    interaction_frame: pd.DataFrame
    seed_track_ids: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def recent_track_count(self) -> int:
        """Return the number of recent-play events represented in the snapshot."""

        return len(self.recent_tracks)


@dataclass(slots=True)
class UserProfileService:
    """Fetch authenticated Spotify history and normalize it for later app use."""

    client: SpotifyAPIClient
    preprocessor: Preprocessor
    default_recent_track_limit: int = 20
    max_seed_tracks: int = 8

    @classmethod
    def from_settings(cls, settings: ProjectSettings) -> "UserProfileService":
        """Build the user-profile service from project settings."""

        return cls(
            client=SpotifyAPIClient.from_settings(settings),
            preprocessor=Preprocessor(settings=settings),
        )

    def build_listening_history(
        self,
        access_token: str,
        recent_track_limit: int | None = None,
    ) -> ListeningHistorySnapshot:
        """Fetch and normalize the current user's recent Spotify listening history."""

        resolved_limit = recent_track_limit or self.default_recent_track_limit
        user_payload = self.client.get_current_user_profile(access_token=access_token)
        recent_payload = self.client.get_current_user_recent_tracks(
            access_token=access_token,
            limit=resolved_limit,
        )

        recent_tracks = self._build_recent_track_summaries(recent_payload)
        track_ids = self._extract_recent_track_ids(recent_payload)
        warnings: list[str] = []
        if not track_ids:
            return ListeningHistorySnapshot(
                user_id=str(user_payload.get("id", "spotify_user")),
                display_name=str(user_payload.get("display_name") or "Spotify User"),
                recent_tracks=recent_tracks,
                track_level_frame=pd.DataFrame(),
                interaction_frame=self._build_interaction_frame(
                    user_id=str(user_payload.get("id", "spotify_user")),
                    recent_payload=recent_payload,
                ),
                seed_track_ids=[],
                warnings=warnings,
            )

        track_metadata_payload = self.client.get_tracks(track_ids, access_token=access_token)
        audio_features_payload: dict[str, Any] = {"audio_features": []}
        try:
            audio_features_payload = self.client.get_audio_features(
                track_ids,
                access_token=access_token,
            )
        except SpotifyAPIClientError as error:
            # Spotify sometimes denies `/audio-features` for user-scoped flows.
            # In that case we continue with basic track metadata instead of
            # failing the entire recent-history section.
            if self._is_audio_features_forbidden_error(error):
                warnings.append(
                    "Spotify audio features were unavailable, so recent history uses basic track metadata only."
                )
            else:
                raise
        artist_ids = self._extract_artist_ids_from_tracks(track_metadata_payload)
        artist_metadata_payload = self.client.get_artists(artist_ids, access_token=access_token)

        track_metadata_frame = self.preprocessor.normalize_track_metadata(track_metadata_payload)
        audio_features_frame = self.preprocessor.normalize_audio_features(audio_features_payload)
        artist_metadata_frame = self.preprocessor.normalize_artist_metadata(artist_metadata_payload)
        recent_events_frame = self._build_recent_events_frame(recent_payload)

        # Reusing the existing track-level table builder keeps downstream schema
        # consistent with the rest of the repository.
        track_level_frame = self.preprocessor.create_track_level_table(
            track_metadata_frame=track_metadata_frame,
            audio_features_frame=audio_features_frame,
            playlist_tracks_frame=recent_events_frame,
            artist_metadata_frame=artist_metadata_frame,
        )
        user_id = str(user_payload.get("id", "spotify_user"))

        return ListeningHistorySnapshot(
            user_id=user_id,
            display_name=str(user_payload.get("display_name") or "Spotify User"),
            recent_tracks=recent_tracks,
            track_level_frame=track_level_frame,
            interaction_frame=self._build_interaction_frame(
                user_id=user_id,
                recent_payload=recent_payload,
            ),
            seed_track_ids=track_ids[: self.max_seed_tracks],
            warnings=warnings,
        )

    def _build_recent_track_summaries(
        self,
        recent_payload: dict[str, Any],
    ) -> list[RecentTrackSummary]:
        """Convert Spotify recent-play items into compact UI summaries."""

        summaries: list[RecentTrackSummary] = []
        for item in recent_payload.get("items", []):
            track_payload = item.get("track") or {}
            artist_names = ", ".join(
                str(artist.get("name", "")).strip()
                for artist in track_payload.get("artists", [])
                if artist.get("name")
            )
            track_id = str(track_payload.get("id", "")).strip()
            if not track_id:
                continue
            summaries.append(
                RecentTrackSummary(
                    track_id=track_id,
                    track_name=str(track_payload.get("name") or track_id),
                    artist_name=artist_names,
                    played_at=str(item.get("played_at", "")),
                )
            )
        return summaries

    def _extract_recent_track_ids(self, recent_payload: dict[str, Any]) -> list[str]:
        """Extract recent unique track IDs in first-seen order."""

        track_ids: list[str] = []
        seen_track_ids: set[str] = set()
        for summary in self._build_recent_track_summaries(recent_payload):
            if summary.track_id in seen_track_ids:
                continue
            seen_track_ids.add(summary.track_id)
            track_ids.append(summary.track_id)
        return track_ids

    def _extract_artist_ids_from_tracks(self, tracks_payload: dict[str, Any]) -> list[str]:
        """Extract unique artist IDs from a Spotify track payload."""

        artist_ids: list[str] = []
        seen_artist_ids: set[str] = set()
        for track_payload in tracks_payload.get("tracks", []):
            if not track_payload:
                continue
            for artist_payload in track_payload.get("artists", []):
                artist_id = str(artist_payload.get("id", "")).strip()
                if not artist_id or artist_id in seen_artist_ids:
                    continue
                seen_artist_ids.add(artist_id)
                artist_ids.append(artist_id)
        return artist_ids

    def _build_recent_events_frame(self, recent_payload: dict[str, Any]) -> pd.DataFrame:
        """Build a simple event-level frame keyed by track ID."""

        event_rows: list[dict[str, Any]] = []
        for summary in self._build_recent_track_summaries(recent_payload):
            event_rows.append(
                {
                    "track_id": summary.track_id,
                    "played_at": summary.played_at,
                }
            )
        return pd.DataFrame(event_rows)

    def _build_interaction_frame(
        self,
        user_id: str,
        recent_payload: dict[str, Any],
    ) -> pd.DataFrame:
        """Aggregate recent listens into implicit user-track interaction strengths."""

        event_frame = self._build_recent_events_frame(recent_payload)
        if event_frame.empty:
            return pd.DataFrame(columns=["user_id", "track_id", "interaction_strength"])

        aggregated_frame = (
            event_frame.groupby("track_id", as_index=False)
            .size()
            .rename(columns={"size": "interaction_strength"})
            .sort_values(["interaction_strength", "track_id"], ascending=[False, True], kind="stable")
            .reset_index(drop=True)
        )
        aggregated_frame["user_id"] = user_id
        return aggregated_frame.loc[:, ["user_id", "track_id", "interaction_strength"]]

    def _is_audio_features_forbidden_error(self, error: SpotifyAPIClientError) -> bool:
        """Return whether a Spotify client error is a 403 from `/audio-features`."""

        error_message = str(error)
        return "audio-features" in error_message and "403" in error_message
