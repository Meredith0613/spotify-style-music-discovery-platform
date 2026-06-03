"""Preprocessing for raw Spotify JSON payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config.settings import ProjectSettings
from utils.io_utils import read_json

from .data_collector import RawCollectionPaths


@dataclass(slots=True)
class ProcessedCollectionPaths:
    """Track processed tabular files produced by preprocessing workflows.

    Attributes:
        playlist_tracks_path: Processed playlist-track table path, when created.
        track_metadata_path: Processed track metadata table path, when created.
        audio_features_path: Processed audio features table path, when created.
        artist_metadata_path: Processed artist metadata table path, when created.
        track_level_path: Curated track-level table path, when created.
        artist_level_path: Curated artist-level table path, when created.
    """

    playlist_tracks_path: Path | None = None
    track_metadata_path: Path | None = None
    audio_features_path: Path | None = None
    artist_metadata_path: Path | None = None
    track_level_path: Path | None = None
    artist_level_path: Path | None = None


@dataclass(slots=True)
class Preprocessor:
    """Normalize raw Spotify payloads into clean pandas DataFrames.

    Attributes:
        settings: Project settings used to resolve processed-data paths.
    """

    settings: ProjectSettings

    def __post_init__(self) -> None:
        """Ensure project directories exist before writing processed outputs."""

        self.settings.ensure_project_directories()

    def preprocess_collection_bundle(
        self,
        raw_paths: RawCollectionPaths,
        output_prefix: str,
    ) -> ProcessedCollectionPaths:
        """Preprocess a full raw collection bundle and save CSV outputs.

        Args:
            raw_paths: Raw JSON paths returned by the data collector.
            output_prefix: Prefix used when naming the processed CSV files.

        Returns:
            Paths to the processed CSV outputs created during preprocessing.
        """

        normalized_tables = self._load_bundle_tables(raw_paths=raw_paths, output_prefix=output_prefix)

        track_level_frame = self.create_track_level_table(
            track_metadata_frame=normalized_tables["track_metadata"][0],
            audio_features_frame=normalized_tables["audio_features"][0],
            playlist_tracks_frame=normalized_tables["playlist_tracks"][0],
            artist_metadata_frame=normalized_tables["artist_metadata"][0],
        )
        track_level_path = self.save_dataframe(
            dataframe=track_level_frame,
            dataset_name="track_level",
            output_prefix=output_prefix,
        )

        artist_level_frame = self.create_artist_level_table(
            artist_metadata_frame=normalized_tables["artist_metadata"][0],
            track_level_frame=track_level_frame,
        )
        artist_level_path = self.save_dataframe(
            dataframe=artist_level_frame,
            dataset_name="artist_level",
            output_prefix=output_prefix,
        )

        return ProcessedCollectionPaths(
            playlist_tracks_path=normalized_tables["playlist_tracks"][1],
            track_metadata_path=normalized_tables["track_metadata"][1],
            audio_features_path=normalized_tables["audio_features"][1],
            artist_metadata_path=normalized_tables["artist_metadata"][1],
            track_level_path=track_level_path,
            artist_level_path=artist_level_path,
        )

    def normalize_track_metadata(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize raw track metadata into a tabular DataFrame.

        Args:
            payload: Raw `tracks` response payload from Spotify.

        Returns:
            A clean DataFrame with one row per track.
        """

        raw_tracks = [track for track in payload.get("tracks", []) if track]
        if not raw_tracks:
            return pd.DataFrame()

        frame = pd.json_normalize(raw_tracks)

        # Lists of artists are collapsed into compact interview-friendly
        # string columns while retaining useful identifiers.
        frame["artist_ids"] = [
            self._join_nested_values(track.get("artists", []), "id")
            for track in raw_tracks
        ]
        frame["artist_names"] = [
            self._join_nested_values(track.get("artists", []), "name")
            for track in raw_tracks
        ]

        renamed_frame = frame.rename(
            columns={
                "id": "track_id",
                "album.id": "album_id",
                "album.name": "album_name",
                "album.release_date": "album_release_date",
                "album.total_tracks": "album_total_tracks",
                "external_urls.spotify": "track_url",
            }
        )

        selected_columns = [
            "track_id",
            "name",
            "artist_ids",
            "artist_names",
            "album_id",
            "album_name",
            "album_release_date",
            "album_total_tracks",
            "duration_ms",
            "explicit",
            "popularity",
            "preview_url",
            "track_number",
            "disc_number",
            "track_url",
        ]
        normalized_frame = self._select_existing_columns(renamed_frame, selected_columns)
        normalized_frame = self.normalize_metadata_fields(normalized_frame)
        return self.clean_missing_values(normalized_frame)

    def normalize_audio_features(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize raw audio feature payloads into a tabular DataFrame.

        Args:
            payload: Raw `audio_features` response payload from Spotify.

        Returns:
            A clean DataFrame with one row per track feature vector.
        """

        raw_audio_features = [item for item in payload.get("audio_features", []) if item]
        if not raw_audio_features:
            return pd.DataFrame()

        frame = pd.json_normalize(raw_audio_features).rename(columns={"id": "track_id"})
        selected_columns = [
            "track_id",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "time_signature",
        ]
        normalized_frame = self._select_existing_columns(frame, selected_columns)
        normalized_frame = self.normalize_metadata_fields(normalized_frame)
        return self.clean_missing_values(normalized_frame)

    def normalize_artist_metadata(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize raw artist metadata into a tabular DataFrame.

        Args:
            payload: Raw `artists` response payload from Spotify.

        Returns:
            A clean DataFrame with one row per artist.
        """

        raw_artists = [artist for artist in payload.get("artists", []) if artist]
        if not raw_artists:
            return pd.DataFrame()

        frame = pd.json_normalize(raw_artists)
        frame["genres"] = [
            ", ".join(artist.get("genres", []))
            for artist in raw_artists
        ]

        renamed_frame = frame.rename(
            columns={
                "id": "artist_id",
                "followers.total": "followers_total",
                "external_urls.spotify": "artist_url",
            }
        )
        selected_columns = [
            "artist_id",
            "name",
            "genres",
            "popularity",
            "followers_total",
            "artist_url",
        ]
        normalized_frame = self._select_existing_columns(renamed_frame, selected_columns)
        normalized_frame = self.normalize_metadata_fields(normalized_frame)
        return self.clean_missing_values(normalized_frame)

    def normalize_playlist_tracks(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize playlist-track items into a tabular DataFrame.

        Args:
            payload: Raw playlist track payload returned by Spotify.

        Returns:
            A clean DataFrame with one row per playlist-track item.
        """

        raw_items = payload.get("items", [])
        if not raw_items:
            return pd.DataFrame()

        frame = pd.json_normalize(raw_items)

        # Nested artist arrays require a small custom extraction step to
        # produce human-readable and join-friendly columns.
        frame["track_artist_ids"] = [
            self._join_nested_values((item.get("track") or {}).get("artists", []), "id")
            for item in raw_items
        ]
        frame["track_artist_names"] = [
            self._join_nested_values((item.get("track") or {}).get("artists", []), "name")
            for item in raw_items
        ]
        frame["playlist_id"] = payload.get("playlist_id", "")

        renamed_frame = frame.rename(
            columns={
                "track.id": "track_id",
                "track.name": "track_name",
                "track.popularity": "track_popularity",
                "track.duration_ms": "track_duration_ms",
                "track.explicit": "track_explicit",
                "track.album.id": "album_id",
                "track.album.name": "album_name",
                "track.album.release_date": "album_release_date",
                "added_by.id": "added_by_user_id",
            }
        )
        selected_columns = [
            "playlist_id",
            "added_at",
            "added_by_user_id",
            "track_id",
            "track_name",
            "track_artist_ids",
            "track_artist_names",
            "track_popularity",
            "track_duration_ms",
            "track_explicit",
            "album_id",
            "album_name",
            "album_release_date",
        ]
        normalized_frame = self._select_existing_columns(renamed_frame, selected_columns)
        normalized_frame = self.normalize_metadata_fields(normalized_frame)
        return self.clean_missing_values(normalized_frame)

    def clean_missing_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in a DataFrame using type-aware defaults.

        Args:
            dataframe: DataFrame containing cleaned Spotify records.

        Returns:
            DataFrame with missing values replaced by stable defaults.
        """

        cleaned_frame = dataframe.copy()

        # Numeric columns are filled with zeros so feature pipelines can run
        # without special-case missing-value handling on every attribute.
        numeric_columns = cleaned_frame.select_dtypes(include=["number"]).columns
        cleaned_frame.loc[:, numeric_columns] = cleaned_frame.loc[:, numeric_columns].fillna(0.0)

        # Boolean flags default to False when Spotify omits a value.
        boolean_columns = cleaned_frame.select_dtypes(include=["bool"]).columns
        cleaned_frame.loc[:, boolean_columns] = cleaned_frame.loc[:, boolean_columns].fillna(False)

        # Text fields use empty strings to keep joins and string operations safe.
        object_columns = cleaned_frame.select_dtypes(include=["object"]).columns
        cleaned_frame.loc[:, object_columns] = cleaned_frame.loc[:, object_columns].fillna("")

        return cleaned_frame

    def normalize_metadata_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize text-heavy metadata fields for consistent downstream use.

        Args:
            dataframe: DataFrame containing Spotify metadata fields.

        Returns:
            DataFrame with normalized string formatting.
        """

        normalized_frame = dataframe.copy()

        # Trimming all string columns upfront prevents hard-to-see whitespace
        # issues from breaking joins, comparisons, and feature extraction.
        object_columns = normalized_frame.select_dtypes(include=["object"]).columns
        for column in object_columns:
            normalized_frame[column] = normalized_frame[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        id_like_columns = {"artist_ids", "track_artist_ids"}
        for column in id_like_columns.intersection(set(normalized_frame.columns)):
            normalized_frame[column] = normalized_frame[column].apply(self._normalize_identifier_list)

        genre_like_columns = {"genres", "artist_genres"}
        for column in genre_like_columns.intersection(set(normalized_frame.columns)):
            normalized_frame[column] = normalized_frame[column].apply(self._normalize_delimited_text)

        name_like_columns = {"artist_names", "track_artist_names"}
        for column in name_like_columns.intersection(set(normalized_frame.columns)):
            normalized_frame[column] = normalized_frame[column].apply(self._normalize_name_list)

        return normalized_frame

    def create_track_level_table(
        self,
        track_metadata_frame: pd.DataFrame,
        audio_features_frame: pd.DataFrame,
        playlist_tracks_frame: pd.DataFrame | None = None,
        artist_metadata_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Create a curated track-level table for modeling and analysis.

        Args:
            track_metadata_frame: Clean track metadata table.
            audio_features_frame: Clean audio feature table.
            playlist_tracks_frame: Optional clean playlist-track table.
            artist_metadata_frame: Optional clean artist metadata table.

        Returns:
            Curated track-level DataFrame with audio and artist context.
        """

        if track_metadata_frame.empty and audio_features_frame.empty:
            return pd.DataFrame()

        track_level_frame = self._start_track_level_table(
            track_metadata_frame=track_metadata_frame,
            audio_features_frame=audio_features_frame,
        )
        track_level_frame = self._merge_audio_features(
            track_level_frame=track_level_frame,
            audio_features_frame=audio_features_frame,
        )
        track_level_frame = self._add_primary_artist_fields(track_level_frame)
        track_level_frame = self._add_playlist_occurrence_counts(
            track_level_frame=track_level_frame,
            playlist_tracks_frame=playlist_tracks_frame,
        )
        track_level_frame = self._merge_artist_summary_fields(
            track_level_frame=track_level_frame,
            artist_metadata_frame=artist_metadata_frame,
        )

        preferred_columns = [
            "track_id",
            "track_name",
            "primary_artist_id",
            "primary_artist_name",
            "artist_ids",
            "artist_names",
            "artist_genres",
            "album_id",
            "album_name",
            "album_release_date",
            "duration_ms",
            "explicit",
            "popularity",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "time_signature",
            "playlist_occurrences",
            "artist_popularity",
            "artist_followers_total",
            "track_url",
            "preview_url",
        ]
        curated_frame = self._select_existing_columns(track_level_frame, preferred_columns)
        curated_frame = self.normalize_metadata_fields(curated_frame)
        return self.clean_missing_values(curated_frame)

    def create_artist_level_table(
        self,
        artist_metadata_frame: pd.DataFrame,
        track_level_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create a curated artist-level table for analysis and modeling.

        Args:
            artist_metadata_frame: Clean artist metadata table.
            track_level_frame: Curated track-level table with primary-artist context.

        Returns:
            Curated artist-level DataFrame with artist metadata and aggregated track signals.
        """

        if artist_metadata_frame.empty and track_level_frame.empty:
            return pd.DataFrame()

        artist_level_frame = self._start_artist_level_table(
            artist_metadata_frame=artist_metadata_frame,
            track_level_frame=track_level_frame,
        )
        artist_level_frame = self._merge_artist_track_aggregates(
            artist_level_frame=artist_level_frame,
            track_level_frame=track_level_frame,
        )

        preferred_columns = [
            "artist_id",
            "name",
            "genres",
            "popularity",
            "followers_total",
            "track_count",
            "avg_track_popularity",
            "avg_danceability",
            "avg_energy",
            "avg_valence",
            "avg_tempo",
            "artist_url",
        ]
        curated_frame = self._select_existing_columns(artist_level_frame, preferred_columns)
        curated_frame = self.normalize_metadata_fields(curated_frame)
        return self.clean_missing_values(curated_frame)

    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        output_prefix: str,
    ) -> Path:
        """Save a processed DataFrame as CSV in the processed data directory.

        Args:
            dataframe: Clean DataFrame generated by a normalization method.
            dataset_name: Logical dataset label such as `track_metadata`.
            output_prefix: Prefix used to group files from one processing run.

        Returns:
            Path to the saved processed CSV file.
        """

        output_path = self.settings.processed_data_dir / f"{output_prefix}_{dataset_name}.csv"
        dataframe.to_csv(output_path, index=False)
        return output_path

    def _load_bundle_tables(
        self,
        raw_paths: RawCollectionPaths,
        output_prefix: str,
    ) -> dict[str, tuple[pd.DataFrame, Path | None]]:
        """Load and normalize every raw dataset in a collection bundle."""

        return {
            "playlist_tracks": self._load_normalize_and_save_if_present(
                raw_path=raw_paths.playlist_tracks_path,
                dataset_name="playlist_tracks",
                output_prefix=output_prefix,
                normalize_operation=self.normalize_playlist_tracks,
            ),
            "track_metadata": self._load_normalize_and_save_if_present(
                raw_path=raw_paths.track_metadata_path,
                dataset_name="track_metadata",
                output_prefix=output_prefix,
                normalize_operation=self.normalize_track_metadata,
            ),
            "audio_features": self._load_normalize_and_save_if_present(
                raw_path=raw_paths.audio_features_path,
                dataset_name="audio_features",
                output_prefix=output_prefix,
                normalize_operation=self.normalize_audio_features,
            ),
            "artist_metadata": self._load_normalize_and_save_if_present(
                raw_path=raw_paths.artist_metadata_path,
                dataset_name="artist_metadata",
                output_prefix=output_prefix,
                normalize_operation=self.normalize_artist_metadata,
            ),
        }

    def _start_track_level_table(
        self,
        track_metadata_frame: pd.DataFrame,
        audio_features_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Choose the best starting table for the curated track view."""

        if not track_metadata_frame.empty:
            return track_metadata_frame.copy()
        return audio_features_frame.copy()

    def _merge_audio_features(
        self,
        track_level_frame: pd.DataFrame,
        audio_features_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach audio features to the track-level table."""

        if audio_features_frame.empty:
            return track_level_frame

        audio_columns = [column for column in audio_features_frame.columns if column != "duration_ms"]

        # Audio features stay in the curated track table so downstream models
        # can read one table instead of re-joining data inside recommender code.
        return track_level_frame.merge(
            audio_features_frame.loc[:, audio_columns],
            on="track_id",
            how="left",
        )

    def _add_primary_artist_fields(self, track_level_frame: pd.DataFrame) -> pd.DataFrame:
        """Create simple primary-artist fields from multi-artist metadata."""

        enriched_frame = track_level_frame.copy()
        enriched_frame["track_name"] = self._get_column_or_default(
            enriched_frame,
            preferred_column="track_name",
            fallback_column="name",
        )
        enriched_frame["primary_artist_id"] = self._get_column_or_default(
            enriched_frame,
            preferred_column="artist_ids",
        ).apply(self._extract_first_token)
        enriched_frame["primary_artist_name"] = self._get_column_or_default(
            enriched_frame,
            preferred_column="artist_names",
        ).apply(self._extract_first_token)
        return enriched_frame

    def _add_playlist_occurrence_counts(
        self,
        track_level_frame: pd.DataFrame,
        playlist_tracks_frame: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Add simple playlist-occurrence counts to each track."""

        if playlist_tracks_frame is None or playlist_tracks_frame.empty:
            return track_level_frame

        playlist_counts = (
            playlist_tracks_frame.groupby("track_id", dropna=False)
            .size()
            .rename("playlist_occurrences")
            .reset_index()
        )
        return track_level_frame.merge(playlist_counts, on="track_id", how="left")

    def _merge_artist_summary_fields(
        self,
        track_level_frame: pd.DataFrame,
        artist_metadata_frame: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Attach primary-artist summary fields to the track table."""

        if artist_metadata_frame is None or artist_metadata_frame.empty:
            return track_level_frame

        artist_lookup = artist_metadata_frame.rename(
            columns={
                "artist_id": "primary_artist_id",
                "name": "primary_artist_name_metadata",
                "genres": "artist_genres",
                "popularity": "artist_popularity",
                "followers_total": "artist_followers_total",
            }
        )
        artist_lookup_columns = [
            "primary_artist_id",
            "artist_genres",
            "artist_popularity",
            "artist_followers_total",
            "primary_artist_name_metadata",
        ]
        for column in artist_lookup_columns:
            if column not in artist_lookup.columns:
                artist_lookup[column] = pd.NA
        return track_level_frame.merge(
            artist_lookup.loc[:, artist_lookup_columns],
            on="primary_artist_id",
            how="left",
        )

    def _start_artist_level_table(
        self,
        artist_metadata_frame: pd.DataFrame,
        track_level_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Choose the best starting table for the curated artist view."""

        if not artist_metadata_frame.empty:
            return artist_metadata_frame.copy()
        return pd.DataFrame(
            {"artist_id": track_level_frame.get("primary_artist_id", pd.Series(dtype=str))}
        )

    def _merge_artist_track_aggregates(
        self,
        artist_level_frame: pd.DataFrame,
        track_level_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach track-derived summary metrics to the artist table."""

        if track_level_frame.empty or "primary_artist_id" not in track_level_frame.columns:
            return artist_level_frame

        aggregation_spec = self._build_artist_aggregation_spec(track_level_frame)
        artist_aggregates = (
            track_level_frame.groupby("primary_artist_id", dropna=False)
            .agg(aggregation_spec)
            .reset_index()
            .rename(columns=self._artist_aggregate_column_names())
        )

        # Artist-level summaries help later recommenders reason about broad
        # artist tendencies without recalculating track averages every time.
        return artist_level_frame.merge(artist_aggregates, on="artist_id", how="outer")

    def _build_artist_aggregation_spec(
        self,
        track_level_frame: pd.DataFrame,
    ) -> dict[str, str | Callable[..., Any]]:
        """Build the aggregation spec for artist-level track summaries."""

        aggregation_candidates: dict[str, str | Callable[..., Any]] = {
            "track_id": pd.Series.nunique,
            "popularity": "mean",
            "danceability": "mean",
            "energy": "mean",
            "valence": "mean",
            "tempo": "mean",
        }
        return {
            column: operation
            for column, operation in aggregation_candidates.items()
            if column in track_level_frame.columns
        }

    def _artist_aggregate_column_names(self) -> dict[str, str]:
        """Return readable names for artist-level aggregate columns."""

        return {
            "primary_artist_id": "artist_id",
            "track_id": "track_count",
            "popularity": "avg_track_popularity",
            "danceability": "avg_danceability",
            "energy": "avg_energy",
            "valence": "avg_valence",
            "tempo": "avg_tempo",
        }

    def _load_normalize_and_save_if_present(
        self,
        raw_path: Path | None,
        dataset_name: str,
        output_prefix: str,
        normalize_operation: Callable[[dict[str, Any]], pd.DataFrame],
    ) -> tuple[pd.DataFrame, Path | None]:
        """Load, normalize, and save a dataset when a raw file is available.

        Args:
            raw_path: Path to the raw JSON dataset, when present.
            dataset_name: Logical dataset label such as `artist_metadata`.
            output_prefix: Prefix used to group files from one processing run.
            normalize_operation: Function that converts raw JSON into a DataFrame.

        Returns:
            Tuple containing the normalized DataFrame and saved CSV path.
        """

        if raw_path is None:
            return pd.DataFrame(), None

        # Centralizing this small workflow keeps bundle preprocessing concise
        # and makes new dataset types easier to add later.
        payload = read_json(raw_path)
        dataframe = normalize_operation(payload)
        saved_path = self.save_dataframe(
            dataframe=dataframe,
            dataset_name=dataset_name,
            output_prefix=output_prefix,
        )
        return dataframe, saved_path

    def _join_nested_values(self, records: list[dict[str, Any]], key: str) -> str:
        """Join values extracted from nested record lists.

        Args:
            records: Nested dictionaries such as artist payloads.
            key: Key to extract from each record.

        Returns:
            A comma-separated string of extracted values.
        """

        values = [str(record.get(key, "")) for record in records if record.get(key)]
        return ", ".join(values)

    def _extract_first_token(self, raw_value: object) -> str:
        """Extract the first comma-delimited token from a metadata field.

        Args:
            raw_value: Raw string containing one or more comma-delimited values.

        Returns:
            The first trimmed token, or an empty string.
        """

        if not isinstance(raw_value, str) or not raw_value.strip():
            return ""
        return raw_value.split(",")[0].strip()

    def _normalize_delimited_text(self, raw_value: object) -> str:
        """Normalize comma-delimited metadata into a stable string format.

        Args:
            raw_value: Raw metadata field value.

        Returns:
            Comma-separated unique values in normalized order.
        """

        if not isinstance(raw_value, str) or not raw_value.strip():
            return ""

        normalized_values = sorted(
            {
                value.strip().lower().replace("  ", " ")
                for value in raw_value.split(",")
                if value.strip()
            }
        )
        return ", ".join(normalized_values)

    def _normalize_identifier_list(self, raw_value: object) -> str:
        """Normalize comma-delimited identifier fields into a stable format.

        Args:
            raw_value: Raw metadata field containing identifiers.

        Returns:
            Comma-separated unique identifiers in lowercase order.
        """

        if not isinstance(raw_value, str) or not raw_value.strip():
            return ""

        normalized_values = sorted(
            {
                value.strip().lower()
                for value in raw_value.split(",")
                if value.strip()
            }
        )
        return ", ".join(normalized_values)

    def _normalize_name_list(self, raw_value: object) -> str:
        """Normalize comma-delimited name fields while preserving casing.

        Args:
            raw_value: Raw metadata field containing names.

        Returns:
            Comma-separated unique names in stable order.
        """

        if not isinstance(raw_value, str) or not raw_value.strip():
            return ""

        normalized_values = sorted(
            {
                value.strip()
                for value in raw_value.split(",")
                if value.strip()
            }
        )
        return ", ".join(normalized_values)

    def _get_column_or_default(
        self,
        dataframe: pd.DataFrame,
        preferred_column: str,
        fallback_column: str | None = None,
    ) -> pd.Series:
        """Return a DataFrame column or a same-length default string series.

        Args:
            dataframe: Source DataFrame being prepared for modeling.
            preferred_column: First-choice column name to retrieve.
            fallback_column: Optional fallback column if the preferred column is absent.

        Returns:
            Series aligned to the DataFrame index.
        """

        if preferred_column in dataframe.columns:
            return dataframe[preferred_column].astype(str)
        if fallback_column is not None and fallback_column in dataframe.columns:
            return dataframe[fallback_column].astype(str)
        return pd.Series([""] * len(dataframe), index=dataframe.index, dtype=str)

    def _select_existing_columns(
        self,
        dataframe: pd.DataFrame,
        preferred_columns: list[str],
    ) -> pd.DataFrame:
        """Select preferred columns that exist in a DataFrame.

        Args:
            dataframe: DataFrame containing normalized Spotify data.
            preferred_columns: Ordered columns to retain when present.

        Returns:
            A column-filtered DataFrame in a stable output order.
        """

        existing_columns = [column for column in preferred_columns if column in dataframe.columns]
        return dataframe.loc[:, existing_columns].copy()
