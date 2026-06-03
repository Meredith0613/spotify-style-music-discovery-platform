"""Preprocess Last.fm-style listening logs into interaction tables.

Usage:
    PYTHONPATH=src python -m data.lastfm_preprocessor
    PYTHONPATH=src python -m data.lastfm_preprocessor --input path/to/listens.csv
    PYTHONPATH=src python -m data.lastfm_preprocessor --input data/raw/userid-timestamp-artid-artname-traid-traname.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from utils.logger import get_logger


DEFAULT_RAW_LASTFM_PATH = Path("data/raw/lastfm_listening_logs.csv")
DEFAULT_PROCESSED_INTERACTIONS_PATH = Path("data/processed/lastfm_interactions.csv")
LASTFM_1K_FILENAME = "userid-timestamp-artid-artname-traid-traname.tsv"
LASTFM_1K_COLUMNS = [
    "user_id",
    "timestamp",
    "artist_id",
    "artist_name",
    "raw_track_id",
    "track_name",
]
OUTPUT_COLUMNS = [
    "user_id",
    "track_id",
    "interaction_strength",
    "artist_name",
    "track_name",
    "listen_count",
    "first_timestamp",
    "last_timestamp",
]


@dataclass(slots=True)
class LastfmColumnConfig:
    """Map raw Last.fm-style column names into the shared preprocessing schema."""

    user_id_column: str
    artist_column: str
    track_column: str
    timestamp_column: str | None = None


@dataclass(slots=True)
class LastfmPreprocessor:
    """Convert raw listening logs into aggregated implicit interaction tables."""

    output_path: Path = DEFAULT_PROCESSED_INTERACTIONS_PATH
    chunk_size: int = 250_000
    merge_frequency: int = 8
    logger_name: str = "data.lastfm_preprocessor"
    _logger: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the module logger lazily so chunk progress is visible in CLI runs."""

        self._logger = get_logger(self.logger_name)

    def preprocess_file(
        self,
        input_path: str | Path,
        *,
        column_config: LastfmColumnConfig | None = None,
    ) -> pd.DataFrame:
        """Load raw listening logs, aggregate them, and save the processed output."""

        resolved_input_path = Path(input_path)
        if self._should_use_explicit_lastfm_tsv_path(resolved_input_path, column_config):
            # Headerless Last.fm 1K-style TSVs have no columns to override safely.
            interactions = self.preprocess_explicit_lastfm_tsv(resolved_input_path)
        else:
            raw_frame = self.load_raw_logs(resolved_input_path)
            interactions = self.preprocess_frame(raw_frame, column_config=column_config)

        self.save_interactions(interactions, self.output_path)
        return interactions

    def load_raw_logs(self, input_path: str | Path) -> pd.DataFrame:
        """Load a generic raw CSV/TSV listening log with delimiter auto-detection."""

        return pd.read_csv(input_path, sep=None, engine="python")

    def preprocess_explicit_lastfm_tsv(self, input_path: str | Path) -> pd.DataFrame:
        """Preprocess the real Last.fm 1K-style TSV in memory-safe chunks.

        The real file is headerless, tab-delimited, and can include raw quote
        characters in artist and track text. This path disables quote handling
        and aggregates incrementally so the full raw log never sits in memory.
        """

        resolved_input_path = Path(input_path)
        running_aggregate = self._empty_aggregate_frame(include_datetime_columns=True)
        pending_chunk_aggregates: list[pd.DataFrame] = []
        total_rows_processed = 0
        total_rows_retained = 0

        for chunk_number, raw_chunk in enumerate(self._iter_explicit_lastfm_chunks(resolved_input_path), start=1):
            total_rows_processed += len(raw_chunk)
            prepared_chunk = self._prepare_explicit_lastfm_chunk(raw_chunk)
            chunk_rows_retained = len(prepared_chunk)
            total_rows_retained += chunk_rows_retained

            if prepared_chunk.empty:
                self._logger.info(
                    "Chunk %s processed | raw_rows=%s | retained_rows=%s | grouped_rows_estimate=%s",
                    chunk_number,
                    len(raw_chunk),
                    chunk_rows_retained,
                    len(running_aggregate),
                )
                continue

            chunk_aggregate = self._aggregate_prepared_frame(prepared_chunk)
            pending_chunk_aggregates.append(chunk_aggregate)
            if len(pending_chunk_aggregates) >= self.merge_frequency:
                running_aggregate = self._merge_aggregate_frames(running_aggregate, pending_chunk_aggregates)
                pending_chunk_aggregates = []

            grouped_rows_estimate = len(running_aggregate) + sum(len(frame) for frame in pending_chunk_aggregates)
            self._logger.info(
                "Chunk %s processed | raw_rows=%s | retained_rows=%s | grouped_rows_estimate=%s",
                chunk_number,
                len(raw_chunk),
                chunk_rows_retained,
                grouped_rows_estimate,
            )

        final_aggregate = self._merge_aggregate_frames(running_aggregate, pending_chunk_aggregates)
        final_interactions = self._finalize_aggregate_frame(final_aggregate)
        self._logger.info(
            "Finished preprocessing | input=%s | raw_rows=%s | retained_rows=%s | output_rows=%s",
            resolved_input_path,
            total_rows_processed,
            total_rows_retained,
            len(final_interactions),
        )
        return final_interactions

    def preprocess_frame(
        self,
        raw_frame: pd.DataFrame,
        *,
        column_config: LastfmColumnConfig | None = None,
    ) -> pd.DataFrame:
        """Aggregate generic raw listen events into one interaction row per user-track pair."""

        if raw_frame.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        resolved_columns = column_config or self._infer_column_config(raw_frame.columns)
        prepared_frame = self._prepare_raw_frame(raw_frame, resolved_columns)
        aggregated_frame = self._aggregate_prepared_frame(prepared_frame)
        return self._finalize_aggregate_frame(aggregated_frame)

    def save_interactions(self, interactions: pd.DataFrame, output_path: str | Path) -> Path:
        """Persist processed interactions to CSV."""

        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        interactions.to_csv(resolved_output_path, index=False)
        return resolved_output_path

    def _iter_explicit_lastfm_chunks(self, input_path: Path) -> object:
        """Yield raw Last.fm chunks using a robust fixed TSV schema."""

        return pd.read_csv(
            input_path,
            sep="\t",
            header=None,
            names=LASTFM_1K_COLUMNS,
            dtype=str,
            chunksize=self.chunk_size,
            engine="python",
            quoting=csv.QUOTE_NONE,
            na_filter=False,
            on_bad_lines="skip",
            encoding="utf-8",
        )

    def _prepare_explicit_lastfm_chunk(self, raw_chunk: pd.DataFrame) -> pd.DataFrame:
        """Normalize one explicit Last.fm chunk before aggregation."""

        prepared_chunk = raw_chunk.copy()
        prepared_chunk["user_id"] = prepared_chunk["user_id"].astype(str).str.strip()
        prepared_chunk["artist_name"] = prepared_chunk["artist_name"].astype(str).str.strip()
        prepared_chunk["track_name"] = prepared_chunk["track_name"].astype(str).str.strip()
        prepared_chunk = prepared_chunk.loc[
            (prepared_chunk["user_id"] != "")
            & (prepared_chunk["artist_name"] != "")
            & (prepared_chunk["track_name"] != "")
        ].copy()
        if prepared_chunk.empty:
            prepared_chunk["track_id"] = pd.Series(dtype=str)
            prepared_chunk["event_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
            return prepared_chunk

        prepared_chunk["track_id"] = [
            self._build_track_id(artist_name=artist_name, track_name=track_name)
            for artist_name, track_name in zip(
                prepared_chunk["artist_name"],
                prepared_chunk["track_name"],
                strict=False,
            )
        ]
        prepared_chunk["event_timestamp"] = pd.to_datetime(
            prepared_chunk["timestamp"],
            utc=True,
            errors="coerce",
        )
        return prepared_chunk

    def _aggregate_prepared_frame(self, prepared_frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a prepared event table into interaction rows."""

        if prepared_frame.empty:
            return self._empty_aggregate_frame(include_datetime_columns=True)

        aggregated_frame = (
            prepared_frame.groupby(["user_id", "track_id"], as_index=False)
            .agg(
                interaction_strength=("track_id", "size"),
                artist_name=("artist_name", "first"),
                track_name=("track_name", "first"),
                first_timestamp=("event_timestamp", "min"),
                last_timestamp=("event_timestamp", "max"),
            )
            .reset_index(drop=True)
        )
        return aggregated_frame

    def _merge_aggregate_frames(
        self,
        base_frame: pd.DataFrame,
        additional_frames: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge aggregated chunk outputs into one consolidated interaction table."""

        frames_to_merge = [
            frame
            for frame in [base_frame, *additional_frames]
            if frame is not None and not frame.empty
        ]
        if not frames_to_merge:
            return self._empty_aggregate_frame(include_datetime_columns=True)

        merged_frame = pd.concat(frames_to_merge, ignore_index=True)
        consolidated_frame = (
            merged_frame.groupby(["user_id", "track_id"], as_index=False)
            .agg(
                interaction_strength=("interaction_strength", "sum"),
                artist_name=("artist_name", "first"),
                track_name=("track_name", "first"),
                first_timestamp=("first_timestamp", "min"),
                last_timestamp=("last_timestamp", "max"),
            )
            .reset_index(drop=True)
        )
        return consolidated_frame

    def _finalize_aggregate_frame(self, aggregate_frame: pd.DataFrame) -> pd.DataFrame:
        """Convert an aggregated interaction frame into the saved output schema."""

        if aggregate_frame.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        finalized_frame = aggregate_frame.copy()
        finalized_frame["listen_count"] = pd.to_numeric(
            finalized_frame["interaction_strength"],
            errors="coerce",
        ).fillna(0).astype(int)
        finalized_frame["interaction_strength"] = finalized_frame["listen_count"].astype(float)
        finalized_frame["first_timestamp"] = self._format_timestamp_series(finalized_frame["first_timestamp"])
        finalized_frame["last_timestamp"] = self._format_timestamp_series(finalized_frame["last_timestamp"])
        finalized_frame = finalized_frame.loc[:, OUTPUT_COLUMNS]
        return finalized_frame.sort_values(
            ["user_id", "interaction_strength", "track_id"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)

    def _empty_aggregate_frame(self, *, include_datetime_columns: bool) -> pd.DataFrame:
        """Build an empty aggregate frame with stable columns for merging."""

        empty_frame = pd.DataFrame(
            {
                "user_id": pd.Series(dtype=str),
                "track_id": pd.Series(dtype=str),
                "interaction_strength": pd.Series(dtype=float),
                "artist_name": pd.Series(dtype=str),
                "track_name": pd.Series(dtype=str),
            }
        )
        if include_datetime_columns:
            empty_frame["first_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
            empty_frame["last_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        else:
            empty_frame["first_timestamp"] = pd.Series(dtype=object)
            empty_frame["last_timestamp"] = pd.Series(dtype=object)
        return empty_frame

    def _should_use_explicit_lastfm_tsv_path(
        self,
        input_path: Path,
        column_config: LastfmColumnConfig | None,
    ) -> bool:
        """Route Last.fm 1K-style headerless TSV files through the explicit parser."""

        return input_path.name == LASTFM_1K_FILENAME or self._looks_like_lastfm_1k_tsv(input_path)

    def _looks_like_lastfm_1k_tsv(self, input_path: Path) -> bool:
        """Return whether a file appears to be a headerless Last.fm 1K TSV."""

        with input_path.open("r", encoding="utf-8", errors="replace") as input_file:
            for line in input_file:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                fields = stripped_line.split("\t")
                if len(fields) != 6:
                    return False

                return (
                    fields[0].startswith("user_")
                    and "T" in fields[1]
                    and fields[1].endswith("Z")
                    and bool(fields[3].strip())
                    and bool(fields[5].strip())
                )

        return False

    def _infer_column_config(self, available_columns: pd.Index) -> LastfmColumnConfig:
        """Infer raw column names from common Last.fm-style aliases."""

        normalized_lookup = {str(column_name).strip().lower(): str(column_name) for column_name in available_columns}

        def resolve_column(candidates: tuple[str, ...], *, required: bool) -> str | None:
            for candidate in candidates:
                if candidate in normalized_lookup:
                    return normalized_lookup[candidate]
            if required:
                raise ValueError(
                    "Could not infer a required Last.fm column. "
                    f"Available columns: {list(available_columns)}"
                )
            return None

        return LastfmColumnConfig(
            user_id_column=str(resolve_column(("user_id", "user", "userid", "user_name"), required=True)),
            artist_column=str(resolve_column(("artist", "artist_name", "artistname"), required=True)),
            track_column=str(resolve_column(("track", "track_name", "song", "title"), required=True)),
            timestamp_column=resolve_column(("timestamp", "played_at", "time", "datetime"), required=False),
        )

    def _prepare_raw_frame(
        self,
        raw_frame: pd.DataFrame,
        column_config: LastfmColumnConfig,
    ) -> pd.DataFrame:
        """Normalize user, artist, track, and timestamp fields before aggregation."""

        prepared_frame = raw_frame.copy()
        prepared_frame["user_id"] = prepared_frame[column_config.user_id_column].astype(str).str.strip()
        prepared_frame["artist_name"] = prepared_frame[column_config.artist_column].astype(str).str.strip()
        prepared_frame["track_name"] = prepared_frame[column_config.track_column].astype(str).str.strip()
        prepared_frame = prepared_frame.loc[
            (prepared_frame["user_id"] != "")
            & (prepared_frame["artist_name"] != "")
            & (prepared_frame["track_name"] != "")
        ].copy()
        prepared_frame["track_id"] = [
            self._build_track_id(artist_name=artist_name, track_name=track_name)
            for artist_name, track_name in zip(
                prepared_frame["artist_name"],
                prepared_frame["track_name"],
                strict=False,
            )
        ]

        if column_config.timestamp_column is not None:
            prepared_frame["event_timestamp"] = pd.to_datetime(
                prepared_frame[column_config.timestamp_column],
                utc=True,
                errors="coerce",
            )
        else:
            prepared_frame["event_timestamp"] = pd.NaT

        return prepared_frame

    def _build_track_id(self, *, artist_name: str, track_name: str) -> str:
        """Create a stable hashed track identifier from artist and track text."""

        normalized_key = f"{self._slugify(artist_name)}::{self._slugify(track_name)}"
        digest = hashlib.sha1(normalized_key.encode("utf-8")).hexdigest()[:16]
        return f"lastfm_{digest}"

    def _slugify(self, value: str) -> str:
        """Normalize raw artist and track strings into stable identifier tokens."""

        compact_value = re.sub(r"\s+", " ", value.strip().lower())
        slug = re.sub(r"[^a-z0-9]+", "_", compact_value).strip("_")
        return slug or "unknown"

    def _format_timestamp_series(self, timestamp_series: pd.Series) -> pd.Series:
        """Convert timestamps into ISO strings while preserving missing values."""

        return timestamp_series.apply(lambda value: value.isoformat() if pd.notna(value) else None)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Last.fm preprocessing entrypoint."""

    parser = argparse.ArgumentParser(description="Preprocess Last.fm-style listening logs into interaction CSVs.")
    parser.add_argument("--input", default=str(DEFAULT_RAW_LASTFM_PATH), help="Path to the raw Last.fm-style CSV/TSV.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PROCESSED_INTERACTIONS_PATH),
        help="Path to save the processed interactions CSV.",
    )
    parser.add_argument("--user-column", default=None, help="Override the raw user ID column name.")
    parser.add_argument("--artist-column", default=None, help="Override the raw artist column name.")
    parser.add_argument("--track-column", default=None, help="Override the raw track column name.")
    parser.add_argument("--timestamp-column", default=None, help="Override the raw timestamp column name.")
    return parser


def main() -> None:
    """Run the Last.fm preprocessing CLI."""

    parser = _build_argument_parser()
    args = parser.parse_args()
    column_config = None
    if args.user_column and args.artist_column and args.track_column:
        column_config = LastfmColumnConfig(
            user_id_column=args.user_column,
            artist_column=args.artist_column,
            track_column=args.track_column,
            timestamp_column=args.timestamp_column,
        )

    preprocessor = LastfmPreprocessor(output_path=Path(args.output))
    interactions = preprocessor.preprocess_file(args.input, column_config=column_config)
    print(f"Saved {len(interactions)} processed interactions to {preprocessor.output_path}")


if __name__ == "__main__":
    main()
