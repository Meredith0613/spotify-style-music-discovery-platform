"""Build a lightweight recommendation catalog from processed Last.fm interactions.

Usage:
    PYTHONPATH=src python -m data.lastfm_catalog_builder
    PYTHONPATH=src python -m data.lastfm_catalog_builder --input data/processed/lastfm_interactions.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INTERACTIONS_PATH = Path("data/processed/lastfm_interactions.csv")
DEFAULT_CATALOG_PATH = Path("data/processed/lastfm_catalog.csv")


@dataclass(slots=True)
class LastfmCatalogBuilder:
    """Construct a lightweight real-data catalog compatible with the recommender stack."""

    output_path: Path = DEFAULT_CATALOG_PATH

    def build_catalog_from_file(self, interactions_path: str | Path) -> pd.DataFrame:
        """Load processed interactions, build the catalog, and save it to disk."""

        interactions = pd.read_csv(interactions_path)
        catalog = self.build_catalog(interactions)
        self.save_catalog(catalog, self.output_path)
        return catalog

    def build_catalog(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Create a deduplicated track catalog with lightweight compatibility features."""

        if interactions.empty:
            return pd.DataFrame(
                columns=[
                    "track_id",
                    "track_name",
                    "artist_name",
                    "primary_artist_name",
                    "popularity_count",
                    "normalized_popularity",
                    "artist_frequency",
                    "artist_track_count",
                    "title_token_count",
                    "popularity",
                    "catalog_popularity",
                    "catalog_artist_frequency",
                    "catalog_artist_track_count",
                    "catalog_title_token_count",
                    "artist_genres",
                ]
            )

        prepared_interactions = interactions.copy()
        prepared_interactions["track_id"] = prepared_interactions["track_id"].astype(str)
        prepared_interactions["artist_name"] = self._normalize_text_series(prepared_interactions["artist_name"])
        prepared_interactions["track_name"] = self._normalize_text_series(prepared_interactions["track_name"])
        prepared_interactions["interaction_strength"] = pd.to_numeric(
            prepared_interactions["interaction_strength"],
            errors="coerce",
        ).fillna(0.0)

        track_catalog = (
            prepared_interactions.groupby("track_id", as_index=False)
            .agg(
                track_name=("track_name", "first"),
                artist_name=("artist_name", "first"),
                popularity_count=("interaction_strength", "sum"),
            )
            .reset_index(drop=True)
        )

        artist_listen_counts = (
            prepared_interactions.groupby("artist_name")["interaction_strength"]
            .sum()
            .astype(float)
        )
        artist_track_counts = (
            prepared_interactions.groupby("artist_name")["track_id"]
            .nunique()
            .astype(float)
        )

        max_popularity = max(float(track_catalog["popularity_count"].max()), 1.0)
        max_artist_frequency = max(float(artist_listen_counts.max()), 1.0)
        max_artist_track_count = max(float(artist_track_counts.max()), 1.0)
        title_token_counts = track_catalog["track_name"].apply(self._count_title_tokens).astype(float)
        max_title_token_count = max(float(title_token_counts.max()), 1.0)

        track_catalog["primary_artist_name"] = track_catalog["artist_name"]
        track_catalog["normalized_popularity"] = track_catalog["popularity_count"].astype(float) / max_popularity
        track_catalog["artist_frequency"] = track_catalog["artist_name"].map(artist_listen_counts).astype(float)
        track_catalog["artist_track_count"] = track_catalog["artist_name"].map(artist_track_counts).astype(float)
        track_catalog["title_token_count"] = title_token_counts
        track_catalog["popularity"] = track_catalog["normalized_popularity"] * 100.0
        track_catalog["catalog_popularity"] = track_catalog["normalized_popularity"]
        track_catalog["catalog_artist_frequency"] = track_catalog["artist_frequency"] / max_artist_frequency
        track_catalog["catalog_artist_track_count"] = track_catalog["artist_track_count"] / max_artist_track_count
        track_catalog["catalog_title_token_count"] = track_catalog["title_token_count"] / max_title_token_count
        track_catalog["artist_genres"] = track_catalog["artist_name"].apply(self._build_artist_identity_token)

        return track_catalog.sort_values(
            ["popularity_count", "track_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

    def save_catalog(self, catalog: pd.DataFrame, output_path: str | Path) -> Path:
        """Persist the processed Last.fm catalog to CSV."""

        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(resolved_output_path, index=False)
        return resolved_output_path

    def _build_artist_identity_token(self, artist_name: str) -> str:
        """Encode artist identity as a lightweight categorical token for content features."""

        normalized_artist_name = re.sub(r"[^a-z0-9]+", "_", str(artist_name).strip().lower()).strip("_")
        return f"artist_{normalized_artist_name or 'unknown'}"

    def _count_title_tokens(self, track_name: object) -> int:
        """Count whitespace-delimited title tokens as a lightweight metadata signal."""

        if pd.isna(track_name):
            return 0

        normalized_track_name = str(track_name).strip()
        if not normalized_track_name:
            return 0
        return len([token for token in normalized_track_name.split() if token])

    def _normalize_text_series(self, text_series: pd.Series) -> pd.Series:
        """Convert present text values to stripped strings while preserving missing as blank."""

        return text_series.where(text_series.notna(), "").astype(str).str.strip()


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Last.fm catalog builder."""

    parser = argparse.ArgumentParser(description="Build a lightweight Last.fm catalog from processed interactions.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INTERACTIONS_PATH),
        help="Path to the processed Last.fm interactions CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CATALOG_PATH),
        help="Path to save the processed Last.fm catalog CSV.",
    )
    return parser


def main() -> None:
    """Run the Last.fm catalog-building CLI."""

    parser = _build_argument_parser()
    args = parser.parse_args()
    builder = LastfmCatalogBuilder(output_path=Path(args.output))
    catalog = builder.build_catalog_from_file(args.input)
    print(f"Saved {len(catalog)} catalog rows to {builder.output_path}")


if __name__ == "__main__":
    main()
