"""Feature engineering for content-based recommendation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureMatrixArtifacts:
    """Store model-ready content features aligned to track identifiers.

    Attributes:
        track_ids: Track identifiers aligned to the feature matrix rows.
        feature_names: Ordered feature names aligned to the matrix columns.
        feature_frame: DataFrame containing standardized model-ready features.
        feature_matrix: Numeric matrix used by content-based recommenders.
        numeric_feature_means: Mean value used to standardize each numeric feature.
        numeric_feature_stds: Standard deviation used to standardize each numeric feature.
    """

    track_ids: list[str]
    feature_names: list[str]
    feature_frame: pd.DataFrame
    feature_matrix: np.ndarray
    numeric_feature_means: dict[str, float]
    numeric_feature_stds: dict[str, float]


@dataclass(slots=True)
class FeatureBuilder:
    """Build standardized content features from cleaned track-level tables.

    Attributes:
        max_genre_features: Maximum number of genre indicator columns to include.
    """

    max_genre_features: int = 20
    content_feature_columns: tuple[str, ...] = field(
        default=(
            "danceability",
            "energy",
            "valence",
            "tempo",
            "acousticness",
            "speechiness",
            "instrumentalness",
            "loudness",
        )
    )
    compatibility_feature_columns: tuple[str, ...] = field(
        default=(
            "catalog_popularity",
            "catalog_artist_frequency",
            "catalog_artist_track_count",
            "catalog_title_token_count",
        )
    )

    def build_content_feature_table(self, track_level_frame: pd.DataFrame) -> pd.DataFrame:
        """Build a numeric feature table from a cleaned track-level table.

        Args:
            track_level_frame: Clean track-level table produced by the preprocessor.

        Returns:
            A DataFrame containing track IDs and numeric content features.
        """

        if track_level_frame.empty:
            return pd.DataFrame(
                columns=["track_id", *self.content_feature_columns, *self.compatibility_feature_columns]
            )

        feature_frame = self._ensure_content_feature_columns(track_level_frame.copy())
        feature_columns = [
            "track_id",
            *self.content_feature_columns,
            *[
                column_name
                for column_name in self.compatibility_feature_columns
                if column_name in feature_frame.columns
            ],
        ]
        feature_table = feature_frame.loc[:, feature_columns].copy()
        feature_table["track_id"] = feature_table["track_id"].astype(str)

        # Genre indicators let the content model incorporate artist metadata
        # without mixing raw string columns directly into the feature matrix.
        if "artist_genres" in track_level_frame.columns:
            genre_feature_frame = self._build_genre_feature_frame(
                track_level_frame["artist_genres"],
            )
            feature_table = pd.concat(
                [feature_table.reset_index(drop=True), genre_feature_frame.reset_index(drop=True)],
                axis=1,
            )

        return feature_table

    def _ensure_content_feature_columns(self, track_level_frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure every expected numeric content feature exists."""

        # The content recommender expects a stable set of columns, even when
        # some raw Spotify payloads are sparse or partially missing.
        for column in self.content_feature_columns:
            if column not in track_level_frame.columns:
                track_level_frame[column] = 0.0
        return track_level_frame

    def standardize_numeric_features(
        self,
        feature_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
        """Standardize numeric features into a model-ready scale.

        Args:
            feature_table: Feature DataFrame returned by `build_content_feature_table`.

        Returns:
            A tuple containing the standardized DataFrame, column means, and column standard deviations.
        """

        if feature_table.empty:
            return feature_table.copy(), {}, {}

        standardized_frame = feature_table.copy()
        numeric_columns = [
            column
            for column in standardized_frame.columns
            if column != "track_id"
        ]

        means: dict[str, float] = {}
        stds: dict[str, float] = {}

        # Z-score standardization keeps continuous features comparable so
        # cosine similarity is not dominated by high-scale attributes like tempo.
        for column in numeric_columns:
            column_values = pd.to_numeric(standardized_frame[column], errors="coerce").fillna(0.0)
            mean_value = float(column_values.mean())
            std_value = float(column_values.std(ddof=0))
            if std_value == 0.0:
                std_value = 1.0

            standardized_frame[column] = (column_values - mean_value) / std_value
            means[column] = mean_value
            stds[column] = std_value

        return standardized_frame, means, stds

    def create_model_ready_feature_matrix(
        self,
        track_level_frame: pd.DataFrame,
    ) -> FeatureMatrixArtifacts:
        """Create a fully standardized feature matrix for recommenders.

        Args:
            track_level_frame: Clean track-level table produced by the preprocessor.

        Returns:
            A `FeatureMatrixArtifacts` object aligned to the input track IDs.
        """

        feature_table = self.build_content_feature_table(track_level_frame)
        standardized_frame, means, stds = self.standardize_numeric_features(feature_table)

        if standardized_frame.empty:
            return FeatureMatrixArtifacts(
                track_ids=[],
                feature_names=[],
                feature_frame=standardized_frame,
                feature_matrix=np.empty((0, 0)),
                numeric_feature_means=means,
                numeric_feature_stds=stds,
            )

        feature_names = [column for column in standardized_frame.columns if column != "track_id"]
        feature_matrix = standardized_frame.loc[:, feature_names].to_numpy(dtype=float)

        return FeatureMatrixArtifacts(
            track_ids=standardized_frame["track_id"].astype(str).tolist(),
            feature_names=feature_names,
            feature_frame=standardized_frame,
            feature_matrix=feature_matrix,
            numeric_feature_means=means,
            numeric_feature_stds=stds,
        )

    def _build_genre_feature_frame(self, genre_series: pd.Series) -> pd.DataFrame:
        """Build one-hot genre indicators from a track-level genre column.

        Args:
            genre_series: Series containing comma-separated genre strings.

        Returns:
            DataFrame containing one-hot genre indicator columns.
        """

        genre_tokens_per_track = [
            self._extract_genre_tokens(value)
            for value in genre_series.fillna("")
        ]
        top_genres = self._select_top_genres(genre_tokens_per_track)

        genre_rows: list[dict[str, float]] = []

        # Multi-hot encoding gives the recommender a lightweight way to use
        # genre context without needing a separate categorical modeling stack.
        for genre_tokens in genre_tokens_per_track:
            genre_rows.append(
                {
                    f"genre_{genre}": 1.0 if genre in genre_tokens else 0.0
                    for genre in top_genres
                }
            )

        return pd.DataFrame(genre_rows)

    def _extract_genre_tokens(self, raw_genre_value: object) -> list[str]:
        """Convert a comma-separated genre string into normalized tokens.

        Args:
            raw_genre_value: Raw genre field value from the track table.

        Returns:
            Normalized genre tokens for one track.
        """

        if not isinstance(raw_genre_value, str) or not raw_genre_value.strip():
            return []

        normalized_tokens = [
            genre.strip().lower().replace(" ", "_")
            for genre in raw_genre_value.split(",")
            if genre.strip()
        ]
        return sorted(set(normalized_tokens))

    def _select_top_genres(self, genre_tokens_per_track: Iterable[list[str]]) -> list[str]:
        """Select the most frequent genre tokens for the feature matrix.

        Args:
            genre_tokens_per_track: Genre tokens extracted per track.

        Returns:
            Ordered list of the most frequent genres to encode.
        """

        genre_counts: dict[str, int] = {}
        for genre_tokens in genre_tokens_per_track:
            for genre in genre_tokens:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        ranked_genres = sorted(
            genre_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [genre for genre, _ in ranked_genres[: self.max_genre_features]]
