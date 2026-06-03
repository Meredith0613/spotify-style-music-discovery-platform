"""Tests for Last.fm-style listening-log preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.lastfm_preprocessor import LASTFM_1K_FILENAME, LastfmColumnConfig, LastfmPreprocessor


def test_lastfm_preprocessor_parses_headerless_tsv_fixture(tmp_path: Path) -> None:
    """The explicit Last.fm TSV path should parse headerless six-column files."""

    input_path = tmp_path / LASTFM_1K_FILENAME
    input_path.write_text(
        "\n".join(
            [
                "user_1\t2009-05-04T23:08:57Z\tartist-id-1\tArtist One\t\tSong Alpha",
                "user_1\t2009-05-05T00:08:57Z\tartist-id-1\tArtist One\t\tSong Alpha",
                "user_2\t2009-05-06T00:08:57Z\tartist-id-2\tArtist Two\t\tSong Beta",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    interactions = LastfmPreprocessor(output_path=tmp_path / "out.csv", chunk_size=2).preprocess_file(input_path)

    assert len(interactions) == 2
    repeated_track = interactions.loc[interactions["track_name"] == "Song Alpha"].iloc[0]
    assert repeated_track["user_id"] == "user_1"
    assert repeated_track["interaction_strength"] == 2.0
    assert repeated_track["listen_count"] == 2
    assert repeated_track["first_timestamp"] == "2009-05-04T23:08:57+00:00"
    assert repeated_track["last_timestamp"] == "2009-05-05T00:08:57+00:00"


def test_lastfm_preprocessor_detects_headerless_tsv_with_nonstandard_filename(tmp_path: Path) -> None:
    """Headerless Last.fm 1K-style samples should not depend on the canonical filename."""

    input_path = tmp_path / "sample_20k.tsv"
    input_path.write_text(
        "\n".join(
            [
                "user_1\t2009-05-04T23:08:57Z\tartist-id-1\tArtist One\t\tSong Alpha",
                "user_2\t2009-05-06T00:08:57Z\tartist-id-2\tArtist Two\t\tSong Beta",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    interactions = LastfmPreprocessor(output_path=tmp_path / "out.csv").preprocess_file(
        input_path,
        column_config=LastfmColumnConfig(
            user_id_column="ignored_user_column",
            artist_column="ignored_artist_column",
            track_column="ignored_track_column",
            timestamp_column="ignored_timestamp_column",
        ),
    )

    assert len(interactions) == 2
    assert set(interactions["user_id"]) == {"user_1", "user_2"}
    assert set(interactions["track_name"]) == {"Song Alpha", "Song Beta"}


def test_lastfm_preprocessor_chunked_aggregation_matches_expected_counts(tmp_path: Path) -> None:
    """Chunked processing should merge repeated user-track listens across chunks correctly."""

    input_path = tmp_path / LASTFM_1K_FILENAME
    input_path.write_text(
        "\n".join(
            [
                "user_1\t2009-05-01T00:00:00Z\tartist-id-1\tArtist One\t\tSong Alpha",
                "user_1\t2009-05-02T00:00:00Z\tartist-id-1\tArtist One\t\tSong Alpha",
                "user_1\t2009-05-03T00:00:00Z\tartist-id-2\tArtist Two\t\tSong Beta",
                "user_1\t2009-05-04T00:00:00Z\tartist-id-1\tArtist One\t\tSong Alpha",
                "user_2\t2009-05-05T00:00:00Z\tartist-id-3\tArtist Three\t\tSong Gamma",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    interactions = LastfmPreprocessor(
        output_path=tmp_path / "out.csv",
        chunk_size=2,
        merge_frequency=1,
    ).preprocess_file(input_path)

    alpha_row = interactions.loc[
        (interactions["user_id"] == "user_1") & (interactions["track_name"] == "Song Alpha")
    ].iloc[0]

    assert len(interactions) == 3
    assert alpha_row["interaction_strength"] == 3.0
    assert alpha_row["listen_count"] == 3
    assert alpha_row["first_timestamp"] == "2009-05-01T00:00:00+00:00"
    assert alpha_row["last_timestamp"] == "2009-05-04T00:00:00+00:00"


def test_lastfm_preprocessor_builds_deterministic_track_ids() -> None:
    """Whitespace and case normalization should not change the generated track ID."""

    preprocessor = LastfmPreprocessor()

    base_track_id = preprocessor._build_track_id(artist_name="Artist One", track_name="Song Alpha")
    normalized_track_id = preprocessor._build_track_id(artist_name="  artist one ", track_name="song   alpha")

    assert base_track_id == normalized_track_id


def test_lastfm_preprocessor_aggregates_repeated_listens_for_generic_frames() -> None:
    """The fallback generic path should still work for small headered inputs."""

    raw_frame = pd.DataFrame(
        [
            {"user_id": "u1", "artist": "Artist A", "track": "Song One", "timestamp": "2024-01-01T00:00:00Z"},
            {"user_id": "u1", "artist": "Artist A", "track": "Song One", "timestamp": "2024-01-02T00:00:00Z"},
            {"user_id": "u1", "artist": "Artist B", "track": "Song Two", "timestamp": "2024-01-03T00:00:00Z"},
        ]
    )

    interactions = LastfmPreprocessor().preprocess_frame(raw_frame)

    assert len(interactions) == 2
    repeated_track = interactions.loc[interactions["track_name"] == "Song One"].iloc[0]
    assert repeated_track["interaction_strength"] == 2.0
    assert repeated_track["listen_count"] == 2
    assert repeated_track["artist_name"] == "Artist A"
    assert repeated_track["first_timestamp"] == "2024-01-01T00:00:00+00:00"
    assert repeated_track["last_timestamp"] == "2024-01-02T00:00:00+00:00"


def test_lastfm_preprocessor_preserves_generic_headered_csv_path(tmp_path: Path) -> None:
    """Headered CSV files should still use generic column inference."""

    input_path = tmp_path / "listens.csv"
    input_path.write_text(
        "\n".join(
            [
                "user_id,artist,track,timestamp",
                "u1,Artist A,Song One,2024-01-01T00:00:00Z",
                "u1,Artist A,Song One,2024-01-02T00:00:00Z",
                "u2,Artist B,Song Two,2024-01-03T00:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    interactions = LastfmPreprocessor(output_path=tmp_path / "out.csv").preprocess_file(input_path)

    assert len(interactions) == 2
    repeated_track = interactions.loc[interactions["track_name"] == "Song One"].iloc[0]
    assert repeated_track["user_id"] == "u1"
    assert repeated_track["interaction_strength"] == 2.0
