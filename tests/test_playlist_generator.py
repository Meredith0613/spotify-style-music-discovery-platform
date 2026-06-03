"""Tests for the mood-aware playlist generator."""

from __future__ import annotations

import pandas as pd

from models.playlist_generator import PlaylistGenerator


def build_playlist_candidates() -> pd.DataFrame:
    """Create a compact candidate table with playlist-relevant audio features."""

    return pd.DataFrame(
        [
            {
                "track_id": "track_a",
                "track_name": "Night Library",
                "artist_name": "Quiet Avenue",
                "danceability": 0.44,
                "energy": 0.32,
                "valence": 0.38,
                "tempo": 92.0,
                "acousticness": 0.74,
                "instrumentalness": 0.25,
            },
            {
                "track_id": "track_b",
                "track_name": "Paper Lanterns",
                "artist_name": "Quiet Avenue",
                "danceability": 0.47,
                "energy": 0.35,
                "valence": 0.41,
                "tempo": 95.0,
                "acousticness": 0.71,
                "instrumentalness": 0.18,
            },
            {
                "track_id": "track_c",
                "track_name": "Window Rain",
                "artist_name": "Still Harbor",
                "danceability": 0.39,
                "energy": 0.24,
                "valence": 0.28,
                "tempo": 80.0,
                "acousticness": 0.86,
                "instrumentalness": 0.41,
            },
        ]
    )


def build_workout_candidates() -> pd.DataFrame:
    """Create tracks that make sequencing behavior easy to observe."""

    return pd.DataFrame(
        [
            {
                "track_id": "opener",
                "track_name": "Start Fast",
                "artist_name": "Pulse Unit",
                "danceability": 0.78,
                "energy": 0.86,
                "valence": 0.72,
                "tempo": 132.0,
                "acousticness": 0.18,
                "instrumentalness": 0.00,
            },
            {
                "track_id": "smooth_next",
                "track_name": "Keep Moving",
                "artist_name": "Pulse Unit",
                "danceability": 0.80,
                "energy": 0.84,
                "valence": 0.70,
                "tempo": 134.0,
                "acousticness": 0.16,
                "instrumentalness": 0.00,
            },
            {
                "track_id": "abrupt_jump",
                "track_name": "Sprint Spike",
                "artist_name": "Voltage Lane",
                "danceability": 0.82,
                "energy": 0.90,
                "valence": 0.71,
                "tempo": 162.0,
                "acousticness": 0.12,
                "instrumentalness": 0.00,
            },
        ]
    )


def test_playlist_generator_returns_tracks_and_explanations() -> None:
    """Generated playlists should include ordered tracks and selection reasons."""

    generator = PlaylistGenerator()
    playlist = generator.generate_playlist(
        candidate_tracks=build_playlist_candidates(),
        mood_label="study",
        max_items=2,
    )

    assert playlist.mood == "study"
    assert len(playlist.tracks) == 2
    assert playlist.track_ids == [track.track_id for track in playlist.tracks]
    assert playlist.tracks[0].explanation.reasons
    assert playlist.tracks[1].explanation.score_breakdown.transition_score >= 0.0


def test_playlist_generator_penalizes_abrupt_sequence_jumps() -> None:
    """The next-track scorer should prefer smoother tempo and energy transitions."""

    generator = PlaylistGenerator()
    playlist = generator.generate_playlist(
        candidate_tracks=build_workout_candidates(),
        mood_label="workout",
        max_items=2,
    )

    assert playlist.track_ids[0] == "opener"
    assert playlist.track_ids[1] == "smooth_next"


def test_playlist_generator_supports_multiple_required_moods() -> None:
    """The built-in generator should expose the required mood profiles."""

    generator = PlaylistGenerator()

    assert {"workout", "study", "happy", "calm", "melancholic"}.issubset(generator.mood_profiles)
