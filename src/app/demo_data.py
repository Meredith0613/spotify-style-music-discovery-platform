"""Static demo datasets used by the Streamlit portfolio app."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class DemoUserProfile:
    """Represent a demo user or preference profile for the app.

    Attributes:
        user_id: Internal user identifier used by the recommenders.
        display_name: Human-readable name shown in the UI.
        summary: Short description of the user's listening taste.
        seed_track_ids: Known liked tracks used for content seeding.
        preferred_mood: Default mood used for playlist generation.
    """

    user_id: str
    display_name: str
    summary: str
    seed_track_ids: list[str]
    preferred_mood: str


def build_demo_track_catalog() -> pd.DataFrame:
    """Create a feature-complete synthetic catalog for the Streamlit demo."""

    return pd.DataFrame(
        [
            {
                "track_id": "track_1",
                "track_name": "Neon Skyline",
                "artist_name": "City Echo",
                "primary_artist_name": "City Echo",
                "danceability": 0.82,
                "energy": 0.78,
                "valence": 0.72,
                "tempo": 118.0,
                "acousticness": 0.15,
                "speechiness": 0.05,
                "instrumentalness": 0.00,
                "loudness": -5.1,
                "artist_genres": "pop, synthpop",
                "popularity": 84.0,
            },
            {
                "track_id": "track_2",
                "track_name": "Midnight Signals",
                "artist_name": "Polar Avenue",
                "primary_artist_name": "Polar Avenue",
                "danceability": 0.69,
                "energy": 0.65,
                "valence": 0.48,
                "tempo": 112.0,
                "acousticness": 0.30,
                "speechiness": 0.06,
                "instrumentalness": 0.02,
                "loudness": -6.8,
                "artist_genres": "indie pop",
                "popularity": 71.0,
            },
            {
                "track_id": "track_3",
                "track_name": "Golden Hour Drive",
                "artist_name": "Sundown Static",
                "primary_artist_name": "Sundown Static",
                "danceability": 0.77,
                "energy": 0.74,
                "valence": 0.83,
                "tempo": 121.0,
                "acousticness": 0.18,
                "speechiness": 0.04,
                "instrumentalness": 0.00,
                "loudness": -5.3,
                "artist_genres": "pop, dance pop",
                "popularity": 88.0,
            },
            {
                "track_id": "track_4",
                "track_name": "Soft Focus",
                "artist_name": "Velvet Transit",
                "primary_artist_name": "Velvet Transit",
                "danceability": 0.54,
                "energy": 0.33,
                "valence": 0.40,
                "tempo": 88.0,
                "acousticness": 0.72,
                "speechiness": 0.04,
                "instrumentalness": 0.21,
                "loudness": -11.2,
                "artist_genres": "ambient, chill",
                "popularity": 54.0,
            },
            {
                "track_id": "track_5",
                "track_name": "Paper Lanterns",
                "artist_name": "Quiet Avenue",
                "primary_artist_name": "Quiet Avenue",
                "danceability": 0.47,
                "energy": 0.35,
                "valence": 0.41,
                "tempo": 95.0,
                "acousticness": 0.71,
                "speechiness": 0.03,
                "instrumentalness": 0.18,
                "loudness": -10.0,
                "artist_genres": "lofi, chillhop",
                "popularity": 49.0,
            },
            {
                "track_id": "track_6",
                "track_name": "Iron Pulse",
                "artist_name": "Voltage Lane",
                "primary_artist_name": "Voltage Lane",
                "danceability": 0.79,
                "energy": 0.91,
                "valence": 0.62,
                "tempo": 136.0,
                "acousticness": 0.08,
                "speechiness": 0.07,
                "instrumentalness": 0.00,
                "loudness": -4.0,
                "artist_genres": "edm, electro_house",
                "popularity": 76.0,
            },
            {
                "track_id": "track_7",
                "track_name": "Harbor Lights",
                "artist_name": "Still Harbor",
                "primary_artist_name": "Still Harbor",
                "danceability": 0.39,
                "energy": 0.24,
                "valence": 0.28,
                "tempo": 80.0,
                "acousticness": 0.86,
                "speechiness": 0.03,
                "instrumentalness": 0.41,
                "loudness": -12.1,
                "artist_genres": "indie_folk, acoustic",
                "popularity": 42.0,
            },
            {
                "track_id": "track_8",
                "track_name": "Velvet Morning",
                "artist_name": "Ash Harbor",
                "primary_artist_name": "Ash Harbor",
                "danceability": 0.36,
                "energy": 0.29,
                "valence": 0.19,
                "tempo": 76.0,
                "acousticness": 0.79,
                "speechiness": 0.04,
                "instrumentalness": 0.12,
                "loudness": -11.5,
                "artist_genres": "acoustic, singer_songwriter",
                "popularity": 39.0,
            },
            {
                "track_id": "track_9",
                "track_name": "City Run",
                "artist_name": "Pulse Unit",
                "primary_artist_name": "Pulse Unit",
                "danceability": 0.81,
                "energy": 0.88,
                "valence": 0.69,
                "tempo": 132.0,
                "acousticness": 0.10,
                "speechiness": 0.05,
                "instrumentalness": 0.00,
                "loudness": -4.6,
                "artist_genres": "dance_pop, electro_pop",
                "popularity": 82.0,
            },
            {
                "track_id": "track_10",
                "track_name": "Static Bloom",
                "artist_name": "Mirror District",
                "primary_artist_name": "Mirror District",
                "danceability": 0.58,
                "energy": 0.52,
                "valence": 0.37,
                "tempo": 104.0,
                "acousticness": 0.44,
                "speechiness": 0.05,
                "instrumentalness": 0.09,
                "loudness": -7.9,
                "artist_genres": "dream_pop, indie_rock",
                "popularity": 46.0,
            },
            {
                "track_id": "track_11",
                "track_name": "Ashes And Echoes",
                "artist_name": "Grey District",
                "primary_artist_name": "Grey District",
                "danceability": 0.40,
                "energy": 0.31,
                "valence": 0.16,
                "tempo": 79.0,
                "acousticness": 0.68,
                "speechiness": 0.04,
                "instrumentalness": 0.06,
                "loudness": -9.8,
                "artist_genres": "alternative, moody_pop",
                "popularity": 44.0,
            },
            {
                "track_id": "track_12",
                "track_name": "Bright Avenue",
                "artist_name": "Golden Habit",
                "primary_artist_name": "Golden Habit",
                "danceability": 0.75,
                "energy": 0.71,
                "valence": 0.86,
                "tempo": 116.0,
                "acousticness": 0.22,
                "speechiness": 0.04,
                "instrumentalness": 0.00,
                "loudness": -5.6,
                "artist_genres": "funk_pop, pop",
                "popularity": 79.0,
            },
        ]
    )


def build_demo_user_profiles() -> dict[str, DemoUserProfile]:
    """Create selectable demo user personas for the Streamlit app."""

    profiles = [
        DemoUserProfile(
            user_id="runner_pop",
            display_name="Runner Pop",
            summary="High-energy pop and dance tracks for workouts and bright commutes.",
            seed_track_ids=["track_1", "track_3", "track_6", "track_9"],
            preferred_mood="workout",
        ),
        DemoUserProfile(
            user_id="study_focus",
            display_name="Study Focus",
            summary="Calm, instrumental, low-distraction tracks for work and deep focus.",
            seed_track_ids=["track_4", "track_5", "track_7"],
            preferred_mood="study",
        ),
        DemoUserProfile(
            user_id="night_melancholy",
            display_name="Late-Night Melancholy",
            summary="Reflective, lower-valence tracks with acoustic and dream-pop textures.",
            seed_track_ids=["track_8", "track_10", "track_11"],
            preferred_mood="melancholic",
        ),
        DemoUserProfile(
            user_id="sunny_mix",
            display_name="Sunny Mix",
            summary="Optimistic pop leaning toward feel-good hooks and moderate danceability.",
            seed_track_ids=["track_1", "track_3", "track_12"],
            preferred_mood="happy",
        ),
        DemoUserProfile(
            user_id="new_listener",
            display_name="New Listener",
            summary="Cold-start profile with no prior history; relies on hybrid fallback behavior.",
            seed_track_ids=[],
            preferred_mood="calm",
        ),
    ]
    return {profile.user_id: profile for profile in profiles}


def build_demo_interactions() -> pd.DataFrame:
    """Create implicit interaction history for collaborative filtering."""

    interaction_rows = [
        {"user_id": "runner_pop", "track_id": "track_1", "interaction_strength": 4.0},
        {"user_id": "runner_pop", "track_id": "track_3", "interaction_strength": 5.0},
        {"user_id": "runner_pop", "track_id": "track_6", "interaction_strength": 5.0},
        {"user_id": "runner_pop", "track_id": "track_9", "interaction_strength": 4.0},
        {"user_id": "runner_pop", "track_id": "track_12", "interaction_strength": 2.0},
        {"user_id": "study_focus", "track_id": "track_4", "interaction_strength": 5.0},
        {"user_id": "study_focus", "track_id": "track_5", "interaction_strength": 4.0},
        {"user_id": "study_focus", "track_id": "track_7", "interaction_strength": 4.0},
        {"user_id": "study_focus", "track_id": "track_10", "interaction_strength": 2.0},
        {"user_id": "night_melancholy", "track_id": "track_8", "interaction_strength": 5.0},
        {"user_id": "night_melancholy", "track_id": "track_10", "interaction_strength": 4.0},
        {"user_id": "night_melancholy", "track_id": "track_11", "interaction_strength": 5.0},
        {"user_id": "night_melancholy", "track_id": "track_7", "interaction_strength": 2.0},
        {"user_id": "sunny_mix", "track_id": "track_1", "interaction_strength": 4.0},
        {"user_id": "sunny_mix", "track_id": "track_3", "interaction_strength": 4.0},
        {"user_id": "sunny_mix", "track_id": "track_12", "interaction_strength": 5.0},
        {"user_id": "sunny_mix", "track_id": "track_2", "interaction_strength": 2.0},
        {"user_id": "club_commuter", "track_id": "track_2", "interaction_strength": 3.0},
        {"user_id": "club_commuter", "track_id": "track_6", "interaction_strength": 4.0},
        {"user_id": "club_commuter", "track_id": "track_9", "interaction_strength": 5.0},
        {"user_id": "club_commuter", "track_id": "track_3", "interaction_strength": 4.0},
        {"user_id": "coffee_reader", "track_id": "track_4", "interaction_strength": 4.0},
        {"user_id": "coffee_reader", "track_id": "track_5", "interaction_strength": 5.0},
        {"user_id": "coffee_reader", "track_id": "track_8", "interaction_strength": 2.0},
        {"user_id": "coffee_reader", "track_id": "track_10", "interaction_strength": 3.0},
    ]
    return pd.DataFrame(interaction_rows)
