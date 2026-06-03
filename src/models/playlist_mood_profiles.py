"""Default mood-profile configuration for playlist generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MoodProfile:
    """Store target feature ranges and sequencing preferences for one mood.

    Attributes:
        label: Human-readable mood label.
        target_feature_values: Target audio feature centers for the mood.
        feature_tolerances: Acceptable distance from each target center.
        sequencing_tempo_tolerance: Largest tempo jump considered smooth.
        sequencing_energy_tolerance: Largest energy jump considered smooth.
    """

    label: str
    target_feature_values: dict[str, float]
    feature_tolerances: dict[str, float]
    sequencing_tempo_tolerance: float
    sequencing_energy_tolerance: float


def build_default_mood_profiles() -> dict[str, MoodProfile]:
    """Return the built-in mood-to-feature mappings used by playlists."""

    return {
        "workout": MoodProfile(
            label="workout",
            target_feature_values={
                "danceability": 0.78,
                "energy": 0.86,
                "valence": 0.72,
                "tempo": 132.0,
                "acousticness": 0.18,
            },
            feature_tolerances={
                "danceability": 0.25,
                "energy": 0.25,
                "valence": 0.30,
                "tempo": 28.0,
                "acousticness": 0.30,
            },
            sequencing_tempo_tolerance=16.0,
            sequencing_energy_tolerance=0.22,
        ),
        "study": MoodProfile(
            label="study",
            target_feature_values={
                "danceability": 0.45,
                "energy": 0.35,
                "valence": 0.40,
                "tempo": 95.0,
                "acousticness": 0.70,
            },
            feature_tolerances={
                "danceability": 0.22,
                "energy": 0.22,
                "valence": 0.25,
                "tempo": 18.0,
                "acousticness": 0.35,
            },
            sequencing_tempo_tolerance=10.0,
            sequencing_energy_tolerance=0.12,
        ),
        "happy": MoodProfile(
            label="happy",
            target_feature_values={
                "danceability": 0.72,
                "energy": 0.70,
                "valence": 0.82,
                "tempo": 118.0,
                "acousticness": 0.30,
            },
            feature_tolerances={
                "danceability": 0.25,
                "energy": 0.25,
                "valence": 0.20,
                "tempo": 24.0,
                "acousticness": 0.30,
            },
            sequencing_tempo_tolerance=14.0,
            sequencing_energy_tolerance=0.18,
        ),
        "calm": MoodProfile(
            label="calm",
            target_feature_values={
                "danceability": 0.42,
                "energy": 0.24,
                "valence": 0.42,
                "tempo": 82.0,
                "acousticness": 0.82,
            },
            feature_tolerances={
                "danceability": 0.20,
                "energy": 0.18,
                "valence": 0.22,
                "tempo": 16.0,
                "acousticness": 0.25,
            },
            sequencing_tempo_tolerance=8.0,
            sequencing_energy_tolerance=0.10,
        ),
        "melancholic": MoodProfile(
            label="melancholic",
            target_feature_values={
                "danceability": 0.38,
                "energy": 0.30,
                "valence": 0.20,
                "tempo": 78.0,
                "acousticness": 0.72,
            },
            feature_tolerances={
                "danceability": 0.20,
                "energy": 0.18,
                "valence": 0.18,
                "tempo": 18.0,
                "acousticness": 0.28,
            },
            sequencing_tempo_tolerance=8.0,
            sequencing_energy_tolerance=0.10,
        ),
        "focus": MoodProfile(
            label="focus",
            target_feature_values={
                "danceability": 0.46,
                "energy": 0.32,
                "valence": 0.38,
                "tempo": 92.0,
                "acousticness": 0.68,
            },
            feature_tolerances={
                "danceability": 0.22,
                "energy": 0.18,
                "valence": 0.22,
                "tempo": 16.0,
                "acousticness": 0.30,
            },
            sequencing_tempo_tolerance=8.0,
            sequencing_energy_tolerance=0.10,
        ),
    }
