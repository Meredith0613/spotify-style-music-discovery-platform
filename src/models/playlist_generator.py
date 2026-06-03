"""Mood-aware playlist generation built on track-level audio features."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .playlist_mood_profiles import MoodProfile, build_default_mood_profiles


@dataclass(slots=True)
class TrackScoreBreakdown:
    """Store score components used while building a playlist sequence.

    Attributes:
        mood_relevance_score: Fit to the requested mood profile.
        diversity_score: Added variety relative to already selected tracks.
        transition_score: Smoothness relative to the previous track.
        final_score: Weighted score used for sequence selection.
    """

    mood_relevance_score: float
    diversity_score: float
    transition_score: float
    final_score: float


@dataclass(slots=True)
class TrackSelectionExplanation:
    """Explain why one track was selected for the playlist.

    Attributes:
        mood_label: Requested playlist mood.
        score_breakdown: Numerical contribution of each playlist component.
        reasons: Short human-readable reasons for the selection.
    """

    mood_label: str
    score_breakdown: TrackScoreBreakdown
    reasons: list[str]


@dataclass(slots=True)
class PlaylistTrack:
    """Represent one ordered playlist track with selection metadata.

    Attributes:
        track_id: Track identifier.
        track_name: Human-readable track name.
        artist_name: Human-readable artist name.
        explanation: Structured explanation for why the track was selected.
    """

    track_id: str
    track_name: str
    artist_name: str
    explanation: TrackSelectionExplanation


@dataclass(slots=True)
class GeneratedPlaylist:
    """Represent a generated playlist with ordered tracks and explanations.

    Attributes:
        name: Display name for the playlist.
        mood: Mood label associated with the playlist.
        tracks: Ordered playlist track objects with explanations.
    """

    name: str
    mood: str
    tracks: list[PlaylistTrack]

    @property
    def track_ids(self) -> list[str]:
        """Return ordered track identifiers for compatibility with app layers."""

        return [track.track_id for track in self.tracks]

    @property
    def explanations(self) -> list[TrackSelectionExplanation]:
        """Return ordered explanations for the generated playlist."""

        return [track.explanation for track in self.tracks]


@dataclass(slots=True)
class PlaylistGenerator:
    """Build mood-aware playlists from track-level candidate tables.

    This class separates the playlist problem into three explainable stages:
    mood mapping defines the target sound, per-track scoring evaluates how well
    candidates fit that sound, and sequencing picks an order that stays smooth
    while still adding variety.

    Attributes:
        mood_profiles: Supported mood profiles keyed by label.
        relevance_weight: Weight for mood-profile fit during sequence scoring.
        diversity_weight: Weight for diversity relative to selected tracks.
        transition_weight: Weight for smooth transitions between neighboring tracks.
        diversity_features: Feature columns used to reward playlist variety.
    """

    mood_profiles: dict[str, MoodProfile] = field(default_factory=dict)
    relevance_weight: float = 0.6
    diversity_weight: float = 0.2
    transition_weight: float = 0.2
    diversity_features: list[str] = field(
        default_factory=lambda: ["danceability", "valence", "acousticness", "instrumentalness"]
    )

    def __post_init__(self) -> None:
        """Populate default mood profiles when none are supplied."""

        if not self.mood_profiles:
            self.mood_profiles = build_default_mood_profiles()

    def generate_playlist(
        self,
        candidate_tracks: pd.DataFrame,
        mood_label: str,
        max_items: int,
    ) -> GeneratedPlaylist:
        """Generate an ordered mood-aware playlist from candidate tracks.

        Args:
            candidate_tracks: Candidate track table containing metadata and audio features.
            mood_label: Desired mood such as `workout`, `study`, or `calm`.
            max_items: Maximum number of tracks to include.

        Returns:
            Generated playlist with ordered tracks and selection explanations.
        """

        if max_items <= 0 or candidate_tracks.empty:
            return GeneratedPlaylist(name=f"{mood_label.title()} Playlist", mood=mood_label, tracks=[])

        mood_profile = self._get_mood_profile(mood_label)
        prepared_candidates = self._prepare_candidate_tracks(candidate_tracks)
        if prepared_candidates.empty:
            return GeneratedPlaylist(name=f"{mood_profile.label.title()} Playlist", mood=mood_profile.label, tracks=[])

        selected_tracks: list[PlaylistTrack] = []
        remaining_candidates = prepared_candidates.copy()

        # The first track anchors the playlist mood, so opener selection focuses
        # mostly on direct mood fit rather than transition smoothness.
        opener_row = self._select_opening_track(remaining_candidates, mood_profile)
        selected_tracks.append(
            self._build_playlist_track(
                selected_track_row=opener_row,
                mood_profile=mood_profile,
                score_breakdown=TrackScoreBreakdown(
                    mood_relevance_score=float(opener_row["mood_relevance_score"]),
                    diversity_score=0.0,
                    transition_score=1.0,
                    final_score=float(opener_row["mood_relevance_score"]),
                ),
                previous_track_row=None,
                selected_track_rows=[opener_row],
            )
        )
        remaining_candidates = self._remove_selected_track(remaining_candidates, opener_row["track_id"])

        while len(selected_tracks) < max_items and not remaining_candidates.empty:
            selected_track_rows = [
                self._get_track_row(prepared_candidates, playlist_track.track_id)
                for playlist_track in selected_tracks
            ]
            previous_track_row = selected_track_rows[-1]
            next_track_row, score_breakdown = self._select_next_track(
                remaining_candidates=remaining_candidates,
                selected_track_rows=selected_track_rows,
                previous_track_row=previous_track_row,
                mood_profile=mood_profile,
            )
            if next_track_row is None or score_breakdown is None:
                break

            selected_tracks.append(
                self._build_playlist_track(
                    selected_track_row=next_track_row,
                    mood_profile=mood_profile,
                    score_breakdown=score_breakdown,
                    previous_track_row=previous_track_row,
                    selected_track_rows=selected_track_rows,
                )
            )
            remaining_candidates = self._remove_selected_track(
                remaining_candidates,
                next_track_row["track_id"],
            )

        return GeneratedPlaylist(
            name=f"{mood_profile.label.title()} Playlist",
            mood=mood_profile.label,
            tracks=selected_tracks,
        )

    def _get_mood_profile(self, mood_label: str) -> MoodProfile:
        """Return the requested mood profile or raise a clear error."""

        normalized_label = mood_label.strip().lower()
        if normalized_label not in self.mood_profiles:
            supported_moods = ", ".join(sorted(self.mood_profiles))
            raise ValueError(f"Unsupported mood '{mood_label}'. Supported moods: {supported_moods}.")
        return self.mood_profiles[normalized_label]

    def _prepare_candidate_tracks(self, candidate_tracks: pd.DataFrame) -> pd.DataFrame:
        """Validate and enrich candidate tracks before playlist scoring."""

        prepared_candidates = candidate_tracks.copy()
        if "artist_name" not in prepared_candidates.columns and "primary_artist_name" in prepared_candidates.columns:
            prepared_candidates["artist_name"] = prepared_candidates["primary_artist_name"]

        required_columns = {"track_id", "track_name", "artist_name", "energy", "tempo", "valence"}
        missing_columns = required_columns.difference(prepared_candidates.columns)
        if missing_columns:
            missing_column_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Candidate tracks are missing required columns: {missing_column_list}.")

        # We coerce modeling features to numeric here so later scoring logic
        # can stay small and focused on recommendation behavior.
        numeric_feature_columns = {
            "danceability",
            "energy",
            "valence",
            "tempo",
            "acousticness",
            "speechiness",
            "instrumentalness",
            "loudness",
            "popularity",
        }
        for column in numeric_feature_columns.intersection(set(prepared_candidates.columns)):
            prepared_candidates[column] = pd.to_numeric(prepared_candidates[column], errors="coerce")

        defaults_by_feature = {
            "danceability": 0.5,
            "energy": 0.5,
            "valence": 0.5,
            "tempo": 120.0,
            "acousticness": 0.5,
            "speechiness": 0.1,
            "instrumentalness": 0.0,
            "loudness": -10.0,
            "popularity": 0.5,
        }
        for feature_name, default_value in defaults_by_feature.items():
            if feature_name not in prepared_candidates.columns:
                prepared_candidates[feature_name] = default_value
            prepared_candidates[feature_name] = prepared_candidates[feature_name].fillna(default_value)

        prepared_candidates["track_name"] = prepared_candidates["track_name"].fillna(
            prepared_candidates["track_id"]
        )
        prepared_candidates["artist_name"] = prepared_candidates["artist_name"].fillna("")
        return prepared_candidates

    def _select_opening_track(
        self,
        remaining_candidates: pd.DataFrame,
        mood_profile: MoodProfile,
    ) -> pd.Series:
        """Select the strongest mood anchor to open the playlist."""

        scored_candidates = remaining_candidates.copy()
        scored_candidates["mood_relevance_score"] = scored_candidates.apply(
            lambda row: self._score_track_for_mood(row, mood_profile),
            axis=1,
        )
        scored_candidates = scored_candidates.sort_values(
            ["mood_relevance_score", "track_id"],
            ascending=[False, True],
            kind="stable",
        )
        return scored_candidates.iloc[0]

    def _select_next_track(
        self,
        remaining_candidates: pd.DataFrame,
        selected_track_rows: list[pd.Series],
        previous_track_row: pd.Series,
        mood_profile: MoodProfile,
    ) -> tuple[pd.Series | None, TrackScoreBreakdown | None]:
        """Choose the next track using relevance, diversity, and smoothness."""

        best_track_row: pd.Series | None = None
        best_score_breakdown: TrackScoreBreakdown | None = None

        for candidate_row in remaining_candidates.itertuples(index=False):
            candidate_series = pd.Series(candidate_row._asdict())
            score_breakdown = self._score_next_track(
                candidate_row=candidate_series,
                selected_track_rows=selected_track_rows,
                previous_track_row=previous_track_row,
                mood_profile=mood_profile,
            )
            if best_score_breakdown is None or score_breakdown.final_score > best_score_breakdown.final_score:
                best_track_row = candidate_series
                best_score_breakdown = score_breakdown

        return best_track_row, best_score_breakdown

    def _score_next_track(
        self,
        candidate_row: pd.Series,
        selected_track_rows: list[pd.Series],
        previous_track_row: pd.Series,
        mood_profile: MoodProfile,
    ) -> TrackScoreBreakdown:
        """Score one candidate as the next track in the sequence."""

        mood_relevance_score = self._score_track_for_mood(candidate_row, mood_profile)
        diversity_score = self._compute_diversity_bonus(candidate_row, selected_track_rows)
        transition_score = self._compute_transition_score(
            previous_track_row=previous_track_row,
            candidate_row=candidate_row,
            mood_profile=mood_profile,
        )
        final_score = (
            self.relevance_weight * mood_relevance_score
            + self.diversity_weight * diversity_score
            + self.transition_weight * transition_score
        )
        return TrackScoreBreakdown(
            mood_relevance_score=mood_relevance_score,
            diversity_score=diversity_score,
            transition_score=transition_score,
            final_score=final_score,
        )

    def _score_track_for_mood(self, track_row: pd.Series, mood_profile: MoodProfile) -> float:
        """Score how well one track matches the requested mood profile."""

        feature_scores: list[float] = []
        for feature_name, target_value in mood_profile.target_feature_values.items():
            tolerance = mood_profile.feature_tolerances[feature_name]
            candidate_value = float(track_row.get(feature_name, target_value))
            normalized_distance = abs(candidate_value - target_value) / max(tolerance, 1e-6)
            feature_scores.append(max(0.0, 1.0 - normalized_distance))

        if not feature_scores:
            return 0.0
        return sum(feature_scores) / len(feature_scores)

    def _compute_diversity_bonus(
        self,
        candidate_row: pd.Series,
        selected_track_rows: list[pd.Series],
    ) -> float:
        """Reward candidates that add variety without depending on tempo jumps."""

        if not selected_track_rows:
            return 0.0

        feature_distances: list[float] = []
        for selected_track_row in selected_track_rows:
            per_track_distances: list[float] = []
            for feature_name in self.diversity_features:
                candidate_value = float(candidate_row.get(feature_name, 0.0))
                selected_value = float(selected_track_row.get(feature_name, 0.0))
                per_track_distances.append(abs(candidate_value - selected_value))
            if per_track_distances:
                feature_distances.append(sum(per_track_distances) / len(per_track_distances))

        if not feature_distances:
            return 0.0

        # Diversity is capped at 1.0 so it adds variety without overwhelming
        # the more important mood-fit and transition-smoothness signals.
        average_distance = sum(feature_distances) / len(feature_distances)
        return min(average_distance, 1.0)

    def _compute_transition_score(
        self,
        previous_track_row: pd.Series,
        candidate_row: pd.Series,
        mood_profile: MoodProfile,
    ) -> float:
        """Reward smooth transitions and penalize abrupt tempo or energy jumps."""

        tempo_gap = abs(float(candidate_row["tempo"]) - float(previous_track_row["tempo"]))
        energy_gap = abs(float(candidate_row["energy"]) - float(previous_track_row["energy"]))

        # Transition smoothness explicitly penalizes sharp tempo or energy jumps
        # so adjacent tracks feel intentionally sequenced rather than random.
        tempo_score = max(
            0.0,
            1.0 - (tempo_gap / mood_profile.sequencing_tempo_tolerance),
        )
        energy_score = max(
            0.0,
            1.0 - (energy_gap / mood_profile.sequencing_energy_tolerance),
        )
        return (tempo_score + energy_score) / 2.0

    def _build_playlist_track(
        self,
        selected_track_row: pd.Series,
        mood_profile: MoodProfile,
        score_breakdown: TrackScoreBreakdown,
        previous_track_row: pd.Series | None,
        selected_track_rows: list[pd.Series],
    ) -> PlaylistTrack:
        """Build the final playlist-track object with explanation text."""

        explanation = self._build_selection_explanation(
            selected_track_row=selected_track_row,
            mood_profile=mood_profile,
            score_breakdown=score_breakdown,
            previous_track_row=previous_track_row,
            selected_track_rows=selected_track_rows,
        )
        return PlaylistTrack(
            track_id=str(selected_track_row["track_id"]),
            track_name=str(selected_track_row["track_name"]),
            artist_name=str(selected_track_row.get("artist_name", "")),
            explanation=explanation,
        )

    def _build_selection_explanation(
        self,
        selected_track_row: pd.Series,
        mood_profile: MoodProfile,
        score_breakdown: TrackScoreBreakdown,
        previous_track_row: pd.Series | None,
        selected_track_rows: list[pd.Series],
    ) -> TrackSelectionExplanation:
        """Build concise human-readable reasons for a track selection."""

        best_fit_features = self._describe_best_mood_fit_features(selected_track_row, mood_profile)
        reasons = [
            f"Matches the {mood_profile.label} mood through {best_fit_features}.",
            f"Mood relevance score: {score_breakdown.mood_relevance_score:.2f}.",
        ]
        if selected_track_rows:
            reasons.append(f"Diversity contribution: {score_breakdown.diversity_score:.2f}.")
        if previous_track_row is not None:
            previous_tempo = float(previous_track_row["tempo"])
            current_tempo = float(selected_track_row["tempo"])
            reasons.append(
                f"Transition smoothness: {score_breakdown.transition_score:.2f} "
                f"with tempo move {previous_tempo:.0f} -> {current_tempo:.0f} BPM."
            )

        return TrackSelectionExplanation(
            mood_label=mood_profile.label,
            score_breakdown=score_breakdown,
            reasons=reasons,
        )

    def _describe_best_mood_fit_features(
        self,
        selected_track_row: pd.Series,
        mood_profile: MoodProfile,
    ) -> str:
        """Describe the strongest mood-aligned features for one track."""

        feature_distances: list[tuple[float, str]] = []
        for feature_name, target_value in mood_profile.target_feature_values.items():
            candidate_value = float(selected_track_row.get(feature_name, target_value))
            feature_distances.append((abs(candidate_value - target_value), feature_name))

        best_feature_names = [name for _, name in sorted(feature_distances)[:2]]
        return " and ".join(best_feature_names)

    def _remove_selected_track(
        self,
        remaining_candidates: pd.DataFrame,
        track_id: str,
    ) -> pd.DataFrame:
        """Remove a chosen track from the remaining candidate pool."""

        return remaining_candidates.loc[remaining_candidates["track_id"] != track_id].reset_index(drop=True)

    def _get_track_row(self, candidate_tracks: pd.DataFrame, track_id: str) -> pd.Series:
        """Return one track row by identifier."""

        return candidate_tracks.loc[candidate_tracks["track_id"] == track_id].iloc[0]
