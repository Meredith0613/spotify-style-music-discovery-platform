"""Reusable demo orchestration logic for the Streamlit portfolio app."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.demo_data import (
    DemoUserProfile,
    build_demo_interactions,
    build_demo_track_catalog,
    build_demo_user_profiles,
)
from features.embedding_builder import ProjectionBuilder
from features.feature_builder import FeatureBuilder, FeatureMatrixArtifacts
from models.collaborative_recommender import CollaborativeRecommender
from models.content_recommender import ContentRecommender
from models.hybrid_recommender import HybridRecommendation, HybridRecommender
from models.interaction_matrix import InteractionMatrixBuilder
from models.playlist_generator import GeneratedPlaylist, PlaylistGenerator


@dataclass(slots=True)
class DemoRecommendationExplanation:
    """Store UI-ready recommendation explanations for one recommended track.

    Attributes:
        track_id: Recommended track identifier.
        track_name: Human-readable track name.
        artist_name: Human-readable artist name.
        summary_lines: Bullet-style explanation lines shown in the app.
        spotify_rationale_lines: Optional Spotify-driven explanation lines.
        spotify_recent_track_labels: Recent Spotify tracks that influenced the mapping.
        spotify_matched_seed_labels: Demo-catalog seeds matched from Spotify history.
        spotify_inferred_mood: Lightweight mood inferred from Spotify listening.
        spotify_taste_signals: Lightweight style or energy signals inferred from Spotify listening.
        spotify_url: Optional Spotify track URL for real-track recommendations.
        album_image_url: Optional album artwork URL for real-track recommendations.
        recommendation_source: Human-readable source label for the recommendation.
    """

    track_id: str
    track_name: str
    artist_name: str
    summary_lines: list[str]
    spotify_rationale_lines: list[str] = field(default_factory=list)
    spotify_recent_track_labels: list[str] = field(default_factory=list)
    spotify_matched_seed_labels: list[str] = field(default_factory=list)
    spotify_inferred_mood: str | None = None
    spotify_taste_signals: list[str] = field(default_factory=list)
    spotify_url: str | None = None
    album_image_url: str | None = None
    recommendation_source: str = "Demo catalog recommendations"


@dataclass(slots=True)
class TasteClusterProjection:
    """Store point data for the taste-cluster visualization.

    Attributes:
        points_frame: Projection points with labels for charting.
        message: Short note explaining whether the view is model-based or fallback.
        used_model_based_projection: Whether optional projection/clustering was available.
    """

    points_frame: pd.DataFrame
    message: str
    used_model_based_projection: bool


@dataclass(slots=True)
class DemoViewState:
    """Bundle all data the Streamlit app needs for one render cycle.

    Attributes:
        profile: Selected demo user profile.
        hybrid_weights: Effective hybrid weight configuration for the UI.
        recommendations: Ordered hybrid recommendations.
        recommendation_table: Tabular recommendation summary for display.
        explanations: UI-ready explanation cards for each recommendation.
        playlist: Mood-based playlist built from the recommendations.
        taste_clusters: Optional taste-cluster projection data.
    """

    profile: DemoUserProfile
    hybrid_weights: dict[str, float]
    recommendations: list[HybridRecommendation]
    recommendation_table: pd.DataFrame
    explanations: list[DemoRecommendationExplanation]
    playlist: GeneratedPlaylist
    taste_clusters: TasteClusterProjection | None


@dataclass(slots=True)
class DemoAppService:
    """Assemble demo data and reusable model logic for the Streamlit app.

    This service keeps business logic outside the Streamlit file. It owns the
    synthetic demo dataset, recommender assembly, explanation formatting,
    playlist generation, and optional taste-map preparation.
    """

    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    interaction_matrix_builder: InteractionMatrixBuilder = field(default_factory=InteractionMatrixBuilder)
    playlist_generator: PlaylistGenerator = field(default_factory=PlaylistGenerator)
    projection_builder: ProjectionBuilder = field(default_factory=ProjectionBuilder)
    track_catalog: pd.DataFrame = field(init=False, repr=False)
    user_profiles: dict[str, DemoUserProfile] = field(init=False, repr=False)
    interactions: pd.DataFrame = field(init=False, repr=False)
    feature_artifacts: FeatureMatrixArtifacts = field(init=False, repr=False)
    content_recommender: ContentRecommender = field(init=False, repr=False)
    collaborative_recommender: CollaborativeRecommender = field(init=False, repr=False)
    popularity_scores: dict[str, float] = field(init=False, repr=False)
    novelty_scores: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the reusable demo catalog, profiles, and trained models."""

        self.track_catalog = build_demo_track_catalog()
        self.user_profiles = build_demo_user_profiles()
        self.interactions = build_demo_interactions()
        self.feature_artifacts = self.feature_builder.create_model_ready_feature_matrix(self.track_catalog)
        self.content_recommender = ContentRecommender(
            feature_artifacts=self.feature_artifacts,
            track_catalog=self.track_catalog,
        )
        self.collaborative_recommender = CollaborativeRecommender().fit(
            self.interaction_matrix_builder.build(self.interactions)
        )
        self.popularity_scores = self._build_popularity_scores()
        self.novelty_scores = self._build_novelty_scores()

    def list_profiles(self) -> list[DemoUserProfile]:
        """Return the available demo profiles in display order."""

        return list(self.user_profiles.values())

    def list_moods(self) -> list[str]:
        """Return the supported playlist moods for UI controls."""

        return sorted(self.playlist_generator.mood_profiles.keys())

    def build_demo_view(
        self,
        user_id: str,
        exploration_level: float,
        recommendation_count: int,
        mood_label: str,
        playlist_length: int,
        include_taste_clusters: bool,
    ) -> DemoViewState:
        """Build all demo outputs for the selected UI state.

        Args:
            user_id: Selected demo user/profile identifier.
            exploration_level: Slider value from 0.0 (familiar) to 1.0 (exploratory).
            recommendation_count: Number of hybrid recommendations to return.
            mood_label: Mood used for playlist generation.
            playlist_length: Desired number of playlist tracks.
            include_taste_clusters: Whether to prepare optional taste-map data.

        Returns:
            Bundle of UI-ready demo data.
        """

        profile = self.user_profiles[user_id]
        return self.build_view_for_profile(
            profile=profile,
            exploration_level=exploration_level,
            recommendation_count=recommendation_count,
            mood_label=mood_label,
            playlist_length=playlist_length,
            include_taste_clusters=include_taste_clusters,
        )

    def build_view_for_profile(
        self,
        profile: DemoUserProfile,
        exploration_level: float,
        recommendation_count: int,
        mood_label: str,
        playlist_length: int,
        include_taste_clusters: bool,
    ) -> DemoViewState:
        """Build all demo outputs for either a static or Spotify-derived profile."""

        hybrid_weights = self._build_hybrid_weights(exploration_level)
        hybrid_recommender = self._build_hybrid_recommender(
            hybrid_weights=hybrid_weights,
            additional_profiles=[profile] if profile.user_id not in self.user_profiles else None,
        )
        candidate_track_ids = self.track_catalog["track_id"].astype(str).tolist()
        recommendations = hybrid_recommender.recommend(
            user_id=profile.user_id,
            candidate_item_ids=candidate_track_ids,
            k=recommendation_count,
        )

        recommendation_table = self._build_recommendation_table(recommendations)
        explanations = self._build_recommendation_explanations(profile, recommendations)
        playlist_candidates = self._build_playlist_candidate_frame(recommendations)
        playlist = self.playlist_generator.generate_playlist(
            candidate_tracks=playlist_candidates,
            mood_label=mood_label,
            max_items=playlist_length,
        )
        taste_clusters = self._build_taste_cluster_projection() if include_taste_clusters else None

        return DemoViewState(
            profile=profile,
            hybrid_weights=hybrid_weights,
            recommendations=recommendations,
            recommendation_table=recommendation_table,
            explanations=explanations,
            playlist=playlist,
            taste_clusters=taste_clusters,
        )

    def _build_hybrid_recommender(
        self,
        hybrid_weights: dict[str, float],
        additional_profiles: list[DemoUserProfile] | None = None,
    ) -> HybridRecommender:
        """Construct a hybrid recommender configured for the current slider state."""

        profiles = list(self.user_profiles.values())
        if additional_profiles:
            profiles.extend(additional_profiles)

        return HybridRecommender(
            collaborative_recommender=self.collaborative_recommender,
            content_recommender=self.content_recommender,
            weights=hybrid_weights,
            popularity_scores=self.popularity_scores,
            novelty_scores=self.novelty_scores,
            user_seed_track_ids_by_user={
                profile.user_id: profile.seed_track_ids
                for profile in profiles
            },
            track_catalog=self.track_catalog,
        )

    def _build_hybrid_weights(self, exploration_level: float) -> dict[str, float]:
        """Map the exploration slider into interpretable hybrid weights.

        Higher exploration values increase novelty and discovery influence while
        gently reducing the familiar collaborative/content dominance.
        """

        bounded_exploration = min(max(float(exploration_level), 0.0), 1.0)
        familiarity_level = 1.0 - bounded_exploration
        return {
            "collaborative": 0.8 + (0.6 * familiarity_level),
            "content": 0.7 + (0.5 * familiarity_level),
            "novelty": 0.2 + (1.0 * bounded_exploration),
            "popularity_prior": 0.35 * familiarity_level,
            "discovery": 0.2 + (0.9 * bounded_exploration),
        }

    def _build_recommendation_table(
        self,
        recommendations: list[HybridRecommendation],
    ) -> pd.DataFrame:
        """Convert hybrid recommendations into a display-friendly table."""

        recommendation_rows = [
            {
                "track_id": recommendation.track_id,
                "track_name": recommendation.track_name,
                "artist_name": recommendation.artist_name,
                "final_score": round(recommendation.final_score, 3),
                "collaborative": round(recommendation.score_breakdown.collaborative_score, 3),
                "content": round(recommendation.score_breakdown.content_score, 3),
                "novelty": round(recommendation.score_breakdown.novelty_score, 3),
                "discovery": round(recommendation.score_breakdown.discovery_score, 3),
                "popularity_prior": round(recommendation.score_breakdown.popularity_prior, 3),
                "cold_start_fallback": recommendation.used_cold_start_fallback,
            }
            for recommendation in recommendations
        ]
        return pd.DataFrame(recommendation_rows)

    def _build_recommendation_explanations(
        self,
        profile: DemoUserProfile,
        recommendations: list[HybridRecommendation],
    ) -> list[DemoRecommendationExplanation]:
        """Build human-readable explanation cards for recommended tracks."""

        explanations: list[DemoRecommendationExplanation] = []

        for recommendation in recommendations:
            summary_lines = [
                (
                    "Final score combines collaborative, content, novelty, discovery, "
                    f"and popularity signals: {recommendation.final_score:.3f}."
                ),
                (
                    "Score breakdown: "
                    f"collaborative={recommendation.score_breakdown.collaborative_score:.2f}, "
                    f"content={recommendation.score_breakdown.content_score:.2f}, "
                    f"novelty={recommendation.score_breakdown.novelty_score:.2f}, "
                    f"discovery={recommendation.score_breakdown.discovery_score:.2f}."
                ),
            ]

            if recommendation.used_cold_start_fallback:
                summary_lines.append(
                    "This recommendation used the cold-start fallback because the profile has limited history."
                )

            if profile.seed_track_ids:
                try:
                    content_explanation = self.content_recommender.explain_recommendation(
                        seed_track_ids=profile.seed_track_ids,
                        candidate_track_id=recommendation.track_id,
                        top_n_features=3,
                    )
                except ValueError:
                    content_explanation = None
                if content_explanation is not None and content_explanation.top_feature_contributions:
                    top_features = ", ".join(
                        contribution.feature_name
                        for contribution in content_explanation.top_feature_contributions
                    )
                    summary_lines.append(f"Closest content features to this profile: {top_features}.")

            explanations.append(
                DemoRecommendationExplanation(
                    track_id=recommendation.track_id,
                    track_name=recommendation.track_name,
                    artist_name=recommendation.artist_name,
                    summary_lines=summary_lines,
                )
            )

        return explanations

    def _build_playlist_candidate_frame(
        self,
        recommendations: list[HybridRecommendation],
    ) -> pd.DataFrame:
        """Build the playlist candidate table from recommendation outputs."""

        if not recommendations:
            return pd.DataFrame(columns=self.track_catalog.columns)

        recommended_track_ids = [recommendation.track_id for recommendation in recommendations]
        playlist_candidates = self.track_catalog.loc[
            self.track_catalog["track_id"].isin(recommended_track_ids)
        ].copy()
        return playlist_candidates.reset_index(drop=True)

    def _build_taste_cluster_projection(self) -> TasteClusterProjection:
        """Build a taste-space visualization using optional projection helpers.

        When optional clustering dependencies are unavailable, the service falls
        back to a simple danceability-energy map so the app still remains demoable.
        """

        try:
            projected_coordinates = self.projection_builder.project_tsne(
                self.feature_artifacts.feature_matrix,
                n_components=2,
                random_state=42,
            )
            cluster_labels = self.projection_builder.cluster_kmeans(
                self.feature_artifacts.feature_matrix,
                n_clusters=min(4, len(self.track_catalog)),
                random_state=42,
            )
            points_frame = self.track_catalog[["track_id", "track_name", "artist_name"]].copy()
            points_frame["projection_x"] = projected_coordinates[:, 0]
            points_frame["projection_y"] = projected_coordinates[:, 1]
            points_frame["cluster_label"] = [f"Cluster {label + 1}" for label in cluster_labels]
            return TasteClusterProjection(
                points_frame=points_frame,
                message="Model-based taste clusters from t-SNE projection and k-means clustering.",
                used_model_based_projection=True,
            )
        except Exception:
            points_frame = self.track_catalog[
                ["track_id", "track_name", "artist_name", "danceability", "energy", "artist_genres"]
            ].copy()
            points_frame["projection_x"] = points_frame["danceability"]
            points_frame["projection_y"] = points_frame["energy"]
            points_frame["cluster_label"] = points_frame["artist_genres"].apply(
                lambda value: str(value).split(",")[0].strip().title() if str(value).strip() else "Mixed"
            )
            return TasteClusterProjection(
                points_frame=points_frame,
                message=(
                    "Optional clustering libraries were unavailable, so this fallback view "
                    "uses danceability and energy as a simple taste map."
                ),
                used_model_based_projection=False,
            )

    def _build_popularity_scores(self) -> dict[str, float]:
        """Normalize track popularity into a 0-to-1 prior used by the hybrid model."""

        return {
            str(row.track_id): float(row.popularity) / 100.0
            for row in self.track_catalog.itertuples(index=False)
        }

    def _build_novelty_scores(self) -> dict[str, float]:
        """Derive novelty as inverse popularity for the demo experience."""

        return {
            track_id: 1.0 - popularity_score
            for track_id, popularity_score in self.popularity_scores.items()
        }
