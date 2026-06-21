"""Streamlit portfolio demo for the Spotify-style music discovery platform."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Adding the repository's src directory to sys.path allows
# `streamlit run src/app/streamlit_app.py` to work from the repo root
# in a fresh local VS Code session before editable installation.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.demo_service import DemoAppService, DemoRecommendationExplanation, DemoViewState
from auth.spotify_auth import SpotifyAuthManager, SpotifyOAuthError
from config.settings import ProjectSettings
from data.spotify_client import SpotifyAPIClientError
from services.spotify_recommendation_adapter import (
    SpotifyRecommendationAdapter,
    SpotifyRecommendationContext,
)
from services.spotify_candidate_service import (
    RecommendationBucket,
    SpotifyCandidateService,
    SpotifyRealRecommendationResult,
)
from services.spotify_explanation_service import SpotifyExplanationService
from services.spotify_playlist_export_service import (
    PLAYLIST_EXPORT_REQUIRED_SCOPE,
    SpotifyPlaylistExportResult,
    SpotifyPlaylistExportService,
)
from services.spotify_reranking_service import SpotifyRerankingResult, SpotifyRerankingService
from services.taste_profile_service import TasteProfileService, TasteProfileSummary
from services.user_profile_service import ListeningHistorySnapshot, UserProfileService


@dataclass(slots=True)
class DemoUIState:
    """Store Streamlit control values for one app render cycle."""

    user_id: str
    exploration_level: float
    recommendation_count: int
    mood_label: str
    ranking_focus: str
    playlist_length: int
    show_taste_clusters: bool


@dataclass(slots=True)
class MusicPersonalityMetrics:
    """Store compact display metrics for the Spotify real-track personality panel."""

    top_artist: str
    top_genre: str
    favorite_era: str
    energy_score: str
    discovery_score: str


def run_app() -> None:
    """Run the Streamlit demo application."""

    import streamlit as st

    settings = ProjectSettings.from_env()
    settings.ensure_project_directories()
    st.set_page_config(page_title="Spotify-Style Music Discovery Platform", layout="wide")
    _inject_demo_styles(st)

    demo_service = _get_demo_service(st)
    auth_manager = _get_auth_manager(st, settings)
    user_profile_service = _get_user_profile_service(st, settings)
    spotify_recommendation_adapter = _get_spotify_recommendation_adapter(st)
    spotify_candidate_service = _get_spotify_candidate_service(st, settings)
    spotify_playlist_export_service = _get_spotify_playlist_export_service(st, settings)
    spotify_explanation_service = _get_spotify_explanation_service(st)
    spotify_reranking_service = _get_spotify_reranking_service(st)
    taste_profile_service = _get_taste_profile_service(st)

    callback_in_progress = _has_auth_callback_params(st)
    callback_processed = bool(st.session_state.get("spotify_callback_processed"))
    if callback_in_progress and callback_processed:
        callback_in_progress = False

    if callback_in_progress and not callback_processed:
        _handle_spotify_auth_callback(st, settings, auth_manager)
        callback_in_progress = _has_auth_callback_params(st)
        callback_processed = bool(st.session_state.get("spotify_callback_processed"))

        # A second clear pass keeps the sidebar from getting stuck in a
        # callback state if query params linger for one more rerun.
        if callback_processed and callback_in_progress:
            _clear_auth_query_params(st)
            callback_in_progress = _has_auth_callback_params(st)

    if callback_processed:
        callback_in_progress = False

    listening_history_snapshot, history_error = _get_listening_history_snapshot(
        streamlit_module=st,
        settings=settings,
        auth_manager=auth_manager,
        user_profile_service=user_profile_service,
    )
    spotify_recommendation_context = _get_spotify_recommendation_context(
        listening_history_snapshot=listening_history_snapshot,
        demo_service=demo_service,
        spotify_recommendation_adapter=spotify_recommendation_adapter,
    )
    ui_state = _render_sidebar(
        st,
        demo_service,
        settings.default_recommendation_count,
        settings,
        auth_manager,
        listening_history_snapshot,
        history_error,
        callback_in_progress,
    )

    spotify_reranking_result: SpotifyRerankingResult | None = None
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None = None
    if spotify_recommendation_context is not None:
        access_token = st.session_state.get("spotify_history_access_token")
        if listening_history_snapshot is not None and access_token:
            try:
                spotify_real_recommendation_result = _build_real_spotify_view(
                    spotify_candidate_service=spotify_candidate_service,
                    access_token=str(access_token),
                    listening_history_snapshot=listening_history_snapshot,
                    ui_state=ui_state,
                )
            except SpotifyAPIClientError as error:
                listening_history_snapshot.warnings.append(
                    f"Real Spotify candidate generation was unavailable: {error}"
                )

        if spotify_real_recommendation_result is not None:
            view_state = spotify_real_recommendation_result.view_state
        else:
            view_state = demo_service.build_view_for_profile(
                profile=spotify_recommendation_context.profile,
                exploration_level=ui_state.exploration_level,
                recommendation_count=ui_state.recommendation_count,
                mood_label=ui_state.mood_label,
                playlist_length=ui_state.playlist_length,
                include_taste_clusters=ui_state.show_taste_clusters,
            )
        if listening_history_snapshot is not None and spotify_real_recommendation_result is None:
            spotify_reranking_result = spotify_reranking_service.rerank_recommendations(
                spotify_context=spotify_recommendation_context,
                recommendations=view_state.recommendations,
                listening_history_snapshot=listening_history_snapshot,
                demo_track_catalog=demo_service.track_catalog,
            )
            if spotify_reranking_result.applied:
                _apply_reranked_recommendations(
                    demo_service=demo_service,
                    view_state=view_state,
                    recommendations=spotify_reranking_result.recommendations,
                    mood_label=ui_state.mood_label,
                    playlist_length=ui_state.playlist_length,
                )
                view_state.recommendation_table = _augment_recommendation_table_with_reranking(
                    recommendation_table=view_state.recommendation_table,
                    reranking_result=spotify_reranking_result,
                )
            view_state.explanations = spotify_explanation_service.enrich_explanations(
                spotify_context=spotify_recommendation_context,
                explanations=view_state.explanations,
                listening_history_snapshot=listening_history_snapshot,
                demo_track_catalog=demo_service.track_catalog,
            )
            if spotify_reranking_result is not None and spotify_reranking_result.applied:
                _append_spotify_reranking_notes(
                    explanations=view_state.explanations,
                    reranking_result=spotify_reranking_result,
                )
    else:
        view_state = demo_service.build_demo_view(
            user_id=ui_state.user_id,
            exploration_level=ui_state.exploration_level,
            recommendation_count=ui_state.recommendation_count,
            mood_label=ui_state.mood_label,
            playlist_length=ui_state.playlist_length,
            include_taste_clusters=ui_state.show_taste_clusters,
        )

    _render_demo_header(
        st,
        view_state=view_state,
        ui_state=ui_state,
        spotify_recommendation_context=spotify_recommendation_context,
        spotify_reranking_result=spotify_reranking_result,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
        listening_history_snapshot=listening_history_snapshot,
        history_error=history_error,
    )

    _render_taste_profile_section(
        st,
        taste_profile_service=taste_profile_service,
        listening_history_snapshot=listening_history_snapshot,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
    )
    _render_recommendation_section(
        st,
        view_state=view_state,
        spotify_recommendation_context=spotify_recommendation_context,
        spotify_reranking_result=spotify_reranking_result,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
    )
    _render_spotify_playlist_export_section(
        st,
        auth_manager=auth_manager,
        spotify_playlist_export_service=spotify_playlist_export_service,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
        listening_history_snapshot=listening_history_snapshot,
        ui_state=ui_state,
    )
    with st.expander("Playlist sequencing details", expanded=False):
        _render_playlist_section(st, view_state)
    with st.expander("Session details", expanded=False):
        _render_recent_history_status(st, listening_history_snapshot, history_error)
        _render_profile_summary(
            st,
            view_state=view_state,
            ui_state=ui_state,
            spotify_recommendation_context=spotify_recommendation_context,
            spotify_reranking_result=spotify_reranking_result,
            spotify_real_recommendation_result=spotify_real_recommendation_result,
        )
        _render_hybrid_weights(st, view_state)
    if spotify_real_recommendation_result is not None:
        _render_spotify_candidate_debug_summary(st, spotify_real_recommendation_result)
    elif ui_state.show_taste_clusters:
        _render_taste_cluster_section(st, view_state)


def _inject_demo_styles(streamlit_module: Any) -> None:
    """Apply a lightweight visual system for cards, badges, and section spacing."""

    streamlit_module.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .demo-hero {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .demo-hero {
            margin-bottom: 1rem;
            border-left: 4px solid #1db954;
        }
        .demo-kicker {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: #0369a1;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .demo-title {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .demo-subtitle {
            color: #334155;
            font-size: 0.98rem;
            margin-bottom: 0.75rem;
        }
        .demo-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            margin: 0 0.35rem 0.35rem 0;
            border-radius: 999px;
            background: #e2e8f0;
            color: #0f172a;
            font-size: 0.76rem;
            font-weight: 600;
        }
        .demo-badge.spotify { background: #dcfce7; color: #166534; }
        .demo-badge.demo { background: #e0f2fe; color: #075985; }
        .demo-badge.mood { background: #fef3c7; color: #92400e; }
        .demo-badge.explore { background: #ede9fe; color: #6d28d9; }
        .demo-badge.rerank { background: #fee2e2; color: #991b1b; }
        .demo-badge.familiar { background: #dbeafe; color: #1e40af; }
        .demo-badge.discovery { background: #fef3c7; color: #92400e; }
        .demo-badge.mood-based { background: #fce7f3; color: #9d174d; }
        .recommendation-card {
            display: flex;
            gap: 0.75rem;
            min-height: 124px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            border: 1px solid #2d383d;
            border-radius: 8px;
            background: #172026;
            color: #f8fafc;
        }
        .recommendation-thumb, .recommendation-thumb-placeholder {
            width: 60px;
            height: 60px;
            flex: 0 0 60px;
            border-radius: 6px;
            object-fit: cover;
            background: #2d383d;
        }
        .recommendation-thumb-placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #b9c7c9;
            font-size: 0.72rem;
        }
        .recommendation-content { min-width: 0; flex: 1; }
        .recommendation-title {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 0.98rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.1rem;
        }
        .recommendation-artist {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #b9c7c9;
            font-size: 0.86rem;
            margin-bottom: 0.3rem;
        }
        .recommendation-card .demo-badge {
            margin-bottom: 0.25rem;
        }
        .recommendation-reason {
            color: #d7e0e2;
            font-size: 0.82rem;
            line-height: 1.3;
            margin-top: 0.2rem;
        }
        .recommendation-link {
            display: inline-block;
            margin-top: 0.35rem;
            color: #62d98b;
            font-size: 0.82rem;
            font-weight: 700;
            text-decoration: none;
        }
        .personality-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem 1rem;
            margin: 0.5rem 0 0.75rem;
        }
        .personality-cluster {
            grid-column: 1 / -1;
            border-left: 3px solid #1db954;
            padding: 0.4rem 0.65rem;
            background: #172b20;
        }
        .personality-item {
            min-width: 0;
            padding: 0.15rem 0;
        }
        .personality-label {
            color: #a8bac8;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .personality-value {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        @media (max-width: 760px) {
            .personality-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        .demo-section-caption {
            color: #64748b;
            margin-top: -0.35rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_demo_header(
    streamlit_module: Any,
    view_state: DemoViewState,
    ui_state: DemoUIState,
    spotify_recommendation_context: SpotifyRecommendationContext | None,
    spotify_reranking_result: SpotifyRerankingResult | None,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None,
    listening_history_snapshot: ListeningHistorySnapshot | None,
    history_error: str | None,
) -> None:
    """Render a cleaner product-style header with mode and status badges."""

    badges = [
        _build_badge_html(
            (
                "Spotify real-track recommendations"
                if spotify_real_recommendation_result is not None
                else "Spotify-driven demo catalog"
                if spotify_recommendation_context is not None
                else "Demo catalog recommendations"
            ),
            "spotify" if spotify_recommendation_context is not None else "demo",
        ),
        _build_badge_html(
            f"Mood: {ui_state.mood_label.replace('_', ' ').title()}",
            "mood",
        ),
        _build_badge_html(
            _format_exploration_badge(ui_state.exploration_level),
            "explore",
        ),
        _build_badge_html(f"Focus: {ui_state.ranking_focus}", "rerank"),
    ]
    if spotify_reranking_result is not None and spotify_reranking_result.applied:
        badges.append(_build_badge_html("Spotify-aware reranking", "rerank"))

    with streamlit_module.container():
        streamlit_module.markdown(
            """
            <div class="demo-hero">
              <div class="demo-kicker">Music Discovery Platform</div>
              <div class="demo-title">Spotify-Style Music Discovery Platform</div>
              <div class="demo-subtitle">
                Personalized music discovery powered by Spotify listening history,
                hybrid ranking, and explainable recommendations.
              </div>
              <div>{badges}</div>
            </div>
            """.replace("{badges}", "".join(badges)),
            unsafe_allow_html=True,
        )

    if spotify_real_recommendation_result is not None:
        streamlit_module.success("Spotify mode is ranking real Spotify tracks from candidate generation.")
        for warning_message in _compact_spotify_candidate_warnings(
            spotify_real_recommendation_result.candidate_set.warnings
        ):
            streamlit_module.info(warning_message)
    elif spotify_recommendation_context is not None:
        streamlit_module.success("Spotify listening is personalizing this session through the demo catalog fallback.")
    elif history_error:
        streamlit_module.warning("Spotify personalization is temporarily unavailable. The demo is still ready to explore.")
    elif listening_history_snapshot is None:
        streamlit_module.info("Demo mode is active. Connect Spotify to personalize recommendations with your recent listening.")

    if spotify_reranking_result is not None and spotify_reranking_result.applied and spotify_reranking_result.message:
        streamlit_module.caption(spotify_reranking_result.message)


def _compact_spotify_candidate_warnings(warnings: list[str]) -> list[str]:
    """Return UI-safe Spotify candidate warnings without raw API details."""

    compact_warnings: list[str] = []
    for warning_message in warnings:
        if "top tracks" in warning_message:
            message = (
                "Some recent artists could not be expanded into top tracks, "
                "so search-based candidates were used instead."
            )
        elif "audio features" in warning_message:
            message = "Spotify audio features were unavailable, so ranking is using metadata-only signals."
        else:
            message = warning_message.split("http", maxsplit=1)[0].strip()
        if message and message not in compact_warnings:
            compact_warnings.append(message)
    return compact_warnings


def _render_spotify_candidate_debug_summary(
    streamlit_module: Any,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult,
) -> None:
    """Render a compact candidate-generation summary for Spotify mode."""

    summary = getattr(spotify_real_recommendation_result.candidate_set, "debug_summary", {}) or {}
    if not summary:
        return
    streamlit_module.caption(
        "Spotify candidates: "
        f"{summary.get('unique_candidate_count', summary.get('candidate_count', summary.get('final_candidate_count', 0)))} unique | "
        f"{summary.get('duplicate_candidates_removed', 0)} duplicates removed | "
        f"{summary.get('candidate_artist_count', 0)} artists | "
        f"{summary.get('top_track_candidate_count', summary.get('top_track_candidates_found', 0))} top-track | "
        f"{summary.get('search_candidate_count', summary.get('search_candidates_found', 0))} search | "
        f"{summary.get('skipped_artist_expansion_count', summary.get('top_track_requests_failed', 0))} skipped artist expansions | "
        f"ranking mode: {str(summary.get('ranking_mode', 'metadata-only')).replace('-', ' ')}"
    )
    if not hasattr(streamlit_module, "expander"):
        return
    with streamlit_module.expander("Spotify ranking debug", expanded=False):
        streamlit_module.write(f"Selected mood: {summary.get('selected_mood', 'unknown')}")
        streamlit_module.write(f"Exploration level: {summary.get('exploration_level', 'unknown')}")
        streamlit_module.write(f"Recommendation count: {summary.get('recommendation_count', 'unknown')}")
        streamlit_module.write(f"Ranking focus: {summary.get('ranking_focus', 'Balanced')}")
        streamlit_module.write(f"Diversity reranking active: {summary.get('diversity_reranking_active', False)}")
        streamlit_module.write(f"ALS signal used: {summary.get('als_score_available', False)}")
        streamlit_module.write(f"Embedding signal used: {summary.get('embedding_score_available', False)}")
        streamlit_module.write(f"Max candidates from one artist: {summary.get('max_candidates_from_single_artist', 0)}")
        streamlit_module.write(f"Search queries used: {summary.get('search_queries_used', 0)}")
        source_breakdown = summary.get("candidate_source_breakdown", {}) or {}
        if source_breakdown:
            streamlit_module.write(f"Source breakdown: {source_breakdown}")
        before_ids = [str(track_id) for track_id in summary.get("top_candidate_ids_before_reranking", [])]
        after_ids = [str(track_id) for track_id in summary.get("top_candidate_ids_after_reranking", [])]
        streamlit_module.write("Top before controls: " + ", ".join(before_ids))
        streamlit_module.write("Top after controls: " + ", ".join(after_ids))
        streamlit_module.write(f"Positions changed: {summary.get('positions_changed_after_reranking', 0)}")


def _get_demo_service(streamlit_module: Any) -> DemoAppService:
    """Return a cached demo service so reruns stay fast in Streamlit."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> DemoAppService:
        """Build the demo service once per Streamlit session."""

        return DemoAppService()

    return load_service()


def _get_auth_manager(
    streamlit_module: Any,
    settings: ProjectSettings,
) -> SpotifyAuthManager:
    """Return a cached Spotify auth manager for Streamlit reruns."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_manager() -> SpotifyAuthManager:
        """Build the auth manager once per Streamlit session."""

        return SpotifyAuthManager.from_settings(settings)

    return load_manager()


def _get_user_profile_service(
    streamlit_module: Any,
    settings: ProjectSettings,
) -> UserProfileService:
    """Return a cached user-profile service for recent-history retrieval."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> UserProfileService:
        """Build the user-profile service once per Streamlit session."""

        return UserProfileService.from_settings(settings)

    return load_service()


def _get_spotify_recommendation_adapter(
    streamlit_module: Any,
) -> SpotifyRecommendationAdapter:
    """Return a cached adapter that maps Spotify history onto demo inputs."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_adapter() -> SpotifyRecommendationAdapter:
        """Build the Spotify recommendation adapter once per Streamlit session."""

        return SpotifyRecommendationAdapter()

    return load_adapter()


def _get_spotify_candidate_service(
    streamlit_module: Any,
    settings: ProjectSettings,
) -> SpotifyCandidateService:
    """Return a cached service that generates real Spotify candidates."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> SpotifyCandidateService:
        """Build the Spotify candidate service once per Streamlit session."""

        return SpotifyCandidateService.from_settings(settings)

    return load_service()


def _get_spotify_playlist_export_service(
    streamlit_module: Any,
    settings: ProjectSettings,
) -> SpotifyPlaylistExportService:
    """Return a cached service that creates Spotify playlists."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> SpotifyPlaylistExportService:
        """Build the playlist export service once per Streamlit session."""

        return SpotifyPlaylistExportService.from_settings(settings)

    return load_service()


def _get_taste_profile_service(streamlit_module: Any) -> TasteProfileService:
    """Return a cached service for taste profile visualization."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> TasteProfileService:
        """Build the taste profile service once per Streamlit session."""

        return TasteProfileService()

    return load_service()


def _build_real_spotify_view(
    spotify_candidate_service: SpotifyCandidateService,
    access_token: str,
    listening_history_snapshot: ListeningHistorySnapshot,
    ui_state: DemoUIState,
) -> SpotifyRealRecommendationResult | None:
    """Build a real Spotify view while tolerating older cached service objects."""

    try:
        return spotify_candidate_service.build_real_spotify_view(
            access_token=access_token,
            listening_history_snapshot=listening_history_snapshot,
            exploration_level=ui_state.exploration_level,
            recommendation_count=ui_state.recommendation_count,
            mood_label=ui_state.mood_label,
            ranking_focus=ui_state.ranking_focus,
            playlist_length=ui_state.playlist_length,
        )
    except TypeError as error:
        if "ranking_focus" not in str(error):
            raise
        return spotify_candidate_service.build_real_spotify_view(
            access_token=access_token,
            listening_history_snapshot=listening_history_snapshot,
            exploration_level=ui_state.exploration_level,
            recommendation_count=ui_state.recommendation_count,
            mood_label=ui_state.mood_label,
            playlist_length=ui_state.playlist_length,
        )


def _get_spotify_explanation_service(
    streamlit_module: Any,
) -> SpotifyExplanationService:
    """Return a cached service that adds Spotify rationale to explanations."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> SpotifyExplanationService:
        """Build the Spotify explanation service once per Streamlit session."""

        return SpotifyExplanationService()

    return load_service()


def _get_spotify_reranking_service(
    streamlit_module: Any,
) -> SpotifyRerankingService:
    """Return a cached service that applies lightweight Spotify-aware reranking."""

    @streamlit_module.cache_resource(show_spinner=False)
    def load_service() -> SpotifyRerankingService:
        """Build the Spotify reranking service once per Streamlit session."""

        return SpotifyRerankingService()

    return load_service()


def _apply_reranked_recommendations(
    demo_service: DemoAppService,
    view_state: DemoViewState,
    recommendations: list[Any],
    mood_label: str,
    playlist_length: int,
) -> None:
    """Refresh dependent view-state outputs after Spotify-aware reranking."""

    view_state.recommendations = recommendations
    view_state.recommendation_table = demo_service._build_recommendation_table(recommendations)
    view_state.explanations = demo_service._build_recommendation_explanations(
        view_state.profile,
        recommendations,
    )
    playlist_candidates = demo_service._build_playlist_candidate_frame(recommendations)
    view_state.playlist = demo_service.playlist_generator.generate_playlist(
        candidate_tracks=playlist_candidates,
        mood_label=mood_label,
        max_items=playlist_length,
    )


def _augment_recommendation_table_with_reranking(
    recommendation_table: pd.DataFrame,
    reranking_result: SpotifyRerankingResult,
) -> pd.DataFrame:
    """Add original-score and reranking-adjustment columns for UI transparency."""

    if recommendation_table.empty:
        return recommendation_table

    augmented_table = recommendation_table.copy()
    augmented_table["model_final_score"] = augmented_table["track_id"].map(
        reranking_result.original_scores_by_track_id
    ).round(3)
    augmented_table["spotify_rerank_adjustment"] = augmented_table["track_id"].map(
        reranking_result.score_adjustments_by_track_id
    ).fillna(0.0).round(3)
    return augmented_table


def _append_spotify_reranking_notes(
    explanations: list[DemoRecommendationExplanation],
    reranking_result: SpotifyRerankingResult,
) -> None:
    """Append one short reranking note so the adjustment remains explainable."""

    for explanation in explanations:
        score_adjustment = reranking_result.score_adjustments_by_track_id.get(explanation.track_id, 0.0)
        if abs(score_adjustment) <= 1e-9:
            continue
        reason_labels = reranking_result.reason_labels_by_track_id.get(explanation.track_id, [])
        if reason_labels:
            explanation.spotify_rationale_lines.append(
                "Spotify-aware reranking applied "
                f"({score_adjustment:+.3f}) for {', '.join(reason_labels[:2])}."
            )
            continue
        explanation.spotify_rationale_lines.append(
            f"Spotify-aware reranking applied ({score_adjustment:+.3f})."
        )


def _render_sidebar(
    streamlit_module: Any,
    demo_service: DemoAppService,
    default_recommendation_count: int,
    settings: ProjectSettings,
    auth_manager: SpotifyAuthManager,
    listening_history_snapshot: ListeningHistorySnapshot | None,
    history_error: str | None,
    callback_in_progress: bool,
) -> DemoUIState:
    """Render sidebar controls and return the selected UI state."""

    sidebar = streamlit_module.sidebar
    _render_auth_sidebar_section(
        streamlit_module=streamlit_module,
        settings=settings,
        auth_manager=auth_manager,
        listening_history_snapshot=listening_history_snapshot,
        history_error=history_error,
        callback_in_progress=callback_in_progress,
    )
    sidebar.header("Demo Controls")

    profiles = demo_service.list_profiles()
    profile_labels = {profile.display_name: profile.user_id for profile in profiles}
    selected_profile_label = sidebar.selectbox(
        "User or preference profile",
        options=list(profile_labels.keys()),
        index=0,
    )
    selected_user_id = profile_labels[selected_profile_label]
    selected_profile = demo_service.user_profiles[selected_user_id]

    # The exploration slider intentionally maps to hybrid weights rather than
    # only changing one novelty term, so users can see a real tradeoff.
    exploration_level = float(
        sidebar.slider(
            "Exploration vs familiarity",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Move right to increase novelty and discovery weighting.",
        )
    )
    recommendation_count = int(
        sidebar.slider(
            "Number of recommendations",
            min_value=3,
            max_value=8,
            value=min(max(default_recommendation_count, 5), 8),
            step=1,
        )
    )
    mood_label = sidebar.selectbox(
        "Playlist mood",
        options=demo_service.list_moods(),
        index=demo_service.list_moods().index(selected_profile.preferred_mood),
    )
    ranking_focus = sidebar.selectbox(
        "Ranking focus",
        options=["Balanced", "Familiar", "Discovery", "Mood-first"],
        index=0,
        help="Adjusts how strongly Spotify real-track ranking favors familiarity, discovery, or mood fit.",
    )
    playlist_length = int(
        sidebar.slider(
            "Playlist length",
            min_value=3,
            max_value=6,
            value=4,
            step=1,
        )
    )
    show_taste_clusters = bool(
        sidebar.checkbox(
            "Show taste clusters",
            value=True,
            help="Displays a taste map when projection data is available.",
        )
    )

    return DemoUIState(
        user_id=selected_user_id,
        exploration_level=exploration_level,
        recommendation_count=recommendation_count,
        mood_label=mood_label,
        ranking_focus=ranking_focus,
        playlist_length=playlist_length,
        show_taste_clusters=show_taste_clusters,
    )


def _render_auth_sidebar_section(
    streamlit_module: Any,
    settings: ProjectSettings,
    auth_manager: SpotifyAuthManager,
    listening_history_snapshot: ListeningHistorySnapshot | None,
    history_error: str | None,
    callback_in_progress: bool,
) -> None:
    """Render the narrow Spotify auth controls without changing the demo flow."""

    sidebar = streamlit_module.sidebar
    sidebar.header("Spotify")

    if not settings.spotify_oauth_available():
        sidebar.caption("Spotify personalization: unavailable")
        sidebar.caption(
            "Add `SPOTIFY_CLIENT_ID` and `SPOTIFY_REDIRECT_URI` to enable Spotify login."
        )
        return

    token = auth_manager.get_token(streamlit_module.session_state)
    if token is None:
        sidebar.caption("Connect recent listening to personalize real Spotify recommendations.")
        callback_processed = bool(streamlit_module.session_state.get("spotify_callback_processed"))

        if callback_processed and not _has_auth_callback_params(streamlit_module):
            streamlit_module.session_state.pop("spotify_callback_processed", None)
            streamlit_module.session_state.pop("spotify_login_url", None)
            callback_processed = False

        login_url = str(
            streamlit_module.session_state.get("spotify_login_url")
            or streamlit_module.session_state.get("spotify_auth_login_url", "")
        ).strip()
        if not login_url and not callback_in_progress:
            # Keep the PKCE verifier in session state before the one-click redirect.
            login_url = auth_manager.get_authorization_url(streamlit_module.session_state)
            streamlit_module.session_state["spotify_login_url"] = login_url
        if callback_in_progress and not callback_processed:
            sidebar.caption("Finishing Spotify login...")
        elif login_url:
            sidebar.link_button("Connect Spotify", login_url, use_container_width=True)
        return

    connected_name = "Spotify listener"
    if listening_history_snapshot is not None:
        connected_name = listening_history_snapshot.display_name or connected_name
        recent_artist_count = len(
            {
                artist_name.strip()
                for track in listening_history_snapshot.recent_tracks
                for artist_name in str(track.artist_name).split(",")
                if artist_name.strip()
            }
        )
        sidebar.success(f"Connected as {connected_name}")
        sidebar.caption(
            f"Recent artists: {recent_artist_count} | Recent tracks: {listening_history_snapshot.recent_track_count}"
        )
        if sidebar.button("Refresh Listening History", use_container_width=True):
            _clear_listening_history_cache(streamlit_module.session_state)
            streamlit_module.rerun()
    elif history_error:
        sidebar.success("Spotify connected")
        sidebar.warning("Connected, but recent listening could not be loaded.")
    else:
        sidebar.success("Spotify connected")
        sidebar.caption("Loading recent listening...")

    if sidebar.button("Log out of Spotify", use_container_width=True):
        auth_manager.clear_token(streamlit_module.session_state)
        streamlit_module.session_state.pop("spotify_login_url", None)
        streamlit_module.session_state.pop("spotify_callback_processed", None)
        streamlit_module.session_state.pop("spotify_playlist_export_result", None)
        _clear_listening_history_cache(streamlit_module.session_state)
        streamlit_module.rerun()


def _handle_spotify_auth_callback(
    streamlit_module: Any,
    settings: ProjectSettings,
    auth_manager: SpotifyAuthManager,
) -> bool:
    """Consume Spotify OAuth callback parameters when the app reruns after login."""

    if not settings.spotify_oauth_available():
        _clear_auth_query_params(streamlit_module)
        return False

    raw_query_params = _read_query_params(streamlit_module)
    callback = auth_manager.parse_callback_parameters(raw_query_params)

    if bool(streamlit_module.session_state.get("spotify_callback_processed")):
        return False

    if callback.error:
        streamlit_module.session_state["spotify_callback_processed"] = True
        streamlit_module.error(
            f"Spotify login could not be completed: {callback.error_description or callback.error}."
        )
        _clear_auth_query_params(streamlit_module)
        return False

    if not callback.has_authorization_code:
        return False

    try:
        auth_manager.complete_authorization(
            code=callback.code,
            state=callback.state,
            session_state=streamlit_module.session_state,
        )
        streamlit_module.session_state.pop("spotify_login_url", None)
        _clear_listening_history_cache(streamlit_module.session_state)
        _clear_auth_query_params(streamlit_module)
        streamlit_module.session_state["spotify_callback_processed"] = True
        streamlit_module.success("Spotify is connected.")
        return True
    except SpotifyOAuthError as error:
        streamlit_module.session_state["spotify_callback_processed"] = True
        streamlit_module.session_state.pop("spotify_login_url", None)
        _clear_auth_query_params(streamlit_module)
        streamlit_module.error(f"Spotify login failed: {error}")
        return False


def _get_listening_history_snapshot(
    streamlit_module: Any,
    settings: ProjectSettings,
    auth_manager: SpotifyAuthManager,
    user_profile_service: UserProfileService,
) -> tuple[ListeningHistorySnapshot | None, str | None]:
    """Return a cached recent-history snapshot for the logged-in Spotify user."""

    if not settings.spotify_oauth_available():
        return None, None

    try:
        token = auth_manager.ensure_valid_token(streamlit_module.session_state)
    except SpotifyOAuthError as error:
        auth_manager.clear_token(streamlit_module.session_state)
        _clear_listening_history_cache(streamlit_module.session_state)
        return None, str(error)

    if token is None:
        _clear_listening_history_cache(streamlit_module.session_state)
        return None, None

    cached_token = streamlit_module.session_state.get("spotify_history_access_token")
    cached_snapshot = streamlit_module.session_state.get("spotify_history_snapshot")
    if cached_token == token.access_token and isinstance(cached_snapshot, ListeningHistorySnapshot):
        return cached_snapshot, None

    try:
        snapshot = user_profile_service.build_listening_history(token.access_token)
    except SpotifyAPIClientError as error:
        _clear_listening_history_cache(streamlit_module.session_state)
        return None, str(error)

    streamlit_module.session_state["spotify_history_access_token"] = token.access_token
    streamlit_module.session_state["spotify_history_snapshot"] = snapshot
    return snapshot, None


def _render_recent_history_status(
    streamlit_module: Any,
    listening_history_snapshot: ListeningHistorySnapshot | None,
    history_error: str | None,
) -> None:
    """Render a compact status block for authenticated recent listening history."""

    if history_error:
        streamlit_module.warning(
            "Recent Spotify listening could not be loaded. You can keep exploring the demo with the built-in profiles."
        )
        return

    if listening_history_snapshot is None:
        return

    streamlit_module.subheader("Spotify Recent History")
    streamlit_module.caption("A quick view of the listening activity currently shaping personalization.")
    for warning_message in listening_history_snapshot.warnings:
        streamlit_module.info(warning_message)

    if not listening_history_snapshot.recent_tracks:
        streamlit_module.info(
            "Spotify is connected, but there was not enough recent listening history to personalize this session yet."
        )
        return

    with streamlit_module.container(border=True):
        streamlit_module.caption(
            f"{listening_history_snapshot.recent_track_count} recent plays loaded for "
            f"{listening_history_snapshot.display_name}."
        )
        recent_track_labels = [
            f"{summary.track_name} - {summary.artist_name}"
            for summary in listening_history_snapshot.recent_tracks[:5]
        ]
        for recent_track_label in recent_track_labels:
            streamlit_module.write(f"- {recent_track_label}")


def _read_query_params(streamlit_module: Any) -> dict[str, Any]:
    """Read Streamlit query parameters in a version-tolerant way."""

    if hasattr(streamlit_module, "query_params"):
        return dict(streamlit_module.query_params)
    return dict(streamlit_module.experimental_get_query_params())  # pragma: no cover


def _has_auth_callback_params(streamlit_module: Any) -> bool:
    """Return whether the current rerun is handling a Spotify OAuth callback."""

    query_params = _read_query_params(streamlit_module)
    return "code" in query_params and "state" in query_params


def _clear_auth_query_params(streamlit_module: Any) -> None:
    """Force clear OAuth query params reliably."""

    if hasattr(streamlit_module, "query_params"):
        try:
            streamlit_module.experimental_set_query_params()
            return
        except Exception:
            streamlit_module.query_params.clear()
            return

    try:
        streamlit_module.experimental_set_query_params()
    except Exception:
        pass


def _clear_listening_history_cache(session_state: Any) -> None:
    """Drop cached recent-history data when auth state changes."""

    session_state.pop("spotify_history_access_token", None)
    session_state.pop("spotify_history_snapshot", None)


def _get_spotify_recommendation_context(
    listening_history_snapshot: ListeningHistorySnapshot | None,
    demo_service: DemoAppService,
    spotify_recommendation_adapter: SpotifyRecommendationAdapter,
) -> SpotifyRecommendationContext | None:
    """Return a Spotify-driven profile when recent listening history is available."""

    if listening_history_snapshot is None or listening_history_snapshot.recent_track_count == 0:
        return None
    return spotify_recommendation_adapter.build_context(
        listening_history_snapshot=listening_history_snapshot,
        demo_track_catalog=demo_service.track_catalog,
    )


def _get_cached_listening_history_snapshot(session_state: Any) -> ListeningHistorySnapshot | None:
    """Return the cached recent-history snapshot already stored in session state."""

    cached_snapshot = session_state.get("spotify_history_snapshot")
    if isinstance(cached_snapshot, ListeningHistorySnapshot):
        return cached_snapshot
    return None


def _render_profile_summary(
    streamlit_module: Any,
    view_state: DemoViewState,
    ui_state: DemoUIState,
    spotify_recommendation_context: SpotifyRecommendationContext | None,
    spotify_reranking_result: SpotifyRerankingResult | None,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None,
) -> None:
    """Render the selected profile summary and seed history."""

    streamlit_module.subheader("Profile")
    streamlit_module.caption("The current recommendation session is grounded in this taste profile.")
    with streamlit_module.container(border=True):
        badge_html = [
            _build_badge_html(
                (
                    "Spotify real-track recommendations"
                    if spotify_real_recommendation_result is not None
                    else "Spotify-driven demo catalog"
                    if spotify_recommendation_context is not None
                    else "Demo catalog recommendations"
                ),
                "spotify" if spotify_recommendation_context is not None else "demo",
            ),
            _build_badge_html(f"Mood: {ui_state.mood_label.replace('_', ' ').title()}", "mood"),
            _build_badge_html(_format_exploration_badge(ui_state.exploration_level), "explore"),
        ]
        if spotify_reranking_result is not None and spotify_reranking_result.applied:
            badge_html.append(_build_badge_html("Spotify-aware reranking", "rerank"))
        streamlit_module.markdown("".join(badge_html), unsafe_allow_html=True)
        streamlit_module.write(view_state.profile.summary)
        if view_state.profile.seed_track_ids:
            streamlit_module.caption(
                "Seed tracks: " + ", ".join(view_state.profile.seed_track_ids)
            )
        else:
            streamlit_module.caption(
                "This profile has limited history, so the hybrid recommender is leaning on its cold-start fallback."
            )


def _render_hybrid_weights(streamlit_module: Any, view_state: DemoViewState) -> None:
    """Render the active hybrid weight configuration as metric cards."""

    streamlit_module.subheader("Hybrid Weighting")
    weight_columns = streamlit_module.columns(5)
    for column, (weight_name, weight_value) in zip(weight_columns, view_state.hybrid_weights.items()):
        column.metric(weight_name.replace("_", " ").title(), f"{weight_value:.2f}")


def _render_taste_profile_section(
    streamlit_module: Any,
    *,
    taste_profile_service: TasteProfileService,
    listening_history_snapshot: ListeningHistorySnapshot | None,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None,
) -> None:
    """Render the Spotify real-track taste profile section."""

    if listening_history_snapshot is None or spotify_real_recommendation_result is None:
        return

    summary = taste_profile_service.build_taste_profile(
        listening_history_snapshot=listening_history_snapshot,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
    )
    personality_metrics = _build_music_personality_metrics(
        summary=summary,
        listening_history_snapshot=listening_history_snapshot,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
    )
    _render_taste_profile_summary(streamlit_module, summary, personality_metrics)


def _render_taste_profile_summary(
    streamlit_module: Any,
    summary: TasteProfileSummary,
    personality_metrics: MusicPersonalityMetrics | None = None,
) -> None:
    """Render a taste profile summary in a Streamlit-compatible way."""

    metrics = personality_metrics or MusicPersonalityMetrics(
        top_artist=summary.top_artists[0] if summary.top_artists else "No data yet",
        top_genre=summary.top_genres[0] if summary.top_genres else "Spotify genre unavailable",
        favorite_era="No year data yet",
        energy_score="Unavailable",
        discovery_score="Unavailable",
    )
    streamlit_module.subheader("Your Music Personality")
    if summary.warning:
        streamlit_module.info(
            "Keep listening to build a more detailed taste map. "
            "The available listening signals are still shown below."
        )

    streamlit_module.markdown(
        """
        <div class="personality-grid">
          <div class="personality-cluster">
            <div class="personality-label">Taste Cluster</div>
            <div class="personality-value">{cluster_label}</div>
          </div>
          <div class="personality-item">
            <div class="personality-label">Top Artist</div>
            <div class="personality-value">{top_artist}</div>
          </div>
          <div class="personality-item">
            <div class="personality-label">Top Genre</div>
            <div class="personality-value">{top_genre}</div>
          </div>
          <div class="personality-item">
            <div class="personality-label">Favorite Era</div>
            <div class="personality-value">{favorite_era}</div>
          </div>
          <div class="personality-item">
            <div class="personality-label">Energy Score</div>
            <div class="personality-value">{energy_score}</div>
          </div>
          <div class="personality-item">
            <div class="personality-label">Discovery Score</div>
            <div class="personality-value">{discovery_score}</div>
          </div>
        </div>
        """.format(
            cluster_label=escape(summary.cluster_label),
            top_artist=escape(metrics.top_artist),
            top_genre=escape(metrics.top_genre),
            favorite_era=escape(metrics.favorite_era),
            energy_score=escape(metrics.energy_score),
            discovery_score=escape(metrics.discovery_score),
        ),
        unsafe_allow_html=True,
    )

    streamlit_module.caption(summary.explanation)
    if not summary.plot_points:
        return

    plot_frame = pd.DataFrame(
        [
            {
                "x": point.x,
                "y": point.y,
                "cluster": f"Cluster {point.cluster_id}",
                "track": point.track_name,
                "artist": point.artist_name,
                "source": "Recent listening" if point.is_recent else "Candidate",
                "point_size": 80 if point.is_recent else 35,
            }
            for point in summary.plot_points
        ]
    )
    if hasattr(streamlit_module, "scatter_chart"):
        try:
            streamlit_module.scatter_chart(
                plot_frame,
                x="x",
                y="y",
                color="cluster",
                size="point_size",
                use_container_width=True,
            )
            return
        except TypeError:
            streamlit_module.scatter_chart(
                plot_frame,
                x="x",
                y="y",
                color="cluster",
                use_container_width=True,
            )
            return
    streamlit_module.dataframe(plot_frame, use_container_width=True)


def _build_music_personality_metrics(
    *,
    summary: TasteProfileSummary,
    listening_history_snapshot: ListeningHistorySnapshot,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult,
) -> MusicPersonalityMetrics:
    """Derive compact personality metrics from already loaded Spotify session data."""

    frames = [
        listening_history_snapshot.track_level_frame,
        spotify_real_recommendation_result.candidate_set.track_catalog,
    ]
    profile_frame = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False)
    top_artist = summary.top_artists[0] if summary.top_artists else "No data yet"
    top_genre = summary.top_genres[0] if summary.top_genres else "Spotify genre unavailable"
    favorite_era = _derive_favorite_era(profile_frame)
    energy_score = _format_average_score(profile_frame, "energy")
    discovery_score = _derive_discovery_score(profile_frame)
    return MusicPersonalityMetrics(
        top_artist=top_artist,
        top_genre=top_genre,
        favorite_era=favorite_era,
        energy_score=energy_score,
        discovery_score=discovery_score,
    )


def _derive_favorite_era(profile_frame: pd.DataFrame) -> str:
    """Return the most common release decade when release metadata is present."""

    if profile_frame.empty:
        return "No year data yet"
    for column_name in ["album_release_date", "release_date", "release_year"]:
        if column_name not in profile_frame.columns:
            continue
        years = pd.to_numeric(
            profile_frame[column_name].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        ).dropna()
        if not years.empty:
            decades = ((years.astype(int) // 10) * 10).astype(str) + "s"
            return str(decades.value_counts().index[0])
    return "No year data yet"


def _format_average_score(profile_frame: pd.DataFrame, column_name: str) -> str:
    """Format a normalized feature average as a compact 0-100 score."""

    if profile_frame.empty or column_name not in profile_frame.columns:
        return "Unavailable"
    values = pd.to_numeric(profile_frame[column_name], errors="coerce").dropna()
    if values.empty:
        return "Unavailable"
    return f"{int(round(float(values.mean()) * 100))}/100"


def _derive_discovery_score(profile_frame: pd.DataFrame) -> str:
    """Estimate discovery from existing novelty or candidate-source signals."""

    if profile_frame.empty:
        return "Unavailable"
    if "catalog_novelty" in profile_frame.columns:
        return _format_average_score(profile_frame, "catalog_novelty")
    if "candidate_sources" in profile_frame.columns:
        source_text = profile_frame["candidate_sources"].fillna("").astype(str).str.lower()
        if len(source_text):
            search_share = float(source_text.str.contains("search").mean())
            return f"{int(round(search_share * 100))}/100"
    return "Unavailable"


def _render_recommendation_section(
    streamlit_module: Any,
    view_state: DemoViewState,
    spotify_recommendation_context: SpotifyRecommendationContext | None,
    spotify_reranking_result: SpotifyRerankingResult | None,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None,
) -> None:
    """Render the recommendation table and explanation controls."""

    streamlit_module.subheader("Recommendations")
    streamlit_module.caption("A compact product-style view of the top-ranked tracks, with full explainability below.")
    if view_state.recommendation_table.empty:
        streamlit_module.info("No recommendations were available for the current setup.")
        return

    recommendation_buckets = (
        _get_spotify_recommendation_buckets(spotify_real_recommendation_result)
        if spotify_real_recommendation_result is not None
        else []
    )
    if recommendation_buckets:
        _render_spotify_bucket_sections(streamlit_module, spotify_real_recommendation_result)
    else:
        _render_recommendation_cards(
            streamlit_module=streamlit_module,
            view_state=view_state,
            spotify_recommendation_context=spotify_recommendation_context,
            spotify_reranking_result=spotify_reranking_result,
            spotify_real_recommendation_result=spotify_real_recommendation_result,
        )
    with streamlit_module.expander("Score details", expanded=False):
        streamlit_module.dataframe(view_state.recommendation_table, use_container_width=True)
    _render_recommendation_explanations(streamlit_module, view_state.explanations)


def _render_spotify_bucket_sections(
    streamlit_module: Any,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult,
) -> None:
    """Render familiar, discovery, and mood-based Spotify recommendation buckets."""

    recommendation_buckets = _get_spotify_recommendation_buckets(spotify_real_recommendation_result)
    if not recommendation_buckets:
        return
    streamlit_module.caption(
        "Spotify mode groups recommendations into Familiar, Discovery, and Mood-Based picks. "
        "The recommendation count controls cards per bucket."
    )
    for bucket in recommendation_buckets:
        bucket_label = bucket.bucket_label
        explanations = bucket.recommendations
        if not explanations:
            continue
        streamlit_module.markdown(f"**{bucket_label}**")
        if bucket.description:
            streamlit_module.caption(bucket.description)
        bucket_columns = streamlit_module.columns(2)
        for index, explanation in enumerate(explanations):
            card_column = bucket_columns[index % len(bucket_columns)]
            with card_column:
                streamlit_module.markdown(
                    _build_recommendation_card_html(
                        explanation=explanation,
                        bucket_label=bucket_label,
                        mood_label=explanation.spotify_inferred_mood or "calm",
                    ),
                    unsafe_allow_html=True,
                )


def _get_spotify_recommendation_buckets(
    spotify_real_recommendation_result: SpotifyRealRecommendationResult,
) -> list[RecommendationBucket]:
    """Return new bucket objects, falling back to legacy bucket dictionaries."""

    recommendation_buckets = getattr(spotify_real_recommendation_result, "recommendation_buckets", []) or []
    if recommendation_buckets:
        return recommendation_buckets

    bucketed_explanations = getattr(spotify_real_recommendation_result, "bucketed_explanations", {}) or {}
    return [
        RecommendationBucket(
            bucket_name=bucket_label.lower().replace(" ", "_"),
            bucket_label=bucket_label,
            description="",
            recommendations=explanations,
        )
        for bucket_label, explanations in bucketed_explanations.items()
    ]


def _render_recommendation_cards(
    streamlit_module: Any,
    view_state: DemoViewState,
    spotify_recommendation_context: SpotifyRecommendationContext | None,
    spotify_reranking_result: SpotifyRerankingResult | None,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None,
) -> None:
    """Render top recommendations as compact cards."""

    explanation_by_track_id = {
        explanation.track_id: explanation
        for explanation in view_state.explanations
    }
    card_columns = streamlit_module.columns(2)
    for index, recommendation in enumerate(view_state.recommendations):
        card_column = card_columns[index % len(card_columns)]
        explanation = explanation_by_track_id.get(recommendation.track_id)
        rationale = _build_recommendation_card_rationale(explanation)
        badges = []
        if spotify_real_recommendation_result is not None:
            badges.append(_build_badge_html("Spotify real track", "spotify"))
        elif spotify_recommendation_context is not None:
            badges.append(_build_badge_html("Demo catalog", "spotify"))
        if recommendation.used_cold_start_fallback:
            badges.append(_build_badge_html("Demo fallback", "demo"))
        if (
            spotify_reranking_result is not None
            and spotify_reranking_result.applied
            and abs(spotify_reranking_result.score_adjustments_by_track_id.get(recommendation.track_id, 0.0)) > 1e-9
        ):
            badges.append(_build_badge_html("Spotify-aware reranking", "rerank"))

        score_line = f"Score {recommendation.final_score:.3f}"
        if (
            spotify_reranking_result is not None
            and spotify_reranking_result.applied
            and recommendation.track_id in spotify_reranking_result.original_scores_by_track_id
        ):
            score_line = (
                f"Score {recommendation.final_score:.3f} "
                f"(model {spotify_reranking_result.original_scores_by_track_id[recommendation.track_id]:.3f})"
            )

        with card_column:
            streamlit_module.markdown(
                _build_recommendation_card_html(
                    explanation=explanation,
                    bucket_label="Recommended",
                    mood_label=view_state.playlist.mood,
                    rank=index + 1,
                    fallback_track_name=recommendation.track_name,
                    fallback_artist_name=recommendation.artist_name,
                    extra_badges="".join(badges),
                    rationale=rationale,
                ),
                unsafe_allow_html=True,
            )


def _render_recommendation_explanations(
    streamlit_module: Any,
    explanations: list[DemoRecommendationExplanation],
) -> None:
    """Render explanation details for one selected recommendation."""

    if not explanations:
        return

    explanation_lookup = {
        f"{explanation.track_name} - {explanation.artist_name}": explanation
        for explanation in explanations
    }
    selected_label = streamlit_module.selectbox(
        "Explain a recommendation",
        options=list(explanation_lookup.keys()),
    )
    selected_explanation = explanation_lookup[selected_label]

    # The explanation view surfaces both the blended hybrid score and the most
    # important content rationale so the demo stays easy to narrate live.
    with streamlit_module.expander("Why this track was recommended", expanded=True):
        if _has_spotify_explanation_context(selected_explanation):
            _render_spotify_recommendation_chain(streamlit_module, selected_explanation)
            if selected_explanation.spotify_inferred_mood:
                streamlit_module.caption(
                    "Inferred listening mood: "
                    + selected_explanation.spotify_inferred_mood.replace("_", " ").title()
                )
            if selected_explanation.spotify_taste_signals:
                streamlit_module.caption(
                    "Taste signals: " + ", ".join(selected_explanation.spotify_taste_signals)
                )
            streamlit_module.markdown("**Model rationale**")
        for summary_line in selected_explanation.summary_lines:
            streamlit_module.write(f"- {summary_line}")


def _render_spotify_playlist_export_section(
    streamlit_module: Any,
    *,
    auth_manager: SpotifyAuthManager,
    spotify_playlist_export_service: SpotifyPlaylistExportService,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult | None,
    listening_history_snapshot: ListeningHistorySnapshot | None,
    ui_state: DemoUIState,
) -> None:
    """Render Spotify playlist export controls for real-track recommendations."""

    if spotify_real_recommendation_result is None:
        streamlit_module.session_state.pop("spotify_playlist_export_result", None)
        return

    streamlit_module.subheader("Playlist Preview")
    track_ids = spotify_playlist_export_service.collect_export_track_ids(
        spotify_real_recommendation_result=spotify_real_recommendation_result,
        include_buckets=True,
    )
    if not track_ids:
        streamlit_module.warning("No real Spotify track IDs are available to export.")
        return

    token = auth_manager.get_token(streamlit_module.session_state)
    if token is None or listening_history_snapshot is None:
        streamlit_module.warning("Connect Spotify before saving recommendations to a playlist.")
        return

    if not SpotifyPlaylistExportService.has_playlist_export_scope(token.scope):
        streamlit_module.warning(
            "Playlist export requires playlist-modify-private scope. Please update your .env and log in again."
        )
        streamlit_module.button(
            "Save recommendations to Spotify",
            disabled=True,
            help=f"Requires the {PLAYLIST_EXPORT_REQUIRED_SCOPE} OAuth scope.",
        )
        return

    preview = _build_playlist_preview_metadata(
        spotify_real_recommendation_result=spotify_real_recommendation_result,
        track_ids=track_ids,
        ui_state=ui_state,
    )
    preview_columns = streamlit_module.columns(4)
    preview_columns[0].metric("Mood", preview["mood"])
    preview_columns[1].metric("Exploration", preview["exploration"])
    preview_columns[2].metric("Tracks", preview["track_count"])
    preview_columns[3].metric("Estimated duration", preview["duration"])
    streamlit_module.caption(f"Buckets included: {preview['buckets']}")
    if streamlit_module.button("Save recommendations to Spotify", use_container_width=True):
        export_result = _export_spotify_recommendations(
            streamlit_module=streamlit_module,
            auth_manager=auth_manager,
            spotify_playlist_export_service=spotify_playlist_export_service,
            spotify_real_recommendation_result=spotify_real_recommendation_result,
            listening_history_snapshot=listening_history_snapshot,
            ui_state=ui_state,
        )
        streamlit_module.session_state["spotify_playlist_export_result"] = export_result

    stored_result = streamlit_module.session_state.get("spotify_playlist_export_result")
    if isinstance(stored_result, SpotifyPlaylistExportResult):
        _render_spotify_playlist_export_result(streamlit_module, stored_result)


def _build_playlist_preview_metadata(
    *,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult,
    track_ids: list[str],
    ui_state: DemoUIState,
) -> dict[str, str]:
    """Build export-preview metadata from already generated recommendations."""

    catalog = spotify_real_recommendation_result.candidate_set.track_catalog
    duration_label = "Unavailable"
    if not catalog.empty and "track_id" in catalog.columns:
        duration_column = "duration_ms" if "duration_ms" in catalog.columns else "track_duration_ms"
        if duration_column in catalog.columns:
            selected_durations = pd.to_numeric(
                catalog.loc[catalog["track_id"].astype(str).isin(track_ids), duration_column],
                errors="coerce",
            ).dropna()
            if not selected_durations.empty:
                total_seconds = int(round(float(selected_durations.sum()) / 1000))
                duration_label = f"{total_seconds // 60}:{total_seconds % 60:02d}"
    buckets = _get_spotify_recommendation_buckets(spotify_real_recommendation_result)
    bucket_labels = [bucket.bucket_label.replace(" Picks", "") for bucket in buckets if bucket.recommendations]
    return {
        "mood": _format_mood_label(ui_state.mood_label),
        "exploration": f"{int(round(ui_state.exploration_level * 100))}%",
        "track_count": str(len(track_ids)),
        "duration": duration_label,
        "buckets": ", ".join(bucket_labels) or "Balanced recommendations",
    }


def _export_spotify_recommendations(
    *,
    streamlit_module: Any,
    auth_manager: SpotifyAuthManager,
    spotify_playlist_export_service: SpotifyPlaylistExportService,
    spotify_real_recommendation_result: SpotifyRealRecommendationResult,
    listening_history_snapshot: ListeningHistorySnapshot,
    ui_state: DemoUIState,
) -> SpotifyPlaylistExportResult:
    """Refresh the user token and export the current real Spotify tracks."""

    try:
        token = auth_manager.ensure_valid_token(streamlit_module.session_state)
    except SpotifyOAuthError:
        return SpotifyPlaylistExportResult(
            playlist_id=None,
            playlist_url=None,
            track_count=0,
            success=False,
            message="Spotify login expired. Please log in again before exporting a playlist.",
        )
    if token is None:
        return SpotifyPlaylistExportResult(
            playlist_id=None,
            playlist_url=None,
            track_count=0,
            success=False,
            message="Connect Spotify before saving recommendations to a playlist.",
        )
    return spotify_playlist_export_service.export_recommendations(
        user_token=token.access_token,
        user_id=listening_history_snapshot.user_id,
        spotify_real_recommendation_result=spotify_real_recommendation_result,
        mood_label=ui_state.mood_label,
        exploration_level=ui_state.exploration_level,
        granted_scopes=token.scope,
        include_buckets=True,
    )


def _render_spotify_playlist_export_result(
    streamlit_module: Any,
    export_result: SpotifyPlaylistExportResult,
) -> None:
    """Render the most recent Spotify playlist export result."""

    if export_result.success:
        streamlit_module.success(f"✅ Playlist created. {export_result.message}")
        if export_result.playlist_url:
            if hasattr(streamlit_module, "link_button"):
                streamlit_module.link_button("Open playlist on Spotify", export_result.playlist_url)
            else:  # pragma: no cover - retained for older Streamlit versions.
                streamlit_module.markdown(f"[Open playlist on Spotify]({export_result.playlist_url})")
        return
    streamlit_module.warning(export_result.message)


def _has_spotify_explanation_context(explanation: DemoRecommendationExplanation) -> bool:
    """Return whether an explanation includes Spotify-enriched rationale fields."""

    return bool(
        explanation.spotify_rationale_lines
        or explanation.spotify_recent_track_labels
        or explanation.spotify_matched_seed_labels
        or explanation.spotify_inferred_mood
        or explanation.spotify_taste_signals
    )


def _render_spotify_recommendation_chain(
    streamlit_module: Any,
    explanation: DemoRecommendationExplanation,
) -> None:
    """Render a compact Spotify-to-recommendation chain for one explanation."""

    rationale_summary = " ".join(explanation.spotify_rationale_lines[:2]).strip()
    if not rationale_summary:
        rationale_summary = "Spotify listening context influenced this recommendation."

    with streamlit_module.container(border=True):
        streamlit_module.markdown("**Spotify-to-recommendation chain**")
        chain_columns = streamlit_module.columns([1.25, 0.2, 1.25, 0.2, 1.5, 0.2, 1.25])
        _render_chain_stage(
            chain_columns[0],
            "Recent Spotify",
            explanation.spotify_recent_track_labels[:2],
            empty_message="Recent listening unavailable",
        )
        chain_columns[1].markdown("**->**")
        _render_chain_stage(
            chain_columns[2],
            "Candidate source"
            if explanation.recommendation_source == "Spotify real-track recommendations"
            else "Matched demo seeds",
            explanation.spotify_matched_seed_labels[:2],
            empty_message="No direct demo seed match",
        )
        chain_columns[3].markdown("**->**")
        _render_chain_stage(
            chain_columns[4],
            "Rationale",
            [rationale_summary],
        )
        chain_columns[5].markdown("**->**")
        _render_chain_stage(
            chain_columns[6],
            "Recommended track",
            [f"{explanation.track_name} - {explanation.artist_name}"],
        )


def _render_chain_stage(
    streamlit_module: Any,
    title: str,
    lines: list[str],
    empty_message: str | None = None,
) -> None:
    """Render one compact stage inside the recommendation chain."""

    streamlit_module.caption(title)
    if lines:
        for line in lines:
            streamlit_module.write(f"- {line}")
        return
    if empty_message:
        streamlit_module.caption(empty_message)


def _render_playlist_section(streamlit_module: Any, view_state: DemoViewState) -> None:
    """Render the mood-aware playlist and sequencing explanations."""

    streamlit_module.subheader("Mood-Based Playlist")
    streamlit_module.caption("A short follow-on sequence built from the current recommendation set.")
    if not view_state.playlist.tracks:
        streamlit_module.info("A playlist could not be generated from the current recommendation set.")
        return

    for playlist_index, playlist_track in enumerate(view_state.playlist.tracks, start=1):
        with streamlit_module.container(border=True):
            streamlit_module.markdown(
                f"**{playlist_index}. {playlist_track.track_name}**  \n"
                f"{playlist_track.artist_name}"
            )
            for reason in playlist_track.explanation.reasons:
                streamlit_module.write(f"- {reason}")


def _render_taste_cluster_section(streamlit_module: Any, view_state: DemoViewState) -> None:
    """Render the optional taste-cluster visualization."""

    streamlit_module.subheader("Taste Map")
    if view_state.taste_clusters is None:
        streamlit_module.info("Taste clusters were not requested for this view.")
        return

    streamlit_module.caption(view_state.taste_clusters.message)
    points_frame = view_state.taste_clusters.points_frame

    try:
        import altair as alt

        chart = (
            alt.Chart(points_frame)
            .mark_circle(size=120)
            .encode(
                x=alt.X("projection_x:Q", title="Taste axis 1"),
                y=alt.Y("projection_y:Q", title="Taste axis 2"),
                color=alt.Color("cluster_label:N", title="Cluster"),
                tooltip=["track_name", "artist_name", "cluster_label"],
            )
            .properties(height=360)
        )
        streamlit_module.altair_chart(chart, use_container_width=True)
    except Exception:
        # If Altair is unavailable, we still show the underlying point data so
        # the demo remains functional in a lightweight local environment.
        streamlit_module.dataframe(
            points_frame[["track_name", "artist_name", "cluster_label"]],
            use_container_width=True,
        )


def _build_badge_html(label: str, tone: str) -> str:
    """Return a compact HTML badge for recruiter-friendly status labels."""

    return f'<span class="demo-badge {tone}">{label}</span>'


def _build_recommendation_card_html(
    *,
    explanation: DemoRecommendationExplanation | None,
    bucket_label: str,
    mood_label: str,
    rank: int | None = None,
    fallback_track_name: str = "Recommendation",
    fallback_artist_name: str = "Unknown artist",
    extra_badges: str = "",
    rationale: str | None = None,
) -> str:
    """Build a compact, media-safe Spotify-style recommendation card."""

    track_name = explanation.track_name if explanation is not None else fallback_track_name
    artist_name = explanation.artist_name if explanation is not None else fallback_artist_name
    album_image_url = explanation.album_image_url if explanation is not None else ""
    spotify_url = explanation.spotify_url if explanation is not None else ""
    card_rationale = rationale or _build_recommendation_card_rationale(explanation)
    bucket_badge = _build_bucket_badge_html(bucket_label)
    mood_badge = _build_mood_badge_html(mood_label)
    mood_fit = _format_mood_fit_label(explanation)
    image_html = (
        f'<img class="recommendation-thumb" src="{escape(album_image_url, quote=True)}" alt="" />'
        if album_image_url
        else '<div class="recommendation-thumb-placeholder">No art</div>'
    )
    spotify_link_html = (
        f'<a class="recommendation-link" href="{escape(spotify_url, quote=True)}" target="_blank">Open in Spotify</a>'
        if spotify_url
        else ""
    )
    rank_prefix = f"{rank}. " if rank is not None else ""
    return """
        <div class="recommendation-card">
          {image_html}
          <div class="recommendation-content">
            <div class="recommendation-title">{rank_prefix}{track_name}</div>
            <div class="recommendation-artist">{artist_name}</div>
            <div>{bucket_badge}{mood_badge}{extra_badges}</div>
            <div class="recommendation-reason"><strong>Why recommended:</strong> {rationale}</div>
            <div class="recommendation-reason">Mood fit: {mood_fit}</div>
            {spotify_link_html}
          </div>
        </div>
        """.format(
        image_html=image_html,
        rank_prefix=escape(rank_prefix),
        track_name=escape(str(track_name)),
        artist_name=escape(str(artist_name)),
        bucket_badge=bucket_badge,
        mood_badge=mood_badge,
        extra_badges=extra_badges,
        rationale=escape(card_rationale),
        mood_fit=escape(mood_fit),
        spotify_link_html=spotify_link_html,
    )


def _build_bucket_badge_html(bucket_label: str) -> str:
    """Return a product-oriented badge for a recommendation bucket."""

    normalized_label = bucket_label.strip().lower()
    if "familiar" in normalized_label:
        return _build_badge_html("🎯 Familiar", "familiar")
    if "discovery" in normalized_label:
        return _build_badge_html("✨ Discovery", "discovery")
    if "mood" in normalized_label:
        return _build_badge_html("🎭 Mood-Based", "mood-based")
    return _build_badge_html(bucket_label, "demo")


def _build_mood_badge_html(mood_label: str) -> str:
    """Return a compact mood badge for recommendation cards."""

    return _build_badge_html(_format_mood_label(mood_label), "mood")


def _format_mood_label(mood_label: str) -> str:
    """Return an emoji-supported mood label for compact product UI."""

    normalized_mood = mood_label.strip().lower()
    mood_labels = {
        "workout": "🔥 Workout",
        "calm": "😌 Calm",
        "happy": "😊 Happy",
        "melancholic": "🌧 Melancholic",
        "party": "🎉 Party",
        "study": "📚 Study",
    }
    return mood_labels.get(normalized_mood, normalized_mood.title())


def _format_mood_fit_label(explanation: DemoRecommendationExplanation | None) -> str:
    """Map an existing explanation mood score into a short display label."""

    if explanation is None:
        return "Unavailable"
    for line in explanation.spotify_rationale_lines:
        if "Mood score:" not in line:
            continue
        try:
            mood_score = float(line.split("Mood score:", maxsplit=1)[1].strip().rstrip("."))
        except ValueError:
            break
        if mood_score >= 0.70:
            return "High"
        if mood_score >= 0.40:
            return "Medium"
        return "Low"
    return "Available" if explanation.spotify_inferred_mood else "Unavailable"


def _format_exploration_badge(exploration_level: float) -> str:
    """Return a compact exploration label for the current slider value."""

    if exploration_level <= 0.25:
        descriptor = "Familiar"
    elif exploration_level >= 0.7:
        descriptor = "Exploratory"
    else:
        descriptor = "Balanced"
    return f"{descriptor} {int(round(exploration_level * 100))}%"


def _build_recommendation_card_rationale(
    explanation: DemoRecommendationExplanation | None,
) -> str:
    """Return a short rationale preview for a recommendation card."""

    if explanation is None:
        return "Built from the current hybrid ranking blend."
    if explanation.spotify_rationale_lines:
        return explanation.spotify_rationale_lines[0]
    if explanation.summary_lines:
        return explanation.summary_lines[0]
    return "Built from the current hybrid ranking blend."


def main() -> None:
    """Provide a module-level entrypoint for Streamlit execution."""

    run_app()


if __name__ == "__main__":
    main()
