"""Tests for real Spotify candidate generation and ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import ProjectSettings
from data.preprocessor import Preprocessor
from data.spotify_client import SpotifyAPIClientError
from app.streamlit_app import _get_spotify_recommendation_buckets, _render_spotify_candidate_debug_summary
from models.hybrid_recommender import HybridRecommendation, HybridScoreBreakdown
from services.spotify_candidate_service import (
    RecommendationBucket,
    SpotifyCandidateService,
    SpotifyCandidateSet,
    top_k_overlap_percent,
)
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary


class FakeSpotifyCandidateClient:
    """Return deterministic Spotify candidate payloads."""

    def get_artist_top_tracks(self, artist_id: str, access_token: str, market: str = "US") -> dict[str, Any]:
        """Return top tracks for the recent artist."""

        return {
            "tracks": [
                build_track_payload("candidate_1", "Bright Real Track", "artist_1", "Aurora Lane", 82),
                build_track_payload("candidate_2", "Deep Cut", "artist_3", "New Artist", 35),
            ]
        }

    def search_tracks(self, query: str, access_token: str, limit: int = 10, market: str = "US") -> dict[str, Any]:
        """Return a search candidate for the recent artist."""

        return {
            "tracks": {
                "items": [
                    build_track_payload("candidate_3", "Search Match", "artist_1", "Aurora Lane", 60),
                ]
            }
        }

    def get_audio_features(self, track_ids: list[str], access_token: str) -> dict[str, Any]:
        """Return candidate audio features."""

        return {
            "audio_features": [
                build_audio_features("candidate_1", energy=0.82, valence=0.75, danceability=0.72),
                build_audio_features("candidate_2", energy=0.30, valence=0.25, danceability=0.42),
                build_audio_features("candidate_3", energy=0.70, valence=0.68, danceability=0.65),
            ]
        }

    def get_artists(self, artist_ids: list[str], access_token: str) -> dict[str, Any]:
        """Return artist metadata for candidates."""

        return {
            "artists": [
                {
                    "id": "artist_1",
                    "name": "Aurora Lane",
                    "genres": ["indie pop"],
                    "popularity": 75,
                    "followers": {"total": 1000},
                },
                {
                    "id": "artist_3",
                    "name": "New Artist",
                    "genres": ["ambient"],
                    "popularity": 40,
                    "followers": {"total": 200},
                },
            ]
        }


class AudioFeaturesUnavailableClient(FakeSpotifyCandidateClient):
    """Simulate Spotify audio feature failure."""

    def get_audio_features(self, track_ids: list[str], access_token: str) -> dict[str, Any]:
        """Raise a Spotify client error for audio features."""

        raise SpotifyAPIClientError("Spotify GET request failed for /audio-features: 403 forbidden")


class NoisyTopTracksClient(FakeSpotifyCandidateClient):
    """Simulate invalid and unavailable artist top-track sources."""

    def __init__(self) -> None:
        """Track which artist IDs were sent to the top-track endpoint."""

        self.top_track_artist_ids: list[str] = []

    def get_artist_top_tracks(self, artist_id: str, access_token: str, market: str = "US") -> dict[str, Any]:
        """Raise a 404 for valid-looking but unavailable Spotify artist IDs."""

        self.top_track_artist_ids.append(artist_id)
        raise SpotifyAPIClientError(
            f"Spotify GET request failed for /artists/{artist_id}/top-tracks: 404 Resource not found"
        )


class FakeStreamlitModule:
    """Capture Streamlit caption calls for debug-summary tests."""

    def __init__(self) -> None:
        """Initialize captured captions."""

        self.captions: list[str] = []

    def caption(self, text: str) -> None:
        """Capture a caption that would be rendered by Streamlit."""

        self.captions.append(text)


class LegacyCandidateSet:
    """Represent an older cached object without debug_summary."""


class FakeSpotifyResult:
    """Small object with the fields used by the Streamlit debug renderer."""

    def __init__(self, candidate_set: object) -> None:
        """Store a candidate set."""

        self.candidate_set = candidate_set


class LegacyBucketResult:
    """Represent an older real recommendation result with only the legacy bucket dict."""

    def __init__(self, bucketed_explanations: dict[str, list[Any]]) -> None:
        """Store legacy bucket explanations."""

        self.bucketed_explanations = bucketed_explanations


def build_settings(tmp_path: Path) -> ProjectSettings:
    """Create settings for the candidate service tests."""

    return ProjectSettings(
        project_root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        interim_data_dir=tmp_path / "data" / "interim",
        processed_data_dir=tmp_path / "data" / "processed",
        artifacts_dir=tmp_path / "artifacts",
        spotify_client_id="client-id",
        spotify_client_secret="client-secret",
        spotify_redirect_uri="http://localhost:8501",
        spotify_api_base_url="https://api.spotify.com/v1",
        spotify_accounts_base_url="https://accounts.spotify.com",
        spotify_request_timeout_seconds=30,
        spotify_default_market="US",
    )


def build_snapshot() -> ListeningHistorySnapshot:
    """Create recent Spotify history with one real artist ID."""

    return ListeningHistorySnapshot(
        user_id="spotify_user",
        display_name="Casey",
        recent_tracks=[
            RecentTrackSummary(
                track_id="recent_1",
                track_name="Recent Song",
                artist_name="Aurora Lane",
                played_at="2024-01-01T00:00:00Z",
            )
        ],
        track_level_frame=pd.DataFrame(
            [
                {
                    "track_id": "recent_1",
                    "track_name": "Recent Song",
                    "primary_artist_id": "0TnOYISbd1XYRBk9myaseg",
                    "primary_artist_name": "Aurora Lane",
                    "artist_name": "Aurora Lane",
                    "artist_genres": "indie pop",
                    "danceability": 0.70,
                    "energy": 0.80,
                    "valence": 0.72,
                    "tempo": 120.0,
                    "acousticness": 0.20,
                    "speechiness": 0.05,
                    "instrumentalness": 0.0,
                    "loudness": -6.0,
                    "popularity": 70,
                }
            ]
        ),
        interaction_frame=pd.DataFrame(),
        seed_track_ids=["recent_1"],
    )


def build_track_payload(
    track_id: str,
    track_name: str,
    artist_id: str,
    artist_name: str,
    popularity: int,
) -> dict[str, Any]:
    """Build a small Spotify track payload."""

    return {
        "id": track_id,
        "name": track_name,
        "artists": [{"id": artist_id, "name": artist_name}],
        "album": {
            "id": f"album_{track_id}",
            "name": "Album",
            "release_date": "2024-01-01",
            "total_tracks": 10,
            "images": [{"url": f"https://images.example/{track_id}.jpg"}],
        },
        "duration_ms": 180000,
        "explicit": False,
        "popularity": popularity,
        "preview_url": "",
        "track_number": 1,
        "disc_number": 1,
        "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
    }


def build_audio_features(track_id: str, energy: float, valence: float, danceability: float) -> dict[str, Any]:
    """Build a small audio feature payload."""

    return {
        "id": track_id,
        "danceability": danceability,
        "energy": energy,
        "key": 5,
        "loudness": -6.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": valence,
        "tempo": 120.0,
        "duration_ms": 180000,
        "time_signature": 4,
    }


def build_service(client: object, tmp_path: Path) -> SpotifyCandidateService:
    """Build a service with a fake Spotify client."""

    return SpotifyCandidateService(
        client=client,  # type: ignore[arg-type]
        preprocessor=Preprocessor(settings=build_settings(tmp_path)),
    )


def test_spotify_candidate_service_returns_real_spotify_tracks(tmp_path: Path) -> None:
    """Spotify mode should rank real Spotify candidate track IDs."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)

    result = service.build_real_spotify_view(
        access_token="token",
        listening_history_snapshot=build_snapshot(),
        exploration_level=0.3,
        recommendation_count=2,
        mood_label="happy",
        playlist_length=2,
    )

    assert result is not None
    assert len(result.view_state.recommendations) == 2
    assert all(recommendation.track_id.startswith("candidate_") for recommendation in result.view_state.recommendations)
    assert result.view_state.explanations[0].spotify_url.startswith("https://open.spotify.com/track/")
    assert result.view_state.explanations[0].album_image_url.startswith("https://images.example/")
    assert "Spotify real-track" in result.view_state.explanations[0].recommendation_source
    assert result.candidate_set.debug_summary["candidate_count"] == 3
    assert result.candidate_set.debug_summary["top_track_candidate_count"] == 2
    assert result.candidate_set.debug_summary["search_candidate_count"] >= 1
    assert result.candidate_set.debug_summary["unique_candidate_count"] == 3
    assert result.candidate_set.debug_summary["duplicate_candidates_removed"] >= 1
    assert result.candidate_set.debug_summary["candidate_source_breakdown"]["recent artist top track"] == 2
    assert result.candidate_set.debug_summary["ranking_mode"] == "audio-feature-based"
    assert [bucket.bucket_name for bucket in result.recommendation_buckets] == [
        "familiar",
        "discovery",
        "mood_based",
    ]
    assert set(result.bucketed_explanations) == {
        "Familiar Picks",
        "Discovery Picks",
        "Mood-Based Picks",
    }


def test_spotify_candidate_service_bucket_generation_returns_distinct_lists(tmp_path: Path) -> None:
    """Familiar, discovery, and mood buckets should rank candidates differently."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)

    result = service.build_real_spotify_view(
        access_token="token",
        listening_history_snapshot=build_snapshot(),
        exploration_level=0.5,
        recommendation_count=2,
        mood_label="study",
        playlist_length=2,
    )

    assert result is not None
    familiar_ids = [explanation.track_id for explanation in result.bucketed_explanations["Familiar Picks"]]
    discovery_ids = [explanation.track_id for explanation in result.bucketed_explanations["Discovery Picks"]]
    mood_ids = [explanation.track_id for explanation in result.bucketed_explanations["Mood-Based Picks"]]
    assert familiar_ids != discovery_ids
    assert mood_ids


def test_recommendation_bucket_can_be_constructed() -> None:
    """The lightweight bucket representation should be safe for UI fallback code."""

    bucket = RecommendationBucket(
        bucket_name="familiar",
        bucket_label="Familiar Picks",
        description="Close to recent listening.",
        recommendations=[],
    )

    assert bucket.bucket_name == "familiar"
    assert bucket.recommendations == []


def test_spotify_bucket_ui_helper_falls_back_to_legacy_bucket_dict() -> None:
    """The UI should keep rendering older cached bucket result objects."""

    buckets = _get_spotify_recommendation_buckets(
        LegacyBucketResult({"Familiar Picks": []})  # type: ignore[arg-type]
    )

    assert buckets[0].bucket_label == "Familiar Picks"
    assert buckets[0].recommendations == []


def test_spotify_candidate_set_debug_summary_defaults_empty() -> None:
    """Candidate sets can be built without explicitly passing debug metadata."""

    candidate_set = SpotifyCandidateSet(candidates=[], track_catalog=pd.DataFrame())

    assert candidate_set.debug_summary == {}


def test_spotify_candidate_debug_summary_renderer_handles_missing_debug_summary() -> None:
    """The UI debug summary should tolerate older cached candidate-set objects."""

    streamlit_module = FakeStreamlitModule()

    _render_spotify_candidate_debug_summary(
        streamlit_module,
        FakeSpotifyResult(candidate_set=LegacyCandidateSet()),  # type: ignore[arg-type]
    )

    assert streamlit_module.captions == []


def test_spotify_candidate_debug_summary_renderer_handles_empty_debug_summary() -> None:
    """The UI debug summary should no-op when debug metadata is empty."""

    streamlit_module = FakeStreamlitModule()

    _render_spotify_candidate_debug_summary(
        streamlit_module,
        FakeSpotifyResult(candidate_set=SpotifyCandidateSet(candidates=[], track_catalog=pd.DataFrame())),  # type: ignore[arg-type]
    )

    assert streamlit_module.captions == []


def test_spotify_candidate_service_degrades_without_audio_features(tmp_path: Path) -> None:
    """Candidate ranking should still work when audio features are unavailable."""

    service = build_service(AudioFeaturesUnavailableClient(), tmp_path)

    result = service.build_real_spotify_view(
        access_token="token",
        listening_history_snapshot=build_snapshot(),
        exploration_level=0.8,
        recommendation_count=2,
        mood_label="study",
        playlist_length=2,
    )

    assert result is not None
    assert len(result.view_state.recommendations) == 2
    assert result.candidate_set.warnings
    assert "metadata-only" in result.candidate_set.warnings[0]


def test_spotify_candidate_service_quiets_invalid_artist_top_track_failures(tmp_path: Path) -> None:
    """Invalid artist IDs and 404 top-track failures should not spam UI warnings."""

    client = NoisyTopTracksClient()
    snapshot = build_snapshot()
    snapshot.track_level_frame = pd.concat(
        [
            snapshot.track_level_frame,
            pd.DataFrame(
                [
                    {
                        "track_id": "recent_bad",
                        "track_name": "Bad Artist ID",
                        "primary_artist_id": "artist_looks_like_lastfm",
                        "primary_artist_name": "Aurora Lane",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    service = build_service(client, tmp_path)

    result = service.build_real_spotify_view(
        access_token="token",
        listening_history_snapshot=snapshot,
        exploration_level=0.4,
        recommendation_count=1,
        mood_label="happy",
        playlist_length=1,
    )

    assert result is not None
    assert client.top_track_artist_ids == ["0TnOYISbd1XYRBk9myaseg"]
    assert result.candidate_set.warnings == [
        "Some recent artists could not be expanded into top tracks, so search-based candidates were used instead."
    ]
    assert result.candidate_set.debug_summary["top_track_requests_failed"] == 1
    assert result.candidate_set.debug_summary["final_candidate_count"] > 0


def test_spotify_candidate_service_exploration_changes_ranked_order(tmp_path: Path) -> None:
    """Exploration should move ranking from familiar/popular to discovery/search."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    snapshot = build_snapshot()
    candidate_catalog = pd.DataFrame(
        [
            {
                "track_id": "familiar",
                "track_name": "Familiar Hit",
                "artist_name": "Aurora Lane",
                "artist_genres": "indie pop",
                "catalog_popularity": 0.95,
                "catalog_novelty": 0.05,
                "candidate_sources": "recent artist top track",
                "ranking_mode": "metadata-only",
            },
            {
                "track_id": "discovery",
                "track_name": "Deep Search Find",
                "artist_name": "New Artist",
                "artist_genres": "ambient",
                "catalog_popularity": 0.10,
                "catalog_novelty": 0.90,
                "candidate_sources": "recent artist search match",
                "ranking_mode": "metadata-only",
            },
        ]
    )
    recommendations = [
        build_recommendation("familiar", "Familiar Hit", "Aurora Lane"),
        build_recommendation("discovery", "Deep Search Find", "New Artist"),
    ]

    low_exploration = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        snapshot,
        exploration_level=0.0,
        mood_label="calm",
    )
    high_exploration = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        snapshot,
        exploration_level=0.9,
        mood_label="calm",
    )

    assert low_exploration[0].track_id == "familiar"
    assert high_exploration[0].track_id == "discovery"


def test_spotify_candidate_service_mood_changes_ranked_order(tmp_path: Path) -> None:
    """Mood should affect ordering even when ranking is metadata-only."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    candidate_catalog = pd.DataFrame(
        [
            {
                "track_id": "happy_track",
                "track_name": "Bright Summer Smile",
                "artist_name": "New Artist",
                "artist_genres": "pop",
                "catalog_popularity": 0.5,
                "catalog_novelty": 0.5,
                "candidate_sources": "recent artist search match",
                "ranking_mode": "metadata-only",
            },
            {
                "track_id": "study_track",
                "track_name": "Soft Ambient Focus",
                "artist_name": "Other Artist",
                "artist_genres": "ambient",
                "catalog_popularity": 0.5,
                "catalog_novelty": 0.5,
                "candidate_sources": "recent artist search match",
                "ranking_mode": "metadata-only",
            },
        ]
    )
    recommendations = [
        build_recommendation("happy_track", "Bright Summer Smile", "New Artist"),
        build_recommendation("study_track", "Soft Ambient Focus", "Other Artist"),
    ]

    happy_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        build_snapshot(),
        exploration_level=0.5,
        mood_label="happy",
    )
    study_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        build_snapshot(),
        exploration_level=0.5,
        mood_label="study",
    )

    assert happy_order[0].track_id == "happy_track"
    assert study_order[0].track_id == "study_track"


def test_spotify_candidate_service_mood_bucket_changes_with_mood(tmp_path: Path) -> None:
    """Changing mood should change the mood-based bucket ordering."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    candidate_set = SpotifyCandidateSet(
        candidates=[],
        track_catalog=pd.DataFrame(
            [
                {
                    "track_id": "happy_track",
                    "track_name": "Bright Summer Smile",
                    "artist_name": "New Artist",
                    "artist_genres": "pop",
                    "catalog_popularity": 0.5,
                    "catalog_novelty": 0.5,
                    "candidate_sources": "recent artist search match",
                    "ranking_mode": "metadata-only",
                    "spotify_url": "https://open.spotify.com/track/happy_track",
                    "album_image_url": "",
                },
                {
                    "track_id": "calm_track",
                    "track_name": "Soft Ambient Calm",
                    "artist_name": "Calm Artist",
                    "artist_genres": "ambient",
                    "catalog_popularity": 0.5,
                    "catalog_novelty": 0.5,
                    "candidate_sources": "recent artist search match",
                    "ranking_mode": "metadata-only",
                    "spotify_url": "https://open.spotify.com/track/calm_track",
                    "album_image_url": "",
                },
            ]
        ),
        source_labels_by_track_id={
            "happy_track": ["recent artist search match"],
            "calm_track": ["recent artist search match"],
        },
    )

    happy_bucket = service._build_bucketed_explanations(
        candidate_catalog=candidate_set.track_catalog,
        listening_history_snapshot=build_snapshot(),
        candidate_set=candidate_set,
        mood_label="happy",
        recommendation_count=2,
    )["Mood-Based Picks"]
    calm_bucket = service._build_bucketed_explanations(
        candidate_catalog=candidate_set.track_catalog,
        listening_history_snapshot=build_snapshot(),
        candidate_set=candidate_set,
        mood_label="calm",
        recommendation_count=2,
    )["Mood-Based Picks"]

    assert happy_bucket[0].track_id == "happy_track"
    assert calm_bucket[0].track_id == "calm_track"


def test_spotify_candidate_service_exploration_changes_bucket_ranking(tmp_path: Path) -> None:
    """Low and high exploration should visibly change bucket order."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    candidate_set = SpotifyCandidateSet(
        candidates=[],
        track_catalog=pd.DataFrame(
            [
                {
                    "track_id": "familiar",
                    "track_name": "Familiar Hit",
                    "artist_name": "Aurora Lane",
                    "artist_genres": "indie pop",
                    "catalog_popularity": 0.95,
                    "catalog_novelty": 0.05,
                    "candidate_sources": "recent artist top track",
                    "ranking_mode": "metadata-only",
                    "spotify_url": "https://open.spotify.com/track/familiar",
                    "album_image_url": "",
                },
                {
                    "track_id": "discovery",
                    "track_name": "Deep Search Find",
                    "artist_name": "New Artist",
                    "artist_genres": "ambient",
                    "catalog_popularity": 0.10,
                    "catalog_novelty": 0.90,
                    "candidate_sources": "recent artist search match",
                    "ranking_mode": "metadata-only",
                    "spotify_url": "https://open.spotify.com/track/discovery",
                    "album_image_url": "",
                },
            ]
        ),
        source_labels_by_track_id={
            "familiar": ["recent artist top track"],
            "discovery": ["recent artist search match"],
        },
    )

    low_exploration_buckets = service._build_recommendation_buckets(
        candidate_catalog=candidate_set.track_catalog,
        listening_history_snapshot=build_snapshot(),
        candidate_set=candidate_set,
        exploration_level=0.0,
        mood_label="calm",
        recommendation_count=2,
    )
    high_exploration_buckets = service._build_recommendation_buckets(
        candidate_catalog=candidate_set.track_catalog,
        listening_history_snapshot=build_snapshot(),
        candidate_set=candidate_set,
        exploration_level=1.0,
        mood_label="calm",
        recommendation_count=2,
    )

    low_familiar = low_exploration_buckets[0].recommendations
    high_discovery = high_exploration_buckets[1].recommendations
    assert low_familiar[0].track_id == "familiar"
    assert high_discovery[0].track_id == "discovery"


def test_spotify_candidate_service_calm_and_workout_change_ranked_order(tmp_path: Path) -> None:
    """Calm and workout should pull different tracks to the top with audio features."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    candidate_catalog = pd.DataFrame(
        [
            {
                "track_id": "workout_track",
                "track_name": "Fast Run",
                "artist_name": "Energy Artist",
                "catalog_popularity": 0.5,
                "catalog_novelty": 0.5,
                "candidate_sources": "recent artist search match",
                "ranking_mode": "audio-feature-based",
                "energy": 0.95,
                "danceability": 0.92,
                "valence": 0.70,
                "tempo": 150.0,
                "acousticness": 0.05,
                "instrumentalness": 0.0,
            },
            {
                "track_id": "calm_track",
                "track_name": "Quiet Room",
                "artist_name": "Calm Artist",
                "catalog_popularity": 0.5,
                "catalog_novelty": 0.5,
                "candidate_sources": "recent artist search match",
                "ranking_mode": "audio-feature-based",
                "energy": 0.20,
                "danceability": 0.25,
                "valence": 0.45,
                "tempo": 74.0,
                "acousticness": 0.90,
                "instrumentalness": 0.30,
            },
        ]
    )
    recommendations = [
        build_recommendation("workout_track", "Fast Run", "Energy Artist"),
        build_recommendation("calm_track", "Quiet Room", "Calm Artist"),
    ]

    workout_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        build_snapshot(),
        exploration_level=0.5,
        mood_label="workout",
    )
    calm_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        build_snapshot(),
        exploration_level=0.5,
        mood_label="calm",
    )

    assert workout_order[0].track_id == "workout_track"
    assert calm_order[0].track_id == "calm_track"


def test_spotify_candidate_service_recommendation_count_controls_output_length(tmp_path: Path) -> None:
    """The recommendation count should control how many cards can render."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)

    result = service.build_real_spotify_view(
        access_token="token",
        listening_history_snapshot=build_snapshot(),
        exploration_level=0.5,
        recommendation_count=1,
        mood_label="happy",
        playlist_length=1,
    )

    assert result is not None
    assert len(result.view_state.recommendations) == 1
    assert len(result.view_state.explanations) == 1
    assert all(len(bucket.recommendations) == 1 for bucket in result.recommendation_buckets)


def test_spotify_candidate_service_debug_summary_tracks_selected_controls(tmp_path: Path) -> None:
    """Result construction should carry mood/exploration controls into debug metadata."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)

    result = service.build_real_spotify_view(
        access_token="token",
        listening_history_snapshot=build_snapshot(),
        exploration_level=1.0,
        recommendation_count=2,
        mood_label="workout",
        ranking_focus="Discovery",
        playlist_length=2,
    )

    assert result is not None
    assert result.candidate_set.debug_summary["selected_mood"] == "workout"
    assert result.candidate_set.debug_summary["exploration_level"] == "1.00"
    assert result.candidate_set.debug_summary["recommendation_count"] == 2
    assert result.candidate_set.debug_summary["ranking_focus"] == "Discovery"
    assert result.candidate_set.debug_summary["top_candidate_ids_before_reranking"]
    assert result.candidate_set.debug_summary["top_candidate_ids_after_reranking"]
    assert result.candidate_set.debug_summary["diversity_reranking_active"] is True
    assert result.candidate_set.debug_summary["advanced_score_used"] is False


def test_spotify_candidate_generation_dedupes_by_id_and_name_artist(tmp_path: Path) -> None:
    """Candidate generation should remove duplicate IDs and normalized title/artist duplicates."""

    class DuplicateCandidateClient(FakeSpotifyCandidateClient):
        def get_artist_top_tracks(self, artist_id: str, access_token: str, market: str = "US") -> dict[str, Any]:
            return {
                "tracks": [
                    build_track_payload("dup_1", "One Song", "artist_1", "Aurora Lane", 80),
                    build_track_payload("dup_1", "One Song", "artist_1", "Aurora Lane", 80),
                ]
            }

        def search_tracks(self, query: str, access_token: str, limit: int = 10, market: str = "US") -> dict[str, Any]:
            return {
                "tracks": {
                    "items": [
                        build_track_payload("dup_2", "One Song - Live Version", "artist_1", "Aurora Lane", 55),
                        build_track_payload("unique_search", "Fresh Search", "artist_2", "Other Artist", 35),
                    ]
                }
            }

        def get_audio_features(self, track_ids: list[str], access_token: str) -> dict[str, Any]:
            return {"audio_features": []}

    service = build_service(DuplicateCandidateClient(), tmp_path)

    candidate_set = service.build_candidate_set(
        access_token="token",
        listening_history_snapshot=build_snapshot(),
    )

    track_ids = set(candidate_set.track_catalog["track_id"].astype(str))
    assert "dup_1" in track_ids
    assert "dup_2" not in track_ids
    assert "unique_search" in track_ids
    assert candidate_set.debug_summary["raw_candidate_count"] > candidate_set.debug_summary["unique_candidate_count"]
    assert candidate_set.debug_summary["duplicate_candidates_removed"] >= 2
    assert candidate_set.debug_summary["max_candidates_from_single_artist"] >= 1
    assert "recent track search match" in candidate_set.debug_summary["candidate_source_breakdown"]


def test_spotify_candidate_service_optional_advanced_scores_skip_and_change_ranking(tmp_path: Path) -> None:
    """Optional ALS/embedding maps should no-op when absent and change ranking when supplied."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    recommendations = [
        build_recommendation("track_a", "Track A", "Artist A"),
        build_recommendation("track_b", "Track B", "Artist B"),
    ]

    unchanged = service._apply_optional_advanced_scores(recommendations)
    changed = service._apply_optional_advanced_scores(
        recommendations,
        als_scores={"track_a": 0.1, "track_b": 0.9},
        embedding_scores={"track_a": 0.1, "track_b": 1.0},
    )

    assert [recommendation.track_id for recommendation in unchanged] == ["track_a", "track_b"]
    assert changed[0].track_id == "track_b"


def test_spotify_candidate_service_familiar_and_discovery_focus_change_top_results(tmp_path: Path) -> None:
    """Ranking focus should materially change the top results when the pool allows."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    snapshot = build_snapshot()
    candidate_catalog = build_control_candidate_catalog()
    recommendations = [
        build_recommendation(str(row.track_id), str(row.track_name), str(row.artist_name))
        for row in candidate_catalog.itertuples(index=False)
    ]

    familiar_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="calm",
        ranking_focus="Familiar",
    )
    discovery_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        candidate_catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="calm",
        ranking_focus="Discovery",
    )

    familiar_top_4 = [recommendation.track_id for recommendation in familiar_order[:4]]
    discovery_top_4 = [recommendation.track_id for recommendation in discovery_order[:4]]
    assert familiar_top_4[0].startswith("familiar")
    assert not discovery_top_4[0].startswith("familiar")
    assert len(set(familiar_top_4).intersection(discovery_top_4)) <= 2


def test_spotify_candidate_service_audio_mood_profiles_rank_controlled_candidates(tmp_path: Path) -> None:
    """Audio-feature mood profiles should choose tracks that match the requested mood."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    workout_track = {
        "track_name": "Fast Run",
        "artist_name": "Energy Artist",
        "ranking_mode": "audio-feature-based",
        "energy": 0.94,
        "danceability": 0.90,
        "valence": 0.70,
        "tempo": 152.0,
        "acousticness": 0.04,
        "instrumentalness": 0.0,
        "catalog_popularity": 0.65,
    }
    calm_track = {
        "track_name": "Quiet Piano Room",
        "artist_name": "Calm Artist",
        "ranking_mode": "audio-feature-based",
        "energy": 0.16,
        "danceability": 0.20,
        "valence": 0.42,
        "tempo": 72.0,
        "acousticness": 0.92,
        "instrumentalness": 0.35,
        "catalog_popularity": 0.35,
    }
    happy_track = {
        "track_name": "Sunshine Smile",
        "artist_name": "Happy Artist",
        "ranking_mode": "audio-feature-based",
        "energy": 0.72,
        "danceability": 0.76,
        "valence": 0.96,
        "tempo": 122.0,
        "acousticness": 0.16,
        "instrumentalness": 0.0,
        "catalog_popularity": 0.62,
    }
    melancholic_track = {
        "track_name": "Lonely Blue Rain",
        "artist_name": "Melancholic Artist",
        "ranking_mode": "audio-feature-based",
        "energy": 0.32,
        "danceability": 0.20,
        "valence": 0.10,
        "tempo": 78.0,
        "acousticness": 0.84,
        "instrumentalness": 0.15,
        "catalog_popularity": 0.42,
    }

    assert service._compute_mood_alignment(workout_track, "workout") > service._compute_mood_alignment(calm_track, "workout")
    assert service._compute_mood_alignment(calm_track, "calm") > service._compute_mood_alignment(workout_track, "calm")
    assert service._compute_mood_alignment(happy_track, "happy") > service._compute_mood_alignment(melancholic_track, "happy")
    assert service._compute_mood_alignment(melancholic_track, "melancholic") > service._compute_mood_alignment(happy_track, "melancholic")


def test_spotify_candidate_service_metadata_mood_fallback_ranks_keywords(tmp_path: Path) -> None:
    """Metadata-only mood fallback should separate keyword-matched candidates."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    workout_track = {
        "track_name": "Gym Power Run",
        "artist_name": "Energy Artist",
        "artist_genres": "workout pop",
        "candidate_sources": "recent track search match",
        "ranking_mode": "metadata-only",
    }
    calm_track = {
        "track_name": "Soft Ambient Sleep",
        "artist_name": "Quiet Artist",
        "artist_genres": "ambient acoustic",
        "candidate_sources": "genre search match",
        "ranking_mode": "metadata-only",
    }

    assert service._compute_mood_alignment(workout_track, "workout") > service._compute_mood_alignment(calm_track, "workout")
    assert service._compute_mood_alignment(calm_track, "study") > service._compute_mood_alignment(workout_track, "study")


def test_spotify_candidate_service_repeated_calls_use_latest_controls(tmp_path: Path) -> None:
    """Repeated ranking calls should not reuse stale mood/exploration debug state."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    snapshot = build_snapshot()
    candidate_catalog = build_control_candidate_catalog()
    profile = build_spotify_real_profile(snapshot, mood_label="calm")

    calm_debug: dict[str, object] = {}
    workout_debug: dict[str, object] = {}
    calm_results = service._rank_candidates(
        candidate_catalog=candidate_catalog,
        listening_history_snapshot=snapshot,
        profile=profile,
        exploration_level=0.0,
        recommendation_count=5,
        mood_label="calm",
        ranking_focus="Familiar",
        debug_summary=calm_debug,
    )
    workout_results = service._rank_candidates(
        candidate_catalog=candidate_catalog,
        listening_history_snapshot=snapshot,
        profile=profile,
        exploration_level=1.0,
        recommendation_count=5,
        mood_label="workout",
        ranking_focus="Discovery",
        debug_summary=workout_debug,
    )

    assert calm_debug["selected_filters"] != workout_debug["selected_filters"]
    assert calm_debug["top_candidate_ids_after_reranking"] != workout_debug["top_candidate_ids_after_reranking"]
    assert [recommendation.track_id for recommendation in calm_results] != [
        recommendation.track_id for recommendation in workout_results
    ]


def test_top_k_overlap_percent_measures_ranked_list_overlap() -> None:
    """Overlap diagnostics should report the shared fraction of two top-k lists."""

    assert top_k_overlap_percent(["a", "b", "c"], ["b", "c", "d"], 3) == 2 / 3
    assert top_k_overlap_percent([], ["b", "c"], 3) == 0.0


def test_mood_first_calm_and_workout_top_10_overlap_is_low(tmp_path: Path) -> None:
    """Calm and workout mood profiles should produce clearly different top tens."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    catalog = build_mood_diverse_candidate_catalog()
    recommendations = build_scored_recommendations(catalog)
    snapshot = build_snapshot()

    calm_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="calm",
        ranking_focus="Mood-first",
    )
    workout_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="workout",
        ranking_focus="Mood-first",
    )

    overlap = top_k_overlap_percent(
        [recommendation.track_id for recommendation in calm_order],
        [recommendation.track_id for recommendation in workout_order],
        10,
    )
    assert overlap < 0.30


def test_mood_first_happy_and_melancholic_top_10_overlap_is_low(tmp_path: Path) -> None:
    """Happy and melancholic mood profiles should produce clearly different top tens."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    catalog = build_mood_diverse_candidate_catalog()
    recommendations = build_scored_recommendations(catalog)
    snapshot = build_snapshot()

    happy_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="happy",
        ranking_focus="Mood-first",
    )
    melancholic_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="melancholic",
        ranking_focus="Mood-first",
    )

    overlap = top_k_overlap_percent(
        [recommendation.track_id for recommendation in happy_order],
        [recommendation.track_id for recommendation in melancholic_order],
        10,
    )
    assert overlap < 0.30


def test_mood_first_changes_more_than_balanced_ranking(tmp_path: Path) -> None:
    """Mood-first should move farther away from base ranking than balanced mode."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    catalog = build_mood_diverse_candidate_catalog()
    recommendations = build_scored_recommendations(catalog)
    snapshot = build_snapshot()
    base_order = [recommendation.track_id for recommendation in recommendations]

    balanced_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="melancholic",
        ranking_focus="Balanced",
    )
    mood_first_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="melancholic",
        ranking_focus="Mood-first",
    )

    balanced_overlap = top_k_overlap_percent(
        base_order,
        [recommendation.track_id for recommendation in balanced_order],
        10,
    )
    mood_first_overlap = top_k_overlap_percent(
        base_order,
        [recommendation.track_id for recommendation in mood_first_order],
        10,
    )
    assert mood_first_overlap < balanced_overlap


def test_mood_profile_ranking_is_deterministic(tmp_path: Path) -> None:
    """Mood profile ranking should be deterministic for identical inputs."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    catalog = build_mood_diverse_candidate_catalog()
    recommendations = build_scored_recommendations(catalog)
    snapshot = build_snapshot()

    first_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="party",
        ranking_focus="Mood-first",
    )
    second_order = service._apply_spotify_ranking_adjustments(
        recommendations,
        catalog,
        snapshot,
        exploration_level=0.5,
        mood_label="party",
        ranking_focus="Mood-first",
    )

    assert [recommendation.track_id for recommendation in first_order] == [
        recommendation.track_id for recommendation in second_order
    ]


def test_mood_debug_summary_includes_profile_diagnostics(tmp_path: Path) -> None:
    """Ranking debug metadata should expose compact mood profile diagnostics."""

    service = build_service(FakeSpotifyCandidateClient(), tmp_path)
    catalog = build_mood_diverse_candidate_catalog()
    debug_summary: dict[str, object] = {}

    service._rank_candidates(
        candidate_catalog=catalog,
        listening_history_snapshot=build_snapshot(),
        profile=build_spotify_real_profile(build_snapshot(), mood_label="workout"),
        exploration_level=0.5,
        recommendation_count=5,
        mood_label="workout",
        ranking_focus="Mood-first",
        debug_summary=debug_summary,
    )

    assert debug_summary["mood_profile_used"] == "workout"
    assert debug_summary["mood_matching_mode"] == "audio_features"
    assert debug_summary["top_mood_scores_after_reranking"]
    assert "positive_mood_signals" in debug_summary


def build_mood_diverse_candidate_catalog() -> pd.DataFrame:
    """Build a large enough fixture to test mood-list overlap targets."""

    rows: list[dict[str, object]] = []
    mood_specs = {
        "workout": {
            "name": "Gym Power Run",
            "genre": "workout dance",
            "energy": 0.92,
            "danceability": 0.88,
            "valence": 0.66,
            "tempo": 150.0,
            "acousticness": 0.05,
            "instrumentalness": 0.0,
            "popularity": 0.55,
        },
        "calm": {
            "name": "Soft Ambient Piano",
            "genre": "ambient acoustic",
            "energy": 0.16,
            "danceability": 0.18,
            "valence": 0.45,
            "tempo": 72.0,
            "acousticness": 0.92,
            "instrumentalness": 0.50,
            "popularity": 0.30,
        },
        "happy": {
            "name": "Sunshine Smile",
            "genre": "happy pop",
            "energy": 0.70,
            "danceability": 0.72,
            "valence": 0.94,
            "tempo": 120.0,
            "acousticness": 0.18,
            "instrumentalness": 0.0,
            "popularity": 0.52,
        },
        "melancholic": {
            "name": "Lonely Blue Rain",
            "genre": "sad acoustic",
            "energy": 0.34,
            "danceability": 0.22,
            "valence": 0.12,
            "tempo": 78.0,
            "acousticness": 0.82,
            "instrumentalness": 0.15,
            "popularity": 0.38,
        },
    }
    for mood_name, spec in mood_specs.items():
        for index in range(10):
            rows.append(
                {
                    "track_id": f"{mood_name}_{index}",
                    "track_name": f"{spec['name']} {index}",
                    "artist_name": f"{mood_name.title()} Artist {index}",
                    "primary_artist_name": f"{mood_name.title()} Artist {index}",
                    "artist_genres": spec["genre"],
                    "catalog_popularity": float(spec["popularity"]) + (index * 0.005),
                    "catalog_novelty": 1.0 - float(spec["popularity"]),
                    "candidate_sources": "recent track search match",
                    "ranking_mode": "audio-feature-based",
                    "energy": spec["energy"],
                    "danceability": spec["danceability"],
                    "valence": spec["valence"],
                    "tempo": spec["tempo"],
                    "acousticness": spec["acousticness"],
                    "instrumentalness": spec["instrumentalness"],
                }
            )
    return pd.DataFrame(rows)


def build_scored_recommendations(candidate_catalog: pd.DataFrame) -> list[HybridRecommendation]:
    """Build base-ranked recommendations that intentionally favor the input order."""

    recommendations: list[HybridRecommendation] = []
    total = len(candidate_catalog)
    for index, row in enumerate(candidate_catalog.itertuples(index=False)):
        score = float(total - index)
        score_breakdown = HybridScoreBreakdown(
            collaborative_score=0.0,
            content_score=score,
            novelty_score=0.0,
            popularity_prior=0.0,
            discovery_score=0.0,
            final_score=score,
        )
        recommendations.append(
            HybridRecommendation(
                item_id=str(row.track_id),
                score=score,
                source="hybrid",
                track_name=str(row.track_name),
                artist_name=str(row.artist_name),
                score_breakdown=score_breakdown,
                used_cold_start_fallback=False,
            )
        )
    return recommendations


def build_control_candidate_catalog() -> pd.DataFrame:
    """Build a controlled candidate set with familiar, discovery, workout, and calm tracks."""

    return pd.DataFrame(
        [
            {
                "track_id": "familiar_1",
                "track_name": "Aurora Hit",
                "artist_name": "Aurora Lane",
                "primary_artist_name": "Aurora Lane",
                "artist_genres": "indie pop",
                "catalog_popularity": 0.95,
                "catalog_novelty": 0.05,
                "candidate_sources": "recent artist top track",
                "ranking_mode": "audio-feature-based",
                "energy": 0.62,
                "danceability": 0.60,
                "valence": 0.72,
                "tempo": 118.0,
                "acousticness": 0.20,
                "instrumentalness": 0.0,
            },
            {
                "track_id": "familiar_2",
                "track_name": "Aurora Single",
                "artist_name": "Aurora Lane",
                "primary_artist_name": "Aurora Lane",
                "artist_genres": "indie pop",
                "catalog_popularity": 0.88,
                "catalog_novelty": 0.12,
                "candidate_sources": "recent artist top track",
                "ranking_mode": "audio-feature-based",
                "energy": 0.58,
                "danceability": 0.55,
                "valence": 0.64,
                "tempo": 112.0,
                "acousticness": 0.25,
                "instrumentalness": 0.0,
            },
            {
                "track_id": "familiar_3",
                "track_name": "Aurora Fan Favorite",
                "artist_name": "Aurora Lane",
                "primary_artist_name": "Aurora Lane",
                "artist_genres": "indie pop",
                "catalog_popularity": 0.80,
                "catalog_novelty": 0.20,
                "candidate_sources": "recent artist top track",
                "ranking_mode": "audio-feature-based",
                "energy": 0.52,
                "danceability": 0.54,
                "valence": 0.62,
                "tempo": 105.0,
                "acousticness": 0.35,
                "instrumentalness": 0.0,
            },
            {
                "track_id": "discovery_1",
                "track_name": "Fresh Search Run",
                "artist_name": "New Artist",
                "primary_artist_name": "New Artist",
                "artist_genres": "dance pop",
                "catalog_popularity": 0.14,
                "catalog_novelty": 0.86,
                "candidate_sources": "recent track search match",
                "ranking_mode": "audio-feature-based",
                "energy": 0.92,
                "danceability": 0.90,
                "valence": 0.70,
                "tempo": 150.0,
                "acousticness": 0.05,
                "instrumentalness": 0.0,
            },
            {
                "track_id": "discovery_2",
                "track_name": "Hidden Club Lift",
                "artist_name": "Other Artist",
                "primary_artist_name": "Other Artist",
                "artist_genres": "electronic",
                "catalog_popularity": 0.18,
                "catalog_novelty": 0.82,
                "candidate_sources": "genre search match",
                "ranking_mode": "audio-feature-based",
                "energy": 0.88,
                "danceability": 0.84,
                "valence": 0.58,
                "tempo": 142.0,
                "acousticness": 0.08,
                "instrumentalness": 0.0,
            },
            {
                "track_id": "calm_1",
                "track_name": "Quiet Study Piano",
                "artist_name": "Calm Artist",
                "primary_artist_name": "Calm Artist",
                "artist_genres": "ambient acoustic",
                "catalog_popularity": 0.28,
                "catalog_novelty": 0.72,
                "candidate_sources": "genre search match",
                "ranking_mode": "audio-feature-based",
                "energy": 0.18,
                "danceability": 0.18,
                "valence": 0.42,
                "tempo": 72.0,
                "acousticness": 0.90,
                "instrumentalness": 0.45,
            },
        ]
    )


def build_spotify_real_profile(snapshot: ListeningHistorySnapshot, mood_label: str) -> Any:
    """Build a minimal real Spotify profile for direct ranking tests."""

    from app.demo_data import DemoUserProfile

    return DemoUserProfile(
        user_id=f"spotify_real::{snapshot.user_id}",
        display_name="Spotify Real",
        summary="Test profile",
        seed_track_ids=snapshot.seed_track_ids,
        preferred_mood=mood_label,
    )


def build_recommendation(track_id: str, track_name: str, artist_name: str) -> HybridRecommendation:
    """Build a same-score recommendation for ranking-control tests."""

    score_breakdown = HybridScoreBreakdown(
        collaborative_score=0.0,
        content_score=1.0,
        novelty_score=0.0,
        popularity_prior=0.0,
        discovery_score=0.0,
        final_score=1.0,
    )
    return HybridRecommendation(
        item_id=track_id,
        score=1.0,
        source="hybrid",
        track_name=track_name,
        artist_name=artist_name,
        score_breakdown=score_breakdown,
        used_cold_start_fallback=False,
    )
