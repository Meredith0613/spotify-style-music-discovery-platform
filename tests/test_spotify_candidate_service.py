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
from services.spotify_candidate_service import RecommendationBucket, SpotifyCandidateService, SpotifyCandidateSet
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
    assert result.candidate_set.debug_summary["search_candidate_count"] == 1
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
