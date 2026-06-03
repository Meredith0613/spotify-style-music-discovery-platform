"""Offline evaluation pipeline for comparing recommender variants."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.demo_data import build_demo_interactions, build_demo_track_catalog
from app.demo_service import DemoAppService
from features.feature_builder import FeatureBuilder
from models.als_recommender import ALSRecommender
from models.collaborative_recommender import CollaborativeRecommender
from models.content_recommender import ContentRecommender
from models.base_recommender import RecommendationResult
from models.hybrid_recommender import HybridRecommender
from models.interaction_matrix import InteractionMatrixArtifacts, InteractionMatrixBuilder
from models.track_embedding_model import TrackEmbeddingModel
from services.advanced_hybrid_ranking_service import AdvancedHybridRankingService, AdvancedHybridWeights
from services.spotify_recommendation_adapter import SpotifyRecommendationAdapter
from services.spotify_reranking_service import SpotifyRerankingService
from services.user_profile_service import ListeningHistorySnapshot, RecentTrackSummary

from .ranking_evaluator import RankingEvaluator


@dataclass(slots=True)
class OfflineEvaluationSplit:
    """Store deterministic train/test interaction splits for offline evaluation."""

    train_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    train_seed_track_ids_by_user: dict[str, list[str]]
    test_track_ids_by_user: dict[str, set[str]]

    @property
    def user_ids(self) -> list[str]:
        """Return evaluated user IDs in stable order."""

        return sorted(self.test_track_ids_by_user)


@dataclass(slots=True)
class OfflineEvaluationResult:
    """Store comparison outputs from the offline evaluation pipeline."""

    split: OfflineEvaluationSplit
    comparison_table: pd.DataFrame
    recommendation_lists_by_model: dict[str, dict[str, list[object]]] = field(default_factory=dict)


@dataclass(slots=True)
class OfflineEvaluator:
    """Evaluate content, collaborative, hybrid, and reranked hybrid variants."""

    track_catalog: pd.DataFrame
    interactions: pd.DataFrame
    hybrid_weights: dict[str, float]
    ranking_evaluator: RankingEvaluator = field(default_factory=RankingEvaluator)
    interaction_matrix_builder: InteractionMatrixBuilder = field(default_factory=InteractionMatrixBuilder)
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    spotify_recommendation_adapter: SpotifyRecommendationAdapter = field(default_factory=SpotifyRecommendationAdapter)
    spotify_reranking_service: SpotifyRerankingService = field(default_factory=SpotifyRerankingService)
    k: int = 3
    holdout_count: int = 1
    min_interactions_per_user: int = 2

    @classmethod
    def from_demo_data(
        cls,
        *,
        k: int = 3,
        holdout_count: int = 1,
    ) -> "OfflineEvaluator":
        """Build the evaluator from the synthetic demo dataset."""

        demo_service = DemoAppService()
        return cls(
            track_catalog=demo_service.track_catalog.copy(),
            interactions=demo_service.interactions.copy(),
            hybrid_weights=demo_service._build_hybrid_weights(0.35),
            k=k,
            holdout_count=holdout_count,
        )

    def build_interaction_split(self) -> OfflineEvaluationSplit:
        """Create a deterministic user-level holdout split from interactions."""

        prepared_interactions = self.interactions.copy()
        if prepared_interactions.empty:
            return OfflineEvaluationSplit(
                train_interactions=pd.DataFrame(columns=self.interactions.columns),
                test_interactions=pd.DataFrame(columns=self.interactions.columns),
                train_seed_track_ids_by_user={},
                test_track_ids_by_user={},
            )

        train_rows: list[dict[str, object]] = []
        test_rows: list[dict[str, object]] = []
        train_seed_track_ids_by_user: dict[str, list[str]] = {}
        test_track_ids_by_user: dict[str, set[str]] = {}

        for user_id, user_frame in prepared_interactions.groupby("user_id", sort=True):
            sorted_user_frame = user_frame.sort_values(
                ["interaction_strength", "track_id"],
                ascending=[False, True],
                kind="stable",
            ).reset_index(drop=True)
            if len(sorted_user_frame) < self.min_interactions_per_user:
                continue

            effective_holdout_count = min(self.holdout_count, len(sorted_user_frame) - 1)
            if effective_holdout_count <= 0:
                continue

            test_frame = sorted_user_frame.iloc[:effective_holdout_count].copy()
            train_frame = sorted_user_frame.iloc[effective_holdout_count:].copy()
            if train_frame.empty:
                continue

            train_rows.extend(train_frame.to_dict(orient="records"))
            test_rows.extend(test_frame.to_dict(orient="records"))
            train_seed_track_ids_by_user[str(user_id)] = train_frame["track_id"].astype(str).tolist()
            test_track_ids_by_user[str(user_id)] = set(test_frame["track_id"].astype(str).tolist())

        return OfflineEvaluationSplit(
            train_interactions=pd.DataFrame(train_rows, columns=self.interactions.columns),
            test_interactions=pd.DataFrame(test_rows, columns=self.interactions.columns),
            train_seed_track_ids_by_user=train_seed_track_ids_by_user,
            test_track_ids_by_user=test_track_ids_by_user,
        )

    def evaluate(self) -> OfflineEvaluationResult:
        """Run the full offline comparison across all supported model variants."""

        split = self.build_interaction_split()
        if split.train_interactions.empty or not split.user_ids:
            return OfflineEvaluationResult(
                split=split,
                comparison_table=pd.DataFrame(
                    columns=["model", f"precision@{self.k}", f"recall@{self.k}", f"ndcg@{self.k}", "evaluated_users"]
                ),
            )

        content_recommender = self._build_content_recommender()
        interaction_artifacts = self.interaction_matrix_builder.build(split.train_interactions)
        collaborative_recommender = self._build_collaborative_recommender_from_artifacts(interaction_artifacts)
        als_recommender = ALSRecommender(n_factors=8, n_iterations=6, random_state=42).fit(interaction_artifacts)
        embedding_model = TrackEmbeddingModel(embedding_dim=8, window_size=2, random_state=42).fit(split.train_interactions)
        advanced_hybrid_with_als = AdvancedHybridRankingService(
            weights=AdvancedHybridWeights(hybrid=1.0, als=0.35, embedding=0.0)
        )
        advanced_hybrid_with_als_embedding = AdvancedHybridRankingService(
            weights=AdvancedHybridWeights(hybrid=1.0, als=0.3, embedding=0.3)
        )
        hybrid_recommender = self._build_hybrid_recommender(
            collaborative_recommender=collaborative_recommender,
            content_recommender=content_recommender,
            train_seed_track_ids_by_user=split.train_seed_track_ids_by_user,
        )
        candidate_track_ids = self.track_catalog["track_id"].astype(str).tolist()

        recommendation_lists_by_model: dict[str, dict[str, list[object]]] = {
            "content_only": {},
            "collaborative_only": {},
            "hybrid": {},
            "ALS_only": {},
            "Word2Vec_similarity_only": {},
            "hybrid_plus_ALS": {},
            "hybrid_plus_ALS_Word2Vec": {},
            "hybrid_spotify_reranked": {},
        }
        metrics_by_model: dict[str, list[dict[str, float]]] = {model_name: [] for model_name in recommendation_lists_by_model}

        for user_id in split.user_ids:
            relevant_track_ids = split.test_track_ids_by_user[user_id]
            train_seed_track_ids = split.train_seed_track_ids_by_user.get(user_id, [])
            if not train_seed_track_ids or not relevant_track_ids:
                continue

            full_hybrid_recommendations = hybrid_recommender.recommend(
                user_id=user_id,
                candidate_item_ids=candidate_track_ids,
                k=len(candidate_track_ids),
            )
            als_scores = als_recommender.score_candidates(
                user_id=user_id,
                candidate_track_ids=candidate_track_ids,
            )
            embedding_scores = embedding_model.score_candidate_similarity(
                seed_track_ids=train_seed_track_ids,
                candidate_track_ids=candidate_track_ids,
            )
            model_recommendations = {
                "content_only": content_recommender.recommend_from_seed_tracks(
                    seed_track_ids=train_seed_track_ids,
                    seen_track_ids=train_seed_track_ids,
                    k=self.k,
                ),
                "collaborative_only": collaborative_recommender.recommend_for_user(
                    user_id=user_id,
                    k=self.k,
                    candidate_track_ids=candidate_track_ids,
                ),
                "hybrid": hybrid_recommender.recommend(
                    user_id=user_id,
                    candidate_item_ids=candidate_track_ids,
                    k=self.k,
                ),
                "ALS_only": als_recommender.recommend_for_user(
                    user_id=user_id,
                    k=self.k,
                    candidate_track_ids=candidate_track_ids,
                ),
                "Word2Vec_similarity_only": self._rank_score_map(
                    score_map=embedding_scores,
                    seen_track_ids=set(train_seed_track_ids),
                    k=self.k,
                    source="word2vec_similarity",
                ),
                "hybrid_plus_ALS": advanced_hybrid_with_als.rerank(
                    full_hybrid_recommendations,
                    als_scores=als_scores,
                    k=self.k,
                ),
                "hybrid_plus_ALS_Word2Vec": advanced_hybrid_with_als_embedding.rerank(
                    full_hybrid_recommendations,
                    als_scores=als_scores,
                    embedding_scores=embedding_scores,
                    k=self.k,
                ),
            }
            model_recommendations["hybrid_spotify_reranked"] = self._rerank_hybrid_recommendations(
                user_id=user_id,
                recommendations=model_recommendations["hybrid"],
                train_seed_track_ids=train_seed_track_ids,
            )

            for model_name, recommendations in model_recommendations.items():
                recommendation_lists_by_model[model_name][user_id] = recommendations
                metrics_by_model[model_name].append(
                    {
                        f"precision@{self.k}": self.ranking_evaluator.precision_at_k(
                            recommendations=recommendations,
                            relevant_item_ids=relevant_track_ids,
                            k=self.k,
                        ),
                        f"recall@{self.k}": self.ranking_evaluator.recall_at_k(
                            recommendations=recommendations,
                            relevant_item_ids=relevant_track_ids,
                            k=self.k,
                        ),
                        f"ndcg@{self.k}": self.ranking_evaluator.ndcg_at_k(
                            recommendations=recommendations,
                            relevant_item_ids=relevant_track_ids,
                            k=self.k,
                        ),
                    }
                )

        comparison_rows = []
        for model_name, model_metrics in metrics_by_model.items():
            if not model_metrics:
                comparison_rows.append(
                    {
                        "model": model_name,
                        f"precision@{self.k}": 0.0,
                        f"recall@{self.k}": 0.0,
                        f"ndcg@{self.k}": 0.0,
                        "evaluated_users": 0,
                    }
                )
                continue

            metrics_frame = pd.DataFrame(model_metrics)
            comparison_rows.append(
                {
                    "model": model_name,
                    f"precision@{self.k}": float(metrics_frame[f"precision@{self.k}"].mean()),
                    f"recall@{self.k}": float(metrics_frame[f"recall@{self.k}"].mean()),
                    f"ndcg@{self.k}": float(metrics_frame[f"ndcg@{self.k}"].mean()),
                    "evaluated_users": len(model_metrics),
                }
            )

        comparison_table = pd.DataFrame(comparison_rows).sort_values(
            [f"ndcg@{self.k}", f"precision@{self.k}", "model"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        numeric_columns = [column_name for column_name in comparison_table.columns if "@" in column_name]
        comparison_table[numeric_columns] = comparison_table[numeric_columns].round(3)

        return OfflineEvaluationResult(
            split=split,
            comparison_table=comparison_table,
            recommendation_lists_by_model=recommendation_lists_by_model,
        )

    def _build_content_recommender(self) -> ContentRecommender:
        """Train the content recommender on the full demo catalog."""

        feature_artifacts = self.feature_builder.create_model_ready_feature_matrix(self.track_catalog)
        return ContentRecommender(
            feature_artifacts=feature_artifacts,
            track_catalog=self.track_catalog,
        )

    def _build_collaborative_recommender(
        self,
        train_interactions: pd.DataFrame,
    ) -> CollaborativeRecommender:
        """Train the collaborative recommender on training interactions only."""

        interaction_artifacts = self.interaction_matrix_builder.build(train_interactions)
        return self._build_collaborative_recommender_from_artifacts(interaction_artifacts)

    def _build_collaborative_recommender_from_artifacts(
        self,
        interaction_artifacts: InteractionMatrixArtifacts,
    ) -> CollaborativeRecommender:
        """Train the collaborative recommender from pre-built matrix artifacts."""

        return CollaborativeRecommender().fit(interaction_artifacts)

    def _rank_score_map(
        self,
        *,
        score_map: dict[str, float],
        seen_track_ids: set[str],
        k: int,
        source: str,
    ) -> list[RecommendationResult]:
        """Rank a plain score map with the same seen-track filtering as recommenders."""

        filtered_scores = {
            str(track_id): float(score)
            for track_id, score in score_map.items()
            if str(track_id) not in seen_track_ids
        }
        ranked_items = sorted(filtered_scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            RecommendationResult(item_id=track_id, score=score, source=source)
            for track_id, score in ranked_items[:k]
        ]

    def _build_hybrid_recommender(
        self,
        collaborative_recommender: CollaborativeRecommender,
        content_recommender: ContentRecommender,
        train_seed_track_ids_by_user: dict[str, list[str]],
    ) -> HybridRecommender:
        """Construct the hybrid recommender for offline evaluation."""

        popularity_scores = {
            str(row.track_id): float(row.popularity) / 100.0
            for row in self.track_catalog.itertuples(index=False)
        }
        novelty_scores = {
            track_id: 1.0 - popularity_score
            for track_id, popularity_score in popularity_scores.items()
        }
        return HybridRecommender(
            collaborative_recommender=collaborative_recommender,
            content_recommender=content_recommender,
            weights=self.hybrid_weights,
            popularity_scores=popularity_scores,
            novelty_scores=novelty_scores,
            user_seed_track_ids_by_user=train_seed_track_ids_by_user,
            track_catalog=self.track_catalog,
        )

    def _rerank_hybrid_recommendations(
        self,
        user_id: str,
        recommendations: list[object],
        train_seed_track_ids: list[str],
    ) -> list[object]:
        """Apply Spotify-aware reranking using train interactions as pseudo recent history."""

        if not recommendations or not train_seed_track_ids:
            return recommendations

        listening_history_snapshot = self._build_simulated_listening_history_snapshot(
            user_id=user_id,
            train_seed_track_ids=train_seed_track_ids,
        )
        spotify_context = self.spotify_recommendation_adapter.build_context(
            listening_history_snapshot=listening_history_snapshot,
            demo_track_catalog=self.track_catalog,
        )
        if spotify_context is None:
            return recommendations

        reranking_result = self.spotify_reranking_service.rerank_recommendations(
            spotify_context=spotify_context,
            recommendations=recommendations,
            listening_history_snapshot=listening_history_snapshot,
            demo_track_catalog=self.track_catalog,
        )
        if not reranking_result.applied:
            return recommendations
        return reranking_result.recommendations

    def _build_simulated_listening_history_snapshot(
        self,
        user_id: str,
        train_seed_track_ids: list[str],
    ) -> ListeningHistorySnapshot:
        """Simulate recent Spotify listening from train interactions for reranking evaluation."""

        selected_tracks = self.track_catalog.loc[
            self.track_catalog["track_id"].astype(str).isin(train_seed_track_ids)
        ].copy()
        ranked_track_ids = [
            track_id
            for track_id in train_seed_track_ids
            if track_id in set(selected_tracks["track_id"].astype(str))
        ]
        track_rows_by_id = {
            str(row.track_id): row
            for row in selected_tracks.itertuples(index=False)
        }

        recent_tracks: list[RecentTrackSummary] = []
        track_level_rows: list[dict[str, object]] = []
        interaction_rows: list[dict[str, object]] = []
        for position, track_id in enumerate(ranked_track_ids):
            track_row = track_rows_by_id[track_id]
            spotify_track_id = f"offline::{track_id}"
            recent_tracks.append(
                RecentTrackSummary(
                    track_id=spotify_track_id,
                    track_name=str(track_row.track_name),
                    artist_name=str(track_row.artist_name),
                    played_at=f"2024-01-01T00:{position:02d}:00Z",
                )
            )
            track_level_row = {
                column_name: getattr(track_row, column_name)
                for column_name in selected_tracks.columns
            }
            track_level_row["track_id"] = spotify_track_id
            track_level_rows.append(track_level_row)
            interaction_rows.append(
                {
                    "user_id": user_id,
                    "track_id": spotify_track_id,
                    "interaction_strength": float(max(len(ranked_track_ids) - position, 1)),
                }
            )

        return ListeningHistorySnapshot(
            user_id=user_id,
            display_name=user_id.replace("_", " ").title(),
            recent_tracks=recent_tracks,
            track_level_frame=pd.DataFrame(track_level_rows),
            interaction_frame=pd.DataFrame(interaction_rows),
            seed_track_ids=[recent_track.track_id for recent_track in recent_tracks],
        )


def run_demo_offline_evaluation(
    *,
    k: int = 3,
    holdout_count: int = 1,
) -> OfflineEvaluationResult:
    """Run the default offline evaluation pipeline on synthetic demo interactions."""

    evaluator = OfflineEvaluator.from_demo_data(k=k, holdout_count=holdout_count)
    return evaluator.evaluate()
