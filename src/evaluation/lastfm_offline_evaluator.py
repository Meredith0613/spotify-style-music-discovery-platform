"""Offline evaluation path for processed Last.fm-style datasets.

Usage:
    PYTHONPATH=src python -c "from evaluation.lastfm_offline_evaluator import run_lastfm_offline_evaluation; result = run_lastfm_offline_evaluation('data/processed/lastfm_interactions.csv', 'data/processed/lastfm_catalog.csv'); print(result.comparison_table.to_string(index=False))"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.demo_service import DemoAppService
from models.als_recommender import ALSRecommender
from models.track_embedding_model import TrackEmbeddingModel
from services.advanced_hybrid_ranking_service import AdvancedHybridRankingService, AdvancedHybridWeights

from .offline_evaluator import OfflineEvaluationResult, OfflineEvaluationSplit, OfflineEvaluator


def compute_lastfm_coverage_statistics(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    min_user_interactions: int,
) -> dict[str, float | int]:
    """Compute dataset coverage and interaction-density statistics."""

    if interactions.empty:
        return {
            "interaction_count": 0,
            "user_count": 0,
            "unique_tracks_in_interactions": 0,
            "unique_tracks_in_catalog": int(catalog["track_id"].nunique()) if "track_id" in catalog.columns else 0,
            "average_interactions_per_user": 0.0,
            "median_interactions_per_user": 0.0,
            "average_users_per_track": 0.0,
            "median_users_per_track": 0.0,
            "tracks_with_more_than_one_listener_pct": 0.0,
            "possible_interactions": 0,
            "observed_interactions": 0,
            "matrix_density": 0.0,
            "matrix_sparsity": 1.0,
            "evaluated_users": 0,
            "average_train_interactions_per_evaluated_user": 0.0,
        }

    prepared = interactions.copy()
    prepared["user_id"] = prepared["user_id"].astype(str)
    prepared["track_id"] = prepared["track_id"].astype(str)
    interaction_count = int(len(prepared))
    user_count = int(prepared["user_id"].nunique())
    track_count = int(prepared["track_id"].nunique())
    catalog_track_count = int(catalog["track_id"].nunique()) if "track_id" in catalog.columns else 0
    interactions_per_user = prepared.groupby("user_id")["track_id"].count()
    users_per_track = prepared.groupby("track_id")["user_id"].nunique()
    possible_interactions = int(user_count * track_count)
    observed_interactions = int(prepared.drop_duplicates(["user_id", "track_id"]).shape[0])
    matrix_density = observed_interactions / possible_interactions if possible_interactions else 0.0
    evaluated_user_counts = interactions_per_user.loc[interactions_per_user >= min_user_interactions]

    return {
        "interaction_count": interaction_count,
        "user_count": user_count,
        "unique_tracks_in_interactions": track_count,
        "unique_tracks_in_catalog": catalog_track_count,
        "average_interactions_per_user": float(interactions_per_user.mean()) if not interactions_per_user.empty else 0.0,
        "median_interactions_per_user": float(interactions_per_user.median()) if not interactions_per_user.empty else 0.0,
        "average_users_per_track": float(users_per_track.mean()) if not users_per_track.empty else 0.0,
        "median_users_per_track": float(users_per_track.median()) if not users_per_track.empty else 0.0,
        "tracks_with_more_than_one_listener_pct": float((users_per_track > 1).mean() * 100.0)
        if not users_per_track.empty
        else 0.0,
        "possible_interactions": possible_interactions,
        "observed_interactions": observed_interactions,
        "matrix_density": float(matrix_density),
        "matrix_sparsity": float(1.0 - matrix_density),
        "evaluated_users": int(len(evaluated_user_counts)),
        "average_train_interactions_per_evaluated_user": float((evaluated_user_counts - 1).mean())
        if not evaluated_user_counts.empty
        else 0.0,
    }


@dataclass(slots=True)
class LastfmOfflineEvaluator(OfflineEvaluator):
    """Evaluate the existing recommender stack on processed Last.fm-style data.

    This wrapper preserves the synthetic evaluation flow and reuses the shared
    metrics and recommender builders. The main Last.fm-specific change is the
    split logic: when timestamps exist, the evaluator holds out the most recent
    user-track interaction; otherwise it falls back to interaction strength.
    """

    max_candidate_tracks: int = 1000

    @classmethod
    def from_processed_data(
        cls,
        *,
        interactions_path: str,
        catalog_path: str,
        k: int = 10,
        min_user_interactions: int = 5,
        holdout_count: int = 1,
    ) -> "LastfmOfflineEvaluator":
        """Build the evaluator from processed Last.fm interactions and catalog files."""

        interactions = pd.read_csv(interactions_path)
        track_catalog = pd.read_csv(catalog_path)
        hybrid_weights = DemoAppService()._build_hybrid_weights(0.35)
        return cls(
            track_catalog=track_catalog,
            interactions=interactions,
            hybrid_weights=hybrid_weights,
            k=k,
            holdout_count=holdout_count,
            min_interactions_per_user=min_user_interactions,
        )

    def build_interaction_split(self) -> OfflineEvaluationSplit:
        """Create a user-level holdout split that prefers the most recent track."""

        prepared_interactions = self.interactions.copy()
        if prepared_interactions.empty:
            return OfflineEvaluationSplit(
                train_interactions=pd.DataFrame(columns=self.interactions.columns),
                test_interactions=pd.DataFrame(columns=self.interactions.columns),
                train_seed_track_ids_by_user={},
                test_track_ids_by_user={},
            )

        timestamp_column = self._resolve_timestamp_column(prepared_interactions)
        if timestamp_column is not None:
            prepared_interactions["_sort_timestamp"] = pd.to_datetime(
                prepared_interactions[timestamp_column],
                utc=True,
                errors="coerce",
            )
        else:
            prepared_interactions["_sort_timestamp"] = pd.NaT

        train_rows: list[dict[str, object]] = []
        test_rows: list[dict[str, object]] = []
        train_seed_track_ids_by_user: dict[str, list[str]] = {}
        test_track_ids_by_user: dict[str, set[str]] = {}

        for user_id, user_frame in prepared_interactions.groupby("user_id", sort=True):
            sorted_user_frame = self._sort_user_frame_for_holdout(user_frame, timestamp_column)
            if len(sorted_user_frame) < self.min_interactions_per_user:
                continue

            effective_holdout_count = min(self.holdout_count, len(sorted_user_frame) - 1)
            if effective_holdout_count <= 0:
                continue

            test_frame = sorted_user_frame.iloc[:effective_holdout_count].copy()
            train_frame = sorted_user_frame.iloc[effective_holdout_count:].copy()
            if train_frame.empty:
                continue

            test_frame = test_frame.drop(columns=["_sort_timestamp"], errors="ignore")
            train_frame = train_frame.drop(columns=["_sort_timestamp"], errors="ignore")

            train_rows.extend(train_frame.to_dict(orient="records"))
            test_rows.extend(test_frame.to_dict(orient="records"))
            train_seed_track_ids_by_user[str(user_id)] = train_frame["track_id"].astype(str).tolist()
            test_track_ids_by_user[str(user_id)] = set(test_frame["track_id"].astype(str).tolist())

        split_columns = [column_name for column_name in self.interactions.columns]
        return OfflineEvaluationSplit(
            train_interactions=pd.DataFrame(train_rows, columns=split_columns),
            test_interactions=pd.DataFrame(test_rows, columns=split_columns),
            train_seed_track_ids_by_user=train_seed_track_ids_by_user,
            test_track_ids_by_user=test_track_ids_by_user,
        )

    def evaluate(self) -> OfflineEvaluationResult:
        """Run Last.fm evaluation with baseline and optional advanced model rows."""

        split = self.build_interaction_split()
        if split.train_interactions.empty or not split.user_ids:
            return OfflineEvaluationResult(
                split=split,
                comparison_table=pd.DataFrame(
                    columns=["model", f"precision@{self.k}", f"recall@{self.k}", f"ndcg@{self.k}", "evaluated_users"]
                ),
            )

        content_recommender = self._build_content_recommender()
        candidate_track_ids = self._build_base_candidate_track_ids()
        benchmark_train_interactions = self._filter_interactions_to_candidate_pool(
            split.train_interactions,
            candidate_track_ids,
        )
        interaction_artifacts = self.interaction_matrix_builder.build(benchmark_train_interactions)
        collaborative_recommender = self._build_collaborative_recommender_from_artifacts(interaction_artifacts)
        als_recommender = ALSRecommender(n_factors=8, n_iterations=2, random_state=42).fit(interaction_artifacts)
        embedding_model = TrackEmbeddingModel(embedding_dim=8, window_size=2, random_state=42).fit(benchmark_train_interactions)
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
        recommendation_lists_by_model: dict[str, dict[str, list[object]]] = {
            "content_only": {},
            "collaborative_only": {},
            "hybrid": {},
            "ALS_only": {},
            "Word2Vec_similarity_only": {},
            "hybrid_plus_ALS": {},
            "hybrid_plus_ALS_Word2Vec": {},
        }
        metrics_by_model: dict[str, list[dict[str, float]]] = {model_name: [] for model_name in recommendation_lists_by_model}

        for user_id in split.user_ids:
            relevant_track_ids = split.test_track_ids_by_user[user_id]
            train_seed_track_ids = split.train_seed_track_ids_by_user.get(user_id, [])
            if not train_seed_track_ids or not relevant_track_ids:
                continue

            user_candidate_track_ids = self._build_user_candidate_track_ids(
                base_candidate_track_ids=candidate_track_ids,
                train_seed_track_ids=train_seed_track_ids,
                relevant_track_ids=relevant_track_ids,
            )
            full_hybrid_recommendations = hybrid_recommender.recommend(
                user_id=user_id,
                candidate_item_ids=user_candidate_track_ids,
                k=len(user_candidate_track_ids),
            )
            als_scores = als_recommender.score_candidates(
                user_id=user_id,
                candidate_track_ids=user_candidate_track_ids,
            )
            embedding_scores = embedding_model.score_candidate_similarity(
                seed_track_ids=train_seed_track_ids,
                candidate_track_ids=user_candidate_track_ids,
            )
            content_scores = content_recommender.score_candidates_from_seed_tracks(
                seed_track_ids=train_seed_track_ids,
                seen_track_ids=train_seed_track_ids,
                candidate_track_ids=user_candidate_track_ids,
            )
            model_recommendations = {
                "content_only": self._rank_score_map(
                    score_map=content_scores,
                    seen_track_ids=train_seed_track_ids,
                    k=self.k,
                    source="content",
                ),
                "collaborative_only": collaborative_recommender.recommend_for_user(
                    user_id=user_id,
                    k=self.k,
                    candidate_track_ids=user_candidate_track_ids,
                ),
                "hybrid": hybrid_recommender.recommend(
                    user_id=user_id,
                    candidate_item_ids=user_candidate_track_ids,
                    k=self.k,
                ),
                "ALS_only": als_recommender.recommend_for_user(
                    user_id=user_id,
                    k=self.k,
                    candidate_track_ids=user_candidate_track_ids,
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

    def _resolve_timestamp_column(self, interactions: pd.DataFrame) -> str | None:
        """Pick the most useful timestamp column for chronological holdout selection."""

        for column_name in ("last_timestamp", "first_timestamp", "timestamp"):
            if column_name in interactions.columns:
                return column_name
        return None

    def _sort_user_frame_for_holdout(self, user_frame: pd.DataFrame, timestamp_column: str | None) -> pd.DataFrame:
        """Sort one user's history using timestamp first, then deterministic fallbacks."""

        if timestamp_column is not None and user_frame["_sort_timestamp"].notna().any():
            return user_frame.sort_values(
                ["_sort_timestamp", "interaction_strength", "track_id"],
                ascending=[False, False, True],
                kind="stable",
            ).reset_index(drop=True)
        return user_frame.sort_values(
            ["interaction_strength", "track_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

    def _build_base_candidate_track_ids(self) -> list[str]:
        """Build a deterministic Last.fm candidate pool for practical benchmarking."""

        catalog = self.track_catalog.copy()
        if "track_id" not in catalog.columns:
            return []
        catalog["track_id"] = catalog["track_id"].astype(str)
        if "popularity" in catalog.columns:
            catalog["_sort_popularity"] = pd.to_numeric(catalog["popularity"], errors="coerce").fillna(0.0)
            catalog = catalog.sort_values(["_sort_popularity", "track_id"], ascending=[False, True], kind="stable")
        else:
            catalog = catalog.sort_values("track_id", kind="stable")
        candidate_limit = max(int(self.max_candidate_tracks), self.k)
        return catalog["track_id"].drop_duplicates().head(candidate_limit).tolist()

    def _build_user_candidate_track_ids(
        self,
        *,
        base_candidate_track_ids: list[str],
        train_seed_track_ids: list[str],
        relevant_track_ids: set[str],
    ) -> list[str]:
        """Include train and held-out tracks in the capped evaluation pool."""

        candidate_ids: list[str] = []
        seen_track_ids: set[str] = set()
        for track_id in [*base_candidate_track_ids, *train_seed_track_ids, *sorted(relevant_track_ids)]:
            normalized_track_id = str(track_id)
            if normalized_track_id in seen_track_ids:
                continue
            seen_track_ids.add(normalized_track_id)
            candidate_ids.append(normalized_track_id)
        return candidate_ids

    def _filter_interactions_to_candidate_pool(
        self,
        interactions: pd.DataFrame,
        candidate_track_ids: list[str],
    ) -> pd.DataFrame:
        """Restrict expensive collaborative benchmarks to the candidate pool."""

        if interactions.empty or not candidate_track_ids:
            return interactions
        candidate_track_id_set = set(candidate_track_ids)
        filtered = interactions.loc[
            interactions["track_id"].astype(str).isin(candidate_track_id_set)
        ].copy()
        return filtered if not filtered.empty else interactions


def run_lastfm_offline_evaluation(
    interactions_path: str,
    catalog_path: str,
    *,
    k: int = 10,
    min_user_interactions: int = 5,
    holdout_count: int = 1,
) -> OfflineEvaluationResult:
    """Run the Last.fm offline evaluation flow on processed inputs."""

    evaluator = LastfmOfflineEvaluator.from_processed_data(
        interactions_path=interactions_path,
        catalog_path=catalog_path,
        k=k,
        min_user_interactions=min_user_interactions,
        holdout_count=holdout_count,
    )
    return evaluator.evaluate()


def generate_lastfm_evaluation_report(
    *,
    result: OfflineEvaluationResult,
    coverage_statistics: dict[str, float | int],
    min_user_interactions: int,
    holdout_count: int,
    k: int,
    max_candidate_tracks: int | None = None,
) -> str:
    """Render a Markdown Last.fm evaluation report."""

    metrics_table = _render_metrics_table(result.comparison_table, k)
    coverage_table = _render_coverage_table(coverage_statistics)
    interpretation_lines = _build_dataset_interpretation(coverage_statistics)
    model_interpretation = _build_model_interpretation(result.comparison_table, k)
    return f"""# Last.fm Offline Evaluation

## Dataset Summary

- Interaction count: {_format_integer(coverage_statistics["interaction_count"])}
- User count: {_format_integer(coverage_statistics["user_count"])}
- Track count: {_format_integer(coverage_statistics["unique_tracks_in_interactions"])}
- Evaluated user count: {_format_integer(coverage_statistics["evaluated_users"])}
- Minimum interaction threshold: {min_user_interactions}
- Holdout strategy: most recent user-track interaction when timestamps exist; otherwise interaction strength fallback

## Dataset Coverage and Interaction Density

{coverage_table}

{interpretation_lines}

## Evaluation Setup

- Train/test split: deterministic user-level holdout.
- `holdout_count`: {holdout_count}
- `K`: {k}
- Candidate pool: {_format_candidate_pool(max_candidate_tracks)}
- Metrics: Precision@{k}, Recall@{k}, and NDCG@{k}.
- Assumptions: processed Last.fm plays are treated as implicit user-track preference signals, and each held-out track is considered relevant for its user.

## Metrics Table

{metrics_table}

## Interpretation

{model_interpretation}

Sparsity and overlap strongly affect collaborative methods. ALS needs enough repeated user-item structure to estimate stable latent factors, while Word2Vec-style embeddings need longer, consistently ordered listening sequences to learn reliable co-occurrence neighborhoods.

## Product Perspective

Offline metrics matter because they provide a repeatable way to compare rankers before user-facing experiments. Stronger architectures do not automatically improve metrics when their required signal is weak, sparse, or mismatched to the split strategy.

Production recommenders often combine multiple signals because each model family covers different failure modes: content similarity handles cold or sparse cases, collaborative models learn crowd behavior, embedding models capture context, and novelty/diversity controls keep recommendations from becoming repetitive.
"""


def write_lastfm_evaluation_report(
    *,
    result: OfflineEvaluationResult,
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    min_user_interactions: int,
    holdout_count: int,
    k: int,
    max_candidate_tracks: int | None = None,
    output_path: str | Path = "evaluation_report_lastfm.md",
) -> Path:
    """Write the Last.fm Markdown report to disk."""

    coverage_statistics = compute_lastfm_coverage_statistics(
        interactions=interactions,
        catalog=catalog,
        min_user_interactions=min_user_interactions,
    )
    report_text = generate_lastfm_evaluation_report(
        result=result,
        coverage_statistics=coverage_statistics,
        min_user_interactions=min_user_interactions,
        holdout_count=holdout_count,
        k=k,
        max_candidate_tracks=max_candidate_tracks,
    )
    resolved_path = Path(output_path)
    resolved_path.write_text(report_text, encoding="utf-8")
    return resolved_path


def _render_coverage_table(statistics: dict[str, float | int]) -> str:
    """Render coverage statistics as a Markdown table."""

    rows = [
        ("Interactions", _format_integer(statistics["interaction_count"])),
        ("Users", _format_integer(statistics["user_count"])),
        ("Tracks in interactions", _format_integer(statistics["unique_tracks_in_interactions"])),
        ("Tracks in catalog", _format_integer(statistics["unique_tracks_in_catalog"])),
        ("Avg interactions/user", _format_float(statistics["average_interactions_per_user"])),
        ("Median interactions/user", _format_float(statistics["median_interactions_per_user"])),
        ("Avg users/track", _format_float(statistics["average_users_per_track"])),
        ("Median users/track", _format_float(statistics["median_users_per_track"])),
        ("Tracks with >1 listener", f"{float(statistics['tracks_with_more_than_one_listener_pct']):.2f}%"),
        ("Matrix density", f"{float(statistics['matrix_density']):.8f}"),
        ("Matrix sparsity", f"{float(statistics['matrix_sparsity']):.8f}"),
        ("Evaluated users", _format_integer(statistics["evaluated_users"])),
        (
            "Avg train interactions/evaluated user",
            _format_float(statistics["average_train_interactions_per_evaluated_user"]),
        ),
    ]
    body = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return "| Statistic | Value |\n| --- | ---: |\n" + body


def _render_metrics_table(comparison_table: pd.DataFrame, k: int) -> str:
    """Render evaluation metrics as a Markdown table."""

    if comparison_table.empty:
        return f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | evaluated_users |\n| --- | ---: | ---: | ---: | ---: |"
    rows = []
    for _, row in comparison_table.iterrows():
        rows.append(
            f"| `{row['model']}` | {float(row[f'precision@{k}']):.3f} | "
            f"{float(row[f'recall@{k}']):.3f} | {float(row[f'ndcg@{k}']):.3f} | "
            f"{int(row['evaluated_users'])} |"
        )
    return f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | evaluated_users |\n| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(rows)


def _build_dataset_interpretation(statistics: dict[str, float | int]) -> str:
    """Interpret whether the dataset has enough collaborative/co-occurrence signal."""

    density = float(statistics["matrix_density"])
    average_users_per_track = float(statistics["average_users_per_track"])
    multi_listener_pct = float(statistics["tracks_with_more_than_one_listener_pct"])
    average_interactions_per_user = float(statistics["average_interactions_per_user"])
    overlap_label = "strong" if average_users_per_track >= 2.0 and multi_listener_pct >= 25.0 else "weak"
    sparsity_label = "high" if density < 0.01 else "moderate"
    als_label = "has useful" if overlap_label == "strong" else "has limited"
    embedding_label = "have useful" if average_interactions_per_user >= 10.0 else "have limited"
    content_label = "may remain competitive" if overlap_label == "weak" else "is less likely to dominate purely because of sparsity"
    return (
        f"Collaborative overlap is **{overlap_label}**: tracks average "
        f"{average_users_per_track:.2f} listeners and {multi_listener_pct:.2f}% of tracks have more than one listener.\n\n"
        f"Matrix sparsity is **{sparsity_label}** at {1.0 - density:.6f}, so most possible user-track pairs are unobserved.\n\n"
        f"ALS **{als_label} latent-factor signal** under this split. Word2Vec-style embeddings **{embedding_label} sequence/co-occurrence signal** from the average user history length. Content methods **{content_label}** when collaborative overlap or sequence structure is thin."
    )


def _build_model_interpretation(comparison_table: pd.DataFrame, k: int) -> str:
    """Summarize advanced model behavior from the metrics table."""

    metric_column = f"ndcg@{k}"
    scores = {
        str(row["model"]): float(row[metric_column])
        for _, row in comparison_table.iterrows()
    }
    collaborative = scores.get("collaborative_only", 0.0)
    als = scores.get("ALS_only", 0.0)
    content = scores.get("content_only", 0.0)
    word2vec = scores.get("Word2Vec_similarity_only", 0.0)
    hybrid = scores.get("hybrid", 0.0)
    hybrid_als = scores.get("hybrid_plus_ALS", 0.0)
    hybrid_als_word2vec = scores.get("hybrid_plus_ALS_Word2Vec", 0.0)
    return (
        f"- ALS {'improved over' if als > collaborative else 'did not improve over'} the current collaborative baseline on NDCG@{k}.\n"
        f"- Word2Vec-style similarity {'improved over' if word2vec > content else 'did not improve over'} the content baseline on NDCG@{k}.\n"
        f"- `hybrid_plus_ALS` {'improved over' if hybrid_als > hybrid else 'did not improve over'} the base hybrid model.\n"
        f"- `hybrid_plus_ALS_Word2Vec` {'improved over' if hybrid_als_word2vec > hybrid else 'did not improve over'} the base hybrid model.\n"
        "- These results should be read alongside density and overlap statistics rather than treated as a final judgment of model quality."
    )


def _format_integer(value: float | int) -> str:
    """Format integer-like report values."""

    return f"{int(value):,}"


def _format_float(value: float | int) -> str:
    """Format floating-point report values."""

    return f"{float(value):,.2f}"


def _format_candidate_pool(max_candidate_tracks: int | None) -> str:
    """Format candidate-pool scope for the report."""

    if max_candidate_tracks is None:
        return "full catalog"
    return (
        f"top {max_candidate_tracks:,} catalog tracks by popularity, with each evaluated user's "
        "train and held-out tracks forced into that user's candidate list"
    )
