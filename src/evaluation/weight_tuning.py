"""Lightweight offline weight tuning utilities for recommender variants."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .offline_evaluator import run_demo_offline_evaluation


@dataclass(frozen=True, slots=True)
class WeightTuningConfig:
    """One interpretable score-blend configuration to compare offline."""

    name: str
    model_name: str
    hybrid_weight: float
    als_weight: float
    embedding_weight: float
    diversity_weight: float
    novelty_weight: float


@dataclass(slots=True)
class WeightTuningResult:
    """Store a tuning table and the best configs under two objectives."""

    tuning_table: pd.DataFrame
    best_by_ndcg: str = ""
    best_by_diversity_adjusted_objective: str = ""


@dataclass(slots=True)
class WeightTuningRunner:
    """Compare simple weight configurations using the synthetic offline evaluator."""

    configs: list[WeightTuningConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Install default configs when callers do not provide their own."""

        if not self.configs:
            self.configs = [
                WeightTuningConfig("content_focus", "content_only", 1.0, 0.0, 0.0, 0.10, 0.10),
                WeightTuningConfig("hybrid_baseline", "hybrid", 1.0, 0.0, 0.0, 0.20, 0.20),
                WeightTuningConfig("als_blend", "hybrid_plus_ALS", 1.0, 0.35, 0.0, 0.20, 0.20),
                WeightTuningConfig(
                    "als_embedding_blend",
                    "hybrid_plus_ALS_Word2Vec",
                    1.0,
                    0.30,
                    0.30,
                    0.25,
                    0.25,
                ),
            ]

    def run(self, *, k: int = 3, holdout_count: int = 1) -> WeightTuningResult:
        """Return a table ranked by NDCG and a diversity-adjusted objective."""

        evaluation_result = run_demo_offline_evaluation(k=k, holdout_count=holdout_count)
        metrics_by_model = evaluation_result.comparison_table.set_index("model").to_dict(orient="index")
        rows: list[dict[str, object]] = []
        ndcg_column = f"ndcg@{k}"

        for config in self.configs:
            metrics = metrics_by_model.get(config.model_name, {})
            ndcg_score = float(metrics.get(ndcg_column, 0.0))
            diversity_adjusted_objective = (
                ndcg_score
                + (0.05 * config.diversity_weight)
                + (0.03 * config.novelty_weight)
            )
            rows.append(
                {
                    "config": config.name,
                    "model": config.model_name,
                    "hybrid_weight": config.hybrid_weight,
                    "als_weight": config.als_weight,
                    "embedding_weight": config.embedding_weight,
                    "diversity_weight": config.diversity_weight,
                    "novelty_weight": config.novelty_weight,
                    f"precision@{k}": float(metrics.get(f"precision@{k}", 0.0)),
                    f"recall@{k}": float(metrics.get(f"recall@{k}", 0.0)),
                    ndcg_column: ndcg_score,
                    "diversity_adjusted_objective": round(diversity_adjusted_objective, 3),
                }
            )

        tuning_table = pd.DataFrame(rows).sort_values(
            ["diversity_adjusted_objective", ndcg_column, "config"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        if tuning_table.empty:
            return WeightTuningResult(tuning_table=tuning_table)

        best_by_ndcg = str(
            tuning_table.sort_values([ndcg_column, "config"], ascending=[False, True], kind="stable")
            .iloc[0]["config"]
        )
        best_by_diversity = str(tuning_table.iloc[0]["config"])
        return WeightTuningResult(
            tuning_table=tuning_table,
            best_by_ndcg=best_by_ndcg,
            best_by_diversity_adjusted_objective=best_by_diversity,
        )


def run_demo_weight_tuning(*, k: int = 3, holdout_count: int = 1) -> WeightTuningResult:
    """Run the default synthetic weight-tuning comparison."""

    return WeightTuningRunner().run(k=k, holdout_count=holdout_count)
