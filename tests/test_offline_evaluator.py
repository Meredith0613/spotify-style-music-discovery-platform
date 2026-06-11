"""Tests for the offline recommendation evaluation pipeline."""

from __future__ import annotations

from evaluation.offline_evaluator import OfflineEvaluator, run_demo_offline_evaluation
from evaluation.weight_tuning import run_demo_weight_tuning


def test_offline_evaluator_builds_deterministic_holdout_split() -> None:
    """The evaluator should hold out one interaction per sufficiently active user."""

    evaluator = OfflineEvaluator.from_demo_data(k=3, holdout_count=1)

    split = evaluator.build_interaction_split()

    assert not split.train_interactions.empty
    assert not split.test_interactions.empty
    assert split.user_ids
    assert all(len(track_ids) == 1 for track_ids in split.test_track_ids_by_user.values())
    assert all(train_seed_track_ids for train_seed_track_ids in split.train_seed_track_ids_by_user.values())


def test_offline_evaluator_returns_model_comparison_table() -> None:
    """The offline evaluation run should compare all requested recommender variants."""

    result = run_demo_offline_evaluation(k=3, holdout_count=1)

    assert not result.comparison_table.empty
    assert set(result.comparison_table["model"]) == {
        "content_only",
        "collaborative_only",
        "hybrid",
        "ALS_only",
        "Word2Vec_similarity_only",
        "hybrid_plus_ALS",
        "hybrid_plus_ALS_Word2Vec",
        "hybrid_spotify_reranked",
    }
    assert "precision@3" in result.comparison_table.columns
    assert "recall@3" in result.comparison_table.columns
    assert "ndcg@3" in result.comparison_table.columns
    assert result.comparison_table["evaluated_users"].min() > 0


def test_weight_tuning_returns_config_table_and_best_configs() -> None:
    """The weight-tuning helper should summarize existing synthetic evaluation rows."""

    result = run_demo_weight_tuning(k=3, holdout_count=1)

    assert not result.tuning_table.empty
    assert {
        "config",
        "model",
        "hybrid_weight",
        "als_weight",
        "embedding_weight",
        "diversity_adjusted_objective",
    }.issubset(result.tuning_table.columns)
    assert result.best_by_ndcg
    assert result.best_by_diversity_adjusted_objective
