"""Testes de integração do logging de experimentos no MLflow."""

from pathlib import Path

import mlflow
import pandas as pd

from datathon_mlet.experiments import log_baseline_vs_thompson_sampling


def _sample_history() -> tuple[pd.Series, pd.Series]:
    action = pd.Series(["cellular", "telephone"] * 20)
    reward = pd.Series([1, 0] * 20)
    return action, reward


def _use_isolated_tracking(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")


def test_logs_one_run_for_baseline_with_expected_params_and_metrics(
    tmp_path: Path,
) -> None:
    _use_isolated_tracking(tmp_path)
    action, reward = _sample_history()

    results = log_baseline_vs_thompson_sampling(
        action,
        reward,
        arms=["cellular", "telephone"],
        baseline_arm="telephone",
        n_seeds=3,
        experiment_name="test_baseline",
    )

    experiment = mlflow.get_experiment_by_name("test_baseline")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    baseline_runs = runs[runs["params.policy"] == "fixed"]

    assert len(baseline_runs) == 1
    assert baseline_runs.iloc[0]["params.arm"] == "telephone"
    assert baseline_runs.iloc[0]["metrics.rounds_used"] == 20.0
    assert baseline_runs.iloc[0]["metrics.conversion_rate"] == 0.0
    assert results.baseline.rounds_used == 20
    assert results.baseline.conversion_rate == 0.0


def test_logs_nested_run_per_seed_and_aggregate_on_parent_run(
    tmp_path: Path,
) -> None:
    _use_isolated_tracking(tmp_path)
    action, reward = _sample_history()
    n_seeds = 4

    results = log_baseline_vs_thompson_sampling(
        action,
        reward,
        arms=["cellular", "telephone"],
        baseline_arm="telephone",
        n_seeds=n_seeds,
        experiment_name="test_thompson_sampling",
    )

    experiment = mlflow.get_experiment_by_name("test_thompson_sampling")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    parent_runs = runs[runs["params.policy"] == "thompson_sampling"]
    seed_runs = runs[runs["params.seed"].notna()]

    assert len(parent_runs) == 1
    assert parent_runs.iloc[0]["params.n_seeds"] == str(n_seeds)
    assert "metrics.conversion_rate_mean" in runs.columns
    assert pd.notna(parent_runs.iloc[0]["metrics.conversion_rate_mean"])
    assert len(seed_runs) == n_seeds
    assert set(seed_runs["params.seed"]) == {"0", "1", "2", "3"}
    assert len(results.ts_conversion_rates) == n_seeds
