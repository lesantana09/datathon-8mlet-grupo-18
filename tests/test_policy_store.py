"""Testes da persistência da política de produção via MLflow."""

from pathlib import Path

import pytest

from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import load_latest_policy, log_policy


@pytest.fixture(autouse=True)
def isolated_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)


def _trained_policy() -> ThompsonSamplingPolicy:
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    policy.update("cellular", 1)
    policy.update("telephone", 0)
    return policy


def test_logged_policy_is_recovered_with_same_posteriors() -> None:
    policy = _trained_policy()

    log_policy(policy)
    loaded = load_latest_policy()

    assert isinstance(loaded, ThompsonSamplingPolicy)
    assert loaded.alpha == policy.alpha
    assert loaded.beta == policy.beta
    assert loaded.recommend() == policy.recommend()


def test_load_returns_most_recent_policy() -> None:
    first = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    first.update("telephone", 1)
    log_policy(first)

    second = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    second.update("cellular", 1)
    log_policy(second)

    loaded = load_latest_policy()

    assert loaded.alpha == second.alpha

