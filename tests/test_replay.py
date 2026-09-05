"""Testes do método de replay para avaliação offline de políticas."""

import pandas as pd

from datathon_mlet.policies import FixedPolicy, ThompsonSamplingPolicy
from datathon_mlet.replay import run_replay


def _sample_history() -> tuple[pd.Series, pd.Series]:
    action = pd.Series(["cellular", "telephone", "cellular", "telephone", "cellular"])
    reward = pd.Series([1, 0, 1, 1, 0])
    return action, reward


def test_fixed_policy_only_counts_matching_rounds() -> None:
    action, reward = _sample_history()
    policy = FixedPolicy(arm="cellular")

    result = run_replay(action, reward, policy, seed=0)

    assert result.rounds_used == 3


def test_fixed_policy_conversion_rate_matches_matching_rounds() -> None:
    action, reward = _sample_history()
    policy = FixedPolicy(arm="cellular")

    result = run_replay(action, reward, policy, seed=0)

    assert result.cumulative_reward == 2
    assert result.conversion_rate == 2 / 3


def test_conversion_rate_is_zero_when_no_rounds_used() -> None:
    action, reward = _sample_history()
    policy = FixedPolicy(arm="unknown_arm")

    result = run_replay(action, reward, policy, seed=0)

    assert result.rounds_used == 0
    assert result.conversion_rate == 0.0


def test_run_replay_is_deterministic_for_same_seed() -> None:
    action, reward = _sample_history()

    result_a = run_replay(
        action, reward, ThompsonSamplingPolicy(arms=["cellular", "telephone"]), seed=7
    )
    result_b = run_replay(
        action, reward, ThompsonSamplingPolicy(arms=["cellular", "telephone"]), seed=7
    )

    assert result_a.reward_history == result_b.reward_history
    assert result_a.rounds_used == result_b.rounds_used


def test_thompson_sampling_updates_policy_only_on_used_rounds() -> None:
    action, reward = _sample_history()
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])

    result = run_replay(action, reward, policy, seed=0)

    total_updates = sum(policy.alpha.values()) + sum(policy.beta.values()) - 4
    assert total_updates == result.rounds_used
