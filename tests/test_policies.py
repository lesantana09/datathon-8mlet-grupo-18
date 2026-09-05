"""Testes das políticas de decisão: baseline e Thompson Sampling."""

import numpy as np

from datathon_mlet.policies import FixedPolicy, ThompsonSamplingPolicy


def test_fixed_policy_always_selects_configured_arm() -> None:
    policy = FixedPolicy(arm="cellular")
    rng = np.random.default_rng(0)

    selections = {policy.select_action(rng) for _ in range(20)}

    assert selections == {"cellular"}


def test_fixed_policy_update_is_noop() -> None:
    policy = FixedPolicy(arm="cellular")

    policy.update("cellular", 1)

    assert policy.arm == "cellular"


def test_thompson_sampling_starts_with_configured_prior() -> None:
    policy = ThompsonSamplingPolicy(
        arms=["cellular", "telephone"], prior_alpha=2.0, prior_beta=3.0
    )

    assert policy.alpha == {"cellular": 2.0, "telephone": 2.0}
    assert policy.beta == {"cellular": 3.0, "telephone": 3.0}


def test_thompson_sampling_update_increments_alpha_on_success() -> None:
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])

    policy.update("cellular", 1)

    assert policy.alpha["cellular"] == 2.0
    assert policy.beta["cellular"] == 1.0


def test_thompson_sampling_update_increments_beta_on_failure() -> None:
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])

    policy.update("cellular", 0)

    assert policy.alpha["cellular"] == 1.0
    assert policy.beta["cellular"] == 2.0


def test_thompson_sampling_favors_arm_with_stronger_posterior() -> None:
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    policy.alpha["cellular"] = 100.0
    policy.beta["telephone"] = 100.0
    rng = np.random.default_rng(42)

    selections = [policy.select_action(rng) for _ in range(50)]

    assert selections.count("cellular") == 50


def test_thompson_sampling_select_action_returns_known_arm() -> None:
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    rng = np.random.default_rng(1)

    for _ in range(20):
        assert policy.select_action(rng) in {"cellular", "telephone"}
