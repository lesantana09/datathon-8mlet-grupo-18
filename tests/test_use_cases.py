"""Testes do caso de uso de recomendação, desacoplado de HTTP/FastAPI."""

from datathon_mlet.policies import FixedPolicy, ThompsonSamplingPolicy
from datathon_mlet.use_cases import recommend_channel


def test_recommend_channel_delegates_to_fixed_policy() -> None:
    policy = FixedPolicy(arm="cellular")

    assert recommend_channel(policy) == "cellular"


def test_recommend_channel_delegates_to_thompson_sampling_posterior() -> None:
    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    policy.alpha["telephone"] = 50.0
    policy.beta["cellular"] = 50.0

    assert recommend_channel(policy) == "telephone"
