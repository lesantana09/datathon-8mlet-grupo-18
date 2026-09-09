"""Casos de uso: orquestram políticas de decisão pra uma interação específica."""

from datathon_mlet.policies import Policy


def recommend_channel(policy: Policy) -> str:
    """Recomenda o canal a oferecer a um cliente."""
    return policy.recommend()
