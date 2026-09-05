"""Políticas de decisão: baseline determinístico e Thompson Sampling."""

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Contrato mínimo de uma política de bandit."""

    def select_action(self, rng: np.random.Generator) -> str:
        """Escolhe um braço."""

    def update(self, arm: str, reward: int) -> None:
        """Atualiza a política com a recompensa observada para um braço."""


class FixedPolicy:
    """Baseline determinístico: sempre escolhe o mesmo braço."""

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def select_action(self, rng: np.random.Generator) -> str:
        return self.arm

    def update(self, arm: str, reward: int) -> None:
        return None


class ThompsonSamplingPolicy:
    """Thompson Sampling Beta-Bernoulli: 1 posterior Beta(alpha, beta) por braço."""

    def __init__(
        self, arms: list[str], prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> None:
        self.alpha = {arm: prior_alpha for arm in arms}
        self.beta = {arm: prior_beta for arm in arms}

    def select_action(self, rng: np.random.Generator) -> str:
        samples = {arm: rng.beta(self.alpha[arm], self.beta[arm]) for arm in self.alpha}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: int) -> None:
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
