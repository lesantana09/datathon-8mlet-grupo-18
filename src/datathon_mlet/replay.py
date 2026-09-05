"""Avaliação offline de políticas via replay (Li et al., 2011).

O canal historicamente atribuído a cada cliente não foi sorteado
aleatoriamente (ver `notebooks/01_eda.ipynb`, passo 6 — confundido com
regime econômico). Por isso este método é uma avaliação offline sobre dado
observacional, não uma medida causal do efeito do braço.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from datathon_mlet.policies import Policy


@dataclass
class ReplayResult:
    """Resultado de uma simulação de replay sobre dado histórico."""

    rounds_used: int
    cumulative_reward: int
    reward_history: list[int] = field(default_factory=list)

    @property
    def conversion_rate(self) -> float:
        if self.rounds_used == 0:
            return 0.0
        return self.cumulative_reward / self.rounds_used


def run_replay(
    action: pd.Series, reward: pd.Series, policy: Policy, seed: int
) -> ReplayResult:
    """Simula `policy` sobre o histórico de `action`/`reward` via replay.

    Só conta uma rodada quando a ação escolhida pela política coincide com a
    ação registrada no histórico para aquele cliente; caso contrário, o
    cliente é descartado (não há recompensa contrafactual conhecida).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(action))
    action_values = action.to_numpy()
    reward_values = reward.to_numpy()

    cumulative_reward = 0
    reward_history: list[int] = []

    for idx in order:
        logged_action = action_values[idx]
        chosen_action = policy.select_action(rng)
        if chosen_action != logged_action:
            continue

        observed_reward = int(reward_values[idx])
        policy.update(chosen_action, observed_reward)
        cumulative_reward += observed_reward
        reward_history.append(observed_reward)

    return ReplayResult(
        rounds_used=len(reward_history),
        cumulative_reward=cumulative_reward,
        reward_history=reward_history,
    )
