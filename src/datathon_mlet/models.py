"""Modelos de dados do domínio do bandit contextual."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    """Contexto, ação e recompensa prontos para baseline e política adaptativa."""

    context: pd.DataFrame
    action: pd.Series
    reward: pd.Series
