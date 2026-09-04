"""Preparação de dados: dataset tratado -> contexto, ação e recompensa."""

from pathlib import Path

import pandas as pd

from datathon_mlet.models import PreparedDataset

ACTION_COLUMN = "contact"
TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ("duration",)
REDUNDANT_CONTEXT_COLUMNS = ("emp.var.rate", "nr.employed")


def load_clean_dataset(path: Path) -> pd.DataFrame:
    """Carrega o dataset tratado na Etapa 1 (ver notebooks/01_eda.ipynb)."""
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame) -> PreparedDataset:
    """Separa contexto, ação (`contact`) e recompensa (`y`) do dataset tratado.

    `contact` é a ação do bandit e nunca entra no contexto — um modelo não
    pode usar a própria decisão como feature dela mesma. `emp.var.rate` e
    `nr.employed` também saem do contexto por correlação 0.91-0.97 com
    `euribor3m` (ver passo 7 do EDA).
    """
    for column in LEAKAGE_COLUMNS:
        if column in df.columns:
            msg = f"coluna de vazamento '{column}' não pode estar no dataset de entrada"
            raise ValueError(msg)

    action = df[ACTION_COLUMN]
    reward = (df[TARGET_COLUMN] == "yes").astype(int)

    drop_columns = [ACTION_COLUMN, TARGET_COLUMN, *REDUNDANT_CONTEXT_COLUMNS]
    context = df.drop(columns=drop_columns)

    return PreparedDataset(context=context, action=action, reward=reward)
