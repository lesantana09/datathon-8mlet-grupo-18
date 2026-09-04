"""Testes de tratamento de dados: contrato de contexto/ação/recompensa."""

import pandas as pd
import pytest

from datathon_mlet.data_prep import prepare_features
from datathon_mlet.models import PreparedDataset


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 45],
            "job": ["admin.", "technician"],
            "contact": ["cellular", "telephone"],
            "emp.var.rate": [1.1, -1.8],
            "nr.employed": [5191.0, 5099.1],
            "euribor3m": [4.857, 1.344],
            "y": ["yes", "no"],
        }
    )


def test_prepare_features_returns_prepared_dataset() -> None:
    result = prepare_features(_sample_df())

    assert isinstance(result, PreparedDataset)


def test_action_excluded_from_context() -> None:
    result = prepare_features(_sample_df())

    assert "contact" not in result.context.columns


def test_target_excluded_from_context() -> None:
    result = prepare_features(_sample_df())

    assert "y" not in result.context.columns


def test_redundant_macro_columns_excluded_from_context() -> None:
    result = prepare_features(_sample_df())

    assert "emp.var.rate" not in result.context.columns
    assert "nr.employed" not in result.context.columns
    assert "euribor3m" in result.context.columns


def test_reward_is_binary_encoding_of_target() -> None:
    result = prepare_features(_sample_df())

    assert result.reward.tolist() == [1, 0]


def test_action_matches_original_contact_column() -> None:
    result = prepare_features(_sample_df())

    assert result.action.tolist() == ["cellular", "telephone"]


def test_duration_column_raises_value_error() -> None:
    df = _sample_df()
    df["duration"] = [120, 45]

    with pytest.raises(ValueError, match="duration"):
        prepare_features(df)
