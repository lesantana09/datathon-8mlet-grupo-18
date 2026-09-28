"""Testes unitários para a sincronização de artefatos e dados com o S3."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datathon_mlet.data_lake_sync import (
    sync_all_artifacts_to_s3,
    upload_processed_dataset,
    upload_raw_dataset,
    upload_trained_policy,
)
from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import load_policy_from_s3


@pytest.fixture
def mock_storage(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mocka o StorageClient para testes isolados sem chamada de rede."""
    mock = MagicMock()
    monkeypatch.setattr("datathon_mlet.data_lake_sync.StorageClient", lambda: mock)
    monkeypatch.setattr("datathon_mlet.policy_store.StorageClient", lambda: mock)
    return mock


def test_upload_raw_dataset_success(mock_storage: MagicMock, tmp_path: Path) -> None:
    """Valida envio do arquivo raw para o S3."""
    fake_csv = tmp_path / "bank-additional-full.csv"
    fake_csv.write_text("col1;col2\n1;2")

    result = upload_raw_dataset(
        bucket="test-bucket",
        local_path=fake_csv,
        s3_key="data/raw/bank-additional-full.csv",
    )

    assert result is True
    mock_storage.upload_bytes.assert_called_once()


def test_upload_raw_dataset_raises_file_not_found(
    mock_storage: MagicMock, tmp_path: Path
) -> None:
    """Valida lançamento de exceção se o arquivo local não existir."""
    fake_csv = tmp_path / "non_existent.csv"

    with pytest.raises(FileNotFoundError):
        upload_raw_dataset(
            bucket="test-bucket",
            local_path=fake_csv,
            s3_key="data/raw/bank-additional-full.csv",
        )


def test_upload_processed_dataset_success(
    mock_storage: MagicMock, tmp_path: Path
) -> None:
    """Valida envio do arquivo parquet tratado para o S3."""
    fake_parquet = tmp_path / "bank_marketing_clean.parquet"
    fake_parquet.write_bytes(b"PAR1fake")

    result = upload_processed_dataset(
        bucket="test-bucket",
        local_path=fake_parquet,
        s3_key="data/processed/bank_marketing_clean.parquet",
    )

    assert result is True
    mock_storage.upload_bytes.assert_called_once()


def test_upload_trained_policy_and_save_load(mock_storage: MagicMock) -> None:
    """Valida serialização e upload da política para o S3."""
    import pickle

    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    policy.update("cellular", 1)

    # Teste de save
    result = upload_trained_policy(policy=policy, bucket="test-bucket")
    assert result is True
    mock_storage.upload_bytes.assert_called_once()

    # Teste de load
    mock_storage.download_bytes.return_value = pickle.dumps(policy)
    loaded = load_policy_from_s3(bucket="test-bucket")
    assert loaded.alpha == policy.alpha
    assert loaded.beta == policy.beta


def test_sync_all_artifacts_to_s3(
    mock_storage: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valida orquestração completa da sincronização de bases e modelo."""
    fake_raw = tmp_path / "fake_raw.csv"
    fake_raw.touch()
    fake_proc = tmp_path / "fake_proc.parquet"
    fake_proc.touch()

    monkeypatch.setattr("datathon_mlet.data_lake_sync.RAW_DATA_PATH", fake_raw)
    monkeypatch.setattr("datathon_mlet.data_lake_sync.PROCESSED_DATA_PATH", fake_proc)

    policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    results = sync_all_artifacts_to_s3(bucket="test-bucket", policy=policy)

    assert results["raw_dataset"] is True
    assert results["processed_dataset"] is True
    assert results["policy_model"] is True
    assert mock_storage.upload_bytes.call_count >= 3
