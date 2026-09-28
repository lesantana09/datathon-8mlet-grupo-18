"""Sincronização de dados brutos, processados e modelos com o Data Lake (AWS S3 / MinIO).

Permite carregar as bases de dados e os modelos serializados para o bucket S3
do projeto, garantindo a integridade dos dados e governança no ambiente de nuvem.
"""

from pathlib import Path

from core.logging import setup_logging
from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import (
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_MODEL_KEY,
    save_policy_to_s3,
)
from integrations import StorageClient

logger = setup_logging()

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = REPO_ROOT / "data/raw/bank-additional-full.csv"
PROCESSED_DATA_PATH = REPO_ROOT / "data/processed/bank_marketing_clean.parquet"

S3_RAW_KEY = "data/raw/bank-additional-full.csv"
S3_PROCESSED_KEY = "data/processed/bank_marketing_clean.parquet"


def upload_raw_dataset(
    bucket: str = DEFAULT_S3_BUCKET,
    local_path: Path = RAW_DATA_PATH,
    s3_key: str = S3_RAW_KEY,
) -> bool:
    """Realiza o upload do dataset bruto para o Data Lake no S3.

    Args:
        bucket: Nome do bucket de destino.
        local_path: Caminho local do arquivo CSV bruto.
        s3_key: Chave de destino no bucket S3.

    Returns:
        bool: True se o upload for concluído com sucesso.

    Raises:
        FileNotFoundError: Se o arquivo local não for encontrado.
    """
    if not local_path.exists():
        msg = f"Arquivo de dataset bruto não encontrado: {local_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    storage = StorageClient()
    logger.info(
        "Enviando dataset bruto (%s) para s3://%s/%s...",
        local_path.name,
        bucket,
        s3_key,
    )
    with open(local_path, "rb") as file_obj:
        data = file_obj.read()
    storage.upload_bytes(data=data, bucket=bucket, file=s3_key)
    logger.info("Dataset bruto enviado com sucesso.")
    return True


def upload_processed_dataset(
    bucket: str = DEFAULT_S3_BUCKET,
    local_path: Path = PROCESSED_DATA_PATH,
    s3_key: str = S3_PROCESSED_KEY,
) -> bool:
    """Realiza o upload do dataset tratado em formato Parquet para o Data Lake no S3.

    Args:
        bucket: Nome do bucket de destino.
        local_path: Caminho local do arquivo Parquet tratado.
        s3_key: Chave de destino no bucket S3.

    Returns:
        bool: True se o upload for concluído com sucesso.

    Raises:
        FileNotFoundError: Se o arquivo local não for encontrado.
    """
    if not local_path.exists():
        msg = f"Arquivo de dataset processado não encontrado: {local_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    storage = StorageClient()
    logger.info(
        "Enviando dataset processado (%s) para s3://%s/%s...",
        local_path.name,
        bucket,
        s3_key,
    )
    with open(local_path, "rb") as file_obj:
        data = file_obj.read()
    storage.upload_bytes(data=data, bucket=bucket, file=s3_key)
    logger.info("Dataset processado enviado com sucesso.")
    return True


def upload_trained_policy(
    policy: ThompsonSamplingPolicy,
    bucket: str = DEFAULT_S3_BUCKET,
    s3_key: str = DEFAULT_S3_MODEL_KEY,
) -> bool:
    """Salva e envia a política treinada para o bucket S3.

    Args:
        policy: Objeto da política Thompson Sampling treinada.
        bucket: Nome do bucket de destino.
        s3_key: Chave de destino no bucket S3.

    Returns:
        bool: True se o upload for concluído com sucesso.
    """
    logger.info("Enviando política treinada para s3://%s/%s...", bucket, s3_key)
    return save_policy_to_s3(policy=policy, bucket=bucket, key=s3_key)


def sync_all_artifacts_to_s3(
    bucket: str = DEFAULT_S3_BUCKET,
    policy: ThompsonSamplingPolicy | None = None,
) -> dict[str, bool]:
    """Sincroniza todas as bases (raw e processed) e a política no bucket S3.

    Args:
        bucket: Nome do bucket de destino.
        policy: Instância opcional da política treinada.

    Returns:
        dict[str, bool]: Dicionário contendo o status de cada envio realizado.
    """
    results: dict[str, bool] = {}

    results["raw_dataset"] = upload_raw_dataset(bucket=bucket)
    results["processed_dataset"] = upload_processed_dataset(bucket=bucket)

    if policy is not None:
        results["policy_model"] = upload_trained_policy(policy=policy, bucket=bucket)

    logger.info("Sincronização com o Data Lake finalizada com sucesso: %s", results)
    return results


def main() -> None:
    """Ponto de entrada para sincronização via linha de comando."""
    from datathon_mlet.train_and_publish_policy import train_policy

    logger.info(
        "Iniciando sincronização completa com o bucket '%s'...", DEFAULT_S3_BUCKET
    )
    policy = train_policy(PROCESSED_DATA_PATH)
    status = sync_all_artifacts_to_s3(bucket=DEFAULT_S3_BUCKET, policy=policy)
    logger.info("Status da sincronização: %s", status)


if __name__ == "__main__":
    main()
