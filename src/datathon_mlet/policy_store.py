"""Persistência da política de produção via MLflow e Data Lake (AWS S3 / MinIO).

A política servida pela API é publicada primariamente no MLflow como artifact simples
(pickle) e secundariamente sincronizada com o bucket S3 do projeto para redundância
e governança no Data Lake corporativo.
"""

import pickle
import tempfile
from pathlib import Path

import mlflow

from core.config import settings
from core.logging import setup_logging
from datathon_mlet.policies import ThompsonSamplingPolicy
from integrations import RegistryClient, StorageClient

logger = setup_logging()

MLFLOW_EXPERIMENT_NAME = settings.MLFLOW_EXPERIMENT_NAME
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
ARTIFACT_FILE_NAME = "policy.pkl"
DEFAULT_S3_BUCKET = "datathon-8mlet-grupo-18"
DEFAULT_S3_MODEL_KEY = "models/policy.pkl"


def save_policy_to_s3(
    policy: ThompsonSamplingPolicy,
    bucket: str = DEFAULT_S3_BUCKET,
    key: str = DEFAULT_S3_MODEL_KEY,
) -> bool:
    """Serializa e faz o upload da política para o bucket S3 / MinIO.

    Args:
        policy: Instância da política a ser persistida.
        bucket: Nome do bucket de destino no S3.
        key: Caminho da chave do objeto no bucket.

    Returns:
        bool: True se o upload foi concluído com sucesso.
    """
    storage = StorageClient()
    payload = pickle.dumps(policy)
    storage.upload_bytes(data=payload, bucket=bucket, file=key)
    logger.info("Política persistida com sucesso no S3 (s3://%s/%s).", bucket, key)
    return True


def load_policy_from_s3(
    bucket: str = DEFAULT_S3_BUCKET,
    key: str = DEFAULT_S3_MODEL_KEY,
) -> ThompsonSamplingPolicy:
    """Baixa e desserializa a política armazenada no bucket S3 / MinIO.

    Args:
        bucket: Nome do bucket onde o modelo está armazenado.
        key: Caminho da chave do objeto no bucket.

    Returns:
        ThompsonSamplingPolicy: Instância restaurada da política.
    """
    storage = StorageClient()
    data = storage.download_bytes(bucket=bucket, file=key)
    policy = pickle.loads(data)
    logger.info("Política carregada com sucesso do S3 (s3://%s/%s).", bucket, key)
    return policy


def log_policy(
    policy: ThompsonSamplingPolicy,
    sync_to_s3: bool = False,
    bucket: str = DEFAULT_S3_BUCKET,
) -> str:
    """Loga a política como artifact num novo run do MLflow e opcionalmente no S3.

    Args:
        policy: Instância da política Thompson Sampling.
        sync_to_s3: Se True, envia uma cópia da política para o bucket S3 do projeto.
        bucket: Nome do bucket S3 caso sync_to_s3 seja verdadeiro.

    Returns:
        str: Identificador do run gerado no MLflow.
    """
    input_example = ["cellular", "telephone"]
    ts_model = ThompsonSamplingPolicy(arms=input_example)
    output_example = ts_model.recommend()
    signature = mlflow.models.infer_signature(input_example, output_example)

    client = RegistryClient()
    with client.start_run() as run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / ARTIFACT_FILE_NAME
            with open(artifact_path, "wb") as f:
                pickle.dump(policy, f)
            client.log_artifact(str(artifact_path))

        mlflow.pyfunc.log_model(
            artifact_path="thompson_sampling_model",
            python_model=policy,
            signature=signature,
            registered_model_name="Thompson-Sampling-Bandit",  # Nome no Registry
        )

    if sync_to_s3:
        try:
            save_policy_to_s3(policy=policy, bucket=bucket)
        except Exception as exc:
            logger.warning("Não foi possível sincronizar política com o S3: %s", exc)

    return run.info.run_id


def load_latest_policy(fallback_to_s3: bool = False) -> ThompsonSamplingPolicy:
    """Carrega a política do run mais recente logado por `log_policy`.

    Caso a busca no MLflow falhe e `fallback_to_s3` seja True, tenta recuperar
    a versão persistida no S3 como estratégia de resiliência.

    Args:
        fallback_to_s3: Se True, tenta recuperar a política do S3 em caso de falha do MLflow.

    Returns:
        ThompsonSamplingPolicy: Instância da política recuperada.

    Raises:
        RuntimeError: Se nenhuma política for localizada ou o MLflow estiver inacessível.
    """
    client = RegistryClient()
    experiment = client.experiment_id

    try:
        if experiment is not None:
            runs = mlflow.search_runs(
                experiment_ids=[experiment],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=ARTIFACT_FILE_NAME
                )
                with open(local_path, "rb") as f:
                    return pickle.load(f)
    except Exception as exc:
        logger.warning("Falha ao buscar política no MLflow: %s", exc)
        if not fallback_to_s3:
            raise

    if fallback_to_s3:
        try:
            logger.info("Tentando carregar política via fallback do S3...")
            return load_policy_from_s3()
        except Exception as s3_exc:
            logger.error("Falha no fallback do S3: %s", s3_exc)

    msg = (
        f"Nenhum run encontrado no experimento '{MLFLOW_EXPERIMENT_NAME}'. Rode "
        "`uv run python -m datathon_mlet.train_and_publish_policy` primeiro."
    )
    raise RuntimeError(msg)

if __name__ == "__main__":
    # Registrando o modelo no MLFLOW
    input_example = ["cellular", "telephone"]
    ts_model = ThompsonSamplingPolicy(arms=input_example)
    output_example = ts_model.recommend()
    signature = mlflow.models.infer_signature(input_example, output_example)

    client = RegistryClient()
    with client.start_run() as run:
        mlflow.pyfunc.log_model(
        artifact_path="thompson_sampling_model",
        python_model=ts_model,
        signature=signature,
        registered_model_name="Thompson-Sampling-Bandit",  # Nome no Registry
    )