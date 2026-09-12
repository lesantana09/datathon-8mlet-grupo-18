"""Persistência da política de produção via MLflow (não a avaliação da Etapa 3).

A política servida pela API é publicada aqui como um artifact simples
(pickle) dentro de um run — sem MLflow Model Registry: um único modelo,
um único consumidor (a própria API), sem promoção de estágio nem
múltiplos consumidores que justifiquem esse custo extra (ver
`docs/decisions/007-api-carrega-policy-do-mlflow.md`).
"""

import os
import pickle
import tempfile
from pathlib import Path

import mlflow

from datathon_mlet.policies import ThompsonSamplingPolicy

EXPERIMENT_NAME = "channel_recommendation_policy"
ARTIFACT_FILE_NAME = "policy.pkl"


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def log_policy(policy: ThompsonSamplingPolicy) -> str:
    """Loga `policy` como artifact num novo run; retorna o `run_id`."""
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / ARTIFACT_FILE_NAME
            with open(artifact_path, "wb") as f:
                pickle.dump(policy, f)
            mlflow.log_artifact(str(artifact_path))

    return run.info.run_id


def load_latest_policy() -> ThompsonSamplingPolicy:
    """Carrega a política do run mais recente logado por `log_policy`.

    Falha explicitamente (sem retreinar como fallback) se o MLflow estiver
    inacessível ou nenhuma política tiver sido publicada ainda — rodar
    `python -m datathon_mlet.train_and_publish_policy` primeiro.
    """
    mlflow.set_tracking_uri(_tracking_uri())
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        msg = (
            f"Nenhum experimento '{EXPERIMENT_NAME}' encontrado no MLflow "
            f"({_tracking_uri()}). Rode "
            "`python -m datathon_mlet.train_and_publish_policy` primeiro."
        )
        raise RuntimeError(msg)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        msg = (
            f"Nenhum run encontrado no experimento '{EXPERIMENT_NAME}'. Rode "
            "`python -m datathon_mlet.train_and_publish_policy` primeiro."
        )
        raise RuntimeError(msg)

    run_id = runs.iloc[0]["run_id"]
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ARTIFACT_FILE_NAME
    )
    with open(local_path, "rb") as f:
        return pickle.load(f)
