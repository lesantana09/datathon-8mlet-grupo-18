"""Treina a política servida pela API e publica no MLflow.

Rode sempre que quiser publicar uma nova versão para a API carregar no
próximo startup:

    uv run python -m datathon_mlet.train_and_publish_policy
"""

from pathlib import Path

from core.logging import setup_logging
from datathon_mlet.data_prep import load_clean_dataset, prepare_features
from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import log_policy
from datathon_mlet.replay import run_replay

logger = setup_logging()

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATASET_PATH = REPO_ROOT / "data/processed/bank_marketing_clean.parquet"
ARMS = ["cellular", "telephone"]
TRAINING_SEED = 0


def train_policy(
    dataset_path: Path,
    arms: list[str] = ARMS,
    seed: int = TRAINING_SEED,
) -> ThompsonSamplingPolicy:
    """Treina uma política Thompson Sampling sobre os dados limpos utilizando replay offline.

    Args:
        dataset_path: Caminho no disco para o arquivo parquet limpo.
        arms: Lista de canais/braços de decisão avaliados.
        seed: Semente aleatória para o sorteio das rodadas de replay.

    Returns:
        ThompsonSamplingPolicy: Instância da política treinada com distribuições Beta atualizadas.
    """
    df_clean = load_clean_dataset(dataset_path)
    prepared = prepare_features(df_clean)

    policy = ThompsonSamplingPolicy(arms=arms)
    run_replay(prepared.action, prepared.reward, policy, seed=seed)
    return policy


def train_and_publish(
    dataset_path: Path = CLEAN_DATASET_PATH,
    arms: list[str] = ARMS,
    seed: int = TRAINING_SEED,
) -> tuple[ThompsonSamplingPolicy, str]:
    """Executa o pipeline completo de treinamento da política e publicação no MLflow.

    Args:
        dataset_path: Caminho para os dados de entrada.
        arms: Lista de canais disponíveis.
        seed: Semente aleatória para o replay.

    Returns:
        tuple[ThompsonSamplingPolicy, str]: Política treinada e o ID do run gerado no MLflow.
    """
    policy = train_policy(dataset_path=dataset_path, arms=arms, seed=seed)
    run_id = log_policy(policy)
    logger.info(f"Política publicada no MLflow com sucesso (run_id={run_id})")
    return policy, run_id


def main() -> None:
    """Ponto de entrada para execução via CLI."""
    _, run_id = train_and_publish()
    logger.info(f"Execução CLI concluída (run_id={run_id})")


if __name__ == "__main__":
    main()
