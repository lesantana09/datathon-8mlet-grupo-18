"""Treina a política servida pela API e publica no MLflow.

Rode sempre que quiser publicar uma nova versão para a API carregar no
próximo startup:

    uv run python -m datathon_mlet.train_and_publish_policy
"""

from pathlib import Path

from datathon_mlet.data_prep import load_clean_dataset, prepare_features
from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import log_policy
from datathon_mlet.replay import run_replay

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATASET_PATH = REPO_ROOT / "data/processed/bank_marketing_clean.parquet"
ARMS = ["cellular", "telephone"]
TRAINING_SEED = 0


def train_policy(dataset_path: Path) -> ThompsonSamplingPolicy:
    df_clean = load_clean_dataset(dataset_path)
    prepared = prepare_features(df_clean)

    policy = ThompsonSamplingPolicy(arms=ARMS)
    run_replay(prepared.action, prepared.reward, policy, seed=TRAINING_SEED)
    return policy


def main() -> None:
    policy = train_policy(CLEAN_DATASET_PATH)
    run_id = log_policy(policy)
    print(f"Política publicada no MLflow (run_id={run_id})")


if __name__ == "__main__":
    main()
