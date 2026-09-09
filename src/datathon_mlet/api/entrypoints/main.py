"""API de recomendação de canal (Etapa 5).

A política é treinada uma vez, no startup, replayando o histórico já
tratado (Etapa 1) e preparado (Etapa 2). Como o Thompson Sampling é
não-contextual (decisão da Etapa 3), `POST /recommendations` recebe o
contexto do cliente mas a recomendação hoje é a mesma para qualquer
cliente — o contrato já aceita contexto para não quebrar numa eventual
extensão contextual futura.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from datathon_mlet.data_prep import load_clean_dataset, prepare_features
from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.replay import run_replay
from datathon_mlet.use_cases import recommend_channel

from ..schemas import ClientContext, RecommendationResponse

REPO_ROOT = Path(__file__).resolve().parents[4]
CLEAN_DATASET_PATH = REPO_ROOT / "data/processed/bank_marketing_clean.parquet"
ARMS = ["cellular", "telephone"]
TRAINING_SEED = 0


def train_policy(dataset_path: Path) -> ThompsonSamplingPolicy:
    df_clean = load_clean_dataset(dataset_path)
    prepared = prepare_features(df_clean)

    policy = ThompsonSamplingPolicy(arms=ARMS)
    run_replay(prepared.action, prepared.reward, policy, seed=TRAINING_SEED)
    return policy


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.policy = train_policy(CLEAN_DATASET_PATH)
    yield


app = FastAPI(title="Datathon MLET — Recomendação de Canal", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(client: ClientContext) -> RecommendationResponse:
    policy: ThompsonSamplingPolicy = app.state.policy
    return RecommendationResponse(recommended_action=recommend_channel(policy))
