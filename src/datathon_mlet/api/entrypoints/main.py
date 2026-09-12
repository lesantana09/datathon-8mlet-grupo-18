"""API de recomendação de canal (Etapa 5).

A política é carregada no startup a partir do MLflow — o run mais recente
publicado por `python -m datathon_mlet.train_and_publish_policy`. A API não
treina mais nada: se o MLflow estiver inacessível ou nenhuma política tiver
sido publicada, o startup falha explicitamente, em vez de silenciosamente
servir um modelo diferente do que está registrado.

Como o Thompson Sampling é não-contextual (decisão da Etapa 3),
`POST /recommendations` recebe o contexto do cliente mas a recomendação hoje
é a mesma para qualquer cliente — o contrato já aceita contexto para não
quebrar numa eventual extensão contextual futura.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import load_latest_policy
from datathon_mlet.use_cases import recommend_channel

from ..schemas import ClientContext, RecommendationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.policy = load_latest_policy()
    yield


app = FastAPI(title="Datathon MLET — Recomendação de Canal", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(client: ClientContext) -> RecommendationResponse:
    policy: ThompsonSamplingPolicy = app.state.policy
    return RecommendationResponse(recommended_action=recommend_channel(policy))
