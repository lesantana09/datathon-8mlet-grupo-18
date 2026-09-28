"""API de recomendação de canal (Etapa 5).

A política é carregada no startup a partir do MLflow — o run mais recente
publicado por `python -m datathon_mlet.train_and_publish_policy`. A API não
treina mais nada: se o MLflow estiver inacessível ou nenhuma política tiver
sido publicada, o startup falha explicitamente, em vez de silenciosamente
servir um modelo diferente do que está registrado.

Como o Thompson Sampling é não-contextual (decisão da Etapa 3),
`POST /recommend` recebe o contexto do cliente mas a recomendação hoje
é a mesma para qualquer cliente — o contrato já aceita contexto para não
quebrar numa eventual extensão contextual futura.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from prometheus_fastapi_instrumentator import Instrumentator

from api.model import router as model
from api.system import router as system
from core.config import settings
from core.logging import APILoggingMiddleware, setup_logging
from datathon_mlet.policy_store import load_latest_policy

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.policy = load_latest_policy()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    summary="API do sistema de recomendação de canal",
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if settings.ENVIRONMENT == "local"
    else settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(APILoggingMiddleware, logger=logger)

# Adding routes
app.include_router(model)
app.include_router(system)

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    # Ponto de entrada padrão para execução local ou via Dockerfile
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == "local",
        workers=settings.WEB_CONCURRENCY,
    )
