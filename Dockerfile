# =============================================================================
# Dockerfile.api — Build multi-stage otimizado para producao
#
# Stages:
#   1. builder  — resolve e instala dependencias via uv numa venv isolada
#   2. runtime  — copia apenas a venv e o codigo; sem ferramentas de build
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1 — builder
# Instala o uv e resolve todas as dependencias de producao numa venv local.
# Ao separar o COPY dos manifestos do COPY do codigo, o Docker reusa o cache
# da layer de instalacao sempre que so o codigo-fonte mudar.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# O instalador requer curl (e certificados) para baixar o arquivo de release
# Atualiza a lista de pacotes e instala as dependências necessárias
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Baixa o instalador mais recente
ADD https://astral.sh/uv/0.11.21/install.sh /uv-installer.sh

# Instala e remove o instalador
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Coloca o instalador na variavel PATH
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# 1. Copia os manifestos ANTES do codigo para maximizar o cache de layers.
#    Se apenas src/ mudar, esta layer de install NAO sera invalidada.
COPY pyproject.toml uv.lock ./

ENV UV_NO_DEV=1

# 2. Instala dependencias de producao (sem grupo [dev]) sem o pacote local.
#    --frozen:             usa o uv.lock sem resolver novamente
#    --no-dev:             exclui ipykernel, optuna, pytest, ruff, etc.
#    --no-install-project: nao instala o pacote local ainda (codigo ausente)

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

# 3. Copia o codigo e instala o pacote local (deps ja estao em cache)
COPY src/ ./src/

RUN uv sync \
    --frozen \
    --no-dev



# -----------------------------------------------------------------------------
# Stage 2 — runtime
# Imagem final enxuta: somente a venv resolvida + codigo-fonte.
# Sem pip, sem uv, sem headers de compilacao.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Ativa a venv copiada do builder sem precisar de "source activate"
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Utilizador sem privilegios — nao roda como root em producao
RUN groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copia a venv ja resolvida e o codigo do stage builder
COPY --from=builder --chown=appuser:appgroup /app/.venv  ./.venv
COPY --from=builder --chown=appuser:appgroup /app/src    ./src
COPY --chown=appuser:appgroup pyproject.toml ./

# Cria os diretórios de artefatos e do SQLite com permissão para o appuser
RUN mkdir -p models data && chown -R appuser:appgroup models data

USER appuser

EXPOSE 8081

# Health check nativo do Docker — usa o readiness probe da propria API.
# start-period=60s: da tempo ao lifespan de baixar os artefatos do MinIO/S3.
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD python -c \
    "import os, urllib.request; port = os.getenv('PORT', '8081'); urllib.request.urlopen(f'http://localhost:{port}/health')" \
    || exit 1

# Ponto de entrada: uvicorn sem --reload (controlado pela variavel ENVIRONMENT no lifespan)
#CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
