# Datathon MLET — FIAP POSTECH

## Visão Geral

Projeto acadêmico final da pós-graduação em **Machine Learning Engineering da FIAP POSTECH**. O Datathon propõe uma solução *end-to-end* em nível de produção para apoiar a escolha adaptativa de um canal de contato (`cellular` ou `telephone`) para clientes elegíveis de uma instituição financeira, maximizando a taxa de conversão em depósitos a prazo.

O projeto implementa uma arquitetura completa de **MLOps**, integrando aprendizado por reforço leve (**Multi-Armed Bandit com Thompson Sampling**), rastreabilidade de experimentos e traces de inferência no **DagsHub (MLflow Remoto)**, persistência no **Data Lake (AWS S3)** e uma **API FastAPI** resiliente com aprendizado contínuo (*online learning*) e *hot reload*.

---

## Problema de Negócio e Formulação

Uma instituição financeira precisa decidir como abordar cada cliente elegível sem depender apenas de uma regra estática. A abordagem deve equilibrar exploração e aproveitamento (*exploration vs. exploitation*), permitindo aprendizado controlado a partir das respostas dos clientes.

- **Contexto:** Atributos demográficos e financeiros conhecidos antes da decisão (17 variáveis — ver Etapa 2).
- **Ação (Braços):** Canal de contato ofertado, `cellular` ou `telephone` (coluna `contact`).
- **Recompensa ($r$):** Resposta binária do cliente ($y \in \{0, 1\}$), indicando adesão ao depósito a prazo.
- **Baseline Determinístico:** Regra fixa arbitrária (`telephone`), simulando uma decisão estática legada.
- **Política Adaptativa:** Thompson Sampling Beta-Bernoulli, mantendo priors $Beta(\alpha, \beta)$ por canal.
- **Avaliação Offline:** Método de Replay (Li et al., 2011) sobre dados históricos com múltiplas sementes estocásticas para avaliação de variabilidade e convergência.

---

## Source

- **Dataset:** bank-additional-full.csv
  - **Contexto:** Campanha de marketing direto de um banco português.
  - **Autores:** S.M. Almeida and R.C. Almeida
  - **Link:** http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
  - **Disponibilizado por:** Henrique Yamahata (https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)

---

## Arquitetura de MLOps & Nuvem

O ecossistema foi desenhado com base nos pilares modernos de engenharia de machine learning:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 DagsHub (MLflow Remoto)                │
                  │  - Experiment Tracking (Runs, Métricas, Artefatos PNG) │
                  │  - MLflow Tracing (Spans de Inferência e Feedback)     │
                  └───────────────────────────▲────────────────────────────┘
                                              │
                                              │ Logs / Traces / Artifacts
                                              │
┌─────────────────────────┐         ┌─────────┴─────────────────────┐         ┌─────────────────────────┐
│       AWS S3            │         │         FastAPI API           │         │   Pipeline de Treino    │
│ (Data Lake Corporativo) │◄───────►┤  (ECS Fargate / Local)        ├────────►│    & Sincronização      │
│ - data/raw/             │  Upload/│ - POST /api/v1/model/recommend│         │ - make dataset          │
│ - data/processed/       │ Download│ - GET  /api/v1/model          │         │ - make sync             │
│ - models/policy.pkl     │         │ - PATCH /api/v1/model         │         │ - make experiment       │
└─────────────────────────┘         │ - PATCH /api/v1/model/feedback│         │ - make train            │
                                    │ - PATCH /api/v1/model/batch   │         └─────────────────────────┘
                                    │ - PUT /api/v1/model/train     │
                                    └───────────────────────────────┘
```

1. **Tracking & Observabilidade (DagsHub / MLflow):**
   - O servidor remoto do **DagsHub** gerencia o rastreamento dos experimentos e o registro da política.
   - **MLflow Tracing (OpenTelemetry):** Spans hierárquicos rastreiam a latência e contexto de cada chamada de inferência e feedback em tempo real.
   - **Artefatos Visuais:** Gráficos de curvas de aprendizado cumulativo (`learning_curves.png`) e densidade posterior Beta (`posterior_distributions.png`) são registrados diretamente nos runs do MLflow.
2. **Data Lake & Persistência (AWS S3):**
   - Bucket: `datathon-8mlet-grupo-18`
   - Armazena as bases brutas (`data/raw/`), tratadas (`data/processed/`) e o backup do modelo (`models/policy.pkl`).
   - Fornece fallback de alta disponibilidade (HA): se o MLflow estiver temporariamente inacessível, a API carrega o modelo diretamente do S3.

---

## Catálogo de Endpoints da API

A API FastAPI roda por padrão em `http://localhost:8081` (ou `8000`) e inclui documentação interativa Swagger em `/docs`.

### 1. `GET /health`
Verifica a saúde do serviço (Liveness/Readiness probe).
- **Exemplo de Requisição:**
```bash
curl -X GET http://127.0.0.1:8081/health
```
- **Resposta:**
```json
{
  "status": "ok",
  "service": "datathon-mlet",
  "version": "1.0.0",
  "ready": true
}
```

---

### 2. `GET /metrics`
Exporta métricas para o Prometheus. Ref: [Prometheus Metrics](https://github.com/trallnag/prometheus-fastapi-instrumentator/tree/master?tab=readme-ov-file).

- **Exemplo de Requisição:**
```bash
curl -X GET http://127.0.0.1:8081/metrics
```
- **Resposta:** Exibe métricas do Prometheus no formato text/plain.

---

### 3. `POST /api/v1/model/recommend`
Recomenda o canal ideal (`cellular` ou `telephone`) para um cliente com base na política adaptativa ativa. A requisição é instrumentada com span no MLflow Tracing.

- **Exemplo de Requisição:**
```bash
curl -X POST http://127.0.0.1:8081/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 37,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "foi_contatado_antes": false
  }'
```
- **Resposta:**
```json
{
  "recommended_action": "cellular"
}
```

---

### 4. `PATCH /api/v1/model/batch`
Gera recomendações em lote para múltiplos clientes com alto throughput, ideal para processamento diário de campanhas de telemarketing.

- **Exemplo de Requisição:**
```bash
curl -X PATCH http://127.0.0.1:8081/api/v1/model/batch \
  -H "Content-Type: application/json" \
  -d '{
    "clients": [
      {
        "age": 42,
        "job": "technician",
        "marital": "single",
        "education": "professional.course",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "month": "jun",
        "day_of_week": "wed",
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "cons.price.idx": 94.465,
        "cons.conf.idx": -41.8,
        "euribor3m": 4.962,
        "foi_contatado_antes": false
      }
    ]
  }'
```
- **Resposta:**
```json
{
  "total": 1,
  "recommendations": [
    {
      "index": 0,
      "recommended_action": "cellular"
    }
  ]
}
```

---

### 5. `PATCH /api/v1/model/feedback` *(Loop Fechado do Bandit)*
Registra o desfecho da interação com o cliente (sucesso/conversão = 1, recusa/falha = 0) para o canal acionado. Atualiza imediatamente em tempo real os parâmetros $\alpha$ e $\beta$ da política em memória (*online continuous learning*).

- **Exemplo de Requisição:**
```bash
curl -X PATCH http://127.0.0.1:8081/api/v1/model/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "arm": "cellular",
    "reward": 1,
    "client_id": "cliente_12345"
  }'
```
- **Resposta:**
```json
{
  "status": "success",
  "arm": "cellular",
  "reward": 1,
  "updated_alpha": 121.0,
  "updated_beta": 45.0,
  "new_posterior_mean": 0.7289
}
```

---

### 6. `GET /api/v1/model/info`
Auditoria e governança: expõe o estado interno dos braços, parâmetros das distribuições Beta e a decisão recomendada da política servida.

- **Exemplo de Requisição:**
```bash
curl http://127.0.0.1:8081/api/v1/model/info
```
- **Resposta:**
```json
{
  "model_type": "ThompsonSamplingPolicy",
  "arms": ["cellular", "telephone"],
  "alpha_params": {
    "cellular": 120.0,
    "telephone": 35.0
  },
  "beta_params": {
    "cellular": 45.0,
    "telephone": 90.0
  },
  "posterior_means": {
    "cellular": 0.727,
    "telephone": 0.280
  },
  "recommended_action": "cellular"
}
```

---

### 7. `PATCH /api/v1/model`
Recarrega sob demanda a política mais recente publicada no MLflow (ou S3 como fallback) e atualiza o modelo em memória atomicamente (*zero-downtime hot reload*), sem necessidade de reiniciar o container.

- **Exemplo de Requisição:**
```bash
curl -X PATCH http://127.0.0.1:8081/api/v1/model
```
- **Resposta:**
```json
{
  "status": "success",
  "message": "Política mais recente recarregada com sucesso a partir do registro.",
  "recommended_action": "cellular",
  "alpha_params": { ... },
  "beta_params": { ... }
}
```

---

### 8. `PUT /api/v1/model/train`
Gatilho de Retreinamento Contínuo (*Continuous Training - CT*). Treina a política com os dados especificados, publica a nova versão no MLflow/S3 e atualiza imediatamente a política servida pela aplicação.

- **Exemplo de Requisição:**
```bash
curl -X PUT http://127.0.0.1:8081/api/v1/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/processed/bank_marketing_clean.parquet",
    "arms": ["cellular", "telephone"],
    "seed": 42
  }'
```
- **Resposta:**
```json
{
  "status": "success",
  "run_id": "4b6e59489fcd4dafa02a0a2b918f76e3",
  "recommended_action": "cellular",
  "alpha_params": { "cellular": 105.0, "telephone": 32.0 },
  "beta_params": { "cellular": 38.0, "telephone": 88.0 }
}
```

---

## Execução Local do Projeto

### 1. Pré-requisitos e Ambiente Python

O projeto utiliza o gerenciador [`uv`](https://docs.astral.sh/uv/) e requer Python entre **3.11** e **3.13**.

```bash
# Clone o repositório
git clone https://github.com/FIAP-ML-Engineering/datathon-8mlet-grupo-18.git
cd datathon-8mlet-grupo-18

# Instale as dependências principais e de desenvolvimento
make dev
```

### 2. Configuração de Variáveis de Ambiente (`.env`)

Copie o arquivo `.env.example` para `.env` na raiz do projeto e atualize as variáveis de ambiente:

```ini
# API
ENVIRONMENT=local
API_HOST=0.0.0.0
API_PORT=8081
API_V1_STR=/api/v1
WEB_CONCURRENCY=1
API_USERNAME = "<USUARIO_API>"
API_PASSWORD = "<SENHA_API>"
# MLFOW
DAGSHUB_APP_TOKEN = "<DAGSHUB_APP_TOKEN>"
MLFLOW_TRACKING_URI = "https://dagshub.com/<USUARIO>/<REPOSITORIO>.mlflow"
MLFLOW_EXPERIMENT_NAME = "datathon-mlet-experiments"
MLFLOW_TRACKING_USERNAME = "<USUARIO_DAGSHUB>"
# AWS
BUCKET_NAME = "datathon-8mlet-grupo-18"
AWS_REGION = "<AWS_REGION>"
AWS_ENDPOINT_URL = "<AWS_ENDPOINT_URL>"
AWS_ACCESS_KEY = "<AWS_ACCESS_KEY>"
AWS_SECRET_KEY = "<AWS_SECRET_KEY>"
# Grafana
GF_SECURITY_ADMIN_USER = "<USUARIO_GRAFANA>"
GF_SECURITY_ADMIN_PASSWORD = "<SENHA_GRAFANA>"
GF_AUTH_ANONYMOUS_ENABLED = true
```

### 3. Sincronização do Data Lake (S3)

Para enviar os datasets (`raw` e `processed`) e o modelo treinado para o bucket S3 `datathon-8mlet-grupo-18`:

```bash
make dataset
make sync
```

### 4. Treinamento e Publicação da Política no MLflow

Para executar o treinamento do modelo via CLI e publicá-lo no DagsHub MLflow:

```bash
make train
```

### 5. Execução dos Experimentos e Geração de Artefatos

Para rodar a comparação entre Baseline e Thompson Sampling com registro das métricas e geração dos gráficos no MLflow:

```bash
make experiment
```

### 6. Execução da API FastAPI

Inicie o servidor Uvicorn:

```bash
make local
```

Acesse o Swagger interativo em: **<http://localhost:8081/docs>**

---

## Testes Automatizados e Qualidade

O projeto conta com uma suíte abrangente de **52 testes automatizados** cobrindo contratos de API, schemas, políticas, geração de gráficos, persistência e sincronização de Data Lake:

```bash
# Executa todos os testes unitários e de integração
make test

# Executa checagem de estilo e formatação
make format

# Executa checagem de estilo e formatação (same as CI)
make pre-commit
```

---

## Estrutura de Diretórios

```text
.
├── data/                               # Datasets e artefatos de dados
│   ├── processed/                      # Dataset limpo (bank_marketing_clean.parquet)
│   └── raw/                            # Dataset original (bank-additional-full.csv)
├── deploy/                             # Scripts, configurações e arquivos para deploy dos serviços
│   ├── docker/                         # Dockerfiles para deploy dos serviços
│   └── terraform/                      # Scripts e configurações Terraform para deploy dos serviços
├── docs/                               # Documentação, decisões de arquitetura (ADRs) e diagramas
├── monitoring/                         # Configurações de monitoramento
│   ├── grafana/                        # Configurações do Grafana
│   ├── loki/                           # Configurações do Loki
│   ├── prometheus/                     # Configurações do Prometheus
│   ├── promtail/                       # Configurações do Promtail
│   ├── provisioning/                   # Arquivos de configuração e scripts para provisionamento dos serviços
│   └── README.md                       # Instruções para provisionamento dos serviços de observabilidade
├── notebooks/                          # Notebooks das etapas 1 a 4 (EDA, preparação, avaliação)
├── src/                                # Código-fonte da aplicação
│   ├── api/                            # FastAPI e rotas
│   ├── core/                           # Configurações Pydantic Settings e logging estruturado
│   ├── datathon_mlet/                  # Domínio da aplicação
│   │   ├── data_prep.py                # Preparação de features e isolamento de leakage
│   │   ├── evaluation.py               # Curvas de aprendizado e densidades Beta posteriores
│   │   ├── experiments.py              # Instrumentação de experimentos MLflow
│   │   ├── policies.py                 # Thompson Sampling e Baseline Determinístico
│   │   ├── policy_store.py             # Persistência híbrida MLflow & AWS S3
│   │   ├── replay.py                   # Replay offline sobre dado histórico
│   │   ├── data_lake_sync.py           # Sincronização de dados e modelos no S3
│   │   └── train_and_publish_policy.py # Script de treino e publicação
│   ├── domain/                         # Schemas Pydantic
│   └── integrations/                   # Clientes externos (RegistryClient e StorageClient)
├── tests/                              # 52 testes automatizados (API, S3, Replay, Plots)
├── .env.example                        # Template de variáveis de ambiente
├── Makefile                            # Makefile para automação de tarefas
├── pyproject.toml                      # Gerenciamento de dependências e ferramentas
└── README.md                           # README do projeto
```

---

## Integrantes do Grupo

- Henrico Bela
- André Leone da Silva
- Walicen Rangel Nunes Dalazuana
- Leandro Santana
- Phillipe Tramontano Santos

## Checklist do projeto

Etapas conforme enunciado oficial do Datathon (`docs/POSTECH - MLET -
DATATHON.pdf`):

- [x] Etapa 0 — Organização do projeto
- [x] Etapa 1 — Base Kaggle e EDA
- [x] Etapa 2 — Preparação da base
- [x] Etapa 3 — Baseline e estratégia algorítmica
- [x] Etapa 4 — Avaliação e casos de teste
- [x] Etapa 5 — Serviço ou interface demonstrável
- [x] Etapa 6 — Arquitetura-alvo em nuvem
- [x] Etapa 7 — Ciclo de vida MLOps
