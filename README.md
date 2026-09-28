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
3. **Infraestrutura como Código & Nuvem (AWS ECS Fargate & Terraform):**
   - Provisionamento 100% automatizado e reproduzível com **Terraform** (`deploy/terraform/`).
   - Execução serverless da API conteinerizada no **AWS ECS Fargate**, em VPC dedicada com subnets públicas distribuídas em múltiplas zonas de disponibilidade (`us-east-2`), Security Group com controle estrito de portas e credenciais protegidas via **AWS Secrets Manager**.

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
git clone https://github.com/lesantana09/datathon-8mlet-grupo-18.git
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

## Deploy na AWS (Infraestrutura como Código com Terraform)

O provisionamento da infraestrutura na nuvem AWS é totalmente gerenciado como código (IaC) através do **Terraform** (`deploy/terraform/`) e orquestrado de forma simplificada pelos comandos do [`Makefile`](file:///e:/stacks/datathon-8mlet-grupo-18/Makefile).

A aplicação é executada como um serviço conteinerizado serverless no **AWS ECS Fargate**, dispensando o gerenciamento manual de instâncias EC2, patches de sistema operacional ou provisionamento de capacidade fixa.

### 1. Arquitetura Provisionada

Ao executar o deploy, o Terraform provisiona os seguintes recursos na região **`us-east-2`** (Ohio):

- **Rede e Conectividade (VPC & Subnets):**
  - **VPC Própria (`aws_vpc.main`):** Bloco CIDR `10.0.0.0/16` com suporte a hostnames DNS.
  - **Subnets Públicas (`aws_subnet.public_1` e `public_2`):** `10.0.1.0/24` (AZ `us-east-2a`) e `10.0.2.0/24` (AZ `us-east-2b`), com atribuição automática de IP público.
  - **Internet Gateway & Route Table (`aws_internet_gateway.gw`, `aws_route_table.rt`):** Rota padrão `0.0.0.0/0` para acesso e saída à internet.
- **Segurança e Controle de Acesso:**
  - **Security Group (`aws_security_group.ecs_sg`):** Libera tráfego de entrada na porta da API (`8081/tcp`) de qualquer origem (`0.0.0.0/0`) e permite tráfego de saída irrestrito para download da imagem Docker e comunicação externa com DagsHub/MLflow e AWS S3.
  - **AWS Secrets Manager (`aws_secretsmanager_secret.docker_hub`):** Guarda as credenciais do Docker Hub (`DOCKER_HUB_USERNAME` e `DOCKER_HUB_TOKEN`) de forma segura, com exclusão imediata configurada (`recovery_window_in_days = 0`).
  - **IAM Roles & Policies (`aws_iam_role.ecs_execution_role`):** Role de execução com a política gerenciada `AmazonECSTaskExecutionRolePolicy` e política inline para leitura do segredo no Secrets Manager.
- **Computação Serverless (AWS ECS Fargate):**
  - **Cluster ECS (`aws_ecs_cluster.main`):** `datathon-cluster`.
  - **Task Definition (`aws_ecs_task_definition.app`):** Modo Fargate com 0.25 vCPU (256 CPU units) e 512 MB de RAM, executando a imagem `ghcr.io/tramontano/datathon-8mlet-grupo-18:1.0.0` com comando `uvicorn api:app --host 0.0.0.0 --port 8081`. Todas as variáveis de ambiente necessárias (API, DagsHub MLflow e AWS S3) são injetadas automaticamente a partir do `.env`.
  - **ECS Service (`aws_ecs_service.main`):** `datathon-service`, mantendo 1 réplica da task ativa com IP público em rede pública.

---

### 2. Pré-requisitos

1. **Terraform CLI** instalado (versão `>= 1.5.0` recomendada) — [Instruções de Instalação](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).
2. **AWS CLI** instalado e autenticado (opcional, para consultar o IP da task via terminal) — [Instalação da AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
3. **Arquivo `.env` configurado** na raiz do projeto contendo as chaves AWS (`AWS_ACCESS_KEY`, `AWS_SECRET_KEY`), credenciais do Docker Hub (`DOCKER_HUB_USERNAME`, `DOCKER_HUB_TOKEN`) e parâmetros de API e MLflow.

> [!NOTE]
> O arquivo [`deploy/terraform/variables.tf`](file:///e:/stacks/datathon-8mlet-grupo-18/deploy/terraform/variables.tf) lê e decodifica diretamente o arquivo `.env` localizado na raiz do projeto. Não é necessário exportar variáveis manualmente nem criar arquivos `.tfvars`.

---

### 3. Passo a Passo do Deploy via Makefile

Todos os comandos de gerenciamento de ciclo de vida da infraestrutura AWS estão disponíveis diretamente no `Makefile`:

```
make aws-init     # 1. Inicializa o Terraform e baixa os provedores
make aws-plan     # 2. Visualiza o plano de execução (dry-run)
make aws-apply    # 3. Provisiona toda a infraestrutura na AWS
make aws-output   # 4. Exibe os nomes do cluster, serviço e query de IP público
make aws-destroy  # 5. Destrói toda a infraestrutura e evita custos
```

#### Passo 1 — Inicializar o Terraform
Inicializa o diretório de trabalho do Terraform e baixa o provedor oficial `hashicorp/aws`:
```bash
make aws-init
```

#### Passo 2 — Planejar e Validar Recursos
Gera o plano de execução (*dry-run*), listando todos os recursos que serão criados ou atualizados sem aplicar modificações na nuvem:
```bash
make aws-plan
```

#### Passo 3 — Aplicar e Provisionar a Infraestrutura
Cria a VPC, Subnets, Security Group, Secrets Manager, IAM Roles, Cluster ECS, Task Definition e inicializa o Serviço Fargate:
```bash
make aws-apply
```
> [!IMPORTANT]
> O Terraform solicitará a confirmação digitando `yes`. Após a confirmação, o provisionamento leva aproximadamente 1 a 2 minutos.

#### Passo 4 — Obter o IP Público da Aplicação
Consulte as saídas geradas pelo Terraform:
```bash
make aws-output
```

Para recuperar o IP público da task ECS em execução via AWS CLI:
```bash
aws ecs list-tasks --cluster datathon-cluster --region us-east-2 --query 'taskArns[0]' --output text | \
xargs -I {} aws ecs describe-tasks --cluster datathon-cluster --region us-east-2 --tasks {} \
--query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | \
xargs -I {} aws ec2 describe-network-interfaces --region us-east-2 --network-interface-ids {} \
--query 'NetworkInterfaces[0].Association.PublicIp' --output text
```


> Você também pode verificar o IP público diretamente pelo **Console da AWS**:
> 1. Acesse o serviço **Elastic Container Service (ECS)** na região **us-east-2**.
> 2. Clique em **Clusters** > **`datathon-cluster`** > aba **Tasks**.
> 3. Clique na Task com status **RUNNING**.
> 4. Na seção **Rede** (*Networking*), copie o valor do campo **IP público** (*Public IP*).

---

### 4. Validação da API na Nuvem

Com o IP público obtido (exemplo: `3.15.XXX.XXX`), teste os endpoints remotos da API:

#### 1. Liveness & Readiness Probe
```bash
curl -X GET http://<IP_PUBLICO>:8081/health
```
**Resposta esperada:**
```json
{
  "status": "ok",
  "service": "datathon-mlet",
  "version": "1.0.0",
  "ready": true
}
```

#### 2. Documentação Swagger Interativa
Abra no navegador para inspecionar os endpoints e schemas:
```text
http://<IP_PUBLICO>:8081/docs
```

#### 3. Teste de Recomendação de Canal (Thompson Sampling)
```bash
curl -X POST http://<IP_PUBLICO>:8081/api/v1/model/recommend \
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

#### 4. Auditoria da Política Ativa em Memória
```bash
curl -X GET http://<IP_PUBLICO>:8081/api/v1/model/info
```

---

### 5. Destruição e Limpeza dos Recursos na AWS

Para encerrar o ambiente e evitar custos residuais na conta AWS após testes ou demonstrações:
```bash
make aws-destroy
```
> Digite `yes` para confirmar a destruição de todos os recursos criados (Serviço ECS, Task, Cluster, Secrets, Roles e VPC).

---

## Testes Automatizados e Qualidade

O projeto conta com uma suíte abrangente de **52 testes automatizados** cobrindo contratos de API, schemas, políticas, geração de gráficos, persistência e sincronização de Data Lake:

```bash
# Executa todos os testes unitários e de integração
make test

# Executa checagem de estilo e formatação
make format

# Executa testes e formatação (same as CI)
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
│   ├── docker/                         # Dockerfile multi-stage para a API de produção
│   │   └── Container                   # Build enxuto em 2 estágios (builder e runtime) com uv
│   └── terraform/                      # Infraestrutura como Código (IaC) para AWS ECS Fargate
│       ├── main.tf                     # Definição dos recursos (VPC, Subnets, ECS, IAM, Secrets)
│       └── variables.tf                # Parser e injeção automática de variáveis lidas do .env
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
