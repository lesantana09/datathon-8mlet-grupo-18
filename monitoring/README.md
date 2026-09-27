# Observabilidade e Monitoramento (`monitoring/`)

Este diretório armazena todos os arquivos de configuração da *stack* de observabilidade e monitoramento em tempo real para o serviço de predição de preços de ações. A infraestrutura de monitoramento foi expandida e agora é composta por **Prometheus**, **Grafana**, **Loki** e **Promtail**, cobrindo métricas de sistema, métricas de negócio, métricas de orquestração de dados e centralização de logs.

---

## Estrutura de Diretórios

O diretório `monitoring/` está estruturado da seguinte forma:

| Diretório | Descrição Principal |
| :--- | :--- |
| `grafana/` | Configurações de provisionamento de *dashboards* e *datasources* do Grafana, permitindo configuração automática na inicialização. |
| `prometheus/` | Configurações de coleta de métricas (*scraping*) do Prometheus. |
| `loki/` | Configuração do servidor de agregação de logs Loki. |
| `promtail/` | Configuração do agente Promtail responsável por coletar logs de contêineres e do sistema, enviando-os para o Loki. |

---

## Descrição dos Módulos e Arquivos

### 1. Grafana (`grafana/`)

O Grafana atua como a principal interface de visualização, provisionado de forma automatizada:

* **`provisioning/datasources/`**: Define as fontes de dados padrão.
  * `prometheus.yml`: Configura o Prometheus local como fonte de métricas principal.
  * `loki.yml`: Configura o Loki para a consulta centralizada de logs.
  * `s3_minio.yml`: Adiciona o S3 como fonte de dados (via plugin `parquet`) para leitura direta de arquivos armazenados no bucket *datathon-8mlet-grupo-18*.
* **`provisioning/dashboards/`**: Contém a definição da organização visual e os arquivos JSON de *dashboards*.
  * `dashboards.yml`: Associa os *dashboards* JSON à pasta lógica "datathon-8mlet-grupo-18" na interface do Grafana, com atualizações automáticas (`allowUiUpdates: true`).
  * `json/FastAPI.json`: Painel focado na observabilidade da API de inferência, exibindo contagem de requisições, latência (p99), status HTTP, RPS e uso de recursos do processo Python.
  * `json/Process Exporter.json`: Painel de métricas de *runtime* (Go/Prometheus), focado em uso de recursos.
  * `json/Docker_logs.json`: Painel integrado ao Loki para visualização centralizada da saída padrão (`stdout`) dos contêineres Docker.

### 2. Prometheus (`prometheus/`)

O Prometheus é o núcleo de armazenamento de séries temporais e coleta de métricas:

* **`prometheus.yml`**: Define o comportamento de coleta com intervalo de 15 segundos. Ele monitora a API de predição (`model:8001`).

### 3. Loki e Promtail (`loki/` e `promtail/`)

Esta dupla é responsável pela centralização e consulta eficiente de logs de toda a infraestrutura:

* **`loki/loki-config.yaml`**: Configura o servidor Loki para operar com armazenamento de objetos compatível com S3 (MinIO) no bucket `loki-data`. Ele também gerencia a retenção, indexação e regras de agregação.
* **`promtail/promtail-config.yaml`**: O agente Promtail é configurado para capturar os logs dos contêineres Docker (via `/var/run/docker.sock`) e extrair *labels* automáticas como o nome do contêiner, além de monitorar logs de sistema local. Todos os logs capturados são encaminhados (*pushed*) para a API do Loki.

---

## Fluxo de Operação e Observabilidade

1. **Coleta de Métricas**:
    * A API FastAPI (configurada em `src/api/main.py`) exporta métricas através de seu *endpoint* `/metrics`.
    * O contêiner do Prometheus faz *scraping* periódico de todos esses *targets* e armazena as séries temporais.
2. **Centralização de Logs**:
    * O Promtail intercepta o fluxo de saída (`stdout`/`stderr`) de todos os contêineres em execução no *host* Docker.
    * Os logs são enriquecidos com metadados (como o nome do serviço) e transmitidos continuamente para o Loki, que os indexa e armazena no MinIO.
3. **Visualização e Monitoramento Contínuo**:
    * O Grafana inicializa com todas as fontes de dados (Prometheus, Loki, MinIO) já conectadas.
    * Os *dashboards* são carregados na pasta "Datathon 8MLET Grupo 18", oferecendo à equipe de MLOps visões unificadas sobre a saúde da API de inferência, o desempenho e a depuração imediata de erros através dos logs centralizados.
