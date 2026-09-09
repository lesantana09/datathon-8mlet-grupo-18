# DEC-003 - Arquitetura do serviço demonstrável (Etapa 5)

- Status: Aceita
- Contexto: Etapa 5 exige um script, notebook interativo ou API básica que
  receba dados de um cliente e retorne a oferta recomendada. Múltiplas
  decisões de arquitetura precisavam ser tomadas: tecnologia, origem da
  política treinada, empacotamento/execução e organização interna do
  código.

## Tecnologia

- Decisão: FastAPI (usuário já tem experiência; enunciado lista
  explicitamente como opção válida).

## Origem da política treinada

- Alternativas consideradas:
  1. Retreinar a política no startup da API (`prepare_features` +
     `run_replay` sobre o parquet, mesma lógica já testada nas Etapas 2-3).
  2. Persistir o estado treinado (`alpha`/`beta`) em um artefato (ex. JSON
     em `artifacts/`) e a API só carregar esse arquivo no startup.
- Decisão: alternativa 1. O bandit é simples (2 braços, ~41k linhas),
  retreinar leva milissegundos. Inventar um formato de serialização próprio
  agora provavelmente seria descartado quando o MLflow assumir esse papel
  na Etapa 7 (registry/artefato versionado) — retrabalho evitável.
- Consequências: a API depende de `data/processed/bank_marketing_clean.parquet`
  existir (gerado pela Etapa 1) para treinar no startup. Etapa 7 deve
  revisitar esta decisão ao introduzir MLflow.

## Empacotamento e execução

- Decisão: contêiner Docker único (sem `docker-compose` por enquanto — só
  há 1 serviço; `docker-compose` fica reservado para quando o MLflow entrar
  como serviço próprio, na Etapa 7).
- O dataset tratado não é versionado, então não entra no build da imagem —
  a imagem só tem código e dependências; o dado é montado em runtime via
  `VOLUME ["/app/data/processed"]` no `Dockerfile` + bind mount no
  `docker run`. Separa ciclo de vida de código (imutável, versionado no
  build) do ciclo de vida de dado (mutável, gerado localmente).
- `Dockerfile` usa `uv` (ferramenta já padrão do projeto) com cache de
  camada: `pyproject.toml`/`uv.lock` copiados e instalados antes do
  código-fonte, para não reinstalar dependências a cada mudança em `src/`.

## Organização interna do código

- Alternativas consideradas: chamar a política diretamente na rota FastAPI
  (acoplando decisão de negócio à camada HTTP) vs. introduzir uma camada de
  caso de uso entre rota e política.
- Decisão: camada de caso de uso (`src/datathon_mlet/use_cases.py`,
  `recommend_channel(policy) -> str`), consumida pela rota. Estrutura de
  pastas definida pelo usuário:
  ```
  src/datathon_mlet/
    api/
      schemas.py            # ClientContext, RecommendationResponse
      entrypoints/
        main.py              # app FastAPI, rotas, treino no startup
    use_cases.py             # lógica de aplicação, sem depender de HTTP/Pydantic
  ```
- Consequências: rota vira adapter fino (parsing HTTP → caso de uso →
  response); lógica de decisão é testável sem subir FastAPI/TestClient
  (`tests/test_use_cases.py`) e reaproveitável por outro tipo de entrypoint
  futuro (script, CLI) sem duplicar código. `use_cases.py` fica como
  arquivo único (não pasta) enquanto houver só 1 caso de uso — vira pasta
  quando um 2º aparecer (evita estrutura vazia sem uso imediato).
