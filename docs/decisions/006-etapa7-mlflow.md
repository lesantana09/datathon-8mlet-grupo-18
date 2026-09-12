# DEC-006 - Ciclo de vida MLOps: MLflow via docker-compose, instrumentação no pacote, sem Model Registry

- Status: Aceita
- Contexto: a Etapa 7 pede MLflow local registrando parâmetros e métricas
  dos experimentos da Etapa 3 (baseline vs Thompson Sampling). Havia duas
  decisões de arquitetura em aberto — onde o tracking roda e onde vive o
  código que loga os runs — mais um achado técnico descoberto só ao testar
  de verdade (não bastava validar sintaxe).
- Decisão 1 (servidor de tracking): **servidor MLflow local via
  `docker-compose`**, não tracking direto em arquivo
  (`file:./mlruns`). Alternativa descartada: apontar o cliente Python
  direto pra uma pasta local sem servidor rodando — mais simples, mas não
  dá pra visualizar os runs numa UI nem antecipa o desenho de 2 serviços
  (API + MLflow) já registrado na Etapa 6
  (`docs/decisions/005-etapa6-arquitetura-aws.md`).
- Decisão 2 (imagem Docker do serviço MLflow): **reusa a mesma imagem da
  API** (`docker-compose.yml`, serviço `mlflow` com `build: .`, só o
  `command:` muda), em vez de uma imagem própria. A primeira versão desta
  decisão criou um `Dockerfile.mlflow` separado (`pip install
  "mlflow>=2.19"` direto) — o usuário perguntou se dava pra reusar o mesmo
  Dockerfile com comandos diferentes por serviço, e isso revelou um
  desvio de versão real: o `Dockerfile.mlflow` tinha resolvido mlflow
  3.16.0 via pip solto, enquanto o `uv.lock` do projeto trava mlflow em
  3.15.2 (é dependência de runtime da API, não só dev). Reusar a mesma
  imagem elimina esse desvio por construção. Consequência: o `Dockerfile`
  ficou sem `CMD` fixo (removido, junto com o `EXPOSE 8081` específico da
  API) — cada serviço define seu comando de start no `docker-compose.yml`,
  o que também facilita a tradução futura pra ECS Task Definitions (1
  `containerDefinition` por serviço, mesma imagem, comando explícito —
  ver Etapa 6). Alternativa cogitada e descartada: um `entrypoint.py`
  Python por serviço — pro lado da API já existe o equivalente
  (`api/entrypoints/main.py` + `uvicorn`); pro lado do MLflow, `mlflow
  server` já é o CLI oficial do próprio pacote, então um wrapper Python
  que só chamasse esse comando via subprocess seria uma camada extra sem
  função própria.
- Achado técnico (descoberto testando, não presumido): o MLflow 3.x
  descontinuou o backend de arquivo puro (`file:...`) tanto no `mlflow
  server` quanto no cliente Python (`mlflow.set_experiment()` já lança
  `MlflowException` nesse caso) — exige um backend de banco. Resolvido com
  SQLite local (`sqlite:////app/mlruns/mlflow.db` no container;
  `sqlite:///{tmp_path}/mlflow.db` nos testes) — continua 100% local, sem
  serviço externo, artefatos ainda em arquivo (`file:/app/mlruns`).
- Decisão 3 (instrumentação): **função tipada no pacote**
  (`src/datathon_mlet/experiments.py`,
  `log_baseline_vs_thompson_sampling`), não código direto no notebook —
  segue a convenção já usada no projeto de manter lógica reutilizável fora
  de notebooks. A função reusa `run_replay` e as políticas existentes sem
  alteração, loga 1 run pro baseline e, pro Thompson Sampling, 1 run pai
  (métricas agregadas: média/desvio-padrão de `conversion_rate` entre
  seeds) com N runs aninhados (1 por seed) — preserva a rastreabilidade
  individual exigida pela regra de reportar variabilidade em simulações
  estocásticas. Retorna um `ExperimentResults` (baseline + taxas de
  conversão do TS) pro notebook reusar no gráfico, evitando recomputar o
  replay duas vezes.
- Decisão 4 (escopo): **só tracking de parâmetros/métricas, sem MLflow
  Model Registry.** Opção apresentada e descartada pelo usuário: empacotar
  a `ThompsonSamplingPolicy` treinada num wrapper `mlflow.pyfunc.PythonModel`
  e registrar cada execução como versão de modelo — mais fiel ao "ciclo de
  vida MLOps" completo, mas fora do que o checklist desta etapa pede
  literalmente, e exigiria código novo só pra essa serialização (a
  política não é um estimador scikit-learn). Fica registrado como possível
  extensão futura, não como pendência.
- Decisão 5 (configuração): **`MLFLOW_TRACKING_URI` via variável de
  ambiente** (`.env.example` versionado, `.env` real ignorado pelo Git),
  lida no notebook via `python-dotenv` com fallback pro default local
  (`http://localhost:5000`). Motivo (usuário pediu): evita hardcode do
  endereço do servidor no código — quando o MLflow rodar em outro
  ambiente (ex. a arquitetura AWS desenhada na Etapa 6), só o `.env` muda,
  não o notebook. `python-dotenv` entrou como dependência explícita de
  `dev` (já vinha transitivo do `mlflow`, mas usá-lo sem declarar seria
  frágil).
- Consequências: `docker-compose.yml` e `Dockerfile` (sem `CMD`) ficam
  reutilizáveis por qualquer serviço futuro que rode o mesmo
  código-fonte. `src/datathon_mlet/experiments.py` é o único lugar que
  sabe logar no MLflow — se o Model Registry entrar depois, é uma extensão
  aditiva, não uma reescrita. Se o grupo decidir versionar modelos de
  verdade no futuro, este registro deve ser revisitado com o design do
  wrapper `pyfunc`.
