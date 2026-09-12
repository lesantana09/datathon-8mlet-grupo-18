# DEC-007 - API carrega a política do MLflow em vez de treinar no startup

- Status: Aceita
- Contexto: até aqui a API treinava a política a cada startup
  (`train_policy` em `api/entrypoints/main.py`: `prepare_features` +
  `run_replay` com `seed=0` sobre o parquet local). Isso foi uma decisão
  consciente da Etapa 5 (`docs/decisions/003-etapa5-api-arquitetura.md`),
  que já previa revisitá-la quando o MLflow existisse. Com a Etapa 7
  concluída, o MLflow está no ar — mas guardava apenas params/metrics do
  experimento de avaliação (20 seeds, baseline vs Thompson Sampling), que
  é uma coisa diferente do modelo servido. Ou seja: não havia nenhum
  artefato de modelo carregável, e o treino da API seguia desconectado do
  MLflow.
- Decisão 1 (formato): a política treinada é publicada como **artifact
  simples (pickle) dentro de um run**, não como MLflow Model no Model
  Registry. Alternativa avaliada e descartada pelo usuário depois de
  questionar o ganho real: o Registry entrega versionamento nomeado,
  promoção Staging→Production e endereçamento estável
  (`models:/<nome>/<stage>`) para múltiplos consumidores — nada disso tem
  onde se aplicar aqui (um modelo, um consumidor, o mesmo repositório, sem
  pipeline de promoção). Em compensação, registrar exigiria um wrapper
  `mlflow.pyfunc.PythonModel` com `predict()` só para satisfazer o formato
  de "flavor" do MLflow, já que `ThompsonSamplingPolicy` não é um
  estimador scikit-learn — código sem nenhum outro uso no projeto. O
  pickle recupera o objeto exatamente como ele é.
- Decisão 2 (resolução da versão): a API carrega o **run mais recente** do
  experimento `channel_recommendation_policy`, em vez de um `run_id` fixo
  em configuração. Mais simples e sem bookkeeping manual; o custo é que
  publicar um run novo muda o que a API serve no próximo restart. Se o
  grupo precisar de fixação explícita depois, é uma extensão pequena
  (variável de ambiente com `run_id`).
- Decisão 3 (falha sem fallback): se o MLflow estiver inacessível ou
  nenhuma política tiver sido publicada, o startup da API **falha com erro
  explícito** apontando o comando a rodar. Alternativa descartada:
  retreinar como fallback — esconderia um problema operacional real e
  serviria silenciosamente um modelo diferente do que está registrado,
  anulando o propósito da mudança. Consequência assumida: a API passou a
  ter uma dependência externa no startup, que antes não existia.
- Decisão 4 (onde o treino mora): saiu da API e virou um passo explícito,
  `src/datathon_mlet/train_and_publish_policy.py`, rodado sob demanda
  (`uv run python -m datathon_mlet.train_and_publish_policy`). Mantém a
  mesma lógica de antes (arms `cellular`/`telephone`, `seed=0`,
  `run_replay` sobre o parquet tratado) — o que muda é quando e por quem
  ela é executada. Isso separa a cadência de treino da cadência de deploy,
  que é o padrão em MLOps e é o que a arquitetura da Etapa 6 pressupõe.
- Detalhes de infraestrutura decorrentes:
  - o servidor MLflow passou a usar `--artifacts-destination
    file:/app/mlruns/artifacts` em vez de `--default-artifact-root
    file:/app/mlruns`. Com `--default-artifact-root` apontando para um
    caminho de arquivo, o cliente resolve o artefato como caminho **do
    próprio sistema de arquivos** — o script rodando no host tentaria
    escrever em `/app/mlruns`, que só existe dentro do container. Com
    `--serve-artifacts` (padrão) e `--artifacts-destination`, upload e
    download passam pelo servidor via HTTP, e host e containers funcionam
    igual, sem volume compartilhado;
  - o serviço `api` no `docker-compose.yml` recebe
    `MLFLOW_TRACKING_URI=http://mlflow:5000` (nome do serviço na rede do
    compose; `localhost` de dentro do container seria o próprio container)
    e um `depends_on: condition: service_healthy` contra um healthcheck do
    serviço `mlflow`, para não competir com o servidor ainda inicializando
    o banco;
  - a fixture `client` de `tests/test_api.py` passou a substituir
    `load_latest_policy` por uma política local. Sem isso, a suíte — hoje
    hermética e sem serviços externos — passaria a exigir um MLflow no ar
    para rodar.
- Consequências: a API ficou sem dependência do parquet (não lê mais dado
  bruto; o volume `./data/processed` no serviço `api` está mantido mas já
  não é usado por ela, e pode ser removido se ninguém for treinar dentro
  do container). O MLflow passou a ser o ponto único de verdade sobre qual
  política está sendo servida, que era exatamente a pendência deixada em
  aberto na Etapa 5. Se no futuro o grupo quiser promoção de versões
  (Staging/Production) ou rollback por versão, aí sim o Model Registry
  passa a ter função — e esta decisão deve ser revisitada.
