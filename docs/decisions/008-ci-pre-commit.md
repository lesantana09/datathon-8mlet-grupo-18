# DEC-008 - Integração contínua: GitHub Actions + pre-commit (Ruff + testes)

- Status: Aceita
- Contexto: o projeto não tinha nenhuma automação de verificação — testes e
  lint rodavam só quando alguém lembrava de rodar manualmente
  (`uv run pytest`, `uv run ruff check .`). Usuário pediu (1) um GitHub
  Actions mínimo rodando os testes a cada push na `main`, e (2) como
  melhoria, um `pre-commit` com testes e uma validação Ruff básica,
  integrado ao mesmo GitHub Actions — evitar duas fontes de verdade
  divergentes (o que roda local vs o que roda no CI).
- Decisão 1 (fonte única de verificação): **`.pre-commit-config.yaml`
  define os checks, e tanto o hook local quanto o CI rodam
  `pre-commit run --all-files`** — via `make pre-commit` (novo alvo no
  `Makefile`, mesmo padrão de `make test`/`make run`), não chamado direto
  no YAML. Alternativa descartada: escrever os comandos (`ruff check .`,
  `pytest`) direto no YAML do GitHub Actions, duplicando o que o
  pre-commit já define — divergiria com o tempo (ex. alguém muda o hook
  local e esquece do CI, ou vice-versa).
- Decisão 2 (hooks escolhidos): **`ruff-check`** (do repositório oficial
  `astral-sh/ruff-pre-commit`, pinado em `v0.16.5` — mesma versão do
  `ruff` travada no `uv.lock`) e um hook local de **`pytest`**
  (`language: system`, roda `uv run pytest` no ambiente do próprio
  projeto, não teria sentido isolar como os hooks de terceiros). Sem
  `ruff format` — o projeto não usa formatação automática do Ruff hoje;
  adicionar essa validação sem essa convenção já estabelecida seria
  escopo além do pedido ("validação Ruff básica").
- Decisão 3 (workflow do GitHub Actions): dispara só em `push` pra `main`
  (pedido explícito, "minimamente"). Usa `astral-sh/setup-uv` (não
  `actions/setup-python` + install manual do `uv`) e `uv sync --extra dev`
  — mesmo fluxo documentado no README pra desenvolvimento local, sem
  reencodar a instalação de dependências de outro jeito no CI. Cache de
  `~/.cache/pre-commit` via `actions/cache`, chaveado pelo hash do
  `.pre-commit-config.yaml`, pra não reinstalar o ambiente do `ruff-check`
  a cada execução.
- Validação: rodado de verdade antes de finalizar — `uv run pre-commit
  run --all-files` localmente (ambos os hooks passando) e o YAML do
  workflow verificado com `actionlint` (0 erros; as tags das actions
  (`actions/checkout@v7`, `astral-sh/setup-uv@v10`, `actions/cache@v6`)
  foram conferidas contra os releases reais no GitHub antes de fixar,
  não presumidas).
- Consequências: qualquer contribuidor precisa rodar `uv run pre-commit
  install` uma vez após clonar (documentado no README, nos 3 fluxos de
  instalação) pra ativar o hook local; sem isso, o commit local não é
  bloqueado, mas o push pra `main` ainda seria pego pelo CI. O workflow
  roda só em push direto — não em Pull Request; se o grupo passar a usar
  PRs pra `main`, vale adicionar o gatilho `pull_request` (não fizemos por
  não ter sido pedido e por não haver ainda um fluxo de PR estabelecido
  neste repositório).
