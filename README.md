# Datathon MLET — FIAP POSTECH

## Visão geral

Projeto acadêmico final da pós-graduação em Machine Learning Engineering da FIAP
POSTECH. O Datathon propõe uma solução end-to-end para apoiar a escolha
adaptativa de um canal, oferta, mensagem ou próximo passo para clientes elegíveis
de uma instituição financeira.

Etapas 0 (organização do repositório), 1 (base Kaggle e EDA) e 2 (preparação
da base) estão concluídas. Ainda não há baseline, política adaptativa,
avaliação, serviço demonstrável, arquitetura em nuvem, MLOps ou apresentação
final — ver checklist abaixo.

## Problema de negócio

Uma instituição financeira precisa decidir como abordar cada cliente elegível sem
depender apenas de uma estratégia fixa. A decisão deve considerar o contexto
disponível e permitir aprendizado controlado a partir das respostas observadas,
respeitando privacidade, governança e limitações de inferência.

## Objetivo da solução

Construir e avaliar, nas próximas etapas, uma solução reprodutível que compare uma
política fixa com uma política adaptativa para recomendar uma ação entre opções
elegíveis. A definição final do problema dependerá da qualidade e das limitações
identificadas nos dados.

## Base de dados

A base escolhida é o conjunto público
[Bank Marketing, no Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data)
(`henriqueyamahata/bank-marketing`). O arquivo usado é `bank-additional-full.csv`
(41.188 linhas, 21 colunas), e a variável-alvo é `y`, que indica adesão a um
depósito a prazo (`p̂ = 0.1127`, base desbalanceada ~89/11). O download é feito
via `kagglehub` no notebook `notebooks/01_eda.ipynb`; o dataset não é versionado
(`data/raw/` e `data/processed/` são ignorados pelo Git, exceto `.gitkeep`).

## Formulação inicial (provisória)

As definições abaixo são hipóteses de trabalho e não representam uma solução já
implementada:

- **Contexto:** atributos permitidos do cliente e da interação disponíveis antes
  da decisão.
- **Braços:** categorias observadas na coluna `contact`, candidatas a representar
  os canais de contato.
- **Recompensa:** resposta Bernoulli derivada de `y`, indicando adesão ou não ao
  depósito a prazo.
- **Baseline:** política fixa baseada no canal com melhor desempenho histórico.
- **Política adaptativa:** Thompson Sampling com recompensa Bernoulli.

Essa formulação foi parcialmente validada na EDA (`notebooks/01_eda.ipynb`):
`contact` (braço candidato) mostrou gap de conversão real entre canais
(cellular 14,7% vs telephone 5,2%), mas esse gap está confundido com regime
econômico — `telephone` concentra 89% dos contatos em maio/junho, período de
`emp.var.rate` positivo, enquanto `cellular` concentra em meses de
`emp.var.rate` negativo (crise 2008–2010). Por isso o resultado é reportado
como associação observacional, não como efeito causal do canal.

## Preparação da base (Etapa 2)

`src/datathon_mlet/data_prep.py` transforma o dataset tratado da Etapa 1 em
contexto, ação e recompensa (`PreparedDataset`, em
`src/datathon_mlet/models.py`), consumido por `notebooks/02_preparacao.ipynb`:

- **`context`** (17 colunas): todas as features do cliente, exceto `contact`
  (é a ação), `y` (é o alvo) e `emp.var.rate`/`nr.employed` (redundantes com
  `euribor3m`, correlação 0,91–0,97 — ver EDA);
- **`action`**: a coluna `contact` (`cellular`/`telephone`), o braço do
  bandit;
- **`reward`**: `y` binarizado (`yes` → 1, `no` → 0).

A função valida a entrada e falha explicitamente (`ValueError`) se a coluna
`duration` estiver presente, evitando reintroduzir vazamento por engano.
Decisão de arquitetura: modelos tipados (`PreparedDataset`) + funções puras,
sem camada de repository/adapter por enquanto — só existe uma fonte de dado
local hoje; a migração fica fácil se a Etapa 5 (API) ou uma troca de fonte
exigir.

## Stack tecnológica

- Python 3.11;
- pandas, NumPy e scikit-learn;
- MLflow;
- FastAPI, Uvicorn e Pydantic;
- Jupyter, matplotlib e seaborn;
- pytest, HTTPX e Ruff;
- AWS como arquitetura-alvo futura;
- `uv` para ambiente e dependências.

## Estrutura do repositório

```text
.
├── artifacts/              # artefatos gerados (não versionados)
├── data/
│   ├── processed/          # dados processados (não versionados)
│   └── raw/                # dados brutos (não versionados)
├── notebooks/              # notebooks definitivos do projeto
├── src/datathon_mlet/      # pacote Python
├── tests/                  # testes automatizados
├── .python-version
├── .gitignore
├── pyproject.toml
└── README.md
```

Os diretórios vazios são mantidos no Git por arquivos `.gitkeep`; dados e
artefatos adicionados a eles permanecem ignorados.

## Instalação e execução local

Há dois caminhos de instalação. O caminho com `uv` é o recomendado por oferecer
um fluxo mais rápido e reprodutível. Escolha apenas um dos caminhos e não misture
seus comandos durante a mesma instalação.

### Caminho recomendado — uv

O [`uv`](https://docs.astral.sh/uv/) é um gerenciador de projetos, ambientes e
dependências Python. O comando `uv sync` cria e gerencia automaticamente o
ambiente `.venv`; portanto, não execute `python -m venv .venv` antes dele.

Instale o `uv` de acordo com seu sistema operacional.

#### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### macOS

Com Homebrew:

```bash
brew install uv
```

Como alternativa, use o instalador oficial:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows com WinGet

```powershell
winget install --id=astral-sh.uv -e
```

#### Windows com PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Após instalar o `uv`, clone o projeto, instale as dependências de execução e de
desenvolvimento e execute as validações:

```bash
git clone <URL_DO_REPOSITORIO>
cd datathon-7mlet-grupo-18

uv sync --extra dev
uv run pytest
uv run ruff check .
```

Nesse fluxo:

- `.venv/` é criado e gerenciado automaticamente pelo `uv`;
- `pyproject.toml` declara as dependências do projeto;
- `uv.lock` registra as versões exatas resolvidas e deve ser versionado;
- `.venv/` é um diretório local e não deve ser versionado;
- os comandos do projeto devem ser executados preferencialmente com `uv run`.

### Notebook de EDA

Com o ambiente instalado (`uv sync --extra dev` ou equivalente), rode:

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

A primeira célula de download baixa a base via `kagglehub` — se pedir
autenticação, configure `~/.kaggle/kaggle.json` ou use `kagglehub.login()`. O
notebook gera `data/processed/bank_marketing_clean.parquet`, consumido pelas
próximas etapas.

### Caminho alternativo — venv e pip

Esse caminho usa as ferramentas tradicionais incluídas no Python. O extra de
desenvolvimento `dev`, declarado no `pyproject.toml`, instala os recursos de
notebooks, testes, lint e visualização.

#### Linux e macOS

```bash
git clone <URL_DO_REPOSITORIO>
cd datathon-7mlet-grupo-18

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest
ruff check .
```

Dependendo da instalação do Python, o executável pode se chamar `python` em vez
de `python3`. Nesse caso, crie o ambiente com:

```bash
python -m venv .venv
```

#### Windows PowerShell

```powershell
git clone <URL_DO_REPOSITORIO>
cd datathon-7mlet-grupo-18

py -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest
ruff check .
```

#### Windows Prompt de Comando

Para criar e ativar o ambiente no Prompt de Comando (`cmd`), use:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Depois de ativá-lo, execute os mesmos comandos `pip`, `pytest` e `ruff` mostrados
no fluxo do PowerShell. Para sair do ambiente virtual em qualquer sistema:

```bash
deactivate
```

### Verificação e solução de problemas

Confira as ferramentas disponíveis com:

```bash
python --version
uv --version
```

O projeto exige Python **3.11 ou superior e anterior ao Python 3.14**, conforme
`requires-python = ">=3.11,<3.14"` no `pyproject.toml`.

- Se `uv` não for encontrado logo após a instalação, feche e abra novamente o
  terminal para atualizar o `PATH`.
- Nunca versione `.venv/`; o diretório já está protegido pelo `.gitignore`.
- Nunca coloque credenciais do Kaggle ou da AWS no README ou em outros arquivos
  versionados.
- Não misture os comandos do caminho `uv` com os do caminho `venv` e `pip` na
  mesma instalação.
- Para instalação, atualização e diagnóstico do `uv`, consulte a
  [documentação oficial](https://docs.astral.sh/uv/).

## Privacidade, governança e limitações

- Dados reais de clientes não serão utilizados; o trabalho usará apenas a base
  pública selecionada e dados permitidos pelo enunciado.
- Atributos proibidos pelo enunciado não serão usados.
- A coluna `duration` foi excluída das features por representar vazamento
  temporal: seu valor só é conhecido após a interação.
- `pdays` foi binarizada (`foi_contatado_antes`) por conter um valor sentinela
  (999 = "nunca contatado antes") em 96,3% das linhas, incompatível com
  tratamento como escala numérica contínua.
- Ausência de dado é codificada no dataset original como categoria
  `"unknown"`, não como nulo técnico; ela é mantida como categoria própria
  (não imputada), já que em `default` chega a 20,9% das linhas.
- O gap de conversão observado entre canais de `contact` está confundido com
  regime econômico (ver "Formulação inicial") e é reportado como associação,
  não como efeito causal.
- Resultados de simulação não serão apresentados como evidência causal.
- Dados brutos, dados processados, credenciais, execuções locais do MLflow e
  artefatos gerados não devem ser versionados.
- A formulação, os critérios de elegibilidade, os riscos de viés e as hipóteses de
  avaliação serão documentados e revistos ao longo do projeto.

## Integrantes

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
- [ ] Etapa 3 — Baseline e estratégia algorítmica
- [ ] Etapa 4 — Avaliação e casos de teste
- [ ] Etapa 5 — Serviço ou interface demonstrável
- [ ] Etapa 6 — Arquitetura-alvo em nuvem
- [ ] Etapa 7 — Ciclo de vida MLOps
- [ ] Etapa 8 — Apresentação final (Demo Day)

Detalhamento e evidências de cada etapa em `.ai/PROJECT_STATUS.md`.
