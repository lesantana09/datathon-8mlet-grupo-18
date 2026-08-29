# Datathon MLET — FIAP POSTECH

## Visão geral

Projeto acadêmico final da pós-graduação em Machine Learning Engineering da FIAP
POSTECH. O Datathon propõe uma solução end-to-end para apoiar a escolha
adaptativa de um canal, oferta, mensagem ou próximo passo para clientes elegíveis
de uma instituição financeira.

Esta Etapa 0 estabelece somente a organização inicial e executável do repositório.
Ainda não há análise exploratória, preparação de dados, treinamento, política
adaptativa, API funcional ou infraestrutura em nuvem.

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

A base inicialmente escolhida é o conjunto público
[Bank Marketing, no Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data).
O arquivo previsto para uso futuro é `bank-additional-full.csv`, e a variável-alvo
é `y`, que indica adesão a um depósito a prazo. O dataset não é baixado nem
versionado nesta etapa.

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

Essa formulação será validada e poderá ser revisada durante a EDA. Em particular,
a base observacional pode não sustentar comparações justas entre canais sem
hipóteses adicionais.

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
- A coluna `duration` será excluída das features por representar vazamento
  temporal: seu valor só é conhecido após a interação.
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

- [x] Etapa 0 — Estrutura inicial do repositório
- [ ] Etapa 1 — Entendimento e aquisição dos dados
- [ ] Etapa 2 — Análise exploratória dos dados
- [ ] Etapa 3 — Preparação e validação dos dados
- [ ] Etapa 4 — Baseline e avaliação offline
- [ ] Etapa 5 — Política adaptativa
- [ ] Etapa 6 — Rastreamento de experimentos e empacotamento
- [ ] Etapa 7 — API e arquitetura-alvo
- [ ] Etapa 8 — Validação final e documentação
