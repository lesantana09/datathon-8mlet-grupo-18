# DEC-001 - Arquitetura da preparação de dados (Etapa 2)

- Status: Aceita
- Contexto: Etapa 2 do Datathon exige contexto (`X`), ação e recompensa
  prontos para o baseline e a política adaptativa da Etapa 3. Era preciso
  decidir o nível de abstração do código em `src/datathon_mlet/` — desde
  funções soltas até uma camada formal de repository/adapter.
- Alternativas consideradas:
  1. Só funções soltas, sem tipo de retorno nomeado.
  2. Repository (leitura de dado) + Adapter (transformação) como camadas
     formais desde já.
  3. Modelos tipados (dataclass) + funções puras, sem camada de indireção
     extra.
- Decisão: alternativa 3. `PreparedDataset` (dataclass frozen com `context`,
  `action`, `reward`) em `src/datathon_mlet/models.py`; `load_clean_dataset`
  e `prepare_features` em `src/datathon_mlet/data_prep.py`.
- Consequências: contrato explícito e testável sem introduzir camadas sem
  uso imediato (só existe uma fonte de dado local hoje). Repository/adapter
  fica como evolução natural se a Etapa 5 (API) ou uma troca de fonte de
  dado exigir — a migração é embrulhar as funções existentes, não reescrever.
