# DEC-002 - Formulação do bandit e avaliação offline (Etapa 3)

- Status: Aceita
- Contexto: Etapa 3 exige baseline (regra fixa) e política adaptativa
  (Thompson Sampling ou Epsilon-Greedy) com métrica comparativa mostrando a
  segunda superando a primeira. Duas decisões precisavam ser tomadas: (1)
  profundidade do bandit — usa o contexto (`X`, Etapa 2) para decidir, ou
  aprende só por braço; (2) como simular decisões sequenciais sobre um
  dataset histórico e fechado, sem inventar contrafactuais (regra
  obrigatória do projeto).

## Profundidade do bandit

- Alternativas consideradas:
  1. Não-contextual: Thompson Sampling Beta-Bernoulli por braço (`contact`),
     sem olhar pro cliente.
  2. Contextual por segmento: 1 Thompson Sampling independente por grupo
     (ex. `poutcome`).
  3. Contextual com modelo: regressão logística bayesiana por braço,
     usando as 17 colunas de contexto.
- Decisão: alternativa 1. O enunciado descreve a recompensa como "esperada
  por braço" (não por cliente); o gap de conversão entre `cellular` e
  `telephone` é forte o bastante para um resultado nítido sem contexto; e a
  simplicidade é prioridade explícita do usuário, que precisa apresentar e
  defender o projeto no Demo Day.
- Consequências: `context` (Etapa 2) não é usado na decisão do bandit nesta
  etapa — fica reservado para o Golden Set (Etapa 4) e como extensão futura
  documentada, não descartado nem escondido.

## Método de avaliação offline

- Problema: o canal historicamente atribuído a cada cliente não foi
  sorteado aleatoriamente (achado da EDA, passo 6 — confundido com regime
  econômico) e cada cliente só tem 1 resultado observado. Não é possível
  saber o que teria acontecido se um cliente tivesse recebido o outro
  canal — inventar esse valor violaria a regra do projeto contra
  contrafactuais fabricados.
- Decisão: método de replay (Li et al., 2011) — embaralha os clientes e só
  conta uma rodada quando a ação escolhida pela política coincide com o
  canal real do cliente no histórico; caso contrário, descarta o cliente.
  Implementado em `src/datathon_mlet/replay.py` (`run_replay`).
- Consequências: nenhum dado inventado, mas o estimador só é
  estatisticamente não-enviesado se a atribuição histórica tivesse sido
  aleatória — não foi. O resultado é reportado como avaliação offline sobre
  dado observacional, nunca como efeito causal do canal (mesmo caveat já
  documentado no README para o achado do `contact`).

## Escolha do baseline

- Problema: o baseline inicialmente escolhido (`FixedPolicy("cellular")`,
  o canal com melhor conversão histórica) é um oráculo — ele já começa
  sabendo a resposta certa. Por teoria de bandit, nenhuma política que
  precisa explorar consegue superar, na média, um oráculo (regret ≥ 0);
  o teste empírico confirmou empate estatístico (Thompson Sampling 14,70%
  vs baseline oráculo 14,74%, 1 seed).
- Alternativas consideradas:
  1. Manter o oráculo e reportar o empate como resultado esperado,
     explicando a teoria de regret.
  2. Trocar o baseline para uma regra fixa arbitrária (`telephone`, sem
     análise de dado por trás), deixando espaço real para a política
     adaptativa aprender e melhorar.
  3. Manter os dois baselines lado a lado (oráculo como teto de referência,
     regra arbitrária como baseline principal).
- Decisão: alternativa 2, escolhida pelo usuário. O enunciado permite
  baseline = "regra fixa" **ou** "melhor braço histórico" — são duas
  opções igualmente válidas, não uma preferencial. Baseline final:
  `FixedPolicy("telephone")`.
- Resultado (`notebooks/03_baseline_vs_ts.ipynb`): baseline = 5,23%
  (determinístico, `rounds_used=15044`); Thompson Sampling = 14,69% ± 0,02
  p.p. (média ± desvio-padrão, 20 seeds) — ganho de ~2,8×.
- Consequências: o teste com o baseline oráculo não é descartado — fica
  documentado no notebook como achado intermediário, evidência de que a
  política adaptativa converge para o braço ótimo em vez de resultado
  cherry-picked.
