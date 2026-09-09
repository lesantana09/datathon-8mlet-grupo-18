# DEC-004 - Formulação do braço: canal, não produto/oferta

- Status: Aceita
- Contexto: o enunciado descreve o problema como "decidir, em diferentes
  canais, qual oferta, mensagem ou próximo passo apresentar para cada
  cliente elegível" (página 2 do PDF). Lida literalmente, essa frase
  sugere que o canal é o cenário da decisão e a variável de decisão
  seria a oferta/mensagem/próximo passo — não o canal em si. Essa
  ambiguidade ficou registrada como pendência desde a Etapa 1
  (`.ai/PROJECT_STATUS.md`, "Decisões pendentes de validação") e precisava
  ser fechada antes da apresentação final, já que muda a forma como o
  grupo explica a formulação escolhida.
- Análise: a base escolhida (Bank Marketing, `henriqueyamahata`) testa
  **uma única oferta/produto** ao longo de toda a campanha — depósito a
  prazo (`y`). Não existe no dataset nenhuma coluna que represente
  variação de produto ou mensagem oferecida; o único atributo que varia
  entre clientes com reflexo direto em decisão de negócio é `contact`
  (`cellular`/`telephone`). Atributos como `housing` e `loan` (o cliente já
  tem financiamento imobiliário ou empréstimo pessoal) foram cogitados como
  possíveis "ofertas", mas são estado pré-existente do cliente — coletados
  antes da campanha atual, não uma ação testada pelo banco nela.
- Decisão: braço do bandit = categorias de `contact` (canal), não
  produto/oferta. Não é uma escolha de conveniência contra o enunciado — é
  a única formulação de braço que a base escolhida sustenta com dados
  reais. O enunciado explicitamente permite essa liberdade, desde que a
  escolha seja explicada ("o grupo pode implementar Thompson Sampling...
  desde que explique a escolha, mostre como o contexto entra na decisão").
- Consequências: ao apresentar o projeto, a defesa da formulação é "a base
  pública escolhida só tinha uma oferta e uma dimensão de decisão
  observável nos dados (canal); formulamos o bandit em torno do que os
  dados sustentam, não do que seria ideal em teoria." Se o grupo trocasse
  de base para uma com múltiplas ofertas/produtos candidatos, essa decisão
  precisaria ser revisitada.
