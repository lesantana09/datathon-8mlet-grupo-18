# DEC-005 - Arquitetura-alvo AWS: ECS Fargate + S3, sem implementação de código

- Status: Aceita
- Contexto: a Etapa 6 do enunciado pede 1-2 parágrafos (mais diagrama
  opcional) descrevendo como a solução rodaria em produção na AWS — não
  pede deploy real. O grupo já tinha, da Etapa 5, uma API FastAPI
  containerizada (Docker) que lê o parquet tratado via volume montado no
  `docker run`. A decisão precisava cobrir dois pontos: (1) qual serviço de
  compute hospedaria o container e (2) onde o dado processado viveria,
  além de (3) se essa etapa deveria incluir código (ex. um `Store` de
  acesso a S3) ou só o desenho da arquitetura.
- Alternativas de compute consideradas:
  - **AWS Lambda** (baixo custo, sem servidor): adequada para a API
    FastAPI isolada (stateless, request/response), mas incompatível com o
    MLflow tracking server que entra na Etapa 7 — é um processo
    persistente com UI própria, e Lambda tem limite de 15 min por
    invocação e não mantém estado entre chamadas. Usar Lambda pra API e
    outro serviço pra MLflow quebraria a coerência da arquitetura que o
    grupo quer apresentar (um cluster comportando os dois containers).
  - **AWS App Runner**: descartado — a AWS anunciou que não aceita mais
    novos clientes a partir de 30/04/2026, recomendando o Amazon ECS
    (Express Mode) como sucessor. Não faz sentido adotar um serviço em
    fim de vida para novos clientes numa arquitetura-alvo.
  - **Amazon ECS (Fargate)**: escolhida. Não exige gerenciar EC2
    (patch, capacidade), roda a mesma imagem Docker da Etapa 5 sem
    adaptação, e o modelo "um cluster, múltiplos serviços/tasks" comporta
    diretamente o MLflow como segundo container na Etapa 7 — o equivalente
    em nuvem do `docker-compose` local já cogitado (e adiado) na Etapa 5.
    ECS Express Mode fica citado como opção pra reduzir configuração
    manual (task definition, load balancer), sem trocar a escolha de
    plataforma.
- Alternativa de storage considerada:
  - **Amazon S3** vs **Amazon RDS**: escolhido S3. O dado hoje é um
    parquet estático, lido uma única vez no startup da API
    (`prepare_features` + `run_replay`); não há necessidade de banco
    transacional, e S3 tem custo menor pra esse padrão de acesso
    (leitura em lote, sem escrita concorrente).
- Decisão sobre escopo de código: **nenhum código foi alterado nesta
  etapa.** Foi levantada a possibilidade de já criar uma abstração de
  storage (`Store`) com injeção de dependência, pronta para múltiplos
  backends (local, S3, outros). Decisão do usuário: adiar — criar uma
  interface para "outros stores" sem um segundo backend real de uso
  imediato viola a convenção do próprio projeto de evitar abstração
  prematura (`CLAUDE.md`, "evite abstrações prematuras e dependências sem
  uso imediato"). A Etapa 6 fica só como documentação (README) e este
  registro de decisão; a troca do volume local por S3 (e uma eventual
  abstração de storage, só quando houver um segundo backend real) fica
  como trabalho futuro, fora do escopo das 9 etapas do enunciado.
- Consequências: o README ganha uma seção "Arquitetura-alvo em nuvem
  (Etapa 6)" com o desenho acima. Nenhum teste, dependência ou módulo novo
  foi adicionado. Se o grupo decidir futuramente implementar o deploy de
  fato, esta decisão deve ser revisitada com detalhes de rede (VPC,
  security groups), IAM e custo real medido — hoje é só o desenho
  conceitual pedido pelo enunciado.
