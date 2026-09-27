from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from core.config import settings
from core.logging import log_route, setup_logging
from datathon_mlet.policies import ThompsonSamplingPolicy
from datathon_mlet.policy_store import load_latest_policy
from datathon_mlet.train_and_publish_policy import CLEAN_DATASET_PATH, train_and_publish
from datathon_mlet.use_cases import recommend_channel
from domain.schemas import (
    BatchRecommendationItem,
    BatchRecommendationsRequest,
    BatchRecommendationsResponse,
    ClientContext,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfoResponse,
    ModelReloadResponse,
    RecommendationResponse,
    TrainRequest,
    TrainResponse,
)
from integrations import RegistryClient

logger = setup_logging()

client = RegistryClient()

router = APIRouter(prefix=settings.API_V1_STR + "/model", tags=["Model"])

# Configuração de autenticação básica
security = HTTPBasic()


def verify_api_key(credentials: HTTPBasicCredentials = Depends(security)):
    """Verifica as credenciais de autenticação básica para as rotas de tarefas."""
    correct_username = settings.API_USERNAME
    correct_password = settings.API_PASSWORD

    if (
        credentials.username != correct_username
        or credentials.password != correct_password
    ):
        logger.warning(
            f"Tentativa de acesso não autorizado com usuário: {credentials.username}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@log_route(logger=logger, include_args=True)
@router.post("/recommendation", response_model=RecommendationResponse)
@client.trace(name="recommend_endpoint", span_type="CHAIN")
async def recommend(
    request: Request,
    client_context: ClientContext,
    credentials: HTTPBasicCredentials = Depends(verify_api_key),
    ) -> RecommendationResponse:
    """Endpoint para recomendação do canal de contato ideal para o cliente.

    Recebe os atributos de contexto do cliente, aciona a política ativa
    registrada em memória e retorna o canal selecionado.

    Args:
        client_context: Payload contendo as features demográficas e financeiras do cliente.

    Returns:
        RecommendationResponse: Resposta contendo a ação recomendada.
    """
    policy: ThompsonSamplingPolicy = request.app.state.policy
    recommended_action = recommend_channel(policy)
    return RecommendationResponse(recommended_action=recommended_action)


@log_route(logger=logger, include_args=True)
@router.get("/", response_model=ModelInfoResponse)
@client.trace(name="model_info_endpoint", span_type="CHAIN")
async def model_info(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(verify_api_key),
    ) -> ModelInfoResponse:
    """Retorna os metadados e parâmetros da política atualmente servida pela API.

    Permite a auditoria do modelo em produção, inspecionando os valores de alpha,
    beta e as médias calculadas a posteriori para cada canal.

    Returns:
        ModelInfoResponse: Estado atual dos braços e decisão ótima do modelo.
    """
    policy = request.app.state.policy
    alpha = getattr(policy, "alpha", {})
    beta = getattr(policy, "beta", {})
    arms = list(alpha.keys()) if alpha else []

    posterior_means = {
        arm: alpha[arm] / (alpha[arm] + beta.get(arm, 1.0)) for arm in alpha
    }
    recommended = policy.recommend() if hasattr(policy, "recommend") else ""

    return ModelInfoResponse(
        model_type=type(policy).__name__,
        arms=arms,
        alpha_params=alpha,
        beta_params=beta,
        posterior_means=posterior_means,
        recommended_action=recommended,
    )


@log_route(logger=logger, include_args=True)
@router.patch("/feedback", response_model=FeedbackResponse)
@client.trace(name="feedback_endpoint", span_type="CHAIN")
async def feedback(
    request: Request,
    event: FeedbackRequest,
    credentials: HTTPBasicCredentials = Depends(verify_api_key),
    ) -> FeedbackResponse:
    """Registra o feedback de conversão observado para um canal (loop fechado do bandit).

    Atualiza em tempo real a distribuição Beta do braço acionado com a nova recompensa,
    promovendo o aprendizado contínuo (online learning) sem necessidade de reprocessamento
    completo de todo o histórico.

    Args:
        event: Dados da interação contendo o canal ofertado e a recompensa obtida (0 ou 1).

    Returns:
        FeedbackResponse: Confirmação da atualização e novos parâmetros do canal.

    Raises:
        HTTPException: Se o canal informado não fizer parte dos braços válidos da política.
    """
    import mlflow

    policy = request.app.state.policy
    if not hasattr(policy, "alpha") or event.arm not in policy.alpha:
        raise HTTPException(
            status_code=400,
            detail=f"Canal '{event.arm}' inválido. Braços disponíveis: {list(getattr(policy, 'alpha', {}).keys())}",
        )

    # Atualiza a política em memória
    policy.update(arm=event.arm, reward=event.reward)

    new_alpha = policy.alpha[event.arm]
    new_beta = policy.beta[event.arm]
    new_mean = new_alpha / (new_alpha + new_beta)

    # Rastreabilidade no MLflow Trace
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes(
            {
                "feedback_arm": event.arm,
                "feedback_reward": event.reward,
                "client_id": event.client_id or "anonymous",
                "updated_alpha": new_alpha,
                "updated_beta": new_beta,
                "new_posterior_mean": new_mean,
            }
        )

    logger.info(
        "Feedback registrado: arm=%s, reward=%d -> novo alpha=%.1f, beta=%.1f (média=%.3f)",
        event.arm,
        event.reward,
        new_alpha,
        new_beta,
        new_mean,
    )

    return FeedbackResponse(
        status="success",
        arm=event.arm,
        reward=event.reward,
        updated_alpha=new_alpha,
        updated_beta=new_beta,
        new_posterior_mean=new_mean,
    )


@log_route(logger=logger, include_args=True)
@router.patch("/", response_model=ModelReloadResponse)
@client.trace(name="model_reload_endpoint", span_type="CHAIN")
async def reload_model(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(verify_api_key),
    ) -> ModelReloadResponse:
    """Recarrega sob demanda a política mais recente publicada no MLflow ou S3.

    Atualiza atomicamente a política servida pela API em memória sem requerer
    o reinício da aplicação ou do container em produção.

    Returns:
        ModelReloadResponse: Confirmação e detalhes da política recarregada.

    Raises:
        HTTPException: Se não for possível recuperar nenhuma política do registro.
    """
    try:
        new_policy = load_latest_policy(fallback_to_s3=True)
        request.app.state.policy = new_policy

        alpha = getattr(new_policy, "alpha", {})
        beta = getattr(new_policy, "beta", {})
        recommended = new_policy.recommend() if hasattr(new_policy, "recommend") else ""

        logger.info("Modelo recarregado com sucesso no endpoint /model.")
        return ModelReloadResponse(
            status="success",
            message="Política mais recente recarregada com sucesso a partir do registro.",
            recommended_action=recommended,
            alpha_params=alpha,
            beta_params=beta,
        )
    except Exception as exc:
        logger.exception("Falha ao recarregar modelo a partir do registro: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Não foi possível recarregar a política do registro: {exc}",
        ) from exc


@log_route(logger=logger, include_args=True)
@router.patch("/batch", response_model=BatchRecommendationsResponse)
@client.trace(name="batch_recommendations_endpoint", span_type="CHAIN")
async def batch_recommendations(
    request: Request,
    request_data: BatchRecommendationsRequest,
    credentials: HTTPBasicCredentials = Depends(verify_api_key),
    ) -> BatchRecommendationsResponse:
    """Gera recomendações de canal em lote para múltiplos clientes.

    Ideal para operações em batch de campanhas de telemarketing diárias,
    minimizando o overhead de tráfego de rede e chamadas HTTP repetitivas.

    Args:
        request_data: Lista com os contextos individuais de cada cliente da base.

    Returns:
        BatchRecommendationsResponse: Lista consolidada de recomendações indexadas.
    """
    import mlflow

    policy: ThompsonSamplingPolicy = request.app.state.policy
    if not request_data.clients:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="A lista de clientes não pode estar vazia.",
        )
    recommendations: list[BatchRecommendationItem] = []

    for index, _ in enumerate(request_data.clients):
        # Para Thompson Sampling determinístico/média posterior na inferência
        action = recommend_channel(policy)
        recommendations.append(
            BatchRecommendationItem(index=index, recommended_action=action)
        )

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes(
            {
                "batch_size": len(request_data.clients),
                "recommended_action": policy.recommend()
                if hasattr(policy, "recommend")
                else "unknown",
            }
        )

    return BatchRecommendationsResponse(
        total=len(recommendations),
        recommendations=recommendations,
    )


@log_route(logger=logger, include_args=True)
@router.put("/train", response_model=TrainResponse)
@client.trace(name="train_endpoint", span_type="CHAIN")
async def train(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(verify_api_key),
    payload: TrainRequest | None = None,
    ) -> TrainResponse:
    """Endpoint para acionamento de treinamento e publicação contínua da política.

    Executa o treinamento da política Thompson Sampling utilizando os dados
    limpos, persiste o artefato correspondente no MLflow e atualiza o estado
    da aplicação em memória (hot-reload), permitindo que novas inferências
    utilizem imediatamente o modelo recém-treinado.

    Args:
        payload: Configurações opcionais de dataset, braços e semente aleatória.

    Returns:
        TrainResponse: Informações da política gerada, parâmetros aprendidos e run_id do MLflow.

    Raises:
        HTTPException: Se o arquivo de dataset indicado não for encontrado no disco.
    """
    request_data = payload or TrainRequest()
    dataset_file = (
        Path(request_data.dataset_path)
        if request_data.dataset_path
        else CLEAN_DATASET_PATH
    )

    if not dataset_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo de dataset não encontrado no caminho: {dataset_file}",
        )

    try:
        new_policy, run_id = train_and_publish(
            dataset_path=dataset_file,
            arms=request_data.arms,
            seed=request_data.seed,
        )
        # Hot-reload da política em memória na aplicação
        request.app.state.policy = new_policy

        return TrainResponse(
            status="success",
            run_id=run_id,
            recommended_action=new_policy.recommend(),
            alpha_params=new_policy.alpha,
            beta_params=new_policy.beta,
        )
    except Exception as exc:
        logger.exception("Falha ao executar treinamento via endpoint /train: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao treinar e publicar política: {exc}",
        ) from exc
