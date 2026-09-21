"""Schemas de entrada e saída da API de recomendação."""

from typing import Literal

from pydantic import BaseModel, Field

Job = Literal[
    "admin.",
    "blue-collar",
    "technician",
    "services",
    "management",
    "retired",
    "entrepreneur",
    "self-employed",
    "housemaid",
    "unemployed",
    "student",
    "unknown",
]
Marital = Literal["married", "single", "divorced", "unknown"]
Education = Literal[
    "university.degree",
    "high.school",
    "basic.9y",
    "professional.course",
    "basic.4y",
    "basic.6y",
    "unknown",
    "illiterate",
]
YesNoUnknown = Literal["no", "yes", "unknown"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]
Poutcome = Literal["nonexistent", "failure", "success"]


class ClientContext(BaseModel):
    """Dados do cliente disponíveis antes da decisão (contexto do bandit)."""

    age: int
    job: Job
    marital: Marital
    education: Education
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    month: Month
    day_of_week: DayOfWeek
    campaign: int
    pdays: int
    previous: int
    poutcome: Poutcome
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    foi_contatado_antes: bool

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "age": 37,
                    "job": "admin.",
                    "marital": "married",
                    "education": "university.degree",
                    "default": "no",
                    "housing": "no",
                    "loan": "no",
                    "month": "may",
                    "day_of_week": "mon",
                    "campaign": 1,
                    "pdays": 999,
                    "previous": 0,
                    "poutcome": "nonexistent",
                    "cons.price.idx": 93.994,
                    "cons.conf.idx": -36.4,
                    "euribor3m": 4.857,
                    "foi_contatado_antes": False,
                }
            ]
        },
    }


class RecommendationResponse(BaseModel):
    """Resposta da API: oferta (canal) recomendada pela política."""

    recommended_action: str = Field(examples=["cellular"])


class TrainRequest(BaseModel):
    """Payload de requisição para acionamento de treinamento e publicação da política."""

    dataset_path: str | None = Field(
        default=None,
        description="Caminho relativo ou absoluto para o arquivo parquet limpo. Se nulo, utiliza o dataset padrão.",
    )
    arms: list[str] = Field(
        default=["cellular", "telephone"],
        description="Lista de braços de decisão disponíveis para a política.",
    )
    seed: int = Field(
        default=0,
        description="Semente aleatória para reprodução do replay de treinamento.",
    )


class TrainResponse(BaseModel):
    """Resposta do processo de treinamento e publicação de nova versão do modelo."""

    status: str = Field(
        examples=["success"], description="Status da execução do treinamento."
    )
    run_id: str = Field(description="Identificador único do run gerado no MLflow.")
    recommended_action: str = Field(
        description="Canal ótimo recomendado pela nova política após o treinamento."
    )
    alpha_params: dict[str, float] = Field(
        description="Valores de alpha da distribuição Beta por braço."
    )
    beta_params: dict[str, float] = Field(
        description="Valores de beta da distribuição Beta por braço."
    )


class FeedbackRequest(BaseModel):
    """Payload para registro de recompensa observada no canal ofertado (loop fechado do bandit)."""

    arm: str = Field(
        examples=["cellular"],
        description="Nome do canal/braço de decisão que foi ofertado ao cliente.",
    )
    reward: int = Field(
        ge=0,
        le=1,
        examples=[1],
        description="Resultado da interação (0 = não conversão/recusa, 1 = conversão/adesão).",
    )
    client_id: str | None = Field(
        default=None,
        examples=["client_98745"],
        description="Identificador opcional do cliente para fins de auditoria e rastreabilidade.",
    )


class FeedbackResponse(BaseModel):
    """Resposta confirmando a atualização dos parâmetros da política após a recompensa."""

    status: str = Field(
        examples=["success"], description="Status da atualização dos parâmetros."
    )
    arm: str = Field(description="Canal que recebeu a atualização de recompensa.")
    reward: int = Field(description="Recompensa observada registrada.")
    updated_alpha: float = Field(
        description="Novo valor de alpha para o canal avaliado."
    )
    updated_beta: float = Field(description="Novo valor de beta para o canal avaliado.")
    new_posterior_mean: float = Field(
        description="Nova média a posteriori calculada para o canal."
    )


class ModelInfoResponse(BaseModel):
    """Metadados e parâmetros da política atualmente servida pela API."""

    model_type: str = Field(
        examples=["ThompsonSamplingPolicy"],
        description="Tipo do algoritmo da política.",
    )
    arms: list[str] = Field(
        description="Lista de braços de decisão registrados na política."
    )
    alpha_params: dict[str, float] = Field(
        description="Parâmetros alpha atuais de cada braço."
    )
    beta_params: dict[str, float] = Field(
        description="Parâmetros beta atuais de cada braço."
    )
    posterior_means: dict[str, float] = Field(
        description="Médias posteriores estimadas por canal."
    )
    recommended_action: str = Field(
        description="Canal recomendado no momento com base nas médias."
    )


class ModelReloadResponse(BaseModel):
    """Resposta da operação de recarregamento do modelo a partir do registro."""

    status: str = Field(
        examples=["success"], description="Status da recarga do modelo."
    )
    message: str = Field(description="Mensagem informativa sobre a sincronização.")
    recommended_action: str = Field(
        description="Canal recomendado pela política recarregada."
    )
    alpha_params: dict[str, float] = Field(
        description="Parâmetros alpha da política recarregada."
    )
    beta_params: dict[str, float] = Field(
        description="Parâmetros beta da política recarregada."
    )


class BatchRecommendationItem(BaseModel):
    """Item individual de recomendação para um cliente no processamento em lote."""

    index: int = Field(description="Posição do cliente na lista enviada.")
    recommended_action: str = Field(
        examples=["cellular"], description="Canal recomendado para o cliente."
    )


class BatchRecommendationsRequest(BaseModel):
    """Payload para solicitação de recomendações de múltiplos clientes em lote."""

    clients: list[ClientContext] = Field(
        min_length=1,
        description="Lista contendo os contextos dos clientes a serem processados.",
    )


class BatchRecommendationsResponse(BaseModel):
    """Resposta contendo a lista de recomendações geradas para o lote de clientes."""

    total: int = Field(description="Total de clientes processados no lote.")
    recommendations: list[BatchRecommendationItem] = Field(
        description="Lista com as recomendações geradas em ordem correspondente.",
    )

