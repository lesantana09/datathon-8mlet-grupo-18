"""Schemas de entrada e saída da API de recomendação."""

from pydantic import BaseModel, Field


class ClientContext(BaseModel):
    """Dados do cliente disponíveis antes da decisão (contexto do bandit)."""

    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    foi_contatado_antes: bool

    model_config = {"populate_by_name": True}


class RecommendationResponse(BaseModel):
    """Resposta da API: oferta (canal) recomendada pela política."""

    recommended_action: str
