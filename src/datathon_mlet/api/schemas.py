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
