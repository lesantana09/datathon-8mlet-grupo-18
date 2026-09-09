"""Testes de contrato da API de recomendação (Etapa 5)."""

import pytest
from fastapi.testclient import TestClient

from datathon_mlet.api.entrypoints.main import app

VALID_CLIENT_PAYLOAD = {
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


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_returns_known_arm(client: TestClient) -> None:
    response = client.post("/recommendations", json=VALID_CLIENT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["recommended_action"] in {"cellular", "telephone"}


def test_recommendations_rejects_missing_field(client: TestClient) -> None:
    incomplete_payload = dict(VALID_CLIENT_PAYLOAD)
    del incomplete_payload["age"]

    response = client.post("/recommendations", json=incomplete_payload)

    assert response.status_code == 422


def test_recommendations_rejects_wrong_type(client: TestClient) -> None:
    invalid_payload = dict(VALID_CLIENT_PAYLOAD)
    invalid_payload["age"] = "trinta e sete"

    response = client.post("/recommendations", json=invalid_payload)

    assert response.status_code == 422


def test_recommendations_rejects_value_outside_allowed_options(
    client: TestClient,
) -> None:
    invalid_payload = dict(VALID_CLIENT_PAYLOAD)
    invalid_payload["marital"] = "namorando"

    response = client.post("/recommendations", json=invalid_payload)

    assert response.status_code == 422
