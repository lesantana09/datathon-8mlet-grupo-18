"""Testes de contrato da API de recomendação (Etapa 5)."""

import pytest
from fastapi.testclient import TestClient

from core.config import settings
from datathon_mlet.api.entrypoints import main
from datathon_mlet.api.entrypoints.main import app
from datathon_mlet.policies import FixedPolicy, ThompsonSamplingPolicy

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
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Substitui o carregamento da política no startup por uma política local.

    A API carrega a política do MLflow no `lifespan`; sem esse patch, a
    suíte passaria a depender de um servidor MLflow no ar (ver
    `docs/decisions/007-api-carrega-policy-do-mlflow.md`).
    """
    monkeypatch.setattr(
        main,
        "load_latest_policy",
        lambda: ThompsonSamplingPolicy(arms=["cellular", "telephone"]),
    )
    monkeypatch.setattr(settings, "USER_DATABASE_PATH", str(tmp_path / "users.db"))

    with TestClient(app) as test_client:
        register_response = test_client.post(
            "/user/register",
            json={"username": "test_user", "password": "senha segura"},
        )
        assert register_response.status_code == 201
        login_response = test_client.post(
            "/user/login",
            json={"username": "test_user", "password": "senha segura"},
        )
        assert login_response.status_code == 200
        test_client.headers.update(
            {"Authorization": f"Bearer {login_response.json()['access_token']}"}
        )
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_protected_endpoint_rejects_missing_token(client: TestClient) -> None:
    client.headers.pop("Authorization")

    response = client.get("/model/info")

    assert response.status_code == 401


def test_logout_revokes_current_token(client: TestClient) -> None:
    response = client.post("/user/logout")

    assert response.status_code == 200
    assert client.get("/model/info").status_code == 401


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


def test_recommendations_follows_injected_policy(client: TestClient) -> None:
    """A rota devolve o que a política manda — não personaliza pelo payload.

    Substitui a política treinada (dado real) por uma enviesada pra
    `telephone`, provando que a decisão vem só da política, nunca do
    contexto do cliente (bandit não-contextual, decisão da Etapa 3).
    """
    app.state.policy = FixedPolicy(arm="telephone")

    response = client.post("/recommendations", json=VALID_CLIENT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["recommended_action"] == "telephone"


def test_train_endpoint_success_and_hot_reloads_policy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Valida se POST /train publica o modelo e atualiza app.state.policy com hot reload."""
    fake_dataset = tmp_path / "fake_clean.parquet"
    fake_dataset.touch()

    mock_policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    mock_policy.alpha["cellular"] = 50.0
    mock_policy.beta["cellular"] = 5.0

    monkeypatch.setattr(
        main,
        "train_and_publish",
        lambda dataset_path, arms, seed: (mock_policy, "fake_run_123"),
    )

    response = client.post("/train", json={"dataset_path": str(fake_dataset)})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["run_id"] == "fake_run_123"
    assert data["recommended_action"] == "cellular"
    assert data["alpha_params"]["cellular"] == 50.0
    # Verifica hot-reload no app.state.policy
    assert app.state.policy.alpha["cellular"] == 50.0


def test_train_endpoint_returns_404_when_dataset_not_found(client: TestClient) -> None:
    """Valida se POST /train retorna 404 quando o arquivo de dataset não existe."""
    response = client.post(
        "/train", json={"dataset_path": "caminho/para/arquivo/inexistente.parquet"}
    )

    assert response.status_code == 404
    assert "não encontrado" in response.json()["detail"]


def test_model_info_returns_current_policy_state(client: TestClient) -> None:
    """Valida se GET /model/info expõe os parâmetros atuais e médias dos braços."""
    response = client.get("/model/info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "ThompsonSamplingPolicy"
    assert set(data["arms"]) == {"cellular", "telephone"}
    assert "cellular" in data["alpha_params"]
    assert "telephone" in data["beta_params"]
    assert "posterior_means" in data
    assert data["recommended_action"] in {"cellular", "telephone"}


def test_feedback_updates_arm_distribution_in_real_time(client: TestClient) -> None:
    """Valida se POST /feedback atualiza os parâmetros da política em memória (loop fechado)."""
    # Consulta estado inicial
    initial_info = client.get("/model/info").json()
    prev_alpha = initial_info["alpha_params"]["cellular"]

    # Envia conversão de sucesso para o braço 'cellular'
    feedback_payload = {
        "arm": "cellular",
        "reward": 1,
        "client_id": "test_client_001",
    }
    response = client.post("/feedback", json=feedback_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["arm"] == "cellular"
    assert data["reward"] == 1
    assert data["updated_alpha"] == prev_alpha + 1

    # Confirma que o estado interno do app.state.policy foi de fato atualizado
    updated_info = client.get("/model/info").json()
    assert updated_info["alpha_params"]["cellular"] == prev_alpha + 1


def test_feedback_rejects_invalid_arm(client: TestClient) -> None:
    """Valida se POST /feedback retorna 400 para canais desconhecidos."""
    response = client.post(
        "/feedback",
        json={"arm": "canal_inexistente", "reward": 1},
    )

    assert response.status_code == 400
    assert "inválido" in response.json()["detail"]


def test_feedback_validates_binary_reward(client: TestClient) -> None:
    """Valida se POST /feedback retorna 422 para recompensas fora de [0, 1]."""
    response = client.post(
        "/feedback",
        json={"arm": "cellular", "reward": 5},
    )

    assert response.status_code == 422


def test_batch_recommendations_returns_ordered_recommendations(
    client: TestClient,
) -> None:
    """Valida se POST /batch-recommendations processa lista de clientes e retorna itens indexados."""
    batch_payload = {
        "clients": [VALID_CLIENT_PAYLOAD, VALID_CLIENT_PAYLOAD, VALID_CLIENT_PAYLOAD]
    }
    response = client.post("/batch-recommendations", json=batch_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["recommendations"]) == 3
    assert data["recommendations"][0]["index"] == 0
    assert data["recommendations"][1]["index"] == 1
    assert data["recommendations"][2]["index"] == 2
    for item in data["recommendations"]:
        assert item["recommended_action"] in {"cellular", "telephone"}


def test_batch_recommendations_rejects_empty_clients_list(client: TestClient) -> None:
    """Valida se POST /batch-recommendations rejeita lista vazia de clientes."""
    response = client.post("/batch-recommendations", json={"clients": []})

    assert response.status_code == 422


def test_model_reload_endpoint_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valida se POST /model/reload atualiza a política em app.state.policy com sucesso."""
    new_policy = ThompsonSamplingPolicy(arms=["cellular", "telephone"])
    new_policy.alpha["cellular"] = 99.0
    new_policy.beta["cellular"] = 1.0

    monkeypatch.setattr(main, "load_latest_policy", lambda fallback_to_s3: new_policy)

    response = client.post("/model/reload")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["recommended_action"] == "cellular"
    assert data["alpha_params"]["cellular"] == 99.0
    assert app.state.policy.alpha["cellular"] == 99.0


def test_model_reload_endpoint_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valida se POST /model/reload retorna 500 caso ocorra falha de busca no registro."""

    def _raise_error(fallback_to_s3: bool):
        raise RuntimeError("MLflow e S3 inacessíveis")

    monkeypatch.setattr(main, "load_latest_policy", _raise_error)

    response = client.post("/model/reload")

    assert response.status_code == 500
    assert "Não foi possível recarregar a política" in response.json()["detail"]
