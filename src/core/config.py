import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Lê a versão do pyproject.toml globalmente para evitar que variáveis temporárias
# (como arquivos abertos) sejam capturadas pela classe Settings
_pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
with _pyproject_path.open("rb") as _f:
    _pyproject_data = tomllib.load(_f)


class Settings(BaseSettings):
    # ==========================================
    # Configurações Globais da Aplicação
    # ==========================================
    PROJECT_NAME: str = _pyproject_data["project"]["name"]
    PROJECT_VERSION: str = _pyproject_data["project"]["version"]
    PROJECT_DESCRIPTION: str = _pyproject_data["project"]["description"]
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:8000"]

    # ==========================================
    # Model Registry & Tracking (DagsHub / MLflow / OPTUNA)
    # ==========================================
    DAGSHUB_APP_TOKEN: str
    MLFLOW_TRACKING_URI: str
    MLFLOW_EXPERIMENT_NAME: str
    MLFLOW_TRACKING_USERNAME: str

    # ==========================================
    # Autenticação
    # ==========================================
    JWT_SECRET_KEY: str | None = None
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60
    USER_DATABASE_PATH: str = "data/users.db"

    @model_validator(mode="after")
    def set_development_jwt_secret(self) -> "Settings":
        """Gera uma chave determinística apenas para o ambiente local."""
        if self.JWT_SECRET_KEY:
            return self
        if self.ENVIRONMENT == "production":
            raise ValueError("JWT_SECRET_KEY é obrigatório em produção.")
        self.JWT_SECRET_KEY = hashlib.sha256(
            self.DAGSHUB_APP_TOKEN.encode("utf-8")
        ).hexdigest()
        return self

    # ==========================================
    # Configurações AWS
    # ==========================================
    AWS_REGION: str
    AWS_ENDPOINT_URL: str
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str

    # Configuração do Pydantic para ler do arquivo .env quando rodar localmente
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignora variáveis no .env que não estejam mapeadas aqui
    )


# Instanciamos a classe uma única vez.
# Isso funciona como um Singleton para toda a aplicação.
settings = Settings()
