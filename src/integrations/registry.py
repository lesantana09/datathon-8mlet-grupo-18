"""
Cliente mlflow customizado para o registry de experimentos.

Exemplo de uso
--------------
    from infrastructure.registry import RegistryClient

    client = RegistryClient()

    @client.trace(span_type="CHAIN", name="teste_trace")
    def teste_trace():
        with client.start_run() as run:
            client.log_params("teste", "teste")
            client.log_metrics("teste", "teste")
            client.log_artifact("teste", "teste")

    teste_trace()

"""

from __future__ import annotations

import functools
import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import (
    log_artifact,
    log_metric,
    log_metrics,
    log_param,
    log_params,
    set_experiment,
    start_run,
)

from core.config import settings
from core.logging import setup_logging

logger = setup_logging()


class RegistryClient(MlflowClient):
    """Cliente CRUD para o registry de experimentos MLflow.

    Encapsula toda a lógica de conexão MLflow.

    Attributes:
        client (mlflow.tracking.MlflowClient): Cliente MLflow conectado ao
            servidor de tracking.
        experiment_name (str): Nome do experimento MLflow em uso.
        experiment_id (str): ID do experimento MLflow em uso.
        trace (function): Função decoradora para rastreamento de experimentos.
        log_artifact (function): Função para logar artifacts.
        log_metrics (function): Função para logar métricas.
        log_params (function): Função para logar parâmetros.
        set_experiment (function): Função para setar o experimento.
        start_run (function): Função para iniciar um run.

    Example:
        >>> client = RegistryClient()
        >>> with client.start_run() as run:
        >>>     client.log_params("teste", "teste")
        >>>     client.log_metrics("teste", "teste")
        >>>     client.log_artifact("teste", "teste")
    """

    def __init__(self) -> None:
        """Inicializa o RegistryClient.

        Configura a autenticação com o DagsHub, conecta o MlflowClient ao
        servidor de tracking remoto,
        e resolve (ou cria) o experimento de MLflow definido em
        `settings.MLFLOW_EXPERIMENT_NAME`.

        Raises:
            ValueError: Se `settings.DAGSHUB_APP_TOKEN` não estiver configurado.
            MlflowException: Se não for possível conectar ou criar o
                experimento no servidor de tracking.
        """
        super().__init__()
        os.environ["MLFLOW_LOGGING_LEVEL"] = "DEBUG"
        os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
        os.environ["MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE"] = "2000"
        os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.DAGSHUB_APP_TOKEN
        self.log_param = log_param
        self.log_params = log_params
        self.log_metric = log_metric
        self.log_metrics = log_metrics
        self.log_artifact = log_artifact
        self.set_experiment = set_experiment
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        self.start_run = start_run

        self.client = mlflow.tracking.MlflowClient()
        self.experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME
        self.experiment_id: str = self._resolve_experiment_id()
        logger.debug(
            "RegistryClient pronto (experiment_id=%s, experiment_name=%s).",
            self.experiment_id,
            self.experiment_name,
        )

    def trace(self, *args, span_type: str = "CHAIN", name: str = None):
        """Custom trace decorator tied to this client instance."""
        if args and callable(args[0]):
            func = args[0]

            @functools.wraps(func)
            def wrapper(*f_args, **f_kwargs):
                mlflow.set_experiment(experiment_id=self.experiment_id)

                @mlflow.trace(span_type=span_type, name=name or func.__name__)
                def executed_func(*inner_args, **inner_kwargs):
                    return func(*inner_args, **inner_kwargs)

                return executed_func(*f_args, **f_kwargs)

            return wrapper

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure the correct experiment context is active during execution
                mlflow.set_experiment(experiment_id=self.experiment_id)

                # Derive span name from function name if not explicitly provided
                span_name = name or func.__name__

                # Use mlflow.trace as a dynamic wrapper
                @mlflow.trace(span_type=span_type, name=span_name)
                def executed_func(*f_args, **f_kwargs):
                    return func(*f_args, **f_kwargs)

                return executed_func(*args, **kwargs)

            return wrapper

        return decorator

    def _resolve_experiment_id(self) -> str:
        """Obtém o ID do experimento de MLflow, criando-o se necessário.

        Returns:
            str: ID do experimento correspondente a `self.experiment_name`.
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            logger.debug(
                "Experimento '%s' já existe (id=%s).",
                self.experiment_name,
                experiment.experiment_id,
            )
            return experiment.experiment_id

        experiment_id = self.client.create_experiment(self.experiment_name)
        logger.info(
            "Experimento '%s' criado (id=%s).", self.experiment_name, experiment_id
        )
        return experiment_id


# Teste rápido via linha de comando (Smoke-test)
if __name__ == "__main__":
    client = RegistryClient()
    print(f"Experiment name: {client.experiment_name}")
    print(f"Experiment ID: {client.experiment_id}")
