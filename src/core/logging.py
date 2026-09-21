"""
src/core/logging.py

Módulo de logging centralizado para a aplicação FastAPI.
Fornece middleware, decoradores e utilitários para logging estruturado.

Uso:
    from src.core.logging import setup_logging, get_structured_logger

    logger = setup_logging()
    app.add_middleware(APILoggingMiddleware, logger=logger)
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================


class JSONFormatter(logging.Formatter):
    """Formatter customizado que estrutura logs em JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Adiciona atributos customizados se presentes
        for attr in [
            "request_id",
            "user_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "error_details",
        ]:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    log_file: str | None = None, level: int = logging.INFO, use_json: bool = True
) -> logging.Logger:
    """
    Configura o logger para a aplicação.

    Args:
        log_file: Caminho do arquivo de log (None = apenas console)
        level: Nível de logging
        use_json: Se True, formata logs em JSON

    Returns:
        Logger configurado
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = (
        JSONFormatter()
        if use_json
        else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(request_id)s] %(method)s %(path)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (opcional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# MIDDLEWARE
# ============================================================================


class APILoggingMiddleware(BaseHTTPMiddleware):
    """Middleware que loga todas as requisições HTTP."""

    def __init__(self, app: ASGIApp, logger: logging.Logger | None = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger("fiap_ml_app")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição e loga informações."""

        # Gera Request ID único
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Coleta informações
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_host = request.client.host if request.client else "unknown"

        start_time = time.time()

        # Log de entrada
        self.logger.info(
            f"Requisição iniciada: {method} {path}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "query_params": query_params,
                "client_host": client_host,
            },
        )

        response = None
        status_code = 500
        error_details = None

        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as exc:
            status_code = 500
            error_details = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Determina nível de log
            if 200 <= status_code < 300:
                log_level = logging.INFO
                log_message = "Requisição processada com sucesso"
            elif 400 <= status_code < 500:
                log_level = logging.WARNING
                log_message = "Erro do cliente"
            else:
                log_level = logging.ERROR
                log_message = "Erro do servidor"

            # Log de saída
            self.logger.log(
                log_level,
                log_message,
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "error_details": error_details,
                },
            )

        return response


# ============================================================================
# DECORADOR
# ============================================================================


def log_route(
    logger: logging.Logger | None = None,
    include_args: bool = False,
    max_arg_length: int = 200,
) -> Callable:
    """
    Decorador para logar execução de rotas específicas.

    Args:
        logger: Logger a usar
        include_args: Se True, loga argumentos
        max_arg_length: Tamanho máximo de argumentos
    """
    _logger = logger or logging.getLogger("fiap_ml_app")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            start_time = time.time()

            _logger.debug(
                f"Função iniciada: {func_name}",
                extra={"args": str(args)[:max_arg_length]} if include_args else {},
            )

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                _logger.debug(
                    f"Função concluída: {func_name}",
                    extra={"duration_ms": round(duration_ms, 2)},
                )

                return result

            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000

                _logger.error(
                    f"Erro na função {func_name}",
                    extra={
                        "duration_ms": round(duration_ms, 2),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            start_time = time.time()

            _logger.debug(f"Função iniciada: {func_name}")

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                _logger.debug(
                    f"Função concluída: {func_name}",
                    extra={"duration_ms": round(duration_ms, 2)},
                )

                return result

            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000

                _logger.error(
                    f"Erro na função {func_name}",
                    extra={
                        "duration_ms": round(duration_ms, 2),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================


class StructuredLogger:
    """Wrapper para logging estruturado."""

    def __init__(self, logger: logging.Logger, request_id: str | None = None):
        self.logger = logger
        self.request_id = request_id or str(uuid.uuid4())

    def log_event(
        self, event_name: str, level: int = logging.INFO, **event_data
    ) -> None:
        """Loga um evento estruturado."""
        self.logger.log(
            level, event_name, extra={"request_id": self.request_id, **event_data}
        )

    def log_database_query(
        self,
        query: str,
        duration_ms: float,
        rows_affected: int = 0,
        error: str | None = None,
    ) -> None:
        """Loga uma query de banco de dados."""
        level = logging.ERROR if error else logging.DEBUG
        self.logger.log(
            level,
            "Database query executed",
            extra={
                "request_id": self.request_id,
                "query": query[:300],
                "duration_ms": round(duration_ms, 2),
                "rows_affected": rows_affected,
                "error": error,
            },
        )

    def log_external_api_call(
        self,
        api_name: str,
        method: str,
        url: str,
        status_code: int,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Loga uma chamada a API externa."""
        level = logging.WARNING if status_code >= 400 else logging.DEBUG
        self.logger.log(
            level,
            f"External API call: {api_name}",
            extra={
                "request_id": self.request_id,
                "api_name": api_name,
                "method": method,
                "url": url,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "error": error,
            },
        )

    def log_cache_operation(
        self, operation: str, key: str, duration_ms: float = 0
    ) -> None:
        """Loga operações de cache."""
        self.logger.debug(
            f"Cache operation: {operation}",
            extra={
                "request_id": self.request_id,
                "operation": operation,
                "key": key,
                "duration_ms": round(duration_ms, 2),
            },
        )

    def log_model_prediction(
        self,
        model_name: str,
        input_shape: str,
        output: str,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Loga predições de modelo ML."""
        level = logging.ERROR if error else logging.INFO
        self.logger.log(
            level,
            f"Model prediction: {model_name}",
            extra={
                "request_id": self.request_id,
                "model_name": model_name,
                "input_shape": input_shape,
                "output": output[:200],
                "duration_ms": round(duration_ms, 2),
                "error": error,
            },
        )

    def log_data_pipeline(
        self,
        pipeline_name: str,
        stage: str,
        duration_ms: float,
        records_processed: int = 0,
        error: str | None = None,
    ) -> None:
        """Loga execução de pipelines de dados."""
        level = logging.ERROR if error else logging.INFO
        self.logger.log(
            level,
            f"Data pipeline: {pipeline_name} - {stage}",
            extra={
                "request_id": self.request_id,
                "pipeline_name": pipeline_name,
                "stage": stage,
                "duration_ms": round(duration_ms, 2),
                "records_processed": records_processed,
                "error": error,
            },
        )


# ============================================================================
# CONTEXT MANAGER
# ============================================================================


class LoggingContext:
    """Context manager para logar operações com duração."""

    def __init__(self, logger: logging.Logger, operation_name: str, **context_data):
        self.logger = logger
        self.operation_name = operation_name
        self.context_data = context_data
        self.start_time: float | None = None

    async def __aenter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Iniciando operação: {self.operation_name}", extra=self.context_data
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000  # type: ignore

        if exc_type is None:
            self.logger.info(
                f"Operação concluída: {self.operation_name}",
                extra={"duration_ms": round(duration_ms, 2), **self.context_data},
            )
        else:
            self.logger.error(
                f"Erro na operação: {self.operation_name}",
                extra={
                    "duration_ms": round(duration_ms, 2),
                    "exception_type": exc_type.__name__,
                    "exception_message": str(exc_val),
                    **self.context_data,
                },
            )

        return False


# ============================================================================
# DEPENDÊNCIA PARA FASTAPI
# ============================================================================


def get_structured_logger(request: Request) -> StructuredLogger:
    """
    Dependência FastAPI que fornece um StructuredLogger com o request_id.

    Uso:
        @app.get("/users/{user_id}")
        async def get_user(
            user_id: int,
            structured_logger: StructuredLogger = Depends(get_structured_logger)
        ):
            structured_logger.log_event("Buscando usuário", user_id=user_id)
            return {"user_id": user_id}
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger = logging.getLogger("fiap_ml_app")
    return StructuredLogger(logger, request_id=request_id)
