"""Módulo de infraestrutura e clientes externos."""

from infrastructure.registry import RegistryClient
from infrastructure.storage import StorageClient

__all__ = ["RegistryClient", "StorageClient"]
