"""Módulo de integração com serviços externos."""

from integrations.registry import RegistryClient
from integrations.storage import StorageClient

__all__ = ["RegistryClient", "StorageClient"]
