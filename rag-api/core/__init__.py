# Archivo: rag-api/core/__init__.py
# Descripción: Módulo de infraestructura compartida
#
# `vllm_client.py` (cliente HTTP + streaming SSE) y `retry.py` (reintentos con
# backoff, cuyo único consumidor era ese cliente) se fueron con el rip-out del
# §7.1: rag-api ya no hace peticiones salientes.

from core.cache import ModelCache

__all__ = [
    "ModelCache",
]
