"""
Módulo de fragmentación de documentos.

Proporciona estrategias de fragmentación conscientes del contexto para dividir documentos
en piezas manejables mientras preserva la coherencia semántica.
"""

from chunking.chunker import ContextAwareChunker
from chunking.strategies import (
    adaptive_chunk,
    semantic_chunk,
    simple_chunk,
)

__all__ = [
    "ContextAwareChunker",
    "adaptive_chunk",
    "semantic_chunk",
    "simple_chunk",
]
