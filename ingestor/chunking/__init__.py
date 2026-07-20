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
from chunking.token_chunker import (
    Chunk,
    build_chunks,
    chunk_docling_document,
    chunk_elements,
    dedupe_chunks,
    detect_boilerplate,
)

__all__ = [
    "ContextAwareChunker",
    "adaptive_chunk",
    "semantic_chunk",
    "simple_chunk",
    # Fragmentación por tokens (la que usa el pipeline en vivo)
    "Chunk",
    "build_chunks",
    "chunk_docling_document",
    "chunk_elements",
    "dedupe_chunks",
    "detect_boilerplate",
]
