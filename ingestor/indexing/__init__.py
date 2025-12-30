"""
Módulo de indexación para búsqueda vectorial y por palabras clave.

Proporciona generación de embeddings y almacenamiento en Qdrant (vectores)
y Whoosh (búsqueda por palabras clave BM25).
"""

from indexing.embeddings import (
    EmbeddingService,
    get_embedding_service,
    validate_and_fix_vectors,
)
from indexing.qdrant import (
    QdrantService,
    ensure_qdrant,
    get_qdrant_service,
    topic_collection,
)
from indexing.whoosh_bm25 import (
    WhooshService,
    ensure_whoosh,
    get_whoosh_service,
)

__all__ = [
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    "validate_and_fix_vectors",
    # Qdrant
    "QdrantService",
    "ensure_qdrant",
    "get_qdrant_service",
    "topic_collection",
    # Whoosh
    "WhooshService",
    "ensure_whoosh",
    "get_whoosh_service",
]
