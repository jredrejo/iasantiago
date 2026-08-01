"""
Módulo de indexación para búsqueda vectorial y por palabras clave.

Proporciona generación de embeddings y almacenamiento en Qdrant: vector denso
para la búsqueda semántica y vector disperso BM25 para la léxica, escritos en el
mismo punto (§7.4; Whoosh se retiró el 2026-08-01).
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
from indexing.sparse import (
    build_sparse_vectors,
    get_sparse_embedder,
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
    # Disperso (BM25)
    "build_sparse_vectors",
    "get_sparse_embedder",
]
