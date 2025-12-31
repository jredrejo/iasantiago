# Archivo: rag-api/retrieval_lib/__init__.py
# Descripción: Módulo de funciones auxiliares para retrieval

from retrieval_lib.fusion import reciprocal_rank_fusion, deduplicate_chunks
from retrieval_lib.search import (
    execute_hybrid_search,
    apply_per_file_limit,
    prepare_query_for_retrieval,
)
from retrieval_lib.citations import build_context_with_citations

__all__ = [
    "reciprocal_rank_fusion",
    "deduplicate_chunks",
    "execute_hybrid_search",
    "apply_per_file_limit",
    "prepare_query_for_retrieval",
    "build_context_with_citations",
]
