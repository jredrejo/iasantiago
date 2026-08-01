"""
Vectores dispersos BM25 para la búsqueda léxica servida por Qdrant (§7.4).

El ingestor escribe, junto al vector denso y en el mismo punto, un vector
disperso BM25. Eso permite que Qdrant sirva la rama léxica y que la escritura
de un documento sea atómica —un solo `upsert`— en vez de las dos escrituras
desacopladas actuales (Qdrant + Whoosh), que es de donde salía la posibilidad
de que un fallo dejara los dos índices en desacuerdo.

`Qdrant/bm25` no es una red neuronal: es tokenizador + stopwords + stemmer
Snowball. Corre en CPU y no compite con la GPU del extractor.
"""

import logging
from typing import List, Optional

from qdrant_client import models

from core.config import SPARSE_LANGUAGE, SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)

_embedder = None


def get_sparse_embedder():
    """Carga perezosa del embebedor disperso, compartida por proceso."""
    global _embedder
    if _embedder is None:
        from fastembed import SparseTextEmbedding

        # El idioma es obligatorio, no cosmético: sin stemmer español
        # "instalaciones" e "instalación" son términos distintos. El índice
        # Whoosh al que esto sustituye no lo tenía (`FINDINGS.md` §7.4,
        # corrección del 2026-08-01).
        _embedder = SparseTextEmbedding(
            model_name="Qdrant/bm25", language=SPARSE_LANGUAGE
        )
        logger.info(
            f"[SPARSE] Embebedor BM25 (Qdrant/bm25, {SPARSE_LANGUAGE}) cargado"
        )
    return _embedder


def embed_documents(texts: List[str], batch_size: int = 256) -> List[models.SparseVector]:
    """
    Vectoriza textos de documento para el índice disperso.

    Usa `embed` (lado documento): la ponderación por frecuencia de término va
    aquí, y el IDF lo aplica Qdrant en consulta gracias a `modifier: idf`.
    """
    embedder = get_sparse_embedder()
    return [
        models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
        for sv in embedder.embed(texts, batch_size=batch_size)
    ]


def build_sparse_vectors(payloads: List[dict]) -> Optional[List[models.SparseVector]]:
    """
    Construye los vectores dispersos de una tanda de payloads.

    Devuelve `None` si algo falla: el llamador debe poder seguir escribiendo el
    vector denso. Perder la rama dispersa de un documento degrada la búsqueda;
    abortar la ingesta entera por ello la rompe.
    """
    try:
        texts = [p.get("text", "") for p in payloads]
        return embed_documents(texts)
    except Exception as e:
        logger.error(f"[SPARSE] No se pudieron construir los vectores: {e}")
        return None


__all__ = [
    "SPARSE_VECTOR_NAME",
    "build_sparse_vectors",
    "embed_documents",
    "get_sparse_embedder",
]
