"""
Búsqueda léxica BM25 servida por Qdrant como vector disperso (§7.4).

Sustituye a la rama Whoosh de `bm25_utils.py` sin cambiar su contrato: misma
firma efectiva y misma forma de retorno (`file_path`, `page`, `chunk_id`,
`text`, `score`), para que `reciprocal_rank_fusion` y `apply_per_file_limit`
no se enteren de cuál de los dos motores les habló.

Por qué existe, en corto: un solo almacén en vez de dos, escrituras atómicas
por documento (elimina la inconsistencia de doble escritura por construcción)
y Whoosh 2.7.4 está abandonado desde 2016. La medida que respalda el cambio
está en `FINDINGS.md` §7.4 — con la salvedad, comprobada el 2026-08-01, de que
el índice Whoosh vivo **no lleva stemmer** (`TEXT(stored=True)` usa el
`StandardAnalyzer` por defecto), así que parte de la ventaja medida es
"stemmer español contra ninguno" y no "Qdrant contra Whoosh".

La consulta se sanea con el mismo `sanitize_query_for_bm25` que usaba Whoosh:
no por sintaxis —aquí no hay parser que romper— sino para conservar el
comportamiento de descarte de queries de sistema/vacías, del que depende la
ruta BM25-solo aguas arriba.
"""

import logging
import time
from typing import List, Optional

from qdrant_client import models

from bm25_utils import sanitize_query_for_bm25
from config.settings import SPARSE_VECTOR_NAME
from qdrant_utils import client, topic_collection

logger = logging.getLogger(__name__)

# El modelo se carga una vez por proceso. `Qdrant/bm25` no es una red neuronal:
# es tokenizador + stopwords + stemmer Snowball, así que cargarlo es barato y
# la inferencia es CPU pura.
_embedder = None


def get_sparse_embedder():
    """Carga perezosa del embebedor disperso, compartida por proceso."""
    global _embedder
    if _embedder is None:
        from fastembed import SparseTextEmbedding

        # `language="spanish"` es obligatorio, no cosmético: sin él
        # "instalaciones" e "instalación" son términos distintos y se pierde
        # justo lo que el corpus necesita.
        _embedder = SparseTextEmbedding(model_name="Qdrant/bm25", language="spanish")
        logger.info("[SPARSE] Embebedor BM25 (Qdrant/bm25, es) cargado")
    return _embedder


def embed_query(query: str) -> Optional[models.SparseVector]:
    """
    Vectoriza la consulta para el lado disperso.

    Usa `query_embed`, no `embed`: en BM25 el lado consulta no lleva ponderación
    por frecuencia de término — esa va en los documentos, y el `modifier: idf`
    de la colección aporta el IDF del lado servidor. Vectorizar la consulta como
    si fuera un documento puntúa mal las consultas con términos repetidos.
    """
    embedder = get_sparse_embedder()
    vectors = list(embedder.query_embed(query))
    if not vectors:
        return None
    sv = vectors[0]
    return models.SparseVector(
        indices=sv.indices.tolist(),
        values=sv.values.tolist(),
    )


# Colecciones ya comprobadas, para no preguntar a Qdrant en cada consulta.
_sparse_checked: dict = {}


def _collection_has_sparse(coll: str) -> bool:
    """
    Comprueba una vez por colección que existe el vector disperso.

    Sin esto, la combinación más fácil de equivocarse —`LEXICAL_BACKEND=qdrant`
    con `QDRANT_COLLECTION_SUFFIX` sin poner— apunta a las colecciones viejas,
    que no lo tienen: cada consulta fallaría, devolvería [] y el servicio
    degradaría a densa-sola **en silencio**, que es exactamente el modo de
    fallo que `PLAN.md` avisa de no aceptar (una tirada informa de éxito sin
    haber hecho nada). Aquí se dice en voz alta y con el arreglo concreto.
    """
    if coll in _sparse_checked:
        return _sparse_checked[coll]

    try:
        info = client.get_collection(coll)
        sparse = info.config.params.sparse_vectors or {}
        ok = SPARSE_VECTOR_NAME in sparse
    except Exception as e:
        logger.error(f"[SPARSE] No se pudo inspeccionar '{coll}': {e}")
        ok = False

    if not ok:
        logger.error(
            f"[SPARSE] LEXICAL_BACKEND=qdrant pero '{coll}' no tiene el vector "
            f"disperso '{SPARSE_VECTOR_NAME}'. La rama léxica queda MUERTA "
            f"(sólo densa). Revisa QDRANT_COLLECTION_SUFFIX o migra con "
            f"scripts/migrate_sparse.py."
        )

    _sparse_checked[coll] = ok
    return ok


def sparse_search_safe(topic: str, query: str, topk: int) -> List[dict]:
    """
    Búsqueda BM25 dispersa en Qdrant, con el mismo contrato que
    `bm25_search_safe`.

    Devuelve [] ante cualquier fallo, igual que la rama Whoosh: la ruta híbrida
    debe degradar a densa-sola, nunca romper la petición.
    """
    clean_query = sanitize_query_for_bm25(query)

    if not clean_query:
        logger.info("[SPARSE] Query descartada (sistema/vacía), retornando []")
        return []

    coll = topic_collection(topic)

    if not _collection_has_sparse(coll):
        return []

    total_start = time.time()

    try:
        sparse_vector = embed_query(clean_query)
        if sparse_vector is None or not sparse_vector.indices:
            logger.info("[SPARSE] Query sin términos indexables, retornando []")
            return []

        search_start = time.time()
        res = client.query_points(
            collection_name=coll,
            query=sparse_vector,
            using=SPARSE_VECTOR_NAME,
            limit=topk,
            with_payload=True,
        ).points
        search_time = time.time() - search_start

        hits = [
            {
                "file_path": p.payload["file_path"],
                "page": p.payload["page"],
                "chunk_id": p.payload["chunk_id"],
                "text": p.payload["text"],
                "score": float(p.score),
            }
            for p in res
        ]

        total_time = time.time() - total_start
        logger.info(
            f"[SPARSE] OK: {search_time * 1000:.1f}ms búsqueda, "
            f"{total_time * 1000:.1f}ms total ({len(hits)} hits) en '{coll}'"
        )

        return hits

    except Exception as e:
        logger.error(f"[SPARSE] Error en '{coll}': {e}", exc_info=True)
        return []
