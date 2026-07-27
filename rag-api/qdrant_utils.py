from qdrant_client import QdrantClient
from config.settings import QDRANT_URL

import logging

logger = logging.getLogger(__name__)

client = QdrantClient(url=QDRANT_URL)


def topic_collection(topic: str) -> str:
    # un nombre de colección por tema
    return f"rag_{topic.lower()}"


def collection_exists(topic: str) -> bool:
    """True si existe la colección Qdrant del tema.

    Sirve para que el llamador degrade con elegancia (contexto vacío) en vez de
    reventar con un 500 cuando el tema no mapea a ninguna colección — p. ej. una
    etiqueta mal escrita en un workspace model de Open WebUI (PLAN.md §7.1).
    """
    coll = topic_collection(topic)
    try:
        return client.collection_exists(coll)
    except Exception as e:
        logger.error(f"[QDRANT] Error comprobando existencia de '{coll}': {e}")
        return False


def search_dense(topic: str, vector: list, topk: int):
    """
    Busca vectores en Qdrant.

    Los IDs de los vectores fueron generados con:
      abs(hash(f"{pdf_path}:{chunk_index}")) % (2**31)

    Qdrant busca automáticamente por similitud de vector,
    no necesita saber cómo se generaron los IDs.
    """
    coll = topic_collection(topic)
    logger.debug(f"[QDRANT] Searching collection '{coll}' with topk={topk}")

    try:
        res = client.search(
            collection_name=coll, query_vector=vector, limit=topk, with_payload=True
        )
        logger.info(f"[QDRANT] Found {len(res)} results in '{coll}'")

        # Log de hits para debugging
        for i, hit in enumerate(res[:3], 1):
            filename = hit.payload.get("file_path", "unknown").split("/")[-1]
            logger.debug(f"  [{i}] {filename} (score={hit.score:.4f}, id={hit.id})")

        return res

    except Exception as e:
        logger.error(f"[QDRANT] Search error in '{coll}': {e}", exc_info=True)
        raise


def get_collection_stats(topic: str) -> dict:
    """Obtiene estadísticas de una colección (útil para debugging)"""
    coll = topic_collection(topic)
    try:
        info = client.get_collection(coll)
        return {
            "collection": coll,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.value,
        }
    except Exception as e:
        logger.error(f"[QDRANT] Error getting stats for '{coll}': {e}")
        return {}
