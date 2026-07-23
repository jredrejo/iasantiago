from whoosh import index
from whoosh.qparser import QueryParser
import re
import time
import logging

logger = logging.getLogger(__name__)


def sanitize_query_for_bm25(query: str, max_length: int = 200) -> str:
    """
    Limpia queries que son demasiado largas o complejas para BM25

    Casos problemáticos:
    - Queries muy largas (>200 chars)
    - Exceso de symbols especiales
    """

    original = query

    # 1. Limpiar símbolos problemáticos para Whoosh
    # Reemplazar caracteres especiales que confunden al parser
    query = re.sub(r"[#@$%&*(){}\[\]<>|\\]+", " ", query)

    # 2. Remover líneas vacías y normalizar espacios
    query = " ".join(query.split())

    # 3. Limitar longitud
    if len(query) > max_length:
        logger.warning(
            f"⚠️  Query muy larga ({len(query)} chars), truncando a {max_length}"
        )
        query = query[:max_length]

    # 4. Si quedó muy corto o vacío, skip BM25
    if len(query.strip()) < 3:
        logger.info("Query demasiado corta tras limpieza, ignorando BM25")
        return ""

    if original != query:
        logger.info(f"Query limpiada: '{original[:60]}...' → '{query[:60]}...'")

    return query


def bm25_search_safe(base: str, topic: str, query: str, topk: int):
    """BM25 search con sanitización segura"""
    # ✅ SANITIZAR QUERY ANTES DE BUSCAR
    clean_query = sanitize_query_for_bm25(query)

    if not clean_query:
        logger.info("[BM25] Query descartada (sistema/vacía), retornando []")
        return []

    total_start = time.time()

    try:
        open_start = time.time()
        idx = index.open_dir(f"{base}/{topic}")
        open_time = time.time() - open_start

        parse_start = time.time()
        qp = QueryParser("text", schema=idx.schema)
        q = qp.parse(clean_query)
        parse_time = time.time() - parse_start

        search_start = time.time()
        with idx.searcher() as s:
            res = s.search(q, limit=topk)
            search_time = time.time() - search_start

            results_start = time.time()
            hits = []
            for r in res:
                hits.append(
                    {
                        "file_path": r["file_path"],
                        "page": r["page"],
                        "chunk_id": r["chunk_id"],
                        "text": r["text"],
                        "score": r.score,
                    }
                )
            results_time = time.time() - results_start

        total_time = time.time() - total_start

        logger.info(f"[BM25] ✅ Clean: {search_time * 1000:.1f}ms ({len(hits)} hits)")

        return hits

    except Exception as e:
        logger.error(f"[BM25] Error: {e}", exc_info=True)
        return []
