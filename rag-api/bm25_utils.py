from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
from whoosh.qparser import QueryParser
import os
from typing import List, Dict
import time
import logging

logger = logging.getLogger(__name__)


def topic_index_dir(base: str, topic: str) -> str:
    return os.path.join(base, topic)


def ensure_whoosh_index(base: str, topic: str):
    path = topic_index_dir(base, topic)
    os.makedirs(path, exist_ok=True)
    if not index.exists_in(path):
        schema = Schema(
            file_path=ID(stored=True),
            page=NUMERIC(stored=True),
            chunk_id=NUMERIC(stored=True),
            text=TEXT(stored=True),
        )
        index.create_in(path, schema)


def add_docs(base: str, topic: str, docs: List[Dict]):
    idx = index.open_dir(topic_index_dir(base, topic))
    writer = idx.writer(limitmb=512, procs=0, multisegment=True)
    for d in docs:
        writer.update_document(
            file_path=d["file_path"],
            page=d["page"],
            chunk_id=d["chunk_id"],
            text=d["text"],
        )
    writer.commit()


def bm25_search(base: str, topic: str, query: str, topk: int) -> List[Dict]:
    """BM25 search con medición de timing detallado"""

    total_start = time.time()

    # 1️⃣ Abrir índice
    open_start = time.time()
    idx = index.open_dir(topic_index_dir(base, topic))
    open_time = time.time() - open_start

    # 2️⃣ Parsear query
    parse_start = time.time()
    qp = QueryParser("text", schema=idx.schema)
    q = qp.parse(query)
    parse_time = time.time() - parse_start

    # 3️⃣ Búsqueda
    search_start = time.time()
    with idx.searcher() as s:
        res = s.search(q, limit=topk)
        search_time = time.time() - search_start

        # 4️⃣ Procesar resultados
        results_start = time.time()
        hits = []
        for r in res:
            hits.append(
                dict(
                    file_path=r["file_path"],
                    page=r["page"],
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r.score,
                )
            )
        results_time = time.time() - results_start

    total_time = time.time() - total_start

    # Log de timing
    logger.info(f"[BM25] query='{query[:50]}...'")
    logger.info(f"   - open_index: {open_time * 1000:.1f}ms")
    logger.info(f"   - parse_query: {parse_time * 1000:.1f}ms")
    logger.info(f"   - search: {search_time * 1000:.1f}ms ⭐ (crítico)")
    logger.info(f"   - process_results: {results_time * 1000:.1f}ms")
    logger.info(f"   - TOTAL: {total_time * 1000:.1f}ms ({len(hits)} hits)")

    # ⚠️ Warning si es lento
    if total_time > 2.0:
        logger.warning(f"⚠️  BM25 MUY LENTO: {total_time:.2f}s")
    elif total_time > 0.5:
        logger.warning(f"⚠️  BM25 lento: {total_time:.3f}s")

    return hits


def sanitize_query_for_bm25(query: str, max_length: int = 200) -> str:
    """
    Limpia queries que son demasiado largas o complejas para BM25

    Casos problemáticos:
    - Queries de sistema reales (no preguntas de usuario)
    - Queries muy largas (>200 chars)
    - Exceso de symbols especiales
    """

    original = query

    # 1. Detectar solo queries de sistema REALMENTE problemáticos
    # Patrones más específicos para evitar falsos positivos
    system_patterns = [
        r"^#+\s*(Task|Generate|Create|System).*(?:title|summarize|concise|emoji)",
        r"^(?:You are|Act as|Generate|Create).*(?:title|summary|response|emoji)",
        r"^System\s*:",
        r"^Assistant\s*:",
        # OpenWebUI specific pattern para generar títulos
        r"^#+\s*Task:\s*Generate.*title.*emoji.*summarizing",
    ]

    for pattern in system_patterns:
        if re.match(pattern, query, re.IGNORECASE):
            logger.warning(f"⚠️  Sistema query detectado, ignorando BM25: {query[:80]}...")
            return ""  # Retornar vacío = skip BM25

    # 2. Solo bloquear patrones CLARAMENTE de código/prompt, no preguntas con markdown
    # Bloquear solo si empieza con múltiples símbolos de código seguidos
    if re.match(r"^(?:```\s*$|>>>|---+$)", query.strip()):
        logger.warning(f"⚠️  Código/prompt detectado, ignorando BM25")
        return ""

    # 3. Limpiar símbolos problemáticos para Whoosh
    # Reemplazar caracteres especiales que confunden al parser
    query = re.sub(r"[#@$%&*(){}\[\]<>|\\]+", " ", query)

    # 4. Remover líneas vacías y normalizar espacios
    query = " ".join(query.split())

    # 5. Limitar longitud
    if len(query) > max_length:
        logger.warning(
            f"⚠️  Query muy larga ({len(query)} chars), truncando a {max_length}"
        )
        query = query[:max_length]

    # 6. Si quedó muy corto o vacío, skip BM25
    if len(query.strip()) < 3:
        logger.info("Query demasiado corta tras limpieza, ignorando BM25")
        return ""

    if original != query:
        logger.info(f"Query limpiada: '{original[:60]}...' → '{query[:60]}...'")

    return query


def bm25_search_safe(base: str, topic: str, query: str, topk: int):
    """BM25 search con sanitización segura"""
    from whoosh import index
    from whoosh.qparser import QueryParser
    import time

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
