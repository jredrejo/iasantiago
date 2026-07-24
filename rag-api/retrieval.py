# Archivo: rag-api/retrieval.py
# Descripción: Módulo de retrieval híbrido (API pública)
#
# Este módulo mantiene la API pública para compatibilidad hacia atrás.
# La lógica interna usa los helpers de retrieval_lib/.

import json
import logging
import os
import time
from typing import Dict, List, Tuple

from config.settings import (
    BM25_BASE_DIR,
    BM25_FALLBACK_TOKEN_THRESHOLD,
    EMBED_DEFAULT,
    EMBED_PER_TOPIC,
    FINAL_TOPK,
    GENERATIVE_TOPK_MULTIPLIER,
    HYBRID_BM25_K,
    HYBRID_DENSE_K,
    MAX_CHUNKS_PER_FILE,
    MAX_CHUNKS_PER_FILE_GENERATIVE,
    RERANK_MODEL,
    TELEMETRY_PATH,
    TELEMETRY_RETENTION_MONTHS,
    TRANSLATE_QUERIES,
    EXPAND_ACRONYMS,
)
from acronyms import expand_acronyms
from core.cache import ModelCache
from retrieval_lib.fusion import deduplicate_chunks, reciprocal_rank_fusion
from retrieval_lib.search import apply_per_file_limit, prepare_query_for_retrieval
from retrieval_lib.citations import build_context_with_citations

# Importaciones de módulos existentes
from qdrant_utils import search_dense
from bm25_utils import bm25_search_safe
from rerank import CrossEncoderReranker
from translation import translate_query, detect_language

logger = logging.getLogger(__name__)


# ============================================================
# FUNCIONES DE ACCESO A MODELOS (usan ModelCache)
# ============================================================


def get_embedder(topic: str):
    """Obtiene el embedder para un tema"""
    return ModelCache.get_embedder(topic, EMBED_PER_TOPIC, EMBED_DEFAULT)


def get_reranker():
    """Obtiene el reranker (singleton)"""
    return ModelCache.get_reranker(RERANK_MODEL)


def count_tokens(text: str) -> int:
    """Cuenta tokens en un texto"""
    return ModelCache.count_tokens(text)


# ============================================================
# FUNCIONES DE CONTEXTO
# ============================================================


def soft_trim_context(chunks: List[Dict], token_limit: int) -> List[Dict]:
    """Recorta chunks para no exceder el límite de tokens"""
    total = 0
    out = []
    for c in chunks:
        t = count_tokens(c["text"])
        if total + t > token_limit:
            break
        total += t
        out.append(c)
    return out


# ============================================================
# BÚSQUEDA HÍBRIDA - FUNCIONES PRINCIPALES
# ============================================================


def _execute_search(
    topic: str,
    query: str,
    dense_k: int,
    bm25_k: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Helper interno: ejecuta búsqueda densa + BM25.

    Args:
        topic: Tema de búsqueda
        query: Query del usuario
        dense_k: Número de resultados densos
        bm25_k: Número de resultados BM25

    Returns:
        Tupla (resultados_densos, resultados_bm25)
    """
    embedder = get_embedder(topic)
    embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)

    # Preparar query (añadir prefijo E5 si es necesario)
    query_for_embedding = prepare_query_for_retrieval(query, embed_name)

    # Generar embedding
    q_vec = embedder.encode([query_for_embedding], normalize_embeddings=True)[
        0
    ].tolist()

    # Búsqueda densa
    dense_hits = search_dense(topic, q_vec, dense_k)
    dense = [
        {
            "file_path": h.payload["file_path"],
            "page": h.payload["page"],
            "chunk_id": h.payload["chunk_id"],
            "text": h.payload["text"],
            "score_dense": float(h.score),
            "score_bm25": 0.0,
        }
        for h in dense_hits
    ]

    # Búsqueda BM25
    bm25 = bm25_search_safe(BM25_BASE_DIR, topic, query, bm25_k)

    return dense, bm25


def hybrid_retrieve(
    topic: str, query: str, final_topk: int, is_generative: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Búsqueda híbrida (densa + BM25) con RRF, dedup y límite por archivo.

    Args:
        topic: Tema de búsqueda
        query: Query del usuario
        final_topk: Número final de chunks
        is_generative: True para modo generativo

    Returns:
        Tupla (chunks_filtrados, metadata)
    """
    # Recuperar más en búsqueda inicial (2x el base)
    dense_k = HYBRID_DENSE_K * 2
    bm25_k = HYBRID_BM25_K * 2

    logger.info(f"[HYBRID] Dense K={dense_k}, BM25 K={bm25_k}, Final topk={final_topk}")

    # Ejecutar búsqueda híbrida
    dense, bm25 = _execute_search(topic, query, dense_k, bm25_k)

    # Fusión RRF y deduplicación
    merged = reciprocal_rank_fusion(dense, bm25, k=60)
    merged = deduplicate_chunks(merged)

    # Límite por archivo según modo
    max_per_file = (
        MAX_CHUNKS_PER_FILE_GENERATIVE if is_generative else MAX_CHUNKS_PER_FILE
    )
    mode_name = "GENERATIVO" if is_generative else "RESPUESTA"
    logger.info(f"[HYBRID] Usando max_per_file={max_per_file} (modo {mode_name})")

    # Filtrar
    filtered = apply_per_file_limit(merged, max_per_file, final_topk)

    unique_files = len(set(r["file_path"] for r in filtered))
    logger.info(f"[HYBRID] Final: {len(filtered)} chunks de {unique_files} archivos")

    return filtered, {
        "dense_k": dense_k,
        "bm25_k": bm25_k,
        "final_topk": final_topk,
    }


# ============================================================
# BÚSQUEDA BM25-ONLY
# ============================================================


def bm25_only(
    topic: str, query: str, final_topk: int, is_generative: bool = False
) -> List[Dict]:
    """Búsqueda BM25 sola, con límite por archivo y topk configurable."""
    # Buscar 3x más de lo necesario para tener margen
    hits = bm25_search_safe(BM25_BASE_DIR, topic, query, final_topk * 3)

    # Límite por archivo según modo
    max_per_file = (
        MAX_CHUNKS_PER_FILE_GENERATIVE if is_generative else MAX_CHUNKS_PER_FILE
    )
    mode_name = "GENERATIVO" if is_generative else "RESPUESTA"
    logger.info(f"[BM25] Usando max_per_file={max_per_file} (modo {mode_name})")

    filtered = apply_per_file_limit(hits, max_per_file, final_topk)

    unique_files = len(set(r["file_path"] for r in filtered))
    logger.info(f"[BM25] Final: {len(filtered)} chunks de {unique_files} archivos")

    return filtered


# ============================================================
# SELECCIÓN DE ESTRATEGIA
# ============================================================


def _prepare_query(query: str) -> Tuple[str, str, str]:
    """
    Helper interno: detecta idioma y traduce si es necesario.

    Returns:
        Tupla (query_traducido, idioma_detectado, query_original)
    """
    detected_lang = detect_language(query)
    original_query = query

    # Traducción desactivada por defecto (kill-switch TRANSLATE_QUERIES): con el
    # embedder cross-lingual la consulta original rinde mejor en un corpus
    # español y conserva la rama BM25 (PLAN.md §3.3). Con el flag activo se
    # restaura la traducción es→en previa.
    if TRANSLATE_QUERIES and detected_lang != "en":
        logger.info(f"Query en {detected_lang}, traduciendo a inglés para retrieval")
        query, _ = translate_query(query, detected_lang, "en")
    elif detected_lang != "en":
        logger.info(
            f"Query en {detected_lang}; sin traducir (TRANSLATE_QUERIES=off)"
        )

    # Expansión de acrónimos de dominio (REBT → Reglamento… , PLAN.md §3.1).
    # Se aplica a la consulta de retrieval; original_query se mantiene intacta.
    if EXPAND_ACRONYMS:
        expanded, matched = expand_acronyms(query)
        if matched:
            logger.info(f"Acrónimos expandidos: {matched}")
            query = expanded

    return query, detected_lang, original_query


def choose_retrieval(
    topic: str,
    query: str,
    is_generative: bool = False,
    final_topk_override: int = None,
) -> Tuple[List[Dict], Dict]:
    """
    Selecciona la estrategia de retrieval (BM25-solo vs híbrido) según el modo.

    Usa variables de entorno para configuración (sin hardcodeo).
    Incluye traducción automática de queries no-inglesas.

    `final_topk_override` sólo lo usa `/v1/eval/offline`: permite recuperar más
    hondo que la profundidad de servicio para distinguir "nunca se recuperó" de
    "se recuperó por debajo del corte". Omitido (None) el comportamiento es
    idéntico al de siempre, que es lo que usa la ruta de chat.
    """
    query, detected_lang, original_query = _prepare_query(query)

    q_tokens = len(query.strip().split())

    # Ajustar TOPK usando multiplicador
    if final_topk_override is not None:
        final_topk = final_topk_override
        logger.info(f"TOPK forzado a {final_topk} (override de evaluación)")
    elif is_generative:
        final_topk = FINAL_TOPK * GENERATIVE_TOPK_MULTIPLIER
        logger.info(
            f"Modo GENERATIVO: recuperando {final_topk} chunks "
            f"(base={FINAL_TOPK}, multiplicador={GENERATIVE_TOPK_MULTIPLIER})"
        )
    else:
        final_topk = FINAL_TOPK
        logger.info(f"Modo RESPUESTA: recuperando {final_topk} chunks")

    # Decidir estrategia
    if q_tokens < BM25_FALLBACK_TOKEN_THRESHOLD:
        logger.info(f"Query corta ({q_tokens} tokens) - usando BM25 solo")
        results = bm25_only(topic, query, final_topk, is_generative)
        return results, {
            "mode": "bm25",
            "topk": final_topk,
            "original_language": detected_lang,
            "original_query": original_query,
        }
    else:
        logger.info(f"Query normal ({q_tokens} tokens) - usando Hybrid")
        results, meta = hybrid_retrieve(
            topic, query, final_topk, is_generative
        )
        meta["mode"] = "hybrid"
        meta["topk"] = final_topk
        meta["original_language"] = detected_lang
        meta["original_query"] = original_query
        return results, meta


# ============================================================
# RERANKING
# ============================================================


def rerank_passages(
    query: str, passages: List[Dict], rerank_topk: int = None
) -> List[Dict]:
    """
    Reordena passages usando el reranker.

    Args:
        query: Query string
        passages: Lista de passages a reordenar
        rerank_topk: Número de passages a retornar (None = todos)

    Returns:
        Passages reordenados
    """
    if not passages:
        return []

    if len(passages) == 1:
        return passages

    reranker = get_reranker()

    # Reranquear todos los passages
    order = reranker.rerank(query, [p["text"] for p in passages], topk=len(passages))
    reranked = [passages[i] for i in order]

    # Recortar si se especificó topk
    if rerank_topk is not None:
        reranked = reranked[:rerank_topk]
        logger.info(
            f"[RERANK] Reordenados {len(passages)} passages, retornando top {rerank_topk}"
        )
    else:
        logger.info(f"[RERANK] Reordenados {len(passages)} passages, retornando todos")

    return reranked


# ============================================================
# CITACIONES (wrapper a retrieval_lib.citations)
# ============================================================


def attach_citations(chunks: List[Dict], topic: str = "") -> Tuple[str, List[Dict]]:
    """
    Construye contexto RAG con citaciones clicables.

    Wrapper para mantener compatibilidad hacia atrás.
    """
    return build_context_with_citations(chunks, topic)


# ============================================================
# TELEMETRÍA
# ============================================================


def _rotate_telemetry_if_needed(path: str) -> None:
    """Rota el JSONL cuando cambia el mes y purga los archivos antiguos.

    El fichero activo permanece en `path` (lo que leen eval/§3.1); al cruzar el
    mes se renombra a `path.YYYY-MM` (os.replace es atómico) y se purgan los
    archivos más viejos que TELEMETRY_RETENTION_MONTHS. Best-effort: cualquier
    fallo se registra, nunca interrumpe la petición.
    """
    from datetime import datetime, timedelta
    from glob import glob

    try:
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            now = datetime.now()
            if (mtime.year, mtime.month) != (now.year, now.month):
                archive = f"{path}.{mtime:%Y-%m}"
                if not os.path.exists(archive):
                    os.replace(path, archive)

        if TELEMETRY_RETENTION_MONTHS > 0:
            cutoff = datetime.now() - timedelta(days=31 * TELEMETRY_RETENTION_MONTHS)
            for archived in glob(f"{path}.*"):
                if datetime.fromtimestamp(os.path.getmtime(archived)) < cutoff:
                    os.remove(archived)
    except Exception as e:
        logger.warning(f"No se pudo rotar/purgar la telemetría: {e}")


def telemetry_log(entry: Dict):
    """Registra telemetría en archivo JSONL (con rotación mensual)."""
    ts = int(time.time() * 1000)
    entry["ts"] = ts
    _rotate_telemetry_if_needed(TELEMETRY_PATH)
    try:
        os.makedirs(os.path.dirname(TELEMETRY_PATH), exist_ok=True)
        with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Antes era `except: pass`, que ocultó una caída del logging de
        # telemetría durante días. Registrar, pero no romper la petición.
        logger.warning(f"No se pudo escribir telemetría en {TELEMETRY_PATH}: {e}")


# ============================================================
# DEBUG (mantenido para compatibilidad)
# ============================================================


def debug_retrieval(topic: str, query: str) -> dict:
    """
    Función de debugging completa - muestra TODO lo que pasa en el retrieval.
    Útil para diagnosticar por qué solo trae de un archivo.
    """
    from qdrant_utils import get_collection_stats

    logger.info("\n" + "=" * 80)
    logger.info("DEBUG RETRIEVAL - ANÁLISIS COMPLETO")
    logger.info("=" * 80)

    # 1. Verificar colección
    logger.info("\nColección Qdrant:")
    stats = get_collection_stats(topic)
    if stats:
        logger.info(f"   - Nombre: {stats['collection']}")
        logger.info(f"   - Total puntos: {stats['points_count']}")
        logger.info(f"   - Tamaño vector: {stats['vector_size']}")
        logger.info(f"   - Distancia: {stats['distance']}")
    else:
        logger.error("   No se pudo obtener stats")

    # 2. Query
    logger.info(f"\nQuery: {query[:100]}...")
    logger.info(f"   - Tokens: ~{count_tokens(query)}")

    # 3. Ejecutar búsquedas
    dense, bm25 = _execute_search(topic, query, HYBRID_DENSE_K, HYBRID_BM25_K)

    logger.info(f"\nBúsqueda Densa: {len(dense)} resultados")
    logger.info(f"Búsqueda BM25: {len(bm25)} resultados")

    # 4. Después del merge
    merged, meta = hybrid_retrieve(topic, query, FINAL_TOPK)

    files_merged = {}
    for c in merged:
        file_path = c["file_path"]
        if file_path not in files_merged:
            files_merged[file_path] = 0
        files_merged[file_path] += 1

    logger.info(
        f"\nDespués del Merge: {len(merged)} chunks de {len(files_merged)} archivos"
    )

    logger.info("=" * 80 + "\n")

    return {
        "collection_stats": stats,
        "dense_hits": len(dense),
        "bm25_hits": len(bm25),
        "merged_results": len(merged),
        "unique_files": len(files_merged),
        "results": merged,
    }
