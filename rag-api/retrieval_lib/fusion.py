# Archivo: rag-api/retrieval_lib/fusion.py
# Descripción: Algoritmos de fusión de resultados (RRF) y deduplicación

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Elimina chunks duplicados basándose en (file_path, chunk_id).

    Args:
        chunks: Lista de chunks con claves 'file_path' y 'chunk_id'

    Returns:
        Lista de chunks sin duplicados, manteniendo el orden original
    """
    seen: Set[Tuple[str, int]] = set()
    dedup: List[Dict] = []

    for chunk in chunks:
        key = (chunk["file_path"], chunk["chunk_id"])
        if key not in seen:
            seen.add(key)
            dedup.append(chunk)

    if len(chunks) != len(dedup):
        logger.info(f"Deduplicación: {len(chunks)} -> {len(dedup)} chunks")

    return dedup


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """
    Implementa Reciprocal Rank Fusion para búsqueda híbrida.

    RRF score = sum(1 / (k + rank_i)) para cada retriever

    Es el algoritmo estándar para combinar rankings de múltiples fuentes,
    donde k=60 es el valor típico en la literatura.

    Args:
        dense_results: Resultados de búsqueda densa (Qdrant)
        bm25_results: Resultados de búsqueda BM25 (Whoosh)
        k: Constante para la fórmula RRF (default 60)

    Returns:
        Resultados fusionados ordenados por score RRF (mayor es mejor)
    """
    rrf_scores: Dict[Tuple[str, int], float] = {}
    result_map: Dict[Tuple[str, int], Dict] = {}

    # Calcular scores desde búsqueda densa
    for rank, result in enumerate(dense_results):
        key = (result["file_path"], result["chunk_id"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = result

    # Calcular scores desde búsqueda BM25
    for rank, result in enumerate(bm25_results):
        key = (result["file_path"], result["chunk_id"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = result

    # Combinar resultados con scores RRF
    merged: List[Dict] = []
    for key, result in result_map.items():
        result["score_rrf"] = rrf_scores[key]
        result["score_hybrid"] = rrf_scores[key]  # Compatibilidad hacia atrás
        merged.append(result)

    # Ordenar por score RRF (mayor es mejor)
    merged.sort(key=lambda x: x["score_rrf"], reverse=True)

    logger.debug(
        f"[RRF] Fusionados {len(dense_results)} denso + "
        f"{len(bm25_results)} BM25 -> {len(merged)} resultados"
    )

    return merged


def weighted_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict]:
    """
    Fusión ponderada simple de resultados.

    Alternativa a RRF cuando se quiere controlar el peso de cada fuente.

    Args:
        dense_results: Resultados de búsqueda densa
        bm25_results: Resultados de búsqueda BM25
        dense_weight: Peso para resultados densos (default 0.6)
        bm25_weight: Peso para resultados BM25 (default 0.4)

    Returns:
        Resultados fusionados ordenados por score combinado
    """
    result_map: Dict[Tuple[str, int], Dict] = {}
    scores: Dict[Tuple[str, int], float] = {}

    # Normalizar y ponderar scores densos
    if dense_results:
        max_dense = max(r.get("score_dense", 1.0) for r in dense_results)
        for result in dense_results:
            key = (result["file_path"], result["chunk_id"])
            normalized = result.get("score_dense", 0) / max_dense if max_dense > 0 else 0
            scores[key] = scores.get(key, 0) + normalized * dense_weight
            if key not in result_map:
                result_map[key] = result

    # Normalizar y ponderar scores BM25
    if bm25_results:
        max_bm25 = max(r.get("score_bm25", 1.0) for r in bm25_results)
        for result in bm25_results:
            key = (result["file_path"], result["chunk_id"])
            normalized = result.get("score_bm25", 0) / max_bm25 if max_bm25 > 0 else 0
            scores[key] = scores.get(key, 0) + normalized * bm25_weight
            if key not in result_map:
                result_map[key] = result

    # Combinar resultados
    merged: List[Dict] = []
    for key, result in result_map.items():
        result["score_hybrid"] = scores[key]
        merged.append(result)

    merged.sort(key=lambda x: x["score_hybrid"], reverse=True)

    return merged
