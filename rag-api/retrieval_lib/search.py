# Archivo: rag-api/retrieval_lib/search.py
# Descripción: Helpers compartidos para funciones de búsqueda

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def apply_per_file_limit(
    results: List[Dict],
    max_per_file: int,
    final_topk: int,
) -> List[Dict]:
    """
    Limita el número de chunks por archivo y aplica topk final.

    Asegura diversidad en los resultados evitando que un solo
    archivo domine los resultados.

    Args:
        results: Lista de chunks ordenados por relevancia
        max_per_file: Máximo de chunks por archivo (0 = sin límite)
        final_topk: Número total de chunks a retornar

    Returns:
        Lista filtrada de chunks
    """
    if max_per_file == 0 and final_topk == 0:
        return results

    file_counts: Dict[str, int] = {}
    filtered: List[Dict] = []

    for chunk in results:
        file_path = chunk["file_path"]
        current_count = file_counts.get(file_path, 0)

        # Verificar límite por archivo
        if max_per_file == 0 or current_count < max_per_file:
            filtered.append(chunk)
            file_counts[file_path] = current_count + 1

        # Verificar límite total
        if final_topk > 0 and len(filtered) >= final_topk:
            break

    unique_files = len(set(r["file_path"] for r in filtered))
    logger.debug(
        f"Filtrado: {len(filtered)} chunks de {unique_files} archivos "
        f"(max_per_file={max_per_file}, topk={final_topk})"
    )

    return filtered


def prepare_query_for_retrieval(
    query: str,
    embed_model_name: str,
) -> str:
    """
    Prepara el query para búsqueda, añadiendo prefijos si es necesario.

    Para modelos E5, añade el prefijo estándar de query.

    Args:
        query: Query original del usuario
        embed_model_name: Nombre del modelo de embeddings

    Returns:
        Query preparado (posiblemente con prefijo)
    """
    if "e5" in embed_model_name.lower():
        query_with_prefix = f"Represent this query for search: {query}"
        logger.debug("[E5] Usando prefijo 'Represent this query for search:'")
        return query_with_prefix

    return query


def execute_hybrid_search(
    topic: str,
    query: str,
    dense_k: int,
    bm25_k: int,
    embedder,
    embed_model_name: str,
    search_dense_fn,
    bm25_search_fn,
    bm25_base_dir: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Ejecuta búsqueda híbrida (densa + BM25).

    Helper que encapsula la lógica común de búsqueda para evitar
    duplicación entre hybrid_retrieve y hybrid_retrieve_enhanced.

    Args:
        topic: Tema de búsqueda
        query: Query del usuario
        dense_k: Número de resultados densos
        bm25_k: Número de resultados BM25
        embedder: Modelo de embeddings
        embed_model_name: Nombre del modelo para detección de E5
        search_dense_fn: Función de búsqueda densa (search_dense)
        bm25_search_fn: Función de búsqueda BM25 (bm25_search_safe)
        bm25_base_dir: Directorio base de índices BM25

    Returns:
        Tupla (resultados_densos, resultados_bm25)
    """
    # Preparar query con prefijo si es necesario (E5)
    query_for_embedding = prepare_query_for_retrieval(query, embed_model_name)

    # Generar embedding del query
    q_vec = embedder.encode([query_for_embedding], normalize_embeddings=True)[
        0
    ].tolist()

    # Búsqueda densa
    dense_hits = search_dense_fn(topic, q_vec, dense_k)
    dense_results = [
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
    bm25_results = bm25_search_fn(bm25_base_dir, topic, query, bm25_k)

    logger.debug(
        f"Búsqueda híbrida: {len(dense_results)} densos, {len(bm25_results)} BM25"
    )

    return dense_results, bm25_results


def translate_if_needed(
    query: str,
    detect_language_fn,
    translate_query_fn,
) -> Tuple[str, str, str]:
    """
    Traduce el query a inglés si es necesario.

    Args:
        query: Query original
        detect_language_fn: Función para detectar idioma
        translate_query_fn: Función para traducir

    Returns:
        Tupla (query_traducido, idioma_original, query_original)
    """
    detected_lang = detect_language_fn(query)
    original_query = query

    if detected_lang != "en":
        logger.info(f"Query en {detected_lang}, traduciendo a inglés para retrieval")
        query, _ = translate_query_fn(query, detected_lang, "en")

    return query, detected_lang, original_query


def count_query_tokens(query: str) -> int:
    """
    Cuenta tokens en el query de forma simple (split por espacios).

    Usado para decidir entre BM25-only y búsqueda híbrida.

    Args:
        query: Query a analizar

    Returns:
        Número aproximado de tokens
    """
    return len(query.strip().split())


def soft_trim_context(
    chunks: List[Dict], token_limit: int, count_tokens_fn
) -> List[Dict]:
    """
    Recorta chunks para no exceder el límite de tokens.

    Args:
        chunks: Lista de chunks ordenados por relevancia
        token_limit: Límite máximo de tokens
        count_tokens_fn: Función para contar tokens

    Returns:
        Chunks que caben en el límite de tokens
    """
    total = 0
    result: List[Dict] = []

    for chunk in chunks:
        chunk_tokens = count_tokens_fn(chunk["text"])
        if total + chunk_tokens > token_limit:
            break
        total += chunk_tokens
        result.append(chunk)

    if len(result) < len(chunks):
        logger.info(
            f"Contexto recortado: {len(chunks)} -> {len(result)} chunks "
            f"(límite: {token_limit} tokens)"
        )

    return result
