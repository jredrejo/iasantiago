# Archivo: rag-api/retrieval_lib/search.py
# Descripción: Helpers compartidos para funciones de búsqueda

import logging
from typing import Dict, List

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
        if "instruct" in embed_model_name.lower():
            # E5-instruct variant uses "Instruct: {task}\nQuery: {q}"
            query_with_prefix = f"Instruct: Retrieve relevant documents\nQuery: {query}"
            logger.debug("[E5-instruct] Using instruct query prefix")
        else:
            # E5 base variant uses "query: "
            query_with_prefix = f"query: {query}"
            logger.debug("[E5] Using query: prefix")
        return query_with_prefix

    return query
