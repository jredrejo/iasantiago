"""
Saneado de consultas para la rama léxica BM25.

Quedó sólo esto cuando se retiró Whoosh (§7.4, 2026-08-01): la búsqueda la
sirve ahora `sparse_utils.py` con el vector disperso de Qdrant. La función se
conserva porque la ruta dispersa depende de su comportamiento de descarte
(consultas de sistema o vacías -> ""), del que cuelga el fallback BM25-solo.
"""

import re
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

    # 1. Limpiar símbolos problemáticos.
    # Venían de la sintaxis del parser de Whoosh; se mantienen porque también
    # son ruido para el tokenizador del BM25 disperso.
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
