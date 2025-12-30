"""
Extracción robusta de números de página con múltiples estrategias.

Proporciona estrategias de respaldo para extraer números de página de
varios tipos de elementos y formatos de metadatos.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RobustPageExtractor:
    """
    Extrae números de página usando múltiples estrategias con validación.

    Orden de estrategias:
    1. Metadatos del elemento (atributo page_number)
    2. Coordenadas del elemento (para PDFs con info de layout)
    3. Análisis de contenido de texto (marcadores de página)
    4. Inferencia basada en posición
    5. Respaldo a valor por defecto
    """

    _cache: Dict[int, int] = {}
    _pdf_layout_cache: Dict[str, List[float]] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Limpia la caché de extracción de páginas."""
        cls._cache.clear()
        cls._pdf_layout_cache.clear()

    @classmethod
    def extract_page_number(
        cls,
        elem: Any,
        pdf_path: Optional[str] = None,
        fallback_page: int = 1,
    ) -> int:
        """
        Extrae número de página con múltiples estrategias de respaldo.

        Args:
            elem: Elemento del cual extraer página
            pdf_path: Ruta opcional del PDF para extracción basada en coordenadas
            fallback_page: Página por defecto si la extracción falla

        Returns:
            Número de página extraído (int >= 1)
        """
        elem_id = id(elem)
        if elem_id in cls._cache:
            return cls._cache[elem_id]

        # Estrategia 1: Metadatos
        page = cls._extract_from_metadata(elem)
        if page is not None:
            cls._cache[elem_id] = page
            return page

        # Estrategia 2: Coordenadas (solo PDF)
        if pdf_path and pdf_path.lower().endswith(".pdf"):
            page = cls._extract_from_coordinates(elem, pdf_path)
            if page is not None:
                cls._cache[elem_id] = page
                return page

        # Estrategia 3: Contenido de texto
        page = cls._extract_from_text_content(elem)
        if page is not None:
            cls._cache[elem_id] = page
            return page

        # Estrategia 4: Inferencia de posición
        page = cls._infer_from_position(elem)
        if page is not None:
            cls._cache[elem_id] = page
            return page

        # Respaldo
        logger.debug(
            f"[PAGE] No se pudo extraer página para {type(elem).__name__}, "
            f"usando respaldo: {fallback_page}"
        )
        cls._cache[elem_id] = fallback_page
        return fallback_page

    @classmethod
    def _extract_from_metadata(cls, elem: Any) -> Optional[int]:
        """Extrae página de metadatos del elemento."""
        try:
            # Verificar metadata.page_number
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                page = elem.metadata.page_number
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

            # Verificar metadata.to_dict()
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "to_dict"):
                meta_dict = elem.metadata.to_dict()
                if "page_number" in meta_dict:
                    page = meta_dict["page_number"]
                    if isinstance(page, (int, float)) and page > 0:
                        return int(page)

            # Verificar elem.page
            if hasattr(elem, "page"):
                page = elem.page
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

            # Verificar dict['page']
            if isinstance(elem, dict) and "page" in elem:
                page = elem["page"]
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

        except Exception as e:
            logger.debug(f"[PAGE] Extracción de metadatos falló: {e}")

        return None

    @classmethod
    def _extract_from_coordinates(cls, elem: Any, pdf_path: str) -> Optional[int]:
        """Extrae página de coordenadas del elemento."""
        try:
            if not hasattr(elem, "metadata") or not hasattr(
                elem.metadata, "coordinates"
            ):
                return None

            coords = elem.metadata.coordinates
            if not hasattr(coords, "points") or not coords.points:
                return None

            # Obtener alturas de página de caché o calcular
            page_heights = cls._get_page_heights(pdf_path)
            if not page_heights:
                return None

            # Obtener coordenada y del elemento
            y_coord = min(p[1] for p in coords.points)

            # Encontrar en qué página cae la coordenada
            cumulative_height = 0
            for page_num, height in enumerate(page_heights, 1):
                if cumulative_height <= y_coord < cumulative_height + height:
                    return page_num
                cumulative_height += height

            # Más allá de todas las páginas
            return len(page_heights)

        except Exception as e:
            logger.debug(f"[PAGE] Extracción de coordenadas falló: {e}")

        return None

    @classmethod
    def _get_page_heights(cls, pdf_path: str) -> List[float]:
        """Obtiene alturas de página para un PDF (cacheado)."""
        if pdf_path in cls._pdf_layout_cache:
            return cls._pdf_layout_cache[pdf_path]

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                heights = [page.height for page in pdf.pages]
                cls._pdf_layout_cache[pdf_path] = heights
                return heights
        except Exception as e:
            logger.debug(f"[PAGE] No se pudieron obtener alturas de página: {e}")
            return []

    @classmethod
    def _extract_from_text_content(cls, elem: Any) -> Optional[int]:
        """Extrae página de marcadores de contenido de texto."""
        try:
            text = None
            if isinstance(elem, dict):
                text = elem.get("text", "")
            elif hasattr(elem, "text"):
                text = elem.text

            if not text:
                return None

            # Buscar marcadores de página en primeros 100 caracteres
            text_start = text[:100].lower()

            # Patrón: "page X" o "página X"
            patterns = [
                r"page\s+(\d+)",
                r"página\s+(\d+)",
                r"pág\.\s*(\d+)",
                r"\bpg\.\s*(\d+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text_start, re.IGNORECASE)
                if match:
                    page = int(match.group(1))
                    if 1 <= page <= 10000:
                        return page

        except Exception as e:
            logger.debug(f"[PAGE] Extracción de contenido de texto falló: {e}")

        return None

    @classmethod
    def _infer_from_position(cls, elem: Any) -> Optional[int]:
        """Infiere página de la posición del elemento (ej. índice del elemento)."""
        try:
            # Verificar metadatos relacionados con posición
            if hasattr(elem, "metadata"):
                meta = elem.metadata

                # Verificar element_index que podría correlacionar con página
                if hasattr(meta, "element_index"):
                    index = meta.element_index
                    if isinstance(index, int) and index >= 0:
                        # Estimación aproximada: ~5-10 elementos por página
                        return max(1, (index // 8) + 1)

        except Exception as e:
            logger.debug(f"[PAGE] Inferencia de posición falló: {e}")

        return None


def extract_page_number(
    elem: Any,
    pdf_path: Optional[str] = None,
    fallback: int = 1,
) -> int:
    """
    Función de conveniencia para extracción de página.

    Args:
        elem: Elemento del cual extraer página
        pdf_path: Ruta opcional del PDF
        fallback: Número de página por defecto

    Returns:
        Número de página extraído
    """
    return RobustPageExtractor.extract_page_number(elem, pdf_path, fallback)
