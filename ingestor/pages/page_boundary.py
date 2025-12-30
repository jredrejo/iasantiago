"""
Detección avanzada de límites de página para atribución precisa de páginas.

Usa análisis de estructura de PDF para determinar con precisión a qué página
pertenece un elemento basándose en coordenadas.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdvancedPageBoundaryDetector:
    """
    Detecta límites de página usando múltiples técnicas:
    - Análisis de estructura visual
    - Marcadores de contenido de texto
    - Metadatos de PDF
    """

    def __init__(self):
        self._boundaries_cache: Dict[str, Dict[int, Dict[str, float]]] = {}

    def detect_boundaries(self, pdf_path: str) -> Dict[int, Dict[str, float]]:
        """
        Detecta límites de página para un PDF.

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Diccionario mapeando números de página a info de límites:
            {
                1: {"height": 792.0, "width": 612.0, "text_top": 72.0, ...},
                2: {...},
            }
        """
        if pdf_path in self._boundaries_cache:
            return self._boundaries_cache[pdf_path]

        boundaries: Dict[int, Dict[str, float]] = {}

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    boundaries[page_num] = {
                        "height": page.height,
                        "width": page.width,
                        "bbox": page.bbox,
                        "text_top": self._get_text_top(page),
                        "text_bottom": self._get_text_bottom(page),
                    }

            self._boundaries_cache[pdf_path] = boundaries

        except Exception as e:
            logger.error(f"[PAGE] Error detectando límites: {e}")

        return boundaries

    def _get_text_top(self, page) -> float:
        """Obtiene la coordenada de texto más arriba en una página."""
        try:
            words = page.extract_words()
            if words:
                return min(word["top"] for word in words)
            return 0.0
        except Exception:
            return 0.0

    def _get_text_bottom(self, page) -> float:
        """Obtiene la coordenada de texto más abajo en una página."""
        try:
            words = page.extract_words()
            if words:
                return max(word["bottom"] for word in words)
            return page.height
        except Exception:
            return page.height

    def assign_precise_page(
        self,
        elem: Any,
        boundaries: Dict[int, Dict[str, float]],
    ) -> int:
        """
        Asigna número de página usando información de límites.

        Args:
            elem: Elemento al cual asignar página
            boundaries: Información de límites de página de detect_boundaries()

        Returns:
            Número de página (indexado desde 1)
        """
        if not boundaries:
            return self._get_element_page(elem)

        try:
            # Intentar obtener coordenadas del elemento
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "coordinates"):
                coords = elem.metadata.coordinates
                if hasattr(coords, "points") and len(coords.points) > 0:
                    # Obtener coordenada y mínima
                    y_coord = min(p[1] for p in coords.points)

                    # Encontrar en qué página cae esta coordenada
                    cumulative_height = 0
                    for page_num in sorted(boundaries.keys()):
                        page_height = boundaries[page_num]["height"]
                        if (
                            cumulative_height
                            <= y_coord
                            < cumulative_height + page_height
                        ):
                            return page_num
                        cumulative_height += page_height

                    # Más allá de todas las páginas conocidas
                    return max(boundaries.keys())

        except Exception as e:
            logger.debug(f"[PAGE] Error asignando página precisa: {e}")

        return self._get_element_page(elem)

    def _get_element_page(self, elem: Any) -> int:
        """Obtiene página del propio atributo page del elemento."""
        if isinstance(elem, dict):
            return elem.get("page", 1)
        if hasattr(elem, "page"):
            return elem.page
        return 1

    def clear_cache(self) -> None:
        """Limpia la caché de límites."""
        self._boundaries_cache.clear()


# Funciones de conveniencia a nivel de módulo

_detector: Optional[AdvancedPageBoundaryDetector] = None


def get_page_boundary_detector() -> AdvancedPageBoundaryDetector:
    """Obtiene o crea el detector de límites global."""
    global _detector
    if _detector is None:
        _detector = AdvancedPageBoundaryDetector()
    return _detector


def detect_page_boundaries(pdf_path: str) -> Dict[int, Dict[str, float]]:
    """Detecta límites de página para un PDF."""
    return get_page_boundary_detector().detect_boundaries(pdf_path)


def assign_precise_page(
    elem: Any,
    pdf_path: str,
) -> int:
    """
    Asigna número de página preciso a un elemento.

    Args:
        elem: Elemento al cual asignar página
        pdf_path: Ruta al archivo PDF

    Returns:
        Número de página (indexado desde 1)
    """
    detector = get_page_boundary_detector()
    boundaries = detector.detect_boundaries(pdf_path)
    return detector.assign_precise_page(elem, boundaries)
