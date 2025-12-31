"""
Fragmentador de documentos consciente del contexto.

Mantiene el contexto a través de los límites de página mientras preserva
atribución precisa de página para cada fragmento.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from chunking.strategies import adaptive_chunk, semantic_chunk, simple_chunk
from pages.page_boundary import AdvancedPageBoundaryDetector
from pages.page_extractor import RobustPageExtractor
from pages.page_validator import PageSequenceValidator

logger = logging.getLogger(__name__)


class ContextAwareChunker:
    """
    Fragmentación avanzada que mantiene contexto a través de límites de página.

    Características:
    - Agrupa elementos por página
    - Lleva contexto de páginas anteriores
    - Soporta múltiples estrategias de fragmentación
    - Valida y corrige números de página
    """

    def __init__(
        self,
        chunk_size: int = 900,
        overlap: int = 120,
        min_chunk_size: int = 100,
    ):
        """
        Inicializa el fragmentador.

        Args:
            chunk_size: Tamaño máximo de fragmento en caracteres
            overlap: Número de caracteres de solapamiento entre fragmentos
            min_chunk_size: Tamaño mínimo de fragmento a incluir
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.page_detector = AdvancedPageBoundaryDetector()
        self._cached_boundaries: Optional[Dict] = None

    def chunk_with_context_preservation(
        self,
        elements: List[Dict[str, Any]],
        pdf_path: Optional[str] = None,
        strategy: str = "adaptive",
    ) -> List[Dict[str, Any]]:
        """
        Divide documento en fragmentos preservando contexto y precisión de página.

        Args:
            elements: Lista de diccionarios de elementos
            pdf_path: Ruta opcional al PDF para detección de límites
            strategy: Estrategia de fragmentación ('adaptive', 'semantic', 'simple')

        Returns:
            Lista de diccionarios de fragmentos
        """
        # Obtener límites de página
        boundaries = self._get_boundaries(pdf_path)

        # Agrupar elementos por página
        page_groups = self._group_by_page(elements, boundaries, pdf_path)

        all_chunks = []

        for page_num in sorted(page_groups.keys()):
            page_elements = page_groups[page_num]

            # Agregar contexto de página anterior
            if page_num > 1 and page_num - 1 in page_groups:
                prev_elements = page_groups[page_num - 1]
                context_elements = self._select_context_elements(prev_elements)
                page_elements = context_elements + page_elements

            # Aplicar estrategia de fragmentación
            page_chunks = self._chunk_page(page_elements, page_num, strategy)

            all_chunks.extend(page_chunks)

        # Validar y corregir fragmentos
        all_chunks = self._validate_chunks(all_chunks, boundaries)

        return all_chunks

    def _get_boundaries(self, pdf_path: Optional[str]) -> Dict:
        """Obtiene límites de página, cacheados para llamadas repetidas."""
        if self._cached_boundaries is not None:
            return self._cached_boundaries

        if pdf_path and pdf_path.lower().endswith(".pdf"):
            self._cached_boundaries = self.page_detector.detect_boundaries(pdf_path)
            return self._cached_boundaries

        return {}

    def _group_by_page(
        self,
        elements: List[Dict],
        boundaries: Dict,
        pdf_path: Optional[str] = None,
    ) -> Dict[int, List[Dict]]:
        """Agrupa elementos por página con detección mejorada."""
        page_groups = defaultdict(list)

        for elem in elements:
            if boundaries:
                page = self.page_detector.assign_precise_page(elem, boundaries)
            else:
                fallback = elem.get("page", 1) if isinstance(elem, dict) else 1
                page = RobustPageExtractor.extract_page_number(
                    elem, pdf_path=pdf_path, fallback_page=fallback
                )

            if not isinstance(page, int) or page < 1:
                logger.warning(f"[CHUNK] Página inválida {page}, usando página 1")
                page = 1

            page_groups[page].append(elem)

        return dict(page_groups)

    def _select_context_elements(self, prev_elements: List[Dict]) -> List[Dict]:
        """Selecciona elementos relevantes de página anterior para contexto."""
        # Obtener encabezados y títulos
        headings = [e for e in prev_elements if e.get("type") in ["heading", "title"]]

        # Obtener elementos de texto
        text_elements = [e for e in prev_elements if e.get("type") == "text"]

        # Construir contexto: encabezados + últimos elementos de texto
        context = headings[:2]
        if len(text_elements) > 3:
            context.extend(text_elements[-3:])
        else:
            context.extend(text_elements)

        # Marcar como elementos de contexto
        for elem in context:
            elem = elem.copy()
            elem["is_context"] = True
            elem["source_page"] = elem.get("page", 1)

        return [e.copy() for e in context]

    def _chunk_page(
        self,
        elements: List[Dict],
        page_num: int,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """Aplica estrategia de fragmentación a elementos de página."""
        # Separar elementos de contexto
        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        context_text = self._prepare_context_text(context_elements)

        if strategy == "adaptive":
            return adaptive_chunk(
                page_elements,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_chunk_size=self.min_chunk_size,
                page_num=page_num,
                context_text=context_text,
            )
        elif strategy == "semantic":
            text_content = "\n\n".join(
                e.get("text", "") for e in page_elements if e.get("type") == "text"
            )
            return semantic_chunk(
                text_content,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_chunk_size=self.min_chunk_size,
                page_num=page_num,
                context_text=context_text,
            )
        else:
            text_content = "\n\n".join(
                e.get("text", "") for e in page_elements if e.get("text", "").strip()
            )
            return simple_chunk(
                text_content,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_chunk_size=self.min_chunk_size,
                page_num=page_num,
                context_text=context_text,
            )

    def _prepare_context_text(self, context_elements: List[Dict]) -> str:
        """Prepara texto de contexto de elementos de página anterior."""
        if not context_elements:
            return ""

        context_texts = []
        for elem in context_elements:
            text = elem.get("text", "").strip()
            if text:
                if elem.get("type") in ["heading", "title"]:
                    context_texts.append(f"## {text}")
                else:
                    context_texts.append(text)

        if context_texts:
            return "\n\n".join(context_texts) + "\n\n"

        return ""

    def _validate_chunks(
        self,
        chunks: List[Dict],
        boundaries: Dict,
    ) -> List[Dict]:
        """Valida y corrige datos de fragmentos."""
        validated = []
        max_page = max(boundaries.keys()) if boundaries else None

        for i, chunk in enumerate(chunks):
            # Validar número de página
            page = chunk.get("page", 1)
            if not isinstance(page, int) or page < 1:
                logger.error(f"[CHUNK] Página inválida en fragmento {i}: {page}")
                chunk["page"] = 1

            if max_page and page > max_page:
                logger.warning(f"[CHUNK] Página {page} excede máximo {max_page}")
                chunk["page"] = max_page

            # Validar texto
            text = chunk.get("text", "").strip()
            if text and len(text) >= self.min_chunk_size:
                chunk["text"] = text
                chunk["chunk_id"] = i
                chunk["char_count"] = len(text)
                validated.append(chunk)
            elif text:
                logger.debug(
                    f"[CHUNK] Omitiendo fragmento pequeño {i}: {len(text)} caracteres"
                )

        logger.info(f"[CHUNK] Validados {len(validated)}/{len(chunks)} fragmentos")

        return validated

    def clear_cache(self) -> None:
        """Limpia límites cacheados."""
        self._cached_boundaries = None


# Función de conveniencia para compatibilidad hacia atrás


def chunk_document(
    elements: List[Dict[str, Any]],
    pdf_path: Optional[str] = None,
    chunk_size: int = 900,
    overlap: int = 120,
    strategy: str = "adaptive",
) -> List[Dict[str, Any]]:
    """
    Fragmenta un documento con preservación de contexto.

    Args:
        elements: Lista de diccionarios de elementos
        pdf_path: Ruta opcional del PDF para detección de límites
        chunk_size: Tamaño máximo de fragmento
        overlap: Solapamiento entre fragmentos
        strategy: Estrategia de fragmentación

    Returns:
        Lista de diccionarios de fragmentos
    """
    chunker = ContextAwareChunker(
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunker.chunk_with_context_preservation(
        elements, pdf_path=pdf_path, strategy=strategy
    )
