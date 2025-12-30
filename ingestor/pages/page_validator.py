"""
Utilidades de validación y corrección de números de página.

Consolida la lógica de validación de páginas de múltiples fuentes en
una única implementación unificada.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def validate_page_number(
    page: Any,
    total_pages: Optional[int] = None,
    context: str = "",
) -> int:
    """
    Normaliza y valida un número de página único.

    Esta es la única fuente de verdad para validación de páginas,
    reemplazando código de validación duplicado en el código base.

    Args:
        page: Número de página (puede ser int, float, str, o None)
        total_pages: Límite superior opcional para ajuste
        context: Cadena de contexto para logging

    Returns:
        Número de página válido (int >= 1, <= total_pages si se proporciona)
    """
    # Convertir a int
    if page is None:
        if context:
            logger.debug(f"[PAGE] {context}: Página None, usando 1")
        return 1

    try:
        page = int(page)
    except (ValueError, TypeError):
        if context:
            logger.warning(f"[PAGE] {context}: Tipo de página inválido {type(page)}, usando 1")
        return 1

    # Ajustar a rango válido
    if page < 1:
        if context:
            logger.warning(f"[PAGE] {context}: Página {page} -> 1 (ajustado)")
        page = 1

    if total_pages and page > total_pages:
        if context:
            logger.warning(f"[PAGE] {context}: Página {page} -> {total_pages} (ajustado)")
        page = total_pages

    return page


def validate_page_numbers(
    elements: List[Dict[str, Any]],
    pdf_path: Optional[str] = None,
    total_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Valida y corrige números de página en una lista de elementos.

    Args:
        elements: Lista de diccionarios de elementos con clave 'page'
        pdf_path: Ruta opcional para contexto de logging
        total_pages: Límite superior opcional para ajuste

    Returns:
        Elementos con números de página validados
    """
    if not elements:
        return elements

    filename = pdf_path.split("/")[-1] if pdf_path else "desconocido"
    validated = []
    issues = []

    for i, elem in enumerate(elements):
        page = elem.get("page", 1)
        context = f"Elemento {i} en {filename}"
        validated_page = validate_page_number(page, total_pages, context)

        if validated_page != page:
            issues.append(f"Elemento {i}: {page} -> {validated_page}")

        # Crear copia con página validada
        validated_elem = elem.copy()
        validated_elem["page"] = validated_page
        validated.append(validated_elem)

    if issues:
        logger.warning(f"[PAGE] Corregidos {len(issues)} problemas de página en {filename}")

    return validated


class PageSequenceValidator:
    """
    Valida y corrige secuencias de números de página a través de chunks de documentos.

    Detecta y corrige:
    - Números de página inválidos (no-int, < 1, > 50000)
    - Grandes brechas en secuencias de páginas
    - Páginas fuera de orden
    - Páginas que exceden el total de páginas del documento
    """

    @staticmethod
    def validate_and_fix(
        chunks: List[Dict[str, Any]],
        total_pages: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Valida secuencia de páginas e intenta correcciones.

        Args:
            chunks: Lista de diccionarios de chunks
            total_pages: Conteo total de páginas opcional

        Returns:
            Tupla de (chunks_corregidos, problemas_encontrados)
        """
        if not chunks:
            return chunks, []

        issues = []
        pages = [c.get("page", 1) for c in chunks]

        # Detectar problemas
        issues.extend(PageSequenceValidator._detect_invalid_pages(pages))
        issues.extend(PageSequenceValidator._detect_large_gaps(pages))
        issues.extend(PageSequenceValidator._detect_out_of_order(pages))

        if total_pages:
            issues.extend(
                PageSequenceValidator._detect_page_overflow(pages, total_pages)
            )

        # Corregir si se encontraron problemas
        if issues:
            logger.warning(f"[PAGE] Encontrados {len(issues)} problemas, intentando correcciones...")
            chunks = PageSequenceValidator._fix_page_numbers(chunks, pages, total_pages)

        return chunks, issues

    @staticmethod
    def _detect_invalid_pages(pages: List[Any]) -> List[str]:
        """Detecta números de página inválidos."""
        issues = []
        for i, page in enumerate(pages):
            if not isinstance(page, int) or page < 1:
                issues.append(f"Chunk {i}: número de página inválido {page}")
            elif page > 50000:
                issues.append(f"Chunk {i}: número de página sospechoso {page}")
        return issues

    @staticmethod
    def _detect_large_gaps(pages: List[Any]) -> List[str]:
        """Detecta grandes brechas en secuencia de páginas."""
        issues = []
        if not pages:
            return issues

        sorted_unique = sorted(set(p for p in pages if isinstance(p, int) and p > 0))

        for i in range(1, len(sorted_unique)):
            gap = sorted_unique[i] - sorted_unique[i - 1]
            if gap > 10:
                issues.append(
                    f"Brecha grande: {gap} páginas entre "
                    f"{sorted_unique[i - 1]} y {sorted_unique[i]}"
                )

        return issues

    @staticmethod
    def _detect_out_of_order(pages: List[Any]) -> List[str]:
        """Detecta páginas fuera de orden."""
        issues = []
        for i in range(1, len(pages)):
            if isinstance(pages[i], int) and isinstance(pages[i - 1], int):
                if pages[i] < pages[i - 1]:
                    issues.append(
                        f"Fuera de orden: chunk {i - 1} página {pages[i - 1]} -> "
                        f"chunk {i} página {pages[i]}"
                    )
        return issues

    @staticmethod
    def _detect_page_overflow(pages: List[Any], total_pages: int) -> List[str]:
        """Detecta páginas que exceden el conteo total."""
        issues = []
        for i, page in enumerate(pages):
            if isinstance(page, int) and page > total_pages:
                issues.append(
                    f"Chunk {i}: página {page} excede total de páginas {total_pages}"
                )
        return issues

    @staticmethod
    def _fix_page_numbers(
        chunks: List[Dict[str, Any]],
        pages: List[Any],
        total_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Intenta corregir problemas de números de página."""
        fixed = [c.copy() for c in chunks]

        # Corrección 1: Reemplazar páginas inválidas con números secuenciales
        last_valid = 1
        for i, chunk in enumerate(fixed):
            page = chunk.get("page", 1)
            if not isinstance(page, int) or page < 1 or page > 50000:
                chunk["page"] = last_valid
                logger.debug(f"[FIX] Chunk {i}: página inválida -> {last_valid}")
            else:
                last_valid = page

        # Corrección 2: Aplicar restricción de total_pages
        if total_pages:
            for i, chunk in enumerate(fixed):
                page = chunk.get("page", 1)
                if isinstance(page, int) and page > total_pages:
                    chunk["page"] = total_pages
                    logger.debug(f"[FIX] Chunk {i}: página {page} -> {total_pages}")

        # Corrección 3: Suavizar grandes saltos
        for i in range(1, len(fixed)):
            prev_page = fixed[i - 1].get("page", 1)
            curr_page = fixed[i].get("page", 1)

            if isinstance(curr_page, int) and isinstance(prev_page, int):
                if curr_page - prev_page > 20:
                    fixed[i]["page"] = prev_page + 1
                    logger.debug(
                        f"[FIX] Chunk {i}: salto grande {prev_page}->{curr_page}, "
                        f"usando {prev_page + 1}"
                    )

        return fixed
