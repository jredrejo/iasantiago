"""
Pipeline de extracción para orquestar múltiples extractores con respaldo.

Proporciona una interfaz unificada para extracción de PDF que automáticamente
intenta múltiples métodos de extracción hasta que uno tenga éxito.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.cache import get_pdf_total_pages
from extraction.base import Element, ExtractionError, check_pdf_has_text

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Orquesta la extracción con cadena de respaldo automática.

    Orden por defecto:
    1. DoclingExtractor (acelerado por GPU, mejor calidad)
    2. TextExtractor (pypdf + pdfplumber, rápido)
    3. OCRExtractor (EasyOCR + Tesseract, para docs escaneados)
    4. UnstructuredExtractor (hi_res, consciente del layout)

    El pipeline valida los resultados después de cada extractor y
    recurre al siguiente si los resultados son insuficientes.
    """

    def __init__(
        self,
        extractors: Optional[List] = None,
        min_chars: int = 500,
        min_avg_chars_per_page: int = 80,
        min_chars_last_resort: int = 10,
    ):
        """
        Inicializa el pipeline de extracción.

        Args:
            extractors: Lista de extractores a usar. Si es None, usa cadena por defecto.
            min_chars: Mínimo de caracteres totales para considerar extracción suficiente.
            min_avg_chars_per_page: Mínimo promedio de caracteres por página.
            min_chars_last_resort: Mínimo de caracteres para el último extractor (último recurso).
        """
        self._extractors = extractors
        self.min_chars = min_chars
        self.min_avg_chars_per_page = min_avg_chars_per_page
        self.min_chars_last_resort = min_chars_last_resort

    @property
    def extractors(self) -> List:
        """Obtiene la lista de extractores, inicializada perezosamente."""
        if self._extractors is None:
            self._extractors = self._create_default_extractors()
        return self._extractors

    def _create_default_extractors(self) -> List:
        """Crea la cadena de extractores por defecto."""
        extractors = []

        # Intentar importar cada extractor
        try:
            from extraction.docling_extractor import DoclingExtractor

            extractors.append(DoclingExtractor())
        except ImportError as e:
            logger.warning(f"[PIPELINE] DoclingExtractor no disponible: {e}")

        try:
            from extraction.text_extractor import TextExtractor

            extractors.append(TextExtractor())
        except ImportError as e:
            logger.warning(f"[PIPELINE] TextExtractor no disponible: {e}")

        try:
            from extraction.ocr_extractor import OCRExtractor

            extractors.append(OCRExtractor())
        except ImportError as e:
            logger.warning(f"[PIPELINE] OCRExtractor no disponible: {e}")

        try:
            from extraction.unstructured_extractor import UnstructuredExtractor

            extractors.append(UnstructuredExtractor(strategy="hi_res"))
        except ImportError as e:
            logger.warning(f"[PIPELINE] UnstructuredExtractor no disponible: {e}")

        if not extractors:
            raise RuntimeError("No hay extractores disponibles")

        return extractors

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae elementos del PDF usando la cadena de respaldo.

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Lista de objetos Element extraídos

        Raises:
            ExtractionError: Si todos los extractores fallan
        """
        pdf_path = Path(pdf_path)
        total_pages = get_pdf_total_pages(str(pdf_path))

        logger.info(f"[PIPELINE] Iniciando extracción para {pdf_path.name}")
        logger.info(f"[PIPELINE] Total de páginas: {total_pages or 'desconocido'}")

        errors = []
        extractor_list = self.extractors
        total_extractors = len(extractor_list)

        for idx, extractor in enumerate(extractor_list):
            is_last = idx == total_extractors - 1

            if not extractor.can_handle(pdf_path):
                logger.info(
                    f"[PIPELINE] {extractor.name} no puede manejar este archivo, omitiendo"
                )
                continue

            try:
                logger.info(f"[PIPELINE] Intentando {extractor.name}...")
                elements = extractor.extract(pdf_path)

                if self._is_sufficient(elements, total_pages, is_last_resort=is_last):
                    elements = self._validate_pages(elements, total_pages)
                    logger.info(
                        f"[PIPELINE] {extractor.name} exitoso con "
                        f"{len(elements)} elementos"
                    )
                    return elements

                logger.info(
                    f"[PIPELINE] {extractor.name} retornó resultados insuficientes, "
                    f"intentando siguiente extractor"
                )

            except Exception as e:
                logger.warning(f"[PIPELINE] {extractor.name} falló: {e}")
                errors.append(f"{extractor.name}: {e}")

        error_msg = (
            f"Todos los extractores fallaron para {pdf_path.name}. "
            f"Errores: {'; '.join(errors)}"
        )
        logger.error(f"[PIPELINE] {error_msg}")
        raise ExtractionError(error_msg)

    def extract_with_strategy(
        self,
        pdf_path: Path,
        strategy: str = "auto",
    ) -> List[Element]:
        """
        Extrae con una estrategia específica.

        Args:
            pdf_path: Ruta al archivo PDF
            strategy: Estrategia de extracción:
                - "auto": Usa cadena completa de respaldo
                - "fast": Usa solo TextExtractor
                - "ocr": Usa solo OCRExtractor
                - "docling": Usa solo DoclingExtractor
                - "hi_res": Usa UnstructuredExtractor con hi_res

        Returns:
            Lista de objetos Element extraídos
        """
        pdf_path = Path(pdf_path)

        if strategy == "auto":
            return self.extract(pdf_path)

        if strategy == "fast":
            from extraction.text_extractor import TextExtractor

            return TextExtractor().extract(pdf_path)

        if strategy == "ocr":
            from extraction.ocr_extractor import OCRExtractor

            return OCRExtractor().extract(pdf_path)

        if strategy == "docling":
            from extraction.docling_extractor import DoclingExtractor

            return DoclingExtractor().extract(pdf_path)

        if strategy == "hi_res":
            from extraction.unstructured_extractor import UnstructuredExtractor

            return UnstructuredExtractor(strategy="hi_res").extract(pdf_path)

        raise ValueError(f"Estrategia desconocida: {strategy}")

    def _is_sufficient(
        self,
        elements: List[Element],
        total_pages: Optional[int],
        is_last_resort: bool = False,
    ) -> bool:
        """
        Verifica si los resultados de extracción son suficientes.

        Args:
            elements: Lista de elementos extraídos
            total_pages: Número total de páginas del PDF
            is_last_resort: Si es True, usa umbral mínimo más bajo (último recurso)
        """
        if not elements:
            return False

        total_chars = sum(len(e.text.strip()) for e in elements)

        # Usar umbral más bajo para el último extractor (último recurso)
        min_chars_threshold = self.min_chars_last_resort if is_last_resort else self.min_chars

        if total_chars < min_chars_threshold:
            logger.info(
                f"[PIPELINE] Caracteres insuficientes: {total_chars} < {min_chars_threshold}"
            )
            return False

        # Solo verificar promedio por página si no es último recurso
        if not is_last_resort and total_pages and total_pages > 0:
            avg_chars = total_chars / total_pages
            if avg_chars < self.min_avg_chars_per_page:
                logger.info(
                    f"[PIPELINE] Promedio de caracteres/página insuficiente: "
                    f"{avg_chars:.0f} < {self.min_avg_chars_per_page}"
                )
                return False

        return True

    def _validate_pages(
        self, elements: List[Element], total_pages: Optional[int]
    ) -> List[Element]:
        """Valida y ajusta los números de página."""
        if not elements:
            return elements

        validated = []
        issues = []

        for i, elem in enumerate(elements):
            page = elem.page

            # Ajustar a rango válido
            if page < 1:
                issues.append(f"Elemento {i}: Página {page} -> 1 (ajustado)")
                page = 1

            if total_pages and page > total_pages:
                issues.append(f"Elemento {i}: Página {page} -> {total_pages} (ajustado)")
                page = total_pages

            # Crear nuevo elemento con página validada
            validated.append(
                Element(
                    text=elem.text,
                    type=elem.type,
                    page=page,
                    source=elem.source,
                    bbox=elem.bbox,
                    metadata=elem.metadata,
                )
            )

        if issues:
            logger.info(f"[PIPELINE] Corregidos {len(issues)} problemas de número de página")

        return validated


# Funciones de conveniencia para compatibilidad hacia atrás


def extract_pdf(pdf_path: str, strategy: str = "auto") -> List[Dict[str, Any]]:
    """
    Extrae elementos del PDF (función compatible hacia atrás).

    Args:
        pdf_path: Ruta al archivo PDF
        strategy: Estrategia de extracción

    Returns:
        Lista de diccionarios de elementos
    """
    pipeline = ExtractionPipeline()
    elements = pipeline.extract_with_strategy(Path(pdf_path), strategy)
    return [e.to_dict() for e in elements]


def extract_elements_best_effort(
    pdf_path: str,
    *,
    has_extractable_text: Optional[bool] = None,
    total_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Punto de entrada unificado de extracción (compatible hacia atrás).

    Esto reemplaza la antigua función extract_elements_best_effort de chunk.py.
    """
    pdf_path_obj = Path(pdf_path)

    if total_pages is None:
        total_pages = get_pdf_total_pages(str(pdf_path))

    if has_extractable_text is None:
        has_extractable_text = check_pdf_has_text(pdf_path_obj)

    pipeline = ExtractionPipeline()

    if has_extractable_text:
        # Intentar extractores basados en texto primero
        try:
            from extraction.unstructured_extractor import UnstructuredExtractor

            extractor = UnstructuredExtractor(strategy="hi_res", enable_tables=False)
            elements = extractor.extract(pdf_path_obj)

            total_chars = sum(len(e.text.strip()) for e in elements)
            avg = total_chars / max(total_pages or 1, 1)

            if elements and total_chars > 500 and avg > 80:
                return [
                    e.to_dict() for e in pipeline._validate_pages(elements, total_pages)
                ]

            logger.info("[EXTRACT] hi_res insuficiente -> pipeline OCR")

        except Exception as e:
            logger.warning(f"[EXTRACT] hi_res falló -> pipeline OCR: {e}")

    # Respaldo al pipeline OCR
    from extraction.ocr_extractor import OCRExtractor

    extractor = OCRExtractor()
    elements = extractor.extract(pdf_path_obj)
    return [e.to_dict() for e in pipeline._validate_pages(elements, total_pages)]


def pdf_to_chunks_with_enhanced_validation(
    pdf_path: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    strategy: str = "adaptive",
) -> List[Dict[str, Any]]:
    """
    Extrae y fragmenta contenido PDF (compatible hacia atrás).

    Esta es una versión simplificada - la lógica completa de fragmentación
    debería estar en el módulo chunking.
    """
    pipeline = ExtractionPipeline()
    elements = pipeline.extract(Path(pdf_path))
    return [e.to_dict() for e in elements]
