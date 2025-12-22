# ingestor/docling_client.py
"""
Cliente para extracción de PDF con Docling (simplificado - llamadas a funciones directas)
Ya no usa HTTP - llama a docling_extractor directamente
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from chunk import check_pdf_has_text
from docling_extractor import extract_elements_from_pdf, DoclingCrashLimitExceeded

logger = logging.getLogger(__name__)


# Re-exportar excepción para que main.py pueda capturarla
__all__ = ["DoclingClient", "DoclingCrashLimitExceeded"]


class DoclingClient:
    """
    Cliente para extracción de PDF con Docling.
    Usa llamadas directas a funciones a docling_extractor (sin HTTP).
    """

    def __init__(self, enable_fallback: bool = True):
        self.enable_fallback = enable_fallback

    def extract_pdf_sync(
        self, pdf_path: Path, fallback_func: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracción síncrona (no se necesita async para llamadas locales).
        Para PDFs escaneados (sin capa de texto), activa fallback de EasyOCR si los resultados son insuficientes.
        """
        try:
            logger.info(f"[DOCLING] Extrayendo: {pdf_path.name}")
            elements = extract_elements_from_pdf(pdf_path)
            logger.info(
                f"[DOCLING] Extraídos {len(elements)} elementos de {pdf_path.name}"
            )

            # Para PDFs escaneados (sin capa de texto), verificar si los resultados son suficientes
            # Si no, usar fallback de pipeline EasyOCR para OCR acelerado por GPU
            if self.enable_fallback and fallback_func:
                has_text_layer = check_pdf_has_text(pdf_path)
                if not has_text_layer:
                    total_text = sum(len(e.get("text", "")) for e in elements)
                    if total_text < 200:
                        logger.warning(
                            f"[DOCLING] PDF escaneado con texto insuficiente ({total_text} caracteres) "
                            "- usando fallback EasyOCR"
                        )
                        return fallback_func(pdf_path)

            return elements

        except Exception as e:
            logger.error(f"[DOCLING] La extracción falló para {pdf_path.name}: {e}")

            if self.enable_fallback and fallback_func:
                logger.warning("[DOCLING] Usando función de fallback externa")
                return fallback_func(pdf_path)

            raise
