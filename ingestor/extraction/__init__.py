"""
Capa de extracción para el módulo ingestor.

Proporciona una interfaz unificada para extraer contenido de archivos PDF
usando múltiples estrategias de extracción con respaldo automático.
"""

from extraction.base import Element, ExtractorProtocol, ExtractionError
from extraction.pipeline import ExtractionPipeline

__all__ = [
    "Element",
    "ExtractorProtocol",
    "ExtractionError",
    "ExtractionPipeline",
]
