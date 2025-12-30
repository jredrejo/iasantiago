"""
Paquete Ingestor de PDF.

Proporciona extracción de documentos, fragmentación e indexación para sistemas RAG.

Módulos principales:
- core: Configuración, gestión GPU, caché, heartbeat
- extraction: Pipeline de extracción de PDF con múltiples estrategias
- pages: Validación y extracción de números de página
- chunking: Estrategias de fragmentación de documentos
- indexing: Servicios de indexación Qdrant y Whoosh
- state: Gestión del estado de procesamiento
"""

# Configuración core
from core.config import (
    BM25_BASE_DIR,
    EMBED_DEFAULT,
    EMBED_PER_TOPIC,
    QDRANT_URL,
    TOPIC_BASE_DIR,
    TOPIC_LABELS,
)

# Pipeline de extracción
from extraction.base import Element, ExtractionError
from extraction.pipeline import (
    ExtractionPipeline,
    extract_pdf,
    pdf_to_chunks_with_enhanced_validation,
)

# Servicios de indexación
from indexing import (
    EmbeddingService,
    QdrantService,
    WhooshService,
    ensure_qdrant,
    ensure_whoosh,
    topic_collection,
    validate_and_fix_vectors,
)

# Gestión de estado
from state import ProcessingState, get_processing_state

__all__ = [
    # Configuración
    "BM25_BASE_DIR",
    "EMBED_DEFAULT",
    "EMBED_PER_TOPIC",
    "QDRANT_URL",
    "TOPIC_BASE_DIR",
    "TOPIC_LABELS",
    # Extracción
    "Element",
    "ExtractionError",
    "ExtractionPipeline",
    "extract_pdf",
    "pdf_to_chunks_with_enhanced_validation",
    # Indexación
    "EmbeddingService",
    "QdrantService",
    "WhooshService",
    "ensure_qdrant",
    "ensure_whoosh",
    "topic_collection",
    "validate_and_fix_vectors",
    # Estado
    "ProcessingState",
    "get_processing_state",
]

__version__ = "2.0.0"
