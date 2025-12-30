"""
Módulos de infraestructura core para el ingestor.

Proporciona gestión unificada de GPU, heartbeat/watchdog, caché y configuración.
"""

from core.config import (
    BM25_BASE_DIR,
    EMBED_DEFAULT,
    EMBED_PER_TOPIC,
    MODEL_CACHE_DIR,
    QDRANT_BATCH_SIZE,
    QDRANT_URL,
    TOPIC_BASE_DIR,
    TOPIC_LABELS,
    ensure_nltk_data,
    get_sent_tokenizer,
    setup_ssl_context,
)
from core.gpu import GPUManager
from core.heartbeat import HeartbeatManager, call_heartbeat, set_heartbeat_callback
from core.cache import ExtractionCache, FileHashCache, get_pdf_total_pages

__all__ = [
    # Configuración
    "TOPIC_LABELS",
    "TOPIC_BASE_DIR",
    "BM25_BASE_DIR",
    "EMBED_PER_TOPIC",
    "EMBED_DEFAULT",
    "QDRANT_URL",
    "QDRANT_BATCH_SIZE",
    "MODEL_CACHE_DIR",
    "setup_ssl_context",
    "ensure_nltk_data",
    "get_sent_tokenizer",
    # GPU
    "GPUManager",
    # Heartbeat
    "HeartbeatManager",
    "call_heartbeat",
    "set_heartbeat_callback",
    # Caché
    "ExtractionCache",
    "FileHashCache",
    "get_pdf_total_pages",
]
