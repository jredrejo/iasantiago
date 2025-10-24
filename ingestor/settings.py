import os

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

TOPIC_LABELS = [
    t.strip()
    for t in os.getenv("TOPIC_LABELS", "Chemistry,Electronics,Programming").split(",")
]
TOPIC_BASE_DIR = os.getenv("TOPIC_BASE_DIR", "/topics")
BM25_BASE_DIR = os.getenv("BM25_BASE_DIR", "/whoosh")

# ============================================================
# EMBEDDINGS (Sentence Transformers)
# ============================================================

EMBED_PER_TOPIC = {
    "Chemistry": os.getenv(
        "EMBED_MODEL_CHEMISTRY",
        os.getenv("EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"),
    ),
    "Electronics": os.getenv(
        "EMBED_MODEL_ELECTRONICS",
        os.getenv("EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"),
    ),
    "Programming": os.getenv(
        "EMBED_MODEL_PROGRAMMING",
        os.getenv("EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"),
    ),
}

EMBED_DEFAULT = os.getenv(
    "EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"
)

# ============================================================
# EXTRACTION ENGINE (HÍBRIDO)
# ============================================================

EXTRACTION_ENGINE = os.getenv("EXTRACTION_ENGINE", "hybrid")
"""
Opciones:
  - "hybrid" (default): Usa MinerU para PDFs complejos, Unstructured para el resto
  - "unstructured": Solo Unstructured.io
  - "mineru": Solo MinerU (solo PDFs)
"""

# ============================================================
# MINERU CONFIGURATION (OPCIONAL)
# ============================================================

MINERU_ENABLED = os.getenv("MINERU_ENABLED", "true").lower() == "true"
"""Si está disponible, usar MinerU para PDFs complejos"""

MINERU_COMPLEXITY_THRESHOLD = float(os.getenv("MINERU_COMPLEXITY_THRESHOLD", "0.6"))
"""
Score 0.0-1.0 para usar MinerU
- < 0.6: Usar Unstructured.io (más rápido)
- >= 0.6: Usar MinerU (más preciso)
"""

MINERU_CHECK_PAGES = int(os.getenv("MINERU_CHECK_PAGES", "3"))
"""Número de páginas a analizar para detectar complejidad"""

# ============================================================
# CACHÉ SQLITES
# ============================================================

LLAVA_CACHE_DB = os.getenv("LLAVA_CACHE_DB", "/tmp/llava_cache/llava_cache.db")
"""Ruta de base de datos SQLite para caché de imágenes/tablas"""

# ============================================================
# vLLM (ANÁLISIS CON LLAVA)
# ============================================================

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "60"))

# ============================================================
# EXTRACCIÓN
# ============================================================

MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "3"))

MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "100"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "100"))

LLM_ANALYSIS_TEMPERATURE = float(os.getenv("LLM_ANALYSIS_TEMPERATURE", "0.3"))

# ============================================================
# QDRANT
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================
# MÉTRICAS
# ============================================================

SAVE_EXTRACTION_METRICS = os.getenv("SAVE_EXTRACTION_METRICS", "true").lower() == "true"
"""Guardar métricas de cada extracción en SQLite"""

# ============================================================
# DEBUG
# ============================================================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE_EXTRACTION = os.getenv("VERBOSE_EXTRACTION", "false").lower() == "true"
