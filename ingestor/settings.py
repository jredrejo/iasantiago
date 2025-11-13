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
# BLACKWELL (RTX 50XX) SPECIFIC SETTINGS
# ============================================================

# Batch size multipliers for Blackwell
BLACKWELL_BATCH_MULTIPLIER = float(os.getenv("BLACKWELL_BATCH_MULTIPLIER", "1.5"))
BLACKWELL_SMALL_MODEL_MULTIPLIER = float(
    os.getenv("BLACKWELL_SMALL_MODEL_MULTIPLIER", "1.5")
)
BLACKWELL_FP8_MULTIPLIER = float(os.getenv("BLACKWELL_FP8_MULTIPLIER", "1.2"))


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
# QDRANT
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))

