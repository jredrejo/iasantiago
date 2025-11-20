import os
import multiprocessing

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


# ============================================================
# TESSERACT OCR OPTIMIZATION
# ============================================================

# Enable OpenMP for Tesseract (multi-threading)
os.environ["OMP_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 2))
os.environ["OMP_THREAD_LIMIT"] = str(multiprocessing.cpu_count())

# Tesseract performance settings
os.environ["TESSERACT_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 2))

# Use faster PSM (Page Segmentation Mode) if possible
# PSM 3 = Fully automatic page segmentation (default, but slower)
# PSM 6 = Assume a single uniform block of text (faster)
# PSM 1 = Automatic page segmentation with OSD (slowest, most accurate)
os.environ["TESSERACT_PSM"] = "6"  # Faster mode

# Enable neural network acceleration (if compiled with support)
os.environ["TESSERACT_ENABLE_LSTM"] = "1"  # LSTM networks are faster

# Configure language priority (Spanish first for your use case)
os.environ["TESSERACT_LANG"] = "spa+eng"
os.environ["OCR_LANGUAGES"] = "spa+eng"
os.environ["UNSTRUCTURED_LANGUAGES"] = "spa,eng"
os.environ["UNSTRUCTURED_FALLBACK_LANGUAGE"] = "eng"

# ============================================================
# UNSTRUCTURED PERFORMANCE SETTINGS
# ============================================================

# Number of parallel processes for unstructured
UNSTRUCTURED_PARALLEL_PROCESSES = max(1, multiprocessing.cpu_count() - 2)

# Batch size for processing multiple PDFs
PDF_BATCH_SIZE = 4  # Process 4 PDFs in parallel

# ============================================================
# GPU-ACCELERATED OCR (Optional - requires special build)
# ============================================================

# If you have a GPU-enabled Tesseract build (requires compilation with CUDA)
# Uncomment these lines:
# os.environ["TESSERACT_USE_GPU"] = "1"
# os.environ["TESSERACT_GPU_DEVICE"] = "0"  # Use first GPU

