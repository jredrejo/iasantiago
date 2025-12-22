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
# PSM 6 = Single uniform block (fastest)
# PSM 11 = Sparse text (good for academic papers)
# PSM 12 = Sparse text with OSD
os.environ["TESSERACT_PSM"] = "6"  # Single uniform block - FASTEST

# Enable neural network acceleration (if compiled with support)
os.environ["TESSERACT_ENABLE_LSTM"] = "1"  # LSTM networks are faster

# Additional Tesseract optimizations
os.environ["TESSERACT_OEM_ENGINE"] = "1"  # Use LSTM engine only
os.environ["TESSERACT_USER_DEFINED_DPI"] = "300"  # Standard DPI
os.environ["TESSERACT_MAX_RECOGNITION_TIME"] = "10"  # Limit per page

# Configure language priority (Spanish first for your use case)
os.environ["TESSERACT_LANG"] = "spa+eng"
os.environ["OCR_LANGUAGES"] = "spa+eng"
os.environ["UNSTRUCTURED_LANGUAGES"] = "spa,eng"
os.environ["UNSTRUCTURED_FALLBACK_LANGUAGE"] = "eng"
