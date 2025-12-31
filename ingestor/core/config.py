"""
Configuración unificada para el módulo ingestor.

Consolida configuraciones de settings.py, configuración SSL, inicialización NLTK,
y configuración basada en variables de entorno.
"""

import logging
import multiprocessing
import os
import ssl
from typing import Callable, Optional

import nltk

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN DE TEMAS
# ============================================================

TOPIC_LABELS = [
    t.strip()
    for t in os.getenv("TOPIC_LABELS", "Chemistry,Electronics,Programming").split(",")
]
TOPIC_BASE_DIR = os.getenv("TOPIC_BASE_DIR", "/topics")
BM25_BASE_DIR = os.getenv("BM25_BASE_DIR", "/whoosh")

# ============================================================
# MODELOS DE EMBEDDING (Sentence Transformers)
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
# CACHÉ DE MODELOS
# ============================================================

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE_DIR

# ============================================================
# QDRANT
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))

# ============================================================
# DISPOSITIVO Y TIPO DE EMBEDDING
# ============================================================

EMBEDDING_DTYPE = os.getenv("EMBEDDING_DTYPE", "float16")

# ============================================================
# PROCESAMIENTO POR LOTES
# ============================================================

LARGE_PDF_BATCH_SIZE = int(os.getenv("LARGE_PDF_BATCH_SIZE", "1000"))
ENCODING_MEGA_BATCH_SIZE = int(os.getenv("ENCODING_MEGA_BATCH_SIZE", "5000"))

# ============================================================
# CONFIGURACIÓN DE TESSERACT OCR
# ============================================================

os.environ["OMP_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 2))
os.environ["OMP_THREAD_LIMIT"] = str(multiprocessing.cpu_count())
os.environ["TESSERACT_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 2))
os.environ["TESSERACT_PSM"] = "6"
os.environ["TESSERACT_ENABLE_LSTM"] = "1"
os.environ["TESSERACT_OEM_ENGINE"] = "1"
os.environ["TESSERACT_USER_DEFINED_DPI"] = "300"
os.environ["TESSERACT_MAX_RECOGNITION_TIME"] = "10"
os.environ["TESSERACT_LANG"] = "spa+eng"
os.environ["OCR_LANGUAGES"] = "spa+eng"
os.environ["UNSTRUCTURED_LANGUAGES"] = "spa,eng"
os.environ["UNSTRUCTURED_FALLBACK_LANGUAGE"] = "eng"

# ============================================================
# CONFIGURACIÓN CUDA PARA UNSTRUCTURED
# ============================================================

if os.getenv("UNSTRUCTURED_ENABLE_CUDA", "true").lower() == "true":
    os.environ.pop("UNSTRUCTURED_DISABLE_CUDA", None)
    logger.info("[CONFIG] CUDA HABILITADO para detección de layout de Unstructured")
else:
    os.environ["UNSTRUCTURED_DISABLE_CUDA"] = "1"
    logger.warning("[CONFIG] CUDA DESHABILITADO para Unstructured")


# ============================================================
# CONFIGURACIÓN DE CONTEXTO SSL
# ============================================================


def setup_ssl_context() -> None:
    """Configura contexto SSL para manejar problemas de certificados en descarga de modelos."""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.info(
            "[SSL] Configurado contexto SSL no verificado para descarga de modelos"
        )
    except Exception as e:
        logger.warning(f"[SSL] No se pudo configurar contexto SSL: {e}")


# ============================================================
# INICIALIZACIÓN DE NLTK
# ============================================================

_nltk_available: bool = False
_cached_sent_tokenizer: Optional[Callable] = None


def ensure_nltk_data() -> bool:
    """Asegura que los datos de NLTK estén disponibles con opciones de respaldo."""
    global _nltk_available
    try:
        nltk.data.find("tokenizers/punkt")
        logger.info("[NLTK] Datos de punkt ya disponibles")
        _nltk_available = True
    except LookupError:
        try:
            logger.info("[NLTK] Descargando datos de punkt...")
            nltk.download("punkt", quiet=True)
            logger.info("[NLTK] Datos de punkt descargados exitosamente")
            _nltk_available = True
        except Exception as e:
            logger.error(f"[NLTK] Error al descargar punkt: {e}")
            logger.warning("[NLTK] Se usará tokenización de frases de respaldo")
            _nltk_available = False
    return _nltk_available


def _fallback_sentence_split(text: str) -> list:
    """Divisor de frases de respaldo cuando NLTK no está disponible."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def get_sent_tokenizer() -> Callable:
    """Obtiene tokenizador de frases en caché (español primero con respaldo en inglés)."""
    global _cached_sent_tokenizer

    if _cached_sent_tokenizer is not None:
        return _cached_sent_tokenizer

    try:
        from nltk.tokenize import sent_tokenize

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        def spanish_tokenize(text: str) -> list:
            try:
                return sent_tokenize(text, language="spanish")
            except Exception:
                return sent_tokenize(text, language="english")

        _cached_sent_tokenizer = spanish_tokenize
        logger.info("[NLTK] Tokenizador de frases en español cargado")
    except Exception as e:
        logger.warning(f"[NLTK] Error al cargar, usando respaldo: {e}")
        _cached_sent_tokenizer = _fallback_sentence_split

    return _cached_sent_tokenizer


# Inicializar NLTK al importar el módulo
ensure_nltk_data()
