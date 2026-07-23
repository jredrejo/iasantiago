"""
Configuración unificada para el módulo ingestor.

Consolida configuraciones de settings.py, configuración SSL,
y configuración basada en variables de entorno.
"""

import logging
import multiprocessing
import os
import ssl
from contextlib import contextmanager

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
# HEARTBEAT Y WATCHDOG
# ============================================================

# Única fuente de verdad del timeout. El watchdog mata el proceso al superarlo;
# el healthcheck de compose deriva su umbral de la misma variable para que
# ambos no discrepen (antes: 1200 s watchdog vs 300 s healthcheck vs 450 s README).
WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT", "1200"))
WATCHDOG_CHECK_INTERVAL = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60"))
HEARTBEAT_FILE = os.getenv("HEARTBEAT_FILE", "/tmp/ingestor_heartbeat")

# Presupuesto de reloj para UNA conversión de Docling. Mientras dura, un
# BackgroundHeartbeat mantiene vivo el proceso (una conversión sana de un
# manual de cientos de páginas tarda legítimamente más que WATCHDOG_TIMEOUT).
# Pasado este tope el heartbeat se detiene y el watchdog mata un cuelgue real.
# 5400s = 90 min: holgado para PDFs sanos grandes, finito para los patológicos.
DOCLING_CONVERT_MAX_SECONDS = int(os.getenv("DOCLING_CONVERT_MAX_SECONDS", "5400"))

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
# FRAGMENTACIÓN
# ============================================================

# Presupuesto de tokens por fragmento.
#
# 512 es el techo duro: los tres modelos de embedding en uso
# (e5-large-instruct, gte-large, instructor-large) truncan ahí, así que subirlo
# sólo devolvería el truncado silencioso que la Fase 1 vino a eliminar.
#
# El valor por defecto es 256 por medición, no por intuición. Barrido sobre el
# tema Programming (13 consultas con verdad de referencia por página):
#
#     métrica        sin fragmentar    512 tokens    256 tokens
#     PageRecall@1          0.6154        0.6154        0.8462
#     PageRecall@3          0.7692        0.6923        0.9231
#     PageMRR               0.6987        0.6841        0.8846
#
# A 512 el fragmento es tan grande que el embedding se diluye: empata en @1 y
# EMPEORA en @3/@5 respecto a no fragmentar. A 256 gana o empata en las 13
# consultas. Antes de tocar esto, repetir el barrido.
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "256"))

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


@contextmanager
def unverified_ssl_context():
    """
    Desactiva temporalmente la verificación TLS y la restaura al salir.

    Sólo debe envolver la descarga de modelos de EasyOCR, que usa un CDN con
    certificados problemáticos. La versión anterior desactivaba la verificación
    globalmente y de forma permanente para todo el proceso, afectando también a
    las descargas de HuggingFace y a cualquier conexión https a Qdrant.
    """
    original = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.info("[SSL] Verificación TLS desactivada temporalmente")
        yield
    except Exception as e:
        logger.warning(f"[SSL] No se pudo configurar contexto SSL: {e}")
        yield
    finally:
        ssl._create_default_https_context = original
        logger.info("[SSL] Verificación TLS restaurada")
