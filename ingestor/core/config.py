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
# Volumen de estado del ingestor: `.processing_state.json` y sus copias.
# Era `/whoosh` (alojaba los índices Whoosh) hasta que se renombró el
# 2026-08-01, al retirarlos. **Perder este directorio obliga a reprocesar el
# corpus entero**, así que no es una caché y no comparte volumen con ninguna.
STATE_BASE_DIR = os.getenv("STATE_BASE_DIR", "/state")

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

# Sufijo de nombre de colección (§7.4). El vector disperso no se puede añadir a
# una colección ya creada —Qdrant 1.15.5 responde "Not existing vector name"—,
# así que las colecciones con denso+disperso se crearon aparte como
# `rag_<tema>_v2`. **En producción vale `_v2`** (está en `.env`); el defecto
# vacío sólo vale para una instalación desde cero. Debe coincidir con rag-api:
# si discrepan, el ingestor escribe en una colección y el servicio lee de otra.
QDRANT_COLLECTION_SUFFIX = os.getenv("QDRANT_COLLECTION_SUFFIX", "")

# ============================================================
# BÚSQUEDA LÉXICA DISPERSA (§7.4)
# ============================================================

# Escribe un vector disperso BM25 junto al denso, en el mismo punto y el mismo
# upsert. **Encendido por defecto desde que Whoosh se retiró (2026-08-01)**: es
# la única rama léxica que queda, así que apagarlo deja la búsqueda en densa
# sola, sin error y sin aviso. Sólo se apaga para diagnosticar.
SPARSE_ENABLED = os.getenv("SPARSE_ENABLED", "true").strip().lower() == "true"
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "bm25")
SPARSE_LANGUAGE = os.getenv("SPARSE_LANGUAGE", "spanish")

# ============================================================
# HEARTBEAT Y WATCHDOG
# ============================================================

# Única fuente de verdad del timeout. El watchdog mata el proceso al superarlo;
# el healthcheck de compose deriva su umbral de la misma variable para que
# ambos no discrepen (antes: 1200 s watchdog vs 300 s healthcheck vs 450 s README).
WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT", "1200"))
WATCHDOG_CHECK_INTERVAL = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60"))
HEARTBEAT_FILE = os.getenv("HEARTBEAT_FILE", "/tmp/ingestor_heartbeat")

# Rastro que deja el watchdog justo antes de matar el proceso. Sin él, un kill
# del watchdog y una caída nativa son indistinguibles: los dos dejan el contador
# de `crash_state` incrementado con el motivo "conversión interrumpida".
# **No va junto a HEARTBEAT_FILE a propósito**: ése vive en /tmp y se lo lleva
# el reinicio de contenedor que provoca el propio watchdog. Va en el volumen de
# estado, que es lo único que sobrevive.
WATCHDOG_KILL_MARKER = os.getenv(
    "WATCHDOG_KILL_MARKER", os.path.join(STATE_BASE_DIR, "watchdog_kill.json")
)

# Fichero "en vuelo": qué PDF estaba procesándose cuando el proceso murió.
# `crash_state.json` sólo cubre docling; una muerte dura en la cadena OCR, en el
# VLM, al fragmentar o al escribir en Qdrant no dejaba **ningún** rastro, así que
# el reinicio volvía a coger el mismo fichero y la ingesta no avanzaba nunca.
# Va en el volumen de estado, no en /tmp: tiene que sobrevivir al reinicio que
# provoca el propio watchdog. Lo consume `initial_scan` al arrancar.
INFLIGHT_FILE = os.getenv("INFLIGHT_FILE", os.path.join(STATE_BASE_DIR, "inflight.json"))

# Margen que el supervisor externo espera POR ENCIMA de WATCHDOG_TIMEOUT antes
# de matar. El watchdog en proceso tiene que disparar primero siempre que pueda:
# es el que sabe atribuir la muerte (escribe el rastro que reatribuye el motivo
# en `crash_state`). El supervisor sólo actúa cuando aquél no puede correr —un
# cuelgue nativo con el GIL retenido—, que es justo el caso del §6.8.
SUPERVISOR_GRACE_SECONDS = int(os.getenv("SUPERVISOR_GRACE_SECONDS", "300"))

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

# Lote de forward-pass del embedder. Era 32 (muy conservador). Con la GPU
# dedicada de 32 GiB y chunks de ~256 tokens, 128 sobra y acelera notablemente;
# si un lote provoca OOM de CUDA, encode() ya degrada a CPU sin abortar (§7.3).
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))

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
