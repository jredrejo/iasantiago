# Archivo: rag-api/config/settings.py
# Descripción: Configuración centralizada desde variables de entorno

import os


def get_int_env(key: str, default: int) -> int:
    """
    Lee una variable de entorno como int, manejando casos de cadena vacía.

    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto si no existe o es inválido

    Returns:
        Valor entero de la variable o el default
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# ============================================================
# TOPICS - Configuración de temas educativos
# ============================================================

TOPIC_LABELS = [
    t.strip()
    for t in os.getenv("TOPIC_LABELS", "Chemistry,Electronics,Programming").split(",")
]

TOPIC_BASE_DIR = os.getenv("TOPIC_BASE_DIR", "/topics")


# ============================================================
# EMBEDDINGS - Modelos de embedding por tema
# ============================================================

# Modelo por defecto
EMBED_DEFAULT = os.getenv(
    "EMBED_MODEL_DEFAULT",
    "intfloat/multilingual-e5-large-instruct",
)

# Modelos específicos por tema
EMBED_PER_TOPIC = {
    "Programming": os.getenv("EMBED_MODEL_PROGRAMMING", EMBED_DEFAULT),
    "Electronics": os.getenv("EMBED_MODEL_ELECTRONICS", EMBED_DEFAULT),
    "Chemistry": os.getenv("EMBED_MODEL_CHEMISTRY", EMBED_DEFAULT),
}

# Modelo de reranking
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "jinaai/jina-reranker-v2-base-multilingual",
)


# ============================================================
# ALMACENAMIENTO - URLs y rutas de bases de datos
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
BM25_BASE_DIR = os.getenv("BM25_BASE_DIR", "/whoosh")


# ============================================================
# LÍMITES DE CONTEXTO - Tokens y chunks
# ============================================================

# Límite suave de tokens en contexto RAG
CTX_TOKENS_SOFT_LIMIT = get_int_env("CTX_TOKENS_SOFT_LIMIT", 4000)

# Límite de tokens para modo generativo (mayor contexto)
CTX_TOKENS_GENERATIVE = get_int_env("CTX_TOKENS_GENERATIVE", 10000)

# Máximo de chunks por archivo en modo respuesta
MAX_CHUNKS_PER_FILE = get_int_env("MAX_CHUNKS_PER_FILE", 3)

# Máximo de chunks por archivo en modo generativo (más contexto)
MAX_CHUNKS_PER_FILE_GENERATIVE = get_int_env("MAX_CHUNKS_PER_FILE_GENERATIVE", 5)


# ============================================================
# BÚSQUEDA HÍBRIDA - Parámetros de retrieval
# ============================================================

# Número de resultados de búsqueda densa (Qdrant)
HYBRID_DENSE_K = get_int_env("HYBRID_DENSE_K", 40)

# Número de resultados de búsqueda BM25 (Whoosh)
HYBRID_BM25_K = get_int_env("HYBRID_BM25_K", 40)

# Número final de resultados después de fusión
FINAL_TOPK = get_int_env("FINAL_TOPK", 5)

# Umbral de tokens para fallback a BM25-only
BM25_FALLBACK_TOKEN_THRESHOLD = get_int_env("BM25_FALLBACK_TOKEN_THRESHOLD", 4)

# Multiplicador de topk para modo generativo
GENERATIVE_TOPK_MULTIPLIER = get_int_env("GENERATIVE_TOPK_MULTIPLIER", 4)


# ============================================================
# vLLM - Configuración del servidor de inferencia
# ============================================================

UPSTREAM_OPENAI_URL = os.getenv("UPSTREAM_OPENAI_URL", "http://vllm:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Límites del modelo
VLLM_MAX_MODEL_LEN = get_int_env("VLLM_MAX_MODEL_LEN", 32768)
VLLM_MAX_TOKENS = get_int_env("VLLM_MAX_TOKENS", 4096)


# ============================================================
# TOKENS DINÁMICOS - Porcentajes para max_tokens
# ============================================================

# Porcentaje del modelo para respuesta en modo generativo
GENERATIVE_MAX_TOKENS_PERCENT = get_int_env("GENERATIVE_MAX_TOKENS_PERCENT", 60)

# Porcentaje del modelo para respuesta en modo normal
RESPONSE_MAX_TOKENS_PERCENT = get_int_env("RESPONSE_MAX_TOKENS_PERCENT", 25)

# Mínimo absoluto de tokens para cualquier respuesta
MIN_RESPONSE_TOKENS = get_int_env("MIN_RESPONSE_TOKENS", 512)


# ============================================================
# TELEMETRÍA - Logging y métricas
# ============================================================

TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "/app/retrieval.jsonl")
