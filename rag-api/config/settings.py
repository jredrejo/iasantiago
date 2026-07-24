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


def get_bool_env(key: str, default: bool) -> bool:
    """Lee una variable de entorno como bool ('1/true/yes/on' → True)."""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# TRADUCCIÓN DE QUERIES
# ============================================================
# Kill-switch para la traducción es→en previa al retrieval. Con el embedder
# multilingüe (e5-instruct, cross-lingual) traducir la consulta perjudica al
# corpus mayoritariamente en español: destruye la rama BM25 y añade ruido de
# traducción (PLAN.md §3.3/§1.4/§4.5). Por defecto DESACTIVADO: la consulta va
# tal cual a densa y BM25. Poner TRANSLATE_QUERIES=true restaura el comportamiento
# anterior sin tocar código.
TRANSLATE_QUERIES = get_bool_env("TRANSLATE_QUERIES", False)

# Expansión de acrónimos de dominio en la consulta (REBT → Reglamento
# Electrotécnico para Baja Tensión, etc.). Ayuda cuando la norma se cita por
# acrónimo pero el PDF usa el nombre completo (PLAN.md §3.1). Kill-switch por si
# alguna vez estorba; por defecto ACTIVADO. La tabla vive en acronyms.py.
EXPAND_ACRONYMS = get_bool_env("EXPAND_ACRONYMS", True)


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
# REVISIONES FIJADAS - Pin de commit para modelos con trust_remote_code
# ============================================================
# Estos modelos ejecutan código remoto (trust_remote_code=True). Fijar la
# revisión al SHA auditado impide que un `pull` introduzca código sin auditar.
# Un modelo que NO esté en el mapa se carga sin pin (revision=None), igual que
# antes: así cambiar el nombre de un modelo nunca fija un SHA equivocado en
# silencio. Actualizar el SHA aquí es una decisión deliberada de auditoría.
PINNED_MODEL_REVISIONS = {
    "intfloat/multilingual-e5-large-instruct": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
    "jinaai/jina-reranker-v2-base-multilingual": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
}


def get_model_revision(model_name: str):
    """
    Devuelve el SHA de commit fijado para un modelo, o None si no está fijado.

    None reproduce el comportamiento previo (carga la rama por defecto). Una
    variable de entorno HF_REVISION_<...> permite sobreescribir el pin sin tocar
    el código, para rotar a un SHA recién auditado.

    Args:
        model_name: Nombre del modelo HF (ej: "jinaai/jina-reranker-v2-...")

    Returns:
        SHA de commit fijado, o None.
    """
    env_key = "HF_REVISION_" + model_name.replace("/", "_").replace("-", "_").upper()
    override = os.getenv(env_key)
    if override and override.strip():
        return override.strip()
    return PINNED_MODEL_REVISIONS.get(model_name)


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

# Default alineado con el volumen real (/data/telemetry). El antiguo default
# /app/retrieval.jsonl escribía dentro de la imagen (root-owned, no persistente).
TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "/data/telemetry/retrieval.jsonl")

# Meses de telemetría a conservar. El JSONL rota por mes; los archivos más
# antiguos que esta ventana se purgan (0 o negativo = sin purga).
TELEMETRY_RETENTION_MONTHS = get_int_env("TELEMETRY_RETENTION_MONTHS", 6)


# ============================================================
# SAMPLING PARAMETERS
# ============================================================

# Response mode (Q&A, explanations)
RESPONSE_TEMPERATURE = float(os.getenv("RESPONSE_TEMPERATURE", "0.4"))
RESPONSE_TOP_P = float(os.getenv("RESPONSE_TOP_P", "0.8"))
RESPONSE_TOP_K = get_int_env("RESPONSE_TOP_K", 20)
RESPONSE_REPETITION_PENALTY = float(os.getenv("RESPONSE_REPETITION_PENALTY", "1.05"))

# Generative mode (exams, exercises)
GENERATIVE_TEMPERATURE = float(os.getenv("GENERATIVE_TEMPERATURE", "0.3"))
GENERATIVE_TOP_P = float(os.getenv("GENERATIVE_TOP_P", "0.75"))
GENERATIVE_TOP_K = get_int_env("GENERATIVE_TOP_K", 15)
GENERATIVE_REPETITION_PENALTY = float(os.getenv("GENERATIVE_REPETITION_PENALTY", "1.1"))
