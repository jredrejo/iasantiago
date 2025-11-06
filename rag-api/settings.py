import os


# Helper para leer variables de entorno de forma segura
def get_int_env(key: str, default: int) -> int:
    """Lee una variable de entorno como int, manejando casos de cadena vacía"""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


TOPIC_LABELS = [
    t.strip()
    for t in os.getenv("TOPIC_LABELS", "Chemistry,Electronics,Programming").split(",")
]
TOPIC_BASE_DIR = os.getenv("TOPIC_BASE_DIR", "/topics")

EMBED_PER_TOPIC = {
    "Programming": os.getenv(
        "EMBED_MODEL_PROGRAMMING",
        os.getenv("EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"),
    ),
    "Electronics": os.getenv(
        "EMBED_MODEL_ELECTRONICS",
        os.getenv("EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"),
    ),
    "Chemistry": os.getenv(
        "EMBED_MODEL_CHEMISTRY",
        os.getenv("EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"),
    ),
}
EMBED_DEFAULT = os.getenv(
    "EMBED_MODEL_DEFAULT", "intfloat/multilingual-e5-large-instruct"
)

RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
BM25_BASE_DIR = os.getenv("BM25_BASE_DIR", "/whoosh")

# Contexto RAG - usar helper seguro
CTX_TOKENS_SOFT_LIMIT = get_int_env("CTX_TOKENS_SOFT_LIMIT", 4000)
CTX_TOKENS_GENERATIVE = get_int_env("CTX_TOKENS_GENERATIVE", 10000)  # ✨ Nueva variable
MAX_CHUNKS_PER_FILE = get_int_env("MAX_CHUNKS_PER_FILE", 3)
HYBRID_DENSE_K = get_int_env("HYBRID_DENSE_K", 40)
HYBRID_BM25_K = get_int_env("HYBRID_BM25_K", 40)
FINAL_TOPK = get_int_env("FINAL_TOPK", 5)
BM25_FALLBACK_TOKEN_THRESHOLD = get_int_env("BM25_FALLBACK_TOKEN_THRESHOLD", 4)

UPSTREAM_OPENAI_URL = os.getenv("UPSTREAM_OPENAI_URL", "http://vllm:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")

TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "/app/retrieval.jsonl")
VLLM_HEALTH_CHECK_TIMEOUT = get_int_env("VLLM_HEALTH_CHECK_TIMEOUT", 15)  # segundos
VLLM_CONNECT_TIMEOUT = get_int_env("VLLM_CONNECT_TIMEOUT", 20)  # segundos
VLLM_STREAM_TIMEOUT = get_int_env("VLLM_STREAM_TIMEOUT", 600)
