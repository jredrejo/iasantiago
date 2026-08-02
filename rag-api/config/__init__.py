# Archivo: rag-api/config/__init__.py
# Descripción: Módulo de configuración

from config.settings import *

__all__ = [
    # Topics
    "TOPIC_LABELS",
    "TOPIC_BASE_DIR",
    # Embeddings
    "EMBED_PER_TOPIC",
    "EMBED_DEFAULT",
    "RERANK_MODEL",
    # Almacenamiento
    "QDRANT_URL",
    "QDRANT_COLLECTION_SUFFIX",
    "SPARSE_VECTOR_NAME",
    # Límites de contexto
    "CTX_TOKENS_SOFT_LIMIT",
    "CTX_TOKENS_GENERATIVE",
    "MAX_CHUNKS_PER_FILE",
    "MAX_CHUNKS_PER_FILE_GENERATIVE",
    # Búsqueda híbrida
    "HYBRID_DENSE_K",
    "HYBRID_BM25_K",
    "FINAL_TOPK",
    "BM25_FALLBACK_TOKEN_THRESHOLD",
    "GENERATIVE_TOPK_MULTIPLIER",
    # vLLM (lo que queda tras el rip-out del §7.1: clave, tokenizador, ventana)
    "OPENAI_API_KEY",
    "VLLM_MODEL",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_MAX_TOKENS",
    # Telemetría
    "TELEMETRY_PATH",
    # Helper
    "get_int_env",
]
