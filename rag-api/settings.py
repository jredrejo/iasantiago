import os

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

CTX_TOKENS_SOFT_LIMIT = int(os.getenv("CTX_TOKENS_SOFT_LIMIT", "6000"))
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "3"))
HYBRID_DENSE_K = int(os.getenv("HYBRID_DENSE_K", "40"))
HYBRID_BM25_K = int(os.getenv("HYBRID_BM25_K", "40"))
FINAL_TOPK = int(os.getenv("FINAL_TOPK", "12"))
BM25_FALLBACK_TOKEN_THRESHOLD = int(os.getenv("BM25_FALLBACK_TOKEN_THRESHOLD", "4"))

UPSTREAM_OPENAI_URL = os.getenv("UPSTREAM_OPENAI_URL", "http://vllm:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")

TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "/app/retrieval.jsonl")
