from typing import List, Dict, Tuple
import tiktoken
from sentence_transformers import SentenceTransformer
from settings import *
from qdrant_utils import search_dense
from bm25_utils import bm25_search
from rerank import CrossEncoderReranker
import os, json, time
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")

_embedder_cache = {}


def get_embedder(topic: str):
    name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
    if name not in _embedder_cache:
        _embedder_cache[name] = SentenceTransformer(
            name,
            trust_remote_code=True,
            device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
        )
    return _embedder_cache[name]


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(RERANK_MODEL)
    return _reranker


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def soft_trim_context(chunks: List[Dict], token_limit: int) -> List[Dict]:
    total = 0
    out = []
    for c in chunks:
        t = count_tokens(c["text"])
        if total + t > token_limit:
            break
        total += t
        out.append(c)
    return out


def hybrid_retrieve(topic: str, query: str) -> Tuple[List[Dict], Dict]:
    embedder = get_embedder(topic)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    dense_hits = search_dense(topic, q_vec, HYBRID_DENSE_K)
    dense = [
        {
            "file_path": h.payload["file_path"],
            "page": h.payload["page"],
            "chunk_id": h.payload["chunk_id"],
            "text": h.payload["text"],
            "score_dense": float(h.score),
            "score_bm25": 0.0,
        }
        for h in dense_hits
    ]

    bm25 = bm25_search(BM25_BASE_DIR, topic, query, HYBRID_BM25_K)

    def norm(scores):
        if not scores:
            return []
        mi, ma = min(scores), max(scores)
        if ma - mi < 1e-6:
            return [1.0] * len(scores)
        return [(s - mi) / (ma - mi) for s in scores]

    key = lambda d: (d["file_path"], d["chunk_id"])
    dense_map = {key(d): d for d in dense}
    for b in bm25:
        k = (b["file_path"], b["chunk_id"])
        if k in dense_map:
            dense_map[k]["score_bm25"] = b["score"]
        else:
            dense_map[k] = {
                "file_path": b["file_path"],
                "page": b["page"],
                "chunk_id": b["chunk_id"],
                "text": b["text"],
                "score_dense": 0.0,
                "score_bm25": b["score"],
            }
    merged = list(dense_map.values())
    nd = norm([m["score_dense"] for m in merged])
    nb = norm([m["score_bm25"] for m in merged])
    for m, a, b in zip(merged, nd, nb):
        m["score_hybrid"] = 0.6 * a + 0.4 * b

    merged.sort(key=lambda x: x["score_hybrid"], reverse=True)
    file_counts = {}
    filtered = []
    for m in merged:
        cnt = file_counts.get(m["file_path"], 0)
        if cnt < MAX_CHUNKS_PER_FILE:
            filtered.append(m)
            file_counts[m["file_path"]] = cnt + 1
        if len(filtered) >= FINAL_TOPK:
            break
    return filtered, {
        "dense_k": HYBRID_DENSE_K,
        "bm25_k": HYBRID_BM25_K,
        "final_topk": FINAL_TOPK,
    }


def bm25_only(topic: str, query: str):
    hits = bm25_search(BM25_BASE_DIR, topic, query, FINAL_TOPK * 3)
    file_counts, filtered = {}, []
    for h in hits:
        c = file_counts.get(h["file_path"], 0)
        if c < MAX_CHUNKS_PER_FILE:
            filtered.append(h)
            file_counts[h["file_path"]] = c + 1
        if len(filtered) >= FINAL_TOPK:
            break
    return filtered


def attach_citations(chunks: List[Dict], topic: str = "") -> Tuple[str, List[Dict]]:
    """
    Construye el contexto con citas que incluyen enlaces a /docs/TOPIC/

    Ejemplo:
      Input: /topics/Programming/documento.pdf
      Output: /docs/Programming/documento.pdf#page=5
    """
    if not chunks:
        return "No se encontró información relevante en la base de datos.", []

    logger.info(
        f"attach_citations: Processing {len(chunks)} chunks with topic='{topic}'"
    )

    context = []
    for i, c in enumerate(chunks):
        filename = os.path.basename(c["file_path"])
        page = c["page"]
        text = c["text"]

        # URL-encode el nombre del archivo (maneja espacios y caracteres especiales)
        encoded_filename = quote(filename, safe=".")

        # Generar URL a /docs/TOPIC/ con ancla de página
        if topic:
            doc_url = f"/docs/{topic}/{encoded_filename}#page={page}"
        else:
            doc_url = f"/docs/{encoded_filename}#page={page}"

        # Formato Markdown: texto + cita con enlace
        citation = f"{text}\n— según [{filename}, p.{page}]({doc_url})"
        context.append(citation)

        logger.info(f"  [{i}] Added citation: {filename}, p.{page}")
        logger.debug(f"       URL: {doc_url}")

    result = "\n\n".join(context)
    logger.info(f"attach_citations: Final context length: {len(result)} chars")

    return result, chunks


def telemetry_log(entry: Dict):
    ts = int(time.time() * 1000)
    entry["ts"] = ts
    try:
        with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def choose_retrieval(topic: str, query: str):
    q_tokens = len(query.strip().split())
    if q_tokens < BM25_FALLBACK_TOKEN_THRESHOLD:
        results = bm25_only(topic, query)
        return results, {"mode": "bm25"}
    else:
        results, meta = hybrid_retrieve(topic, query)
        meta["mode"] = "hybrid"
        return results, meta


def rerank_passages(query: str, passages: List[Dict]) -> List[Dict]:
    if not passages:
        return []

    if len(passages) == 1:
        return passages

    reranker = get_reranker()
    order = reranker.rerank(
        query, [p["text"] for p in passages], topk=min(FINAL_TOPK, len(passages))
    )
    return [passages[i] for i in order]
