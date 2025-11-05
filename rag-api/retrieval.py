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


def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks (same file_path + chunk_id)"""
    seen = set()
    dedup = []
    for c in chunks:
        key = (c["file_path"], c["chunk_id"])
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    logger.info(f"Deduplication: {len(chunks)} → {len(dedup)} chunks")
    return dedup


def get_embedder(topic: str):
    name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
    if name not in _embedder_cache:
        try:
            logger.info(f"Loading embedder: {name}")
            embedder = SentenceTransformer(
                name,
                trust_remote_code=True,
                device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
            )
            _embedder_cache[name] = embedder
            logger.info(f"✓ Embedder {name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedder {name}: {e}", exc_info=True)
            raise RuntimeError(f"Cannot load embedding model {name}: {e}")
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

    # ✅ DEDUPLICAR AQUÍ
    merged = deduplicate_chunks(merged)

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
    Versión mejorada: contexto RAG con citas OBLIGATORIAS y fáciles de copiar
    """
    if not chunks:
        return "No se encontró información relevante.", []

    context_parts = []

    for i, c in enumerate(chunks, start=1):
        filename = os.path.basename(c["file_path"])
        page = c["page"]
        text = c["text"]
        encoded_filename = quote(filename, safe=".")

        if topic:
            doc_url = f"/docs/{topic}/{encoded_filename}#page={page}"
        else:
            doc_url = f"/docs/{encoded_filename}#page={page}"

        # FORMATO MEJORADO: Markdown inline citation
        chunk_with_citation = f"""{text}

**Cita:** [{filename}, p.{page}]({doc_url})"""

        context_parts.append(chunk_with_citation)
        logger.info(f"[{i}] {filename}, p.{page}")

    # Separador MUY claro
    context_body = (
        "\n\n" + "─" * 70 + "\n\n".join(["FRAGMENTO DE CONTEXTO RAG:"] + context_parts)
    )

    # Instrucciones REFORZADAS para el LLM
    instructions = f"""

{"─" * 70}
⚠️  INSTRUCCIONES DE CITACIÓN (CRÍTICO)
{"─" * 70}

REGLA 1: SIEMPRE cita las fuentes del contexto anterior.
REGLA 2: Las citas DEBEN estar en formato markdown: [archivo.pdf, p.N](/ruta/completa)
REGLA 3: COPIA EXACTAMENTE el formato de las citas del contexto.
REGLA 4: NO INVENTES información que no esté en el contexto.
REGLA 5: Si NO está en el contexto, di: "No encontré información sobre esto"

FORMATO CORRECTO:
"...información relevante [archivo.pdf, p.13](/docs/TOPIC/archivo.pdf#page=13)"

FORMATOS INCORRECTOS:
❌ "...información [FUENTE 1]" ← falta URL
❌ "...información [archivo.pdf]" ← falta página y URL
❌ "...información" ← falta cita completamente
❌ "Según el archivo:" ← vago, sin número de página

{"─" * 70}
"""

    result = context_body + instructions
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


def attach_citations_explicit(
    chunks: List[Dict], topic: str = ""
) -> Tuple[str, List[Dict]]:
    """
    Formatea contexto RAG de forma MÁS EXPLÍCITA para que el modelo lo vea claro.
    """
    if not chunks:
        empty_msg = "No se encontró información relevante en la base de datos."
        logger.warning(f"attach_citations_explicit: {empty_msg}")
        return empty_msg, []

    logger.info(
        f"attach_citations_explicit: Processing {len(chunks)} chunks with topic='{topic}'"
    )

    context_parts = []

    for i, c in enumerate(chunks, start=1):
        filename = os.path.basename(c["file_path"])
        page = c["page"]
        text = c["text"]

        encoded_filename = quote(filename, safe=".")

        if topic:
            doc_url = f"/docs/{topic}/{encoded_filename}#page={page}"
        else:
            doc_url = f"/docs/{encoded_filename}#page={page}"

        fragment = f"""[{i}] {text}

**Fuente:** [{filename}, p.{page}]({doc_url})"""

        context_parts.append(fragment)
        logger.info(f"  [{i}] {filename}, p.{page}")

    result = f"""
╔════════════════════════════════════════════════════════════════╗
║            CONTEXTO RAG - INFORMACIÓN DE DOCUMENTOS             ║
╚════════════════════════════════════════════════════════════════╝

{chr(10).join([""] + context_parts)}

╔════════════════════════════════════════════════════════════════╗
║                    FIN DE CONTEXTO RAG                          ║
╚════════════════════════════════════════════════════════════════╝

INSTRUCCIONES CRÍTICAS:
- DEBES usar la información anterior para responder
- Si la respuesta está en los fragmentos [1] a [{len(chunks)}], úsala y cita correctamente
- Si NO está en los fragmentos, responde: "No encontré información sobre esto en los documentos"
- NUNCA inventes información fuera de estos fragmentos
- Las citas DEBEN estar en formato: [archivo.pdf, p.N](/docs/TOPIC/archivo.pdf#page=N)
"""

    logger.info(f"attach_citations_explicit: Final context length: {len(result)} chars")
    return result, chunks


def validate_context_usage(retrieved_chunks: List[Dict], model_response: str) -> Dict:
    """Valida si el modelo usó el contexto o alucinó"""
    import re

    context_files = set(c["file_path"] for c in retrieved_chunks)
    citations = re.findall(r"\[([^]]+\.pdf),\s*p\.(\d+)\]", model_response)
    cited_files = set(filename for filename, _ in citations)

    coverage = len(cited_files & context_files) / max(len(context_files), 1)
    has_no_found = "No encontré información" in model_response

    result = {
        "context_files": list(context_files),
        "cited_files": list(cited_files),
        "coverage": round(coverage, 2),
        "said_not_found": has_no_found,
        "citation_count": len(citations),
    }

    logger.info(f"📊 Context validation: {result}")
    return result


# En retrieval.py, modifica hybrid_retrieve para añadir logging detallado

def hybrid_retrieve_debug(topic: str, query: str) -> Tuple[List[Dict], Dict]:
    logger.info("=" * 80)
    logger.info(f"🔍 HYBRID RETRIEVAL DEBUG")
    logger.info("=" * 80)
    logger.info(f"Query: {query}\n")
    
    embedder = get_embedder(topic)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    
    logger.info("📌 DENSE SEARCH (Qdrant):")
    dense_hits = search_dense(topic, q_vec, HYBRID_DENSE_K)
    
    logger.info(f"  Retornó {len(dense_hits)} hits")
    for i, h in enumerate(dense_hits[:5], 1):  # Top 5
        filename = os.path.basename(h.payload["file_path"])
        score = h.score
        text_preview = h.payload["text"][:100].replace("\n", " ")
        logger.info(f"  [{i}] {filename}, p.{h.payload['page']}, score={score:.4f}")
        logger.info(f"       Text: {text_preview}...")
    
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

    logger.info("\n📌 BM25 SEARCH (Whoosh):")
    bm25 = bm25_search(BM25_BASE_DIR, topic, query, HYBRID_BM25_K)
    
    logger.info(f"  Retornó {len(bm25)} hits")
    for i, b in enumerate(bm25[:5], 1):  # Top 5
        filename = os.path.basename(b["file_path"])
        score = b["score"]
        text_preview = b["text"][:100].replace("\n", " ")
        logger.info(f"  [{i}] {filename}, p.{b['page']}, score={score:.4f}")
        logger.info(f"       Text: {text_preview}...")

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
    
    logger.info("\n📌 DESPUÉS DE MERGE Y NORMALIZACIÓN (Top 10):")
    for i, m in enumerate(merged[:10], 1):
        filename = os.path.basename(m["file_path"])
        logger.info(f"  [{i}] {filename}, p.{m['page']}")
        logger.info(f"       Dense: {m['score_dense']:.4f}, BM25: {m['score_bm25']:.4f}, Hybrid: {m['score_hybrid']:.4f}")
        logger.info(f"       Text: {m['text'][:80].replace(chr(10), ' ')}...")
    
    merged = deduplicate_chunks(merged)
    
    file_counts = {}
    filtered = []
    for m in merged:
        cnt = file_counts.get(m["file_path"], 0)
        if cnt < MAX_CHUNKS_PER_FILE:
            filtered.append(m)
            file_counts[m["file_path"]] = cnt + 1
        if len(filtered) >= FINAL_TOPK:
            break
    
    logger.info(f"\n✅ FINAL FILTERED (después MAX_CHUNKS_PER_FILE={MAX_CHUNKS_PER_FILE}, FINAL_TOPK={FINAL_TOPK}):")
    for i, f in enumerate(filtered, 1):
        filename = os.path.basename(f["file_path"])
        logger.info(f"  [{i}] {filename}, p.{f['page']}, hybrid_score={f['score_hybrid']:.4f}")
    
    logger.info("=" * 80)
    
    return filtered, {
        "dense_k": HYBRID_DENSE_K,
        "bm25_k": HYBRID_BM25_K,
        "final_topk": FINAL_TOPK,
    }


# En app.py, reemplaza:
# retrieved, meta = choose_retrieval(topic, user_msg)
# CON:
# retrieved, meta = hybrid_retrieve_debug(topic, user_msg)  # Temporal para debug
