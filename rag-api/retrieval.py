from typing import List, Dict, Tuple
import tiktoken
from sentence_transformers import SentenceTransformer
from settings import *
from qdrant_utils import search_dense
from bm25_utils import bm25_search, bm25_search_safe
from rerank import CrossEncoderReranker
import os, json, time
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")
_embedder_cache = {}
_reranker = None


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
    """Versión básica con valores por defecto de .env"""
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

    bm25 = bm25_search_safe(BM25_BASE_DIR, topic, query, HYBRID_BM25_K)

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
    merged = deduplicate_chunks(merged)

    file_counts = {}
    filtered = []
    for m in merged:
        cnt = file_counts.get(m["file_path"], 0)
        if MAX_CHUNKS_PER_FILE == 0 or cnt < MAX_CHUNKS_PER_FILE:
            filtered.append(m)
            file_counts[m["file_path"]] = cnt + 1
        if len(filtered) >= FINAL_TOPK:
            break
    return filtered, {
        "dense_k": HYBRID_DENSE_K,
        "bm25_k": HYBRID_BM25_K,
        "final_topk": FINAL_TOPK,
    }


def hybrid_retrieve_enhanced(topic: str, query: str, final_topk: int):
    """Versión con topk configurable - SIN HARDCODEO"""
    embedder = get_embedder(topic)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    # Recuperar MÁS en búsqueda inicial (2x el base de .env)
    dense_k = HYBRID_DENSE_K * 2
    bm25_k = HYBRID_BM25_K * 2

    logger.info(f"[HYBRID] Dense K={dense_k}, BM25 K={bm25_k}, Final topk={final_topk}")

    dense_hits = search_dense(topic, q_vec, dense_k)
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

    bm25 = bm25_search_safe(BM25_BASE_DIR, topic, query, bm25_k)

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
    merged = deduplicate_chunks(merged)

    # Filtrar por archivo usando MAX_CHUNKS_PER_FILE de .env
    file_counts = {}
    filtered = []
    for m in merged:
        cnt = file_counts.get(m["file_path"], 0)
        if MAX_CHUNKS_PER_FILE == 0 or cnt < MAX_CHUNKS_PER_FILE:
            filtered.append(m)
            file_counts[m["file_path"]] = cnt + 1
        if len(filtered) >= final_topk:
            break

    logger.info(
        f"[HYBRID] Final: {len(filtered)} chunks de "
        f"{len(set(r['file_path'] for r in filtered))} archivos"
    )

    return filtered, {
        "dense_k": dense_k,
        "bm25_k": bm25_k,
        "final_topk": final_topk,
    }


def bm25_only(topic: str, query: str):
    """Versión básica con valores por defecto de .env"""
    hits = bm25_search_safe(BM25_BASE_DIR, topic, query, FINAL_TOPK * 3)
    file_counts, filtered = {}, []
    for h in hits:
        c = file_counts.get(h["file_path"], 0)
        if MAX_CHUNKS_PER_FILE == 0 or c < MAX_CHUNKS_PER_FILE:
            filtered.append(h)
            file_counts[h["file_path"]] = c + 1
        if len(filtered) >= FINAL_TOPK:
            break
    return filtered


def bm25_only_enhanced(topic: str, query: str, final_topk: int):
    """Versión con topk configurable - SIN HARDCODEO"""
    # Buscar 3x más de lo necesario para tener margen
    hits = bm25_search_safe(BM25_BASE_DIR, topic, query, final_topk * 3)

    file_counts, filtered = {}, []
    for h in hits:
        c = file_counts.get(h["file_path"], 0)
        if MAX_CHUNKS_PER_FILE == 0 or c < MAX_CHUNKS_PER_FILE:
            filtered.append(h)
            file_counts[h["file_path"]] = c + 1
        if len(filtered) >= final_topk:
            break

    logger.info(
        f"[BM25] Final: {len(filtered)} chunks de "
        f"{len(set(r['file_path'] for r in filtered))} archivos"
    )

    return filtered


def choose_retrieval(topic: str, query: str):
    """Versión básica para backward compatibility"""
    q_tokens = len(query.strip().split())
    if q_tokens < BM25_FALLBACK_TOKEN_THRESHOLD:
        results = bm25_only(topic, query)
        return results, {"mode": "bm25"}
    else:
        results, meta = hybrid_retrieve(topic, query)
        meta["mode"] = "hybrid"
        return results, meta


def choose_retrieval_enhanced(topic: str, query: str, is_generative: bool = False):
    """
    Versión mejorada que ajusta parámetros según el modo
    USANDO VARIABLES DE ENTORNO (sin hardcodeo)
    """
    q_tokens = len(query.strip().split())

    # Ajustar TOPK usando multiplicador de .env
    if is_generative:
        # Para generación, multiplicar el topk base
        final_topk = FINAL_TOPK * GENERATIVE_TOPK_MULTIPLIER
        logger.info(
            f"🎯 Modo GENERATIVO: recuperando {final_topk} chunks "
            f"(base={FINAL_TOPK}, multiplicador={GENERATIVE_TOPK_MULTIPLIER})"
        )
    else:
        # Para respuestas, usar el topk normal
        final_topk = FINAL_TOPK
        logger.info(f"💬 Modo RESPUESTA: recuperando {final_topk} chunks")

    # Decidir estrategia: BM25 solo vs Hybrid
    if q_tokens < BM25_FALLBACK_TOKEN_THRESHOLD:
        logger.info(f"Query corta ({q_tokens} tokens) - usando BM25 solo")
        results = bm25_only_enhanced(topic, query, final_topk)
        return results, {"mode": "bm25", "topk": final_topk}
    else:
        logger.info(f"Query normal ({q_tokens} tokens) - usando Hybrid")
        results, meta = hybrid_retrieve_enhanced(topic, query, final_topk)
        meta["mode"] = "hybrid"
        meta["topk"] = final_topk
        return results, meta


def rerank_passages(query: str, passages: List[Dict]) -> List[Dict]:
    """Reordena passages usando el reranker"""
    if not passages:
        return []

    if len(passages) == 1:
        return passages

    reranker = get_reranker()
    order = reranker.rerank(
        query, [p["text"] for p in passages], topk=min(FINAL_TOPK, len(passages))
    )
    return [passages[i] for i in order]


def attach_citations(chunks: List[Dict], topic: str = "") -> Tuple[str, List[Dict]]:
    """
    Versión mejorada: contexto RAG con citas OBLIGATORIAS y fáciles de copiar
    Las citas incluyen URLs COMPLETAS para que el modelo las copie exactamente
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

        # Solo el texto del documento, sin referencias internas
        chunk_with_citation = f"""{text}

FUENTE:
[{filename}, p.{page}]({doc_url})"""

        context_parts.append(chunk_with_citation)
        logger.info(f"[{i}] {filename}, p.{page} → {doc_url}")

    # Separador MUY claro
    separator = "=" * 70
    context_body = "\n\n" + separator + "\n"
    context_body += "CONTEXTO RAG - INFORMACIÓN DE DOCUMENTOS\n"
    context_body += separator + "\n\n"
    context_body += "\n\n".join(context_parts)
    context_body += "\n\n" + separator

    # Instrucciones claras para el LLM - SIN referencias internas
    instructions = f"""

{separator}
INSTRUCCIONES PARA RESPUESTAS CON FUENTES
{separator}

Usa la información de los documentos anteriores para responder:

1. Cada sección del texto incluye su fuente al final en este formato:
   [archivo.pdf, p.N](/docs/TOPIC/archivo.pdf#page=N)

2. Cuando uses información de un documento, cita la fuente EXACTAMENTE como aparece:
   Fuente: [archivo.pdf, p.N](/docs/TOPIC/archivo.pdf#page=N)

3. 🔗 IMPORTANTE: Las citas deben ser enlaces clicables en formato markdown.
   NO escribas las URLs como texto plano.

4. NUNCA menciones "fragmento", "sección", "parte" o referencias internas.
   Solo habla del contenido y cita el archivo y página.

5. Si usas información del documento Manual.pdf página 42, termina con:
   Fuente: [Manual.pdf, p.42](/docs/Chemistry/Manual.pdf#page=42)

EJEMPLO DE RESPUESTA:
"La ley de Ohm establece que V = I × R, donde V es el voltaje, I es la corriente y R es la resistencia.
Esta relación fundamental permite calcular cualquier parámetro eléctrico si se conocen los otros dos.
Fuente: [Electrónica_Básica.pdf, p.15](/docs/Electronics/Electrónica_Básica.pdf#page=15)"

FORMATO CORRECTO DE CITAS:
✅ Fuente: [archivo.pdf, p.25](/docs/TOPIC/archivo.pdf#page=25)  ← Enlace clicable
❌ Fuente: [archivo.pdf, p.25]                             ← No es enlace
❌ Fuente: archivo.pdf, p.25                             ← No es enlace

REGLAS IMPORTANTES:
- NO inventes información que no esté en los documentos
- NO uses referencias internas como "según el documento 3" o "en el fragmento"
- Sé natural y conversacional, solo citando las fuentes
- SOLO permite markdown para: bloques de código (```), fórmulas LaTeX ($formula$) y citas [enlaces](/docs/...)
- NO uses negritas (**), títulos (##) o listas (-)

{separator}
"""

    result = context_body + instructions
    return result, chunks


def telemetry_log(entry: Dict):
    """Registra telemetría en archivo JSONL"""
    ts = int(time.time() * 1000)
    entry["ts"] = ts
    try:
        with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


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


def debug_retrieval(topic: str, query: str) -> dict:
    """
    Función de debugging completa - muestra TODO lo que pasa en el retrieval
    Útil para diagnosticar por qué solo trae de un archivo
    """
    import os
    from qdrant_utils import get_collection_stats

    logger.info("\n" + "=" * 80)
    logger.info("🔍 DEBUG RETRIEVAL - ANÁLISIS COMPLETO")
    logger.info("=" * 80)

    # 1. Verificar colección
    logger.info("\n📦 Colección Qdrant:")
    stats = get_collection_stats(topic)
    if stats:
        logger.info(f"   - Nombre: {stats['collection']}")
        logger.info(f"   - Total puntos: {stats['points_count']}")
        logger.info(f"   - Tamaño vector: {stats['vector_size']}")
        logger.info(f"   - Distancia: {stats['distance']}")
    else:
        logger.error("   ❌ No se pudo obtener stats - ¿colección existe?")

    # 2. Obtener embeddings del query
    logger.info("\n🔎 Query:")
    logger.info(f"   - Texto: {query[:100]}...")
    logger.info(f"   - Tokens: ~{count_tokens(query)}")

    embedder = get_embedder(topic)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    logger.info(f"   - Embedding model: {type(embedder).__name__}")
    logger.info(f"   - Vector shape: {len(q_vec)}")
    logger.info(f"   - Vector norm: {(q_vec**2).sum() ** 0.5:.4f} (should be ~1.0)")

    # 3. Búsqueda densa
    logger.info("\n🔎 Búsqueda Densa (Dense Vector Search):")
    logger.info(f"   - K: {HYBRID_DENSE_K}")
    q_vec_list = q_vec.tolist()
    dense_hits = search_dense(topic, q_vec_list, HYBRID_DENSE_K)
    logger.info(f"   - Resultados: {len(dense_hits)}")

    if dense_hits:
        # Analizar diversidad de archivos
        files = {}
        for i, h in enumerate(dense_hits):
            file_path = h.payload["file_path"]
            if file_path not in files:
                files[file_path] = []
            files[file_path].append((i, h.score, h.id))

        logger.info(f"   - Archivos únicos: {len(files)}")
        for file_path, hits_list in sorted(files.items(), key=lambda x: -len(x[1]))[:5]:
            filename = os.path.basename(file_path)
            logger.info(f"       • {filename}: {len(hits_list)} hits")
            for idx, score, point_id in hits_list[:2]:
                logger.info(f"         - score={score:.4f}, id={point_id}")
    else:
        logger.error("   ❌ NO HAY RESULTADOS - PROBLEMA GRAVE")

    # 4. Búsqueda BM25
    logger.info("\n📚 Búsqueda BM25 (Keyword Search):")
    logger.info(f"   - K: {HYBRID_BM25_K}")
    bm25_hits = bm25_search(BM25_BASE_DIR, topic, query, HYBRID_BM25_K)
    logger.info(f"   - Resultados: {len(bm25_hits)}")

    if bm25_hits:
        files_bm25 = {}
        for h in bm25_hits:
            file_path = h["file_path"]
            if file_path not in files_bm25:
                files_bm25[file_path] = 0
            files_bm25[file_path] += 1

        logger.info(f"   - Archivos únicos: {len(files_bm25)}")
        for file_path, count in sorted(files_bm25.items(), key=lambda x: -x[1])[:5]:
            filename = os.path.basename(file_path)
            logger.info(f"       • {filename}: {count} hits")

    # 5. Después del merge
    logger.info("\n🔀 Después del Merge (Hybrid):")

    merged, meta = hybrid_retrieve(topic, query)

    files_merged = {}
    for c in merged:
        file_path = c["file_path"]
        if file_path not in files_merged:
            files_merged[file_path] = 0
        files_merged[file_path] += 1

    logger.info(f"   - Total chunks: {len(merged)}")
    logger.info(f"   - Archivos únicos: {len(files_merged)}")
    for file_path, count in sorted(files_merged.items(), key=lambda x: -x[1]):
        filename = os.path.basename(file_path)
        chunks_info = [c for c in merged if c["file_path"] == file_path]
        avg_score = sum(c.get("score_hybrid", 0) for c in chunks_info) / len(
            chunks_info
        )
        logger.info(f"       • {filename}: {count} chunks (avg_score={avg_score:.4f})")

    logger.info("=" * 80 + "\n")

    return {
        "collection_stats": stats,
        "dense_hits": len(dense_hits),
        "bm25_hits": len(bm25_hits),
        "merged_results": len(merged),
        "unique_files": len(files_merged),
        "results": merged,
    }


# ============================================================
# CÓMO USAR EN app.py:
# ============================================================
# En app.py, en la función chat_completions(), reemplaza:
#
#     retrieved, meta = choose_retrieval(topic, user_msg)
#
# CON (temporalmente para debugging):
#
#     debug_info = debug_retrieval(topic, user_msg)
#     retrieved = debug_info["results"]
#     meta = {"mode": "debug"}
#
# Verás todos los detalles en los logs
