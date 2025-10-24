import os, json, asyncio, httpx, time, logging
from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from settings import *
from retrieval import (
    choose_retrieval,
    attach_citations,
    soft_trim_context,
    rerank_passages,
    telemetry_log,
)
from token_utils import extract_topic_from_model_name
from eval import aggregate_eval

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_models_loaded():
    """Intenta cargar modelos al startup"""
    logger.info("Checking if embedding models are available...")

    try:
        from settings import EMBED_PER_TOPIC, EMBED_DEFAULT, RERANK_MODEL
        from retrieval import get_embedder, get_reranker

        # Pre-load embedders
        for topic in EMBED_PER_TOPIC.keys():
            logger.info(f"Pre-loading embedder for {topic}...")
            embedder = get_embedder(topic)
            logger.info(f"✓ Embedder for {topic} loaded")

        # Pre-load reranker
        logger.info("Pre-loading reranker...")
        reranker = get_reranker()
        logger.info("✓ Reranker loaded")

    except Exception as e:
        logger.error(f"Error loading models at startup: {e}", exc_info=True)
        raise


app = FastAPI(title="IASantiago RAG API")
ensure_models_loaded()


@app.get("/healthz")
async def healthz():
    return {"ok": True, "topics": TOPIC_LABELS}


# ---------- OpenAI compat: /v1/models ----------
@app.get("/v1/models")
async def list_models(request: Request):
    models = [
        {
            "id": f"topic:{t}",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "iasantiago",
        }
        for t in TOPIC_LABELS
    ]
    return {"object": "list", "data": models}


# Payloads OpenAI-like
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = True
    iasantiago_mode: Optional[str] = "explica"  # "explica" | "guia" | "examen"


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest, request: Request, x_email: str = Header(None)
):
    # x_email viene desde oauth2-proxy (nginx)
    logger.info(f"Usuario: {x_email}")

    topic = extract_topic_from_model_name(req.model, TOPIC_LABELS[0])
    user_msg = next((m.content for m in req.messages[::-1] if m.role == "user"), "")
    sys_prompt = open(
        "/app/templates/system_prompts/default.txt", "r", encoding="utf-8"
    ).read()

    # Retrieval
    retrieved, meta = choose_retrieval(topic, user_msg)
    logger.info(f"Retrieved {len(retrieved)} chunks for topic '{topic}'")

    # Rerank (only if we have results)
    if retrieved:
        retrieved = rerank_passages(user_msg, retrieved)
        # Límite dinámico de tokens en contexto
        retrieved = soft_trim_context(retrieved, CTX_TOKENS_SOFT_LIMIT)

    # Contexto y citas embebidas (incluye topic en URLs)
    context_text, cited = attach_citations(retrieved, topic)

    # Prompt final - contexto RAG en el user message, system prompt separado
    prompt = f"[Contexto RAG]\n{context_text}\n\n[Pregunta]\n{user_msg}"

    # Telemetría
    telemetry_log(
        {
            "query": user_msg,
            "topic": topic,
            "mode": meta.get("mode"),
            "dense_k": meta.get("dense_k"),
            "bm25_k": meta.get("bm25_k"),
            "final_topk": meta.get("final_topk"),
            "retrieved": [
                {
                    "file_path": r["file_path"],
                    "page": r["page"],
                    "chunk_id": r["chunk_id"],
                }
                for r in retrieved
            ],
        }
    )

    # Llamada a vLLM (OpenAI compat)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": os.getenv("VLLM_MODEL", ""),
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stream": req.stream,
    }

    async def stream_generator():
        """Generator that forwards SSE stream from vLLM to Open WebUI"""
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{UPSTREAM_OPENAI_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as r:
                    r.raise_for_status()
                    # Forward raw bytes to preserve SSE formatting
                    async for chunk in r.aiter_bytes():
                        yield chunk

            except httpx.HTTPStatusError as e:
                error_data = {
                    "error": {
                        "message": f"vLLM error: {str(e)}",
                        "type": "upstream_error",
                        "code": e.response.status_code,
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
            except Exception as e:
                error_data = {
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "internal_error",
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()

    if req.stream:
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        # Non-streaming response
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            try:
                resp = await client.post(
                    f"{UPSTREAM_OPENAI_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                return Response(
                    content=resp.content,
                    media_type=resp.headers.get("Content-Type", "application/json"),
                )
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"vLLM error: {e.response.text}",
                )


# ---------- Offline eval ----------
class EvalCase(BaseModel):
    query: str
    topic: str
    relevant_files: List[str] = Field(default_factory=list)


@app.post("/v1/eval/offline")
async def eval_offline(cases: List[EvalCase]):
    rows = []
    for c in cases:
        retrieved, meta = choose_retrieval(c.topic, c.query)
        # Attach citations with topic for proper URLs
        context_text, cited = attach_citations(retrieved, c.topic)
        rows.append(
            {
                "query": c.query,
                "topic": c.topic,
                "relevant_files": c.relevant_files,
                "retrieved": retrieved,
                "context": context_text,
            }
        )
    agg = aggregate_eval(rows)
    return {
        "aggregate": agg,
        "details": [
            {
                "query": r["query"],
                "topic": r["topic"],
                "pred_files": list(
                    dict.fromkeys([x["file_path"] for x in r["retrieved"]])
                ),
                "relevant_files": r["relevant_files"],
            }
            for r in rows
        ],
    }
