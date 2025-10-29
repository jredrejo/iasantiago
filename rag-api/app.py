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
    count_tokens,
)
from token_utils import extract_topic_from_model_name
from eval import aggregate_eval

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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


async def check_vllm_health(max_retries=3) -> bool:
    """Verifica si vLLM está disponible antes de enviar requests"""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(
                    f"{UPSTREAM_OPENAI_URL.replace('/v1', '')}/health"
                )
                if resp.status_code == 200:
                    return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"vLLM health check failed (attempt {attempt+1}/{max_retries}): {e}"
                )
                await asyncio.sleep(2**attempt)
            else:
                logger.error(f"vLLM is not responding after {max_retries} attempts")
                return False
    return False


async def call_vllm_with_retry(
    payload: dict, headers: dict, max_retries=3, timeout=300.0
):
    """Llamar a vLLM con reintentos exponenciales para requests no-streaming"""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                resp = await client.post(
                    f"{UPSTREAM_OPENAI_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"vLLM connection failed after {max_retries} attempts: {e}"
                )
                raise HTTPException(
                    status_code=503, detail=f"vLLM service unavailable: {str(e)}"
                )

            wait_time = 2**attempt  # Backoff exponencial: 1s, 2s, 4s
            logger.warning(
                f"vLLM connection failed (attempt {attempt+1}/{max_retries}), "
                f"retrying in {wait_time}s... Error: {type(e).__name__}: {e}"
            )
            await asyncio.sleep(wait_time)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"vLLM HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"vLLM error: {e.response.text}",
            )


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest, request: Request, x_email: str = Header(None)
):
    logger.info(f"👤 Usuario: {x_email}")
    logger.info(f"📨 Mensajes recibidos: {len(req.messages)}")

    topic = extract_topic_from_model_name(req.model, TOPIC_LABELS[0])

    # Obtener último mensaje del usuario para retrieval
    user_msg = next((m.content for m in req.messages[::-1] if m.role == "user"), "")

    # System prompt
    sys_prompt = open(
        "/app/templates/system_prompts/default.txt", "r", encoding="utf-8"
    ).read()

    # Retrieval
    retrieved, meta = choose_retrieval(topic, user_msg)
    logger.info(f"📚 Retrieved {len(retrieved)} chunks for topic '{topic}'")

    if retrieved:
        retrieved = rerank_passages(user_msg, retrieved)
        retrieved = soft_trim_context(retrieved, CTX_TOKENS_SOFT_LIMIT)

    context_text, cited = attach_citations(retrieved, topic)

    # Telemetría
    telemetry_log(
        {
            "query": user_msg,
            "topic": topic,
            "mode": meta.get("mode"),
            "num_messages": len(req.messages),
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

    # ============================================================
    # CONSTRUCCIÓN DE MENSAJES CON HISTORIAL COMPLETO
    # ============================================================

    messages = []

    # 1. System prompt enriquecido con contexto RAG
    if (
        context_text
        and context_text != "No se encontró información relevante en la base de datos."
    ):
        enhanced_system = f"""{sys_prompt}

[Contexto RAG - Información relevante de la base de datos]
{context_text}

Usa este contexto para responder las preguntas del usuario. Siempre cita las fuentes usando los enlaces proporcionados."""
    else:
        enhanced_system = sys_prompt

    messages.append({"role": "system", "content": enhanced_system})

    # 2. TODO el historial de conversación del usuario
    for msg in req.messages:
        # Filtrar system prompts que venga del cliente (Open WebUI)
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    # ============================================================
    # ANÁLISIS DE TOKENS Y WARNINGS
    # ============================================================
    system_tokens = count_tokens(enhanced_system)
    context_tokens = count_tokens(context_text)
    history_tokens = sum(
        count_tokens(m["content"]) for m in messages if m["role"] != "system"
    )
    total_input_tokens = system_tokens + history_tokens

    logger.info(f"📊 Token breakdown:")
    logger.info(f"   - System prompt: ~{system_tokens} tokens")
    logger.info(f"   - RAG context: ~{context_tokens} tokens")
    logger.info(f"   - Conversation history: ~{history_tokens} tokens")
    logger.info(f"   - TOTAL INPUT: ~{total_input_tokens} tokens")

    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    if total_input_tokens > max_model_len * 0.7:
        logger.warning(
            f"⚠️  Input muy largo ({total_input_tokens} tokens, "
            f"{(total_input_tokens/max_model_len)*100:.1f}% del límite), "
            f"podría causar OOM o respuestas truncadas"
        )

    logger.info(
        f"🚀 Enviando a vLLM: {len(messages)} mensajes "
        f"(system + {len(req.messages)} historial)"
    )

    # Verificar salud de vLLM antes de enviar
    if not await check_vllm_health():
        raise HTTPException(
            status_code=503,
            detail="vLLM service is not responding. Please try again in a few moments.",
        )

    # Payload para vLLM
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "messages": messages,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stream": req.stream,
        "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "8192")),
    }

    # Log del tamaño del payload
    payload_size = len(json.dumps(payload))
    logger.info(f"📦 Payload size: {payload_size:,} bytes ({payload_size/1024:.1f} KB)")

    async def stream_generator():
        """Generator que reenvía el stream SSE de vLLM con reintentos"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    async with client.stream(
                        "POST",
                        f"{UPSTREAM_OPENAI_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as r:
                        r.raise_for_status()
                        logger.info(
                            f"✓ Stream establecido con vLLM (attempt {attempt+1})"
                        )

                        async for chunk in r.aiter_bytes():
                            yield chunk

                        logger.info("✓ Stream completado exitosamente")
                        return  # Éxito, salir

            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"❌ Stream falló después de {max_retries} intentos: {e}"
                    )
                    error_data = {
                        "error": {
                            "message": f"vLLM connection failed after {max_retries} retries: {str(e)}",
                            "type": "connection_error",
                            "code": 503,
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n".encode()
                    return

                wait_time = 2**attempt
                logger.warning(
                    f"⚠️  Stream interrupted (attempt {attempt+1}/{max_retries}), "
                    f"retrying in {wait_time}s... Error: {type(e).__name__}"
                )
                await asyncio.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"❌ vLLM HTTP error: {e.response.status_code} - {e.response.text}"
                )
                error_data = {
                    "error": {
                        "message": f"vLLM error: {str(e)}",
                        "type": "upstream_error",
                        "code": e.response.status_code,
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                return

            except Exception as e:
                logger.error(f"❌ Streaming error inesperado: {str(e)}", exc_info=True)
                error_data = {
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "internal_error",
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                return

    if req.stream:
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response con reintentos
        resp = await call_vllm_with_retry(payload, headers)
        return Response(
            content=resp.content,
            media_type=resp.headers.get("Content-Type", "application/json"),
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
