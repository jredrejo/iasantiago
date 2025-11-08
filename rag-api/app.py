import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from retrieval import (
    attach_citations,
    choose_retrieval,
    choose_retrieval_enhanced,
    count_tokens,
    rerank_passages,
    soft_trim_context,
    telemetry_log,
)
from settings import *
from token_utils import extract_topic_from_model_name

from eval import aggregate_eval

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# VARIABLES GLOBALES PARA TRACKING DE MODELO
# ============================================================
_current_vllm_model = None
_model_check_lock = asyncio.Lock()


def detect_generative_intent(user_message: str) -> bool:
    """
    Detecta si el usuario quiere GENERAR contenido (examen, ejercicios, etc.)
    vs. simplemente RESPONDER una pregunta con el contexto.
    """
    generative_keywords = [
        # Creación de exámenes
        r"\b(crea|elabora|genera|diseña|prepara|haz|hacer)\b.*\b(examen|test|prueba|evaluaci[oó]n)\b",
        r"\b(preguntas?)\b.*\b(sobre|de|acerca)\b",
        r"\b\d+\s*(preguntas?|ejercicios?|cuestiones?)\b",  # "10 preguntas"
        # Creación de ejercicios
        r"\b(ejercicios?|actividades?|pr[aá]cticas?)\b",
        # Creación de contenido educativo
        r"\b(resume|sintetiza|organiza)\b.*\b(en|como)\b.*\b(esquema|mapa|lista)\b",
        r"\blistado\b.*\b(de|con)\b",
        # Comandos explícitos
        r"^(crea|elabora|genera|diseña|prepara|haz)\b",
    ]

    message_lower = user_message.lower()

    for pattern in generative_keywords:
        if re.search(pattern, message_lower):
            logger.info(f"🎯 Intención GENERATIVA detectada: '{pattern}'")
            return True

    logger.info("💬 Intención de RESPUESTA detectada (default)")
    return False


def load_system_prompt(is_generative: bool) -> str:
    """Carga el prompt correcto según la intención"""
    if is_generative:
        path = "/app/templates/system_prompts/generative.txt"
        logger.info("📝 Usando prompt GENERATIVO (crear contenido)")
    else:
        path = "/app/templates/system_prompts/default.txt"
        logger.info("💬 Usando prompt DEFAULT (responder con contexto)")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"❌ Prompt no encontrado: {path}")
        with open(
            "/app/templates/system_prompts/default.txt", "r", encoding="utf-8"
        ) as f:
            return f.read()


def ensure_models_loaded():
    """Intenta cargar modelos al startup"""
    logger.info("Checking if embedding models are available...")

    try:
        from retrieval import get_embedder, get_reranker
        from settings import EMBED_DEFAULT, EMBED_PER_TOPIC, RERANK_MODEL

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
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = True
    iasantiago_mode: Optional[str] = "explica"  # "explica" | "guia" | "examen"


# ============================================================
# FUNCIONES PARA MANEJO DE MODELOS EN vLLM
# ============================================================


async def wait_for_model_ready(
    model_name: str, max_wait_seconds: int = 300, check_interval: float = 2.0
) -> bool:
    """
    Espera a que un modelo esté listo en vLLM.
    """
    vllm_url = os.getenv("UPSTREAM_OPENAI_URL", "http://vllm:8000/v1")
    vllm_base_url = vllm_url.replace("/v1", "")
    timeout = httpx.Timeout(10.0, connect=5.0)

    elapsed = 0
    attempt = 0

    logger.info(
        f"⏳ Waiting for model '{model_name}' to be ready (max {max_wait_seconds}s)..."
    )

    while elapsed < max_wait_seconds:
        attempt += 1
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    health_resp = await client.get(f"{vllm_base_url}/health")
                    if health_resp.status_code != 200:
                        logger.debug(
                            f"[{attempt}] vLLM health check failed (status {health_resp.status_code})"
                        )
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval
                        continue
                except Exception as e:
                    logger.debug(f"[{attempt}] vLLM health check error: {e}")
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval
                    continue

                try:
                    models_resp = await client.get(f"{vllm_url}/models")
                    if models_resp.status_code != 200:
                        logger.debug(
                            f"[{attempt}] Could not fetch models list (status {models_resp.status_code})"
                        )
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval
                        continue

                    models_data = models_resp.json()
                    available_models = [m["id"] for m in models_data.get("data", [])]

                    if model_name in available_models:
                        logger.info(
                            f"✅ Model '{model_name}' is READY (took {elapsed}s)"
                        )
                        return True
                    else:
                        logger.debug(
                            f"[{attempt}] Model '{model_name}' not in list yet. "
                            f"Available: {available_models}. "
                            f"Waiting... ({elapsed}s/{max_wait_seconds}s)"
                        )
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval

                except Exception as e:
                    logger.debug(f"[{attempt}] Error parsing models response: {e}")
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

        except Exception as e:
            logger.debug(f"[{attempt}] Unexpected error checking model readiness: {e}")
            await asyncio.sleep(check_interval)
            elapsed += check_interval

    logger.error(
        f"❌ Model '{model_name}' did not become ready after {max_wait_seconds}s"
    )
    return False


async def check_vllm_health(max_retries=3) -> bool:
    """Verifica si vLLM está disponible antes de enviar requests"""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(
                    f"{UPSTREAM_OPENAI_URL.replace('/v1', '')}/health",
                    timeout=httpx.Timeout(10.0),
                )
                if resp.status_code == 200:
                    return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"vLLM health check failed (attempt {attempt + 1}/{max_retries}): {e}"
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

            wait_time = 2**attempt
            logger.warning(
                f"vLLM connection failed (attempt {attempt + 1}/{max_retries}), "
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

    # ============================================================
    # DETECTAR Y ESPERAR A CAMBIO DE MODELO
    # ============================================================
    global _current_vllm_model

    requested_model = VLLM_MODEL

    async with _model_check_lock:
        if _current_vllm_model is None:
            _current_vllm_model = requested_model
            logger.info(f"🎯 Initial model set to: {_current_vllm_model}")
        elif _current_vllm_model != requested_model:
            logger.warning(
                f"⚠️  Model change detected: {_current_vllm_model} → {requested_model}"
            )
            logger.info(f"⏳ Waiting for model to be ready...")

            model_ready = await wait_for_model_ready(
                requested_model, max_wait_seconds=300
            )

            if not model_ready:
                logger.error(f"❌ Model '{requested_model}' failed to load in time")
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{requested_model}' is not ready. "
                    f"It is currently loading (changing from '{_current_vllm_model}'). "
                    f"Please try again in a moment.",
                )

            _current_vllm_model = requested_model
            logger.info(f"✅ Switched to model: {_current_vllm_model}")

    topic = extract_topic_from_model_name(req.model, TOPIC_LABELS[0])

    # Obtener último mensaje del usuario para retrieval
    user_msg = next((m.content for m in req.messages[::-1] if m.role == "user"), "")

    # System prompt
    is_generative = detect_generative_intent(user_msg)
    sys_prompt = load_system_prompt(is_generative)

    # Ajustar límites de contexto según el modo
    if is_generative:
        context_token_limit = CTX_TOKENS_GENERATIVE
        # En generativo, reservar espacio para la respuesta larga
        # Reducir contexto si es necesario para dejar espacio
        max_context_for_generation = (
            VLLM_MAX_MODEL_LEN - VLLM_MAX_TOKENS - 1000
        )  # 1000 tokens para system prompt
        context_token_limit = min(context_token_limit, max_context_for_generation)
        logger.info(
            f"🎯 Modo GENERATIVO: Límite de contexto ajustado a {context_token_limit} tokens"
        )
    else:
        context_token_limit = CTX_TOKENS_SOFT_LIMIT

    # Retrieval con parámetros ajustados
    retrieved, meta = choose_retrieval_enhanced(topic, user_msg, is_generative)

    # Retrieval
    logger.info(f"📚 Retrieved {len(retrieved)} chunks for topic '{topic}'")
    if retrieved:
        logger.info(
            f"Before rerank: {[(r['file_path'], r['page'], r['chunk_id']) for r in retrieved]}"
        )
        # ✅ RERANKER IMPROVEMENT: Rerank ALL passages, don't limit at reranker
        # Token trimming will happen after reranking for better quality
        retrieved = rerank_passages(
            user_msg, retrieved, rerank_topk=None
        )  # Return all reranked
        logger.info(
            f"After rerank: {[(r['file_path'], r['page'], r['chunk_id']) for r in retrieved]}"
        )
        # Trim by tokens AFTER reranking to maintain quality
        retrieved = soft_trim_context(retrieved, context_token_limit)

    context_text, cited = attach_citations(retrieved, topic)

    if (
        not retrieved
        or context_text == "No se encontró información relevante en la base de datos."
    ):
        logger.warning("⚠️  NO context found - modelo está en riesgo de alucinar")

    # Verificar que el contexto no sea vacío
    if (
        context_text
        and context_text != "No se encontró información relevante en la base de datos."
    ):
        logger.info(f"✅ Context provided: {len(retrieved)} chunks")
        logger.debug(f"Context preview: {context_text[:200]}...")
    else:
        logger.warning("⚠️  NO RAG context available - must answer 'No encontré...'")

    # Telemetría
    telemetry_log(
        {
            "query": user_msg,
            "original_language": meta.get("original_language"),
            "translated_query": (
                meta.get("original_query")
                if meta.get("original_language") != "en"
                else None
            ),
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

    # CRÍTICO: Verificar si el system prompt es demasiado largo y truncarlo si es necesario
    max_system_tokens = int(
        VLLM_MAX_MODEL_LEN * 0.25
    )  # Máximo 25% del modelo para system prompt
    current_system_tokens = count_tokens(enhanced_system)

    if current_system_tokens > max_system_tokens:
        logger.warning(
            f"⚠️ System prompt demasiado largo ({current_system_tokens} tokens). "
            f"Truncando a {max_system_tokens} tokens para dejar espacio para la respuesta."
        )

        # Crear una versión más corta del system prompt
        if context_text:
            # Versión minimalista con contexto
            enhanced_system = f"""Eres un asistente docente experto. Responde usando el contexto proporcionado.

Contexto RAG:
{context_text}

Responde usando solo información del contexto. Cita las fuentes con formato: [archivo.pdf, p.X](/docs/TOPIC/archivo.pdf#page=X)"""
        else:
            # Versión minimalista sin contexto
            enhanced_system = """Eres un asistente docente experto. Responde usando el contexto proporcionado y cita las fuentes con formato: [archivo.pdf, p.X](/docs/TOPIC/archivo.pdf#page=X)"""

        new_tokens = count_tokens(enhanced_system)
        logger.info(
            f"✅ System prompt truncado: {current_system_tokens} → {new_tokens} tokens"
        )

    messages.append({"role": "system", "content": enhanced_system})

    # 2. TODO el historial de conversación del usuario
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    # ============================================================
    # CÁLCULO DINÁMICO DE max_tokens - 100% DESDE .env
    # ============================================================

    # Calcular tokens de input
    system_tokens = count_tokens(enhanced_system)
    context_tokens = count_tokens(context_text)
    history_tokens = sum(
        count_tokens(m["content"]) for m in messages if m["role"] != "system"
    )
    total_input_tokens = system_tokens + history_tokens

    # Margen de seguridad
    safety_margin = 100
    available_tokens = VLLM_MAX_MODEL_LEN - total_input_tokens - safety_margin

    # Cálculo desde .env - sin hardcodeo
    if is_generative:
        # Modo generativo: usar el % configurado en .env
        desired_max_tokens = min(
            VLLM_MAX_TOKENS,
            int(VLLM_MAX_MODEL_LEN * (GENERATIVE_MAX_TOKENS_PERCENT / 100.0)),
        )

        # CRÍTICO: En modo generativo, garantizar MÍNIMO tokens para respuesta
        # Para 40 preguntas necesitamos ~15k tokens mínimo
        min_tokens_for_generation = int(VLLM_MAX_MODEL_LEN * 0.45)  # 45% del modelo
        desired_max_tokens = max(desired_max_tokens, min_tokens_for_generation)

        logger.info(
            f"🎯 MODO GENERATIVO: "
            f"Objetivo {desired_max_tokens} tokens "
            f"({GENERATIVE_MAX_TOKENS_PERCENT}% de {VLLM_MAX_MODEL_LEN}, "
            f"mínimo garantizado: {min_tokens_for_generation})"
        )
    else:
        # Modo respuesta: usar el % configurado en .env
        desired_max_tokens = min(
            VLLM_MAX_TOKENS,
            int(VLLM_MAX_MODEL_LEN * (RESPONSE_MAX_TOKENS_PERCENT / 100.0)),
        )
        logger.info(
            f"💬 MODO RESPUESTA: "
            f"Objetivo {desired_max_tokens} tokens "
            f"({RESPONSE_MAX_TOKENS_PERCENT}% de {VLLM_MAX_MODEL_LEN})"
        )

    # Usar el mínimo entre lo deseado y lo disponible
    max_tokens = max(MIN_RESPONSE_TOKENS, min(desired_max_tokens, available_tokens))

    # CRÍTICO: Si en modo generativo no tenemos suficientes tokens, es un problema
    if is_generative and max_tokens < 10000:
        logger.error(
            f"❌ MODO GENERATIVO: Solo {max_tokens} tokens disponibles "
            f"(se necesitan ~15k para 40 preguntas). "
            f"Input: {total_input_tokens}, disponibles: {available_tokens}"
        )
        # Intentar liberar espacio reduciendo contexto RAG
        logger.warning("⚠️  Considerar reducir CTX_TOKENS_GENERATIVE en .env")

    # ============================================================
    # ANÁLISIS DE TOKENS Y WARNINGS
    # ============================================================
    logger.info(f"📊 Token breakdown:")
    logger.info(f"   - Model max length: {VLLM_MAX_MODEL_LEN} tokens (from .env)")
    logger.info(f"   - vLLM max_tokens limit: {VLLM_MAX_TOKENS} (from .env)")
    logger.info(f"   - System prompt: ~{system_tokens} tokens")
    logger.info(f"   - RAG context: ~{context_tokens} tokens")
    logger.info(f"   - Conversation history: ~{history_tokens} tokens")
    logger.info(f"   - TOTAL INPUT: ~{total_input_tokens} tokens")
    logger.info(f"   - Available for response: {available_tokens} tokens")
    logger.info(f"   - Desired max_tokens: {desired_max_tokens} tokens")
    logger.info(f"   - FINAL max_tokens: {max_tokens} tokens ✅")

    # Validación crítica
    if available_tokens < MIN_RESPONSE_TOKENS:
        logger.error(
            f"❌ Input demasiado largo: {total_input_tokens} tokens "
            f"(límite modelo: {VLLM_MAX_MODEL_LEN}). "
            f"Solo quedan {available_tokens} tokens para respuesta "
            f"(mínimo requerido: {MIN_RESPONSE_TOKENS})."
        )
        raise HTTPException(
            status_code=400,
            detail=f"El contexto de entrada es demasiado largo ({total_input_tokens} tokens). "
            f"El modelo solo soporta {VLLM_MAX_MODEL_LEN} tokens totales. "
            f"Por favor, reduce el historial de conversación o el tamaño de la consulta.",
        )

    # Warning si estamos usando mucho del contexto
    input_percent = (total_input_tokens / VLLM_MAX_MODEL_LEN) * 100
    if input_percent > 70:
        logger.warning(
            f"⚠️  Input muy largo ({total_input_tokens} tokens, "
            f"{input_percent:.1f}% del límite), "
            f"podría afectar calidad de respuesta"
        )

    logger.info(
        f"🚀 Enviando a vLLM: {len(messages)} mensajes "
        f"(system + {len(req.messages)} historial)"
    )

    # Verificar salud de vLLM
    if not await check_vllm_health():
        raise HTTPException(
            status_code=503,
            detail="vLLM service is not responding. Please try again in a few moments.",
        )

    # ============================================================
    # PAYLOAD CON max_tokens CALCULADO DINÁMICAMENTE
    # ============================================================
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.7,
        "top_p": req.top_p if req.top_p is not None else 0.95,
        "stream": req.stream,
        "max_tokens": max_tokens,
    }

    payload_size = len(json.dumps(payload))
    logger.info(
        f"📦 Payload size: {payload_size:,} bytes ({payload_size / 1024:.1f} KB)"
    )

    async def stream_generator():
        """Generator que reenvía el stream SSE de vLLM con reintentos"""
        max_retries = 3
        streaming_timeout = httpx.Timeout(600.0, connect=20.0)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=streaming_timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{UPSTREAM_OPENAI_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as r:
                        if r.status_code == 404:
                            logger.error(
                                f"❌ 404: Model '{payload['model']}' not found or not ready"
                            )
                            error_data = {
                                "error": {
                                    "message": f"Model '{payload['model']}' is not available. "
                                    f"It may still be loading after a model switch. "
                                    f"Please try again.",
                                    "type": "model_not_ready",
                                    "code": 404,
                                }
                            }
                            yield f"data: {json.dumps(error_data)}\n\n".encode()
                            return

                        r.raise_for_status()
                        logger.info(
                            f"✓ Stream establecido con vLLM (attempt {attempt + 1})"
                        )

                        async for chunk in r.aiter_bytes():
                            yield chunk

                        logger.info("✓ Stream completado exitosamente")
                        return

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
                    f"⚠️  Stream interrupted (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time}s... Error: {type(e).__name__}"
                )
                await asyncio.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                try:
                    error_text = await e.response.aread()
                    error_text = error_text.decode("utf-8", errors="ignore")
                except Exception:
                    error_text = "<unreadable>"

                logger.error(
                    f"❌ vLLM HTTP error: {e.response.status_code} - {error_text}"
                )
                error_data = {
                    "error": {
                        "message": f"vLLM error: HTTP {e.response.status_code} - {error_text}",
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
