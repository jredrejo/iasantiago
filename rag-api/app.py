# Archivo: rag-api/app.py
# Descripción: API FastAPI para RAG de IASantiago
#
# Este archivo contiene los endpoints HTTP. La lógica de negocio
# está delegada a los módulos core/, chat/, y retrieval_lib/.

import json
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Importaciones de módulos refactorizados
from config.settings import (
    CTX_TOKENS_GENERATIVE,
    CTX_TOKENS_SOFT_LIMIT,
    OPENAI_API_KEY,
    TOPIC_LABELS,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_TOKENS,
    VLLM_MODEL,
    GENERATIVE_MAX_TOKENS_PERCENT,
    RESPONSE_MAX_TOKENS_PERCENT,
    MIN_RESPONSE_TOKENS,
)
from core.vllm_client import get_vllm_client
from chat.intent import (
    detect_generative_intent,
    load_system_prompt,
    get_last_user_message,
)
from chat.token_calculator import TokenCalculator
from chat.context_builder import ContextBuilder
from retrieval import (
    attach_citations,
    choose_retrieval,
    choose_retrieval_enhanced,
    count_tokens,
    rerank_passages,
    soft_trim_context,
    telemetry_log,
    get_embedder,
    get_reranker,
)
from token_utils import extract_topic_from_model_name
from eval import aggregate_eval

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# INICIALIZACIÓN
# ============================================================


def ensure_models_loaded():
    """Precarga modelos al startup"""
    logger.info("Verificando disponibilidad de modelos de embedding...")

    try:
        from config.settings import EMBED_PER_TOPIC, RERANK_MODEL

        # Precargar embedders
        for topic in EMBED_PER_TOPIC.keys():
            logger.info(f"Precargando embedder para {topic}...")
            get_embedder(topic)
            logger.info(f"Embedder para {topic} cargado")

        # Precargar reranker
        logger.info("Precargando reranker...")
        get_reranker()
        logger.info("Reranker cargado")

    except Exception as e:
        logger.error(f"Error cargando modelos al startup: {e}", exc_info=True)
        raise


# Crear aplicación FastAPI
app = FastAPI(title="IASantiago RAG API")
ensure_models_loaded()

# Instanciar componentes
vllm_client = get_vllm_client()
token_calculator = TokenCalculator(
    model_max_len=VLLM_MAX_MODEL_LEN,
    max_tokens_limit=VLLM_MAX_TOKENS,
    generative_percent=GENERATIVE_MAX_TOKENS_PERCENT,
    response_percent=RESPONSE_MAX_TOKENS_PERCENT,
    min_response_tokens=MIN_RESPONSE_TOKENS,
)
context_builder = ContextBuilder(
    max_system_percent=0.25,
    model_max_len=VLLM_MAX_MODEL_LEN,
)


# ============================================================
# MODELOS PYDANTIC
# ============================================================


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = True
    iasantiago_mode: Optional[str] = "explica"


class EvalCase(BaseModel):
    query: str
    topic: str
    relevant_files: List[str] = Field(default_factory=list)


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/healthz")
async def healthz():
    """Health check endpoint"""
    return {"ok": True, "topics": TOPIC_LABELS}


@app.get("/v1/models")
async def list_models(request: Request):
    """Lista modelos disponibles (compatible con OpenAI API)"""
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


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest,
    request: Request,
    x_email: str = Header(None),
):
    """
    Endpoint principal de chat con RAG.

    Flujo:
    1. Detectar modelo y esperar si hay cambio
    2. Detectar intención (generativa vs respuesta)
    3. Ejecutar retrieval híbrido
    4. Construir contexto y mensajes
    5. Calcular tokens
    6. Enviar a vLLM (streaming o no)
    """
    logger.info(f"Usuario: {x_email}")
    logger.info(f"Mensajes recibidos: {len(req.messages)}")

    # 1. Asegurar que el modelo esté listo
    await vllm_client.ensure_model_ready(VLLM_MODEL)

    # 2. Extraer topic y mensaje del usuario
    topic = extract_topic_from_model_name(req.model, TOPIC_LABELS[0])
    user_msg = get_last_user_message(req.messages)

    # 3. Detectar intención y cargar prompt
    is_generative = detect_generative_intent(user_msg)
    sys_prompt = load_system_prompt(is_generative)

    # 4. Ajustar límites de contexto según modo
    if is_generative:
        context_token_limit = CTX_TOKENS_GENERATIVE
        # Reducir contexto para dejar espacio a respuesta
        max_context = VLLM_MAX_MODEL_LEN - VLLM_MAX_TOKENS - 1000
        context_token_limit = min(context_token_limit, max_context)
        logger.info(f"Modo GENERATIVO: Límite de contexto {context_token_limit} tokens")
    else:
        context_token_limit = CTX_TOKENS_SOFT_LIMIT

    # 5. Retrieval
    retrieved, meta = choose_retrieval_enhanced(topic, user_msg, is_generative)
    logger.info(f"Recuperados {len(retrieved)} chunks para '{topic}'")

    if retrieved:
        # Reranking
        retrieved = rerank_passages(user_msg, retrieved, rerank_topk=None)
        # Trim por tokens después del reranking
        retrieved = soft_trim_context(retrieved, context_token_limit)

    # 6. Construir contexto con citaciones
    context_text, cited = attach_citations(retrieved, topic)
    context_builder.log_context_status(context_text, len(retrieved))

    # 7. Telemetría
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

    # 8. Construir mensajes
    enhanced_system = context_builder.build_enhanced_system_prompt(
        sys_prompt, context_text
    )
    messages = context_builder.build_messages(enhanced_system, req.messages)

    # 9. Calcular tokens
    budget = token_calculator.calculate_budget(
        enhanced_system, context_text, messages, is_generative
    )

    # Validar que hay espacio para respuesta
    if budget.available_for_response < MIN_RESPONSE_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"El contexto de entrada es demasiado largo ({budget.total_input} tokens). "
            f"El modelo solo soporta {VLLM_MAX_MODEL_LEN} tokens totales. "
            f"Por favor, reduce el historial de conversación.",
        )

    # 10. Verificar salud de vLLM
    if not await vllm_client.check_health():
        raise HTTPException(
            status_code=503,
            detail="vLLM no responde. Por favor intente de nuevo.",
        )

    # 11. Preparar payload
    logger.info(f"Enviando a vLLM: {len(messages)} mensajes")

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.7,
        "top_p": req.top_p if req.top_p is not None else 0.95,
        "stream": req.stream,
        "max_tokens": budget.max_tokens,
    }

    payload_size = len(json.dumps(payload))
    logger.info(
        f"Tamaño payload: {payload_size:,} bytes ({payload_size / 1024:.1f} KB)"
    )

    # 12. Enviar a vLLM
    if req.stream:
        return StreamingResponse(
            vllm_client.stream_chat_completion(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        resp = await vllm_client.chat_completion(payload)
        return Response(
            content=resp.content,
            media_type=resp.headers.get("Content-Type", "application/json"),
        )


@app.post("/v1/eval/offline")
async def eval_offline(cases: List[EvalCase]):
    """Evaluación offline del sistema de retrieval"""
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
