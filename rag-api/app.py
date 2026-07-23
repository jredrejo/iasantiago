# Archivo: rag-api/app.py
# Descripción: API FastAPI para RAG de IASantiago
#
# Este archivo contiene los endpoints HTTP. La lógica de negocio
# está delegada a los módulos core/, chat/, y retrieval_lib/.

import contextlib
import json
import logging
import os
import time
from typing import Dict, List, Optional

import httpx

from chat.context_builder import ContextBuilder
from chat.intent import (
    detect_generative_intent,
    get_last_user_message,
    load_system_prompt,
)
from chat.token_calculator import TokenCalculator

# Importaciones de módulos refactorizados
from config.settings import (
    CTX_TOKENS_GENERATIVE,
    CTX_TOKENS_SOFT_LIMIT,
    EMBED_DEFAULT,
    FINAL_TOPK,
    GENERATIVE_MAX_TOKENS_PERCENT,
    GENERATIVE_REPETITION_PENALTY,
    GENERATIVE_TEMPERATURE,
    GENERATIVE_TOP_K,
    GENERATIVE_TOP_P,
    MIN_RESPONSE_TOKENS,
    OPENAI_API_KEY,
    RESPONSE_MAX_TOKENS_PERCENT,
    RESPONSE_REPETITION_PENALTY,
    RESPONSE_TEMPERATURE,
    RESPONSE_TOP_K,
    RESPONSE_TOP_P,
    TOPIC_BASE_DIR,
    TOPIC_LABELS,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_TOKENS,
    VLLM_MODEL,
)
from core.vllm_client import VLLMClient
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from retrieval import (
    attach_citations,
    choose_retrieval,
    count_tokens,
    get_embedder,
    get_reranker,
    rerank_passages,
    soft_trim_context,
    telemetry_log,
)
from token_utils import extract_topic_from_model_name

from eval import (
    aggregate_eval,
    build_content_alias_map,
    dedupe_files,
    dedupe_pages,
    normalize_file,
    normalize_page,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# AUTENTICACIÓN
# ============================================================

security = HTTPBearer()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica que el token Bearer coincida con OPENAI_API_KEY"""
    if credentials.credentials != OPENAI_API_KEY:
        logger.warning(f"Intento de autenticación fallido desde {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    return credentials.credentials


# ============================================================
# LIFESPAN Y CLIENTE HTTPX COMPARTIDO
# ============================================================

# Cliente httpx compartido para todas las peticiones
_shared_httpx_client: Optional[httpx.AsyncClient] = None
# Cliente vLLM global instanciado durante lifespan
_vllm_client_instance = None


def get_httpx_client() -> httpx.AsyncClient:
    """Obtiene el cliente httpx compartido"""
    global _shared_httpx_client
    if _shared_httpx_client is None:
        raise RuntimeError("httpx client not initialized - lifespan not started")
    return _shared_httpx_client


def get_vllm_client_instance():
    """Obtiene la instancia global del cliente vLLM"""
    global _vllm_client_instance
    if _vllm_client_instance is None:
        raise RuntimeError("vllm client not initialized - lifespan not started")
    return _vllm_client_instance


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan de FastAPI: precarga modelos al inicio, limpia al cierre"""
    global _shared_httpx_client, _vllm_client_instance

    # Startup
    logger.info("FastAPI startup: precargando modelos...")
    try:
        ensure_models_loaded()
        logger.info("Modelos precargados correctamente")

        # Crear cliente httpx compartido
        _shared_httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        logger.info("Cliente httpx compartido creado")

        # Instanciar vllm_client con el cliente compartido
        _vllm_client_instance = VLLMClient(httpx_client=_shared_httpx_client)
        logger.info("Cliente vLLM instanciado con httpx compartido")

    except Exception as e:
        logger.error(f"Error en startup: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("FastAPI shutdown: limpiando recursos...")
    if _shared_httpx_client:
        await _shared_httpx_client.aclose()
        logger.info("Cliente httpx cerrado")


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


# Crear aplicación FastAPI con lifespan
app = FastAPI(title="IASantiago RAG API", lifespan=lifespan)

# Instanciar componentes (que no dependen de lifespan)
token_calculator = TokenCalculator(
    model_max_len=VLLM_MAX_MODEL_LEN,
    max_tokens_limit=VLLM_MAX_TOKENS,
    generative_percent=GENERATIVE_MAX_TOKENS_PERCENT,
    response_percent=RESPONSE_MAX_TOKENS_PERCENT,
    min_response_tokens=MIN_RESPONSE_TOKENS,
)
context_builder = ContextBuilder(
    max_context_tokens=CTX_TOKENS_SOFT_LIMIT,
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
    temperature: Optional[float] = None  # Use .env defaults (RESPONSE_TEMPERATURE or GENERATIVE_TEMPERATURE)
    top_p: Optional[float] = None  # Use .env defaults (RESPONSE_TOP_P or GENERATIVE_TOP_P)
    stream: Optional[bool] = True


class EvalCase(BaseModel):
    query: str
    topic: str
    # Ground truth PRIMARIO: "fichero.pdf#12" (la ruta puede ir completa o no,
    # se compara por nombre base). La página es invariante frente a cambios de
    # chunking, así que es lo que debe decidir si un chunker nuevo mejora.
    relevant_pages: List[str] = Field(default_factory=list)
    # Ground truth secundario y grueso. Si se omite, se deriva de relevant_pages
    # para no tener que escribir dos veces lo mismo.
    relevant_files: List[str] = Field(default_factory=list)


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/healthz")
async def healthz():
    """Health check endpoint"""
    return {"ok": True, "topics": TOPIC_LABELS}


@app.get("/v1/models")
async def list_models(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
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
    api_key: str = Depends(verify_api_key),
):
    """
    Endpoint principal de chat con RAG.

    Flujo:
    1. Detectar intención (generativa vs respuesta)
    2. Ejecutar retrieval híbrido
    3. Construir contexto y mensajes
    4. Calcular tokens
    5. Verificar salud de vLLM y enviar (streaming o no)
    """
    logger.info(f"Usuario: {x_email}")
    logger.info(f"Mensajes recibidos: {len(req.messages)}")

    # 1. Extraer topic y mensaje del usuario
    topic = extract_topic_from_model_name(req.model, TOPIC_LABELS[0])
    user_msg = get_last_user_message(req.messages)

    # 2. Detectar intención y cargar prompt
    is_generative = detect_generative_intent(user_msg)
    sys_prompt = load_system_prompt(is_generative)

    # 3. Ajustar límites de contexto según modo
    if is_generative:
        context_token_limit = CTX_TOKENS_GENERATIVE
        # Reducir contexto para dejar espacio a respuesta
        max_context = VLLM_MAX_MODEL_LEN - VLLM_MAX_TOKENS - 1000
        context_token_limit = min(context_token_limit, max_context)
        logger.info(f"Modo GENERATIVO: Límite de contexto {context_token_limit} tokens")
        effective_temp = (
            GENERATIVE_TEMPERATURE if req.temperature is None else req.temperature
        )
        effective_top_p = GENERATIVE_TOP_P if req.top_p is None else req.top_p
        effective_top_k = GENERATIVE_TOP_K
        effective_rep_penalty = GENERATIVE_REPETITION_PENALTY
    else:
        context_token_limit = CTX_TOKENS_SOFT_LIMIT
        effective_temp = (
            RESPONSE_TEMPERATURE if req.temperature is None else req.temperature
        )
        effective_top_p = RESPONSE_TOP_P if req.top_p is None else req.top_p
        effective_top_k = RESPONSE_TOP_K
        effective_rep_penalty = RESPONSE_REPETITION_PENALTY

    logger.info(
        f"Muestreo: temperature={effective_temp}, top_p={effective_top_p} ({'GENERATIVO' if is_generative else 'RESPUESTA'})"
    )

    # 4. Retrieval
    retrieved, meta = choose_retrieval(topic, user_msg, is_generative)
    logger.info(f"Recuperados {len(retrieved)} chunks para '{topic}'")

    if retrieved:
        # Reranking
        retrieved = rerank_passages(user_msg, retrieved, rerank_topk=None)
        # Trim por tokens después del reranking
        retrieved = soft_trim_context(retrieved, context_token_limit)

    # 5. Construir contexto con citaciones
    context_text, cited = attach_citations(retrieved, topic)
    context_builder.log_context_status(context_text, len(retrieved))

    # 6. Telemetría
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

    # 7. Construir mensajes (contexto en user message para prefix caching)
    messages = context_builder.build_messages(
        sys_prompt, req.messages, context_text, context_token_limit
    )

    # 8. Calcular tokens
    budget = token_calculator.calculate_budget(
        system_prompt, context_text, messages, is_generative
    )

    # Validar que hay espacio para respuesta
    if budget.available_for_response < MIN_RESPONSE_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"El contexto de entrada es demasiado largo ({budget.total_input} tokens). "
            f"El modelo solo soporta {VLLM_MAX_MODEL_LEN} tokens totales. "
            f"Por favor, reduce el historial de conversación.",
        )

    # 9. Verificar salud de vLLM
    if not await get_vllm_client_instance().check_health():
        raise HTTPException(
            status_code=503,
            detail="vLLM no responde. Por favor intente de nuevo.",
        )

    # 10. Preparar payload
    logger.info(f"Enviando a vLLM: {len(messages)} mensajes")

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": effective_temp,
        "top_p": effective_top_p,
        "top_k": effective_top_k,
        "repetition_penalty": effective_rep_penalty,
        "stream": req.stream,
        "max_tokens": budget.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    payload_size = len(json.dumps(payload))
    logger.info(
        f"Tamaño payload: {payload_size:,} bytes ({payload_size / 1024:.1f} KB)"
    )

    # 11. Enviar a vLLM
    if req.stream:
        return StreamingResponse(
            get_vllm_client_instance().stream_chat_completion(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        resp = await get_vllm_client_instance().chat_completion(payload)
        return Response(
            content=resp.content,
            media_type=resp.headers.get("Content-Type", "application/json"),
        )


def _eval_warnings(
    rows: List[Dict], file_aliases: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Denuncia ground truth que no puede puntuar.

    Existe por lo ocurrido con `eval/cases.sample.json`: apuntaba a un
    `sample1.pdf` inexistente y con rutas de host, así que Recall y MRR daban
    0.0 de forma permanente y el fichero pasó meses aparentando medir algo.
    Un 0.0 por ground truth roto y un 0.0 por retrieval malo son
    indistinguibles en el número; aquí se separan.

    `file_aliases` canoniza los duplicados byte-idénticos para no avisar en
    falso de que un fichero "no aparece" cuando lo que se recuperó fue su copia.
    """
    aliases = file_aliases or {}

    def canon(name: str) -> str:
        return aliases.get(name, name)

    warnings = []

    # Todos los archivos vistos en toda la tanda: si una referencia no aparece
    # en ninguna, lo más probable es que esté mal escrita o en otro tema.
    seen_files = {canon(f) for r in rows for f in dedupe_files(r["retrieved"])}

    for r in rows:
        q = r["query"][:60]

        if not r["relevant_pages"] and not r["relevant_files"]:
            warnings.append(f"'{q}': sin ground truth; excluido de las métricas")
            continue

        for p in r["relevant_pages"]:
            if "#" not in p:
                warnings.append(
                    f"'{q}': '{p}' no lleva '#pagina'; no puede casar con ninguna página"
                )

        for f in {normalize_file(x) for x in r["relevant_pages"] + r["relevant_files"]}:
            if canon(f) not in seen_files:
                warnings.append(
                    f"'{q}': '{f}' no aparece en ningún resultado de la tanda; "
                    f"revisa el nombre y el tema"
                )

    return warnings


def _resolve_file_aliases(rows: List[Dict]) -> Dict[str, str]:
    """Agrupa los ficheros byte-idénticos (§3.-1) que intervienen en esta tanda.

    Sólo se hashea el conjunto de ficheros implicados —los recuperados y los
    nombrados por el ground truth—, no todo el corpus. Los recuperados traen su
    ruta real de contenedor; los que sólo están en el golden se buscan bajo
    `TOPIC_BASE_DIR/<tema>` (si están anidados y no se encuentran, se ignoran:
    lo importante es cazar el caso en que ambas copias aparecen recuperadas).
    """
    paths_by_name: Dict[str, str] = {}
    for r in rows:
        for c in r["retrieved"]:
            name = normalize_file(c["file_path"])
            paths_by_name.setdefault(name, c["file_path"])
        for ref in r["relevant_pages"] + r["relevant_files"]:
            name = normalize_file(ref)
            paths_by_name.setdefault(
                name, os.path.join(TOPIC_BASE_DIR, r["topic"], name)
            )
    return build_content_alias_map(paths_by_name)


@app.post("/v1/eval/offline")
async def eval_offline(
    cases: List[EvalCase],
    rerank: bool = True,
    final_topk: Optional[int] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Evaluación offline del sistema de retrieval.

    Recorre la MISMA cadena que `/v1/chat/completions` en modo RESPUESTA
    (retrieval → reranking → recorte por tokens). Antes usaba una familia de
    funciones paralela que no reordenaba y recuperaba la mitad de candidatos,
    así que medía un orden que ningún alumno llegaba a ver.

    `?rerank=false` mide la salida previa al reranker: separa un fallo de
    recuperación (no aparece) de uno de ordenación (aparece mal colocado). Es
    además mucho más rápido, porque el reranker jina corre en CPU.

    `?final_topk=N` recupera más hondo que la profundidad de servicio, para
    distinguir "nunca se recuperó" de "se recuperó por debajo del corte de 18".
    Con override se omite el recorte por tokens, que existe para caber en el
    contexto y sólo volvería a cortar lo que se quería ver. NO cambia nada de
    la ruta de chat: es un parámetro de esta petición.
    """
    rows = []
    for c in cases:
        retrieved, meta = choose_retrieval(
            c.topic, c.query, is_generative=False, final_topk_override=final_topk
        )
        if retrieved:
            if rerank:
                retrieved = rerank_passages(c.query, retrieved, rerank_topk=None)
            if final_topk is None:
                retrieved = soft_trim_context(retrieved, CTX_TOKENS_SOFT_LIMIT)
        context_text, cited = attach_citations(retrieved, c.topic)

        # Si sólo se dio ground truth de páginas, derivar el de archivos.
        relevant_files = c.relevant_files or list(
            dict.fromkeys(normalize_file(p) for p in c.relevant_pages)
        )

        rows.append(
            {
                "query": c.query,
                "topic": c.topic,
                "relevant_files": relevant_files,
                "relevant_pages": c.relevant_pages,
                "retrieved": retrieved,
                "context": context_text,
            }
        )

    file_aliases = _resolve_file_aliases(rows)
    agg = aggregate_eval(rows, file_aliases=file_aliases)
    return {
        "aggregate": agg,
        "config": {
            "rerank": rerank,
            "final_topk": final_topk if final_topk is not None else FINAL_TOPK,
            "final_topk_overridden": final_topk is not None,
            "context_token_limit": (
                None if final_topk is not None else CTX_TOKENS_SOFT_LIMIT
            ),
            "embed_model": EMBED_DEFAULT,
            "page_tolerance": agg["page_tolerance"],
            "duplicate_groups": agg["duplicate_groups"],
        },
        "warnings": _eval_warnings(rows, file_aliases),
        "details": [
            {
                "query": r["query"],
                "topic": r["topic"],
                "pred_pages": dedupe_pages(r["retrieved"]),
                "relevant_pages": [normalize_page(p) for p in r["relevant_pages"]],
                "pred_files": dedupe_files(r["retrieved"]),
                "relevant_files": [normalize_file(f) for f in r["relevant_files"]],
            }
            for r in rows
        ],
    }
