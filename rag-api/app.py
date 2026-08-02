# Archivo: rag-api/app.py
# Descripción: API FastAPI para RAG de IASantiago
#
# Este archivo contiene los endpoints HTTP. La lógica de negocio
# está delegada a los módulos core/ y retrieval_lib/.
#
# **rag-api es un servicio de retrieval puro** desde el rip-out del §7.1
# (2026-08-02). La orquestación del chat —historial, prompt de sistema, muestreo,
# streaming contra vLLM— la hace Open WebUI: sus 18 modelos de workspace apuntan
# directamente a vLLM y su Filter (inlet) llama a `POST /retrieve` para inyectar
# el contexto. Aquí ya no hay cliente de vLLM, ni SSE, ni superficie compatible
# con la API de OpenAI (`/v1/models`, `/v1/chat/completions`): ver PLAN.md punto
# 6 y FINDINGS.md §7.1.

import contextlib
import hashlib
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import quote

# Importaciones de módulos refactorizados
from config.settings import (
    CTX_TOKENS_GENERATIVE,
    CTX_TOKENS_SOFT_LIMIT,
    EMBED_DEFAULT,
    FINAL_TOPK,
    OPENAI_API_KEY,
    TOPIC_BASE_DIR,
    TOPIC_LABELS,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_TOKENS,
)
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from retrieval import (
    attach_citations,
    choose_retrieval,
    get_embedder,
    get_reranker,
    rerank_passages,
    soft_trim_context,
    telemetry_log,
)
from qdrant_utils import collection_exists

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
# LIFESPAN
# ============================================================
# El cliente httpx compartido y el cliente de vLLM vivían aquí; los dos se
# fueron con el rip-out del §7.1. rag-api ya no hace ninguna petición saliente:
# habla con Qdrant (cliente propio) y carga modelos locales, nada más.


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan de FastAPI: precarga modelos al inicio.

    El preload no es opcional: `/healthz` sólo responde 200 después, y el
    healthcheck de compose lo usa como señal de readiness.
    """
    logger.info("FastAPI startup: precargando modelos...")
    try:
        ensure_models_loaded()
        logger.info("Modelos precargados correctamente")
    except Exception as e:
        logger.error(f"Error en startup: {e}", exc_info=True)
        raise

    yield

    logger.info("FastAPI shutdown")


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


# ============================================================
# MODELOS PYDANTIC
# ============================================================


class RetrieveRequest(BaseModel):
    """Petición al servicio de retrieval puro (§7.1).

    Es el contrato que consume el Filter (inlet) de Open WebUI: le pasa la
    consulta y el tema, y recibe el contexto ya montado con citaciones. Ni
    historial ni vLLM viven aquí — eso lo orquesta Open WebUI.
    """

    query: str
    topic: str
    # Override de profundidad; None => FINAL_TOPK (o su múltiplo generativo).
    top_k: Optional[int] = None
    # Modo generativo (examen): recupera más hondo, igual que la ruta de chat.
    generative: bool = False
    # Reranking jina (CPU). Desactivable para depurar orden vs recuperación.
    rerank: bool = True


class Citation(BaseModel):
    file: str
    page: int
    chunk_id: str
    url: str


class RetrieveResponse(BaseModel):
    context: str
    citations: List[Citation]
    meta: Dict


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


# `GET /v1/models` y `POST /v1/chat/completions` vivían aquí. Los retiró el
# rip-out del §7.1 (2026-08-02): eran la superficie compatible con OpenAI que
# publicaba los nueve modelos falsos `topic:X` y orquestaba el chat contra vLLM.
# Hoy esa orquestación es de Open WebUI y el único consumidor de rag-api es su
# Filter, que llama a `POST /retrieve`.
#
# Con ellos se fueron `chat/` entero (regex de intención, prompt de sistema,
# cálculo de presupuesto de tokens, montaje de mensajes), `core/vllm_client.py`
# (streaming SSE y reintentos), `core/retry.py` y `token_utils.py`.
#
# **`/v1/eval/offline` NO es parte de esa superficie** aunque comparta el prefijo
# `/v1`: es el banco de evaluación y se queda.


def _build_citations(chunks: List[Dict], topic: str) -> List[Dict]:
    """Lista estructurada de fuentes, con la misma URL clicable que el contexto.

    Duplica el esquema de `retrieval_lib/citations.py` a propósito: el contexto
    de texto lleva los enlaces embebidos para el LLM, y esta lista los expone en
    JSON para el Filter (o una futura UI de fuentes de Open WebUI).
    """
    out = []
    for c in chunks:
        filename = os.path.basename(c["file_path"])
        page = c["page"]
        encoded = quote(filename, safe=".")
        url = f"/docs/{topic}/{encoded}#page={page}" if topic else f"/docs/{encoded}#page={page}"
        out.append(
            {
                "file": filename,
                "page": page,
                # chunk_id llega como int desde el payload de Qdrant; se
                # normaliza a str para un contrato estable de la API.
                "chunk_id": str(c["chunk_id"]),
                "url": url,
            }
        )
    return out


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    req: RetrieveRequest,
    x_email: str = Header(None),
    api_key: str = Depends(verify_api_key),
):
    """Servicio de retrieval puro (§7.1) — sin vLLM, sin historial.

    Es **la única ruta de servicio** desde el rip-out del §7.1 (2026-08-02).
    Recorre retrieval → reranking → recorte por tokens → citaciones y devuelve
    el contexto en JSON. Open WebUI lo orquesta: su Filter (inlet) llama aquí,
    inyecta el `context` en el último mensaje del usuario y hace streaming
    directo contra vLLM.

    El modo generativo llega como parámetro (`generative`) porque lo elige el
    usuario al escoger el modelo "- Generador"; la ruta `topic:X` que se retiró
    lo adivinaba con una regex de intención sobre el texto de la pregunta.
    """
    user_ref = (
        hashlib.sha256(x_email.encode("utf-8")).hexdigest()[:12] if x_email else "anon"
    )
    logger.info(f"[/retrieve] Usuario: {user_ref}, topic={req.topic}, gen={req.generative}")

    # Tema que no está en TOPIC_LABELS → 400. Es un error del llamador (una
    # etiqueta mal escrita en un workspace model, un topic_map incompleto), y
    # devolverlo como contexto vacío lo disfraza de "el corpus no tiene la
    # respuesta": indistinguible de la operación normal para quien lo lee en la
    # UI. Eso escondió los cuatro modelos "- Generador" rotos del 2026-08-02
    # (PLAN.md punto 8). El detalle lleva el tema recibido y los válidos para
    # que el mensaje baste sin abrir el log.
    if req.topic not in TOPIC_LABELS:
        logger.error(
            f"[/retrieve] Tema desconocido '{req.topic}'; válidos: {TOPIC_LABELS}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_topic",
                "topic": req.topic,
                "valid_topics": TOPIC_LABELS,
            },
        )

    # Tema válido pero sin colección → NO es culpa del llamador (falta ingestar,
    # o la colección se borró), así que se mantiene la degradación elegante del
    # §7.1: contexto vacío en vez de 500, el modelo responde igual. Se registra
    # como ERROR, no WARNING, porque un tema configurado sin colección es un
    # fallo de operación que alguien tiene que ver.
    if not collection_exists(req.topic):
        logger.error(
            f"[/retrieve] Tema '{req.topic}' sin colección Qdrant; devuelvo vacío"
        )
        return {
            "context": "",
            "citations": [],
            "meta": {
                "topic": req.topic,
                "mode": None,
                "generative": req.generative,
                "num_chunks": 0,
                "final_topk": None,
                "original_language": None,
                "context_token_limit": None,
                # Ya no es "unknown_topic": ese caso sale por el 400 de arriba.
                # Aquí el tema es válido y lo que falta es la colección.
                "error": "missing_collection",
            },
        }

    # Límite de contexto: mismo cálculo que la ruta de chat.
    if req.generative:
        context_token_limit = min(
            CTX_TOKENS_GENERATIVE, VLLM_MAX_MODEL_LEN - VLLM_MAX_TOKENS - 1000
        )
    else:
        context_token_limit = CTX_TOKENS_SOFT_LIMIT

    retrieved, meta = choose_retrieval(
        req.topic, req.query, req.generative, final_topk_override=req.top_k
    )
    logger.info(f"[/retrieve] Recuperados {len(retrieved)} chunks para '{req.topic}'")

    if retrieved:
        if req.rerank:
            retrieved = rerank_passages(req.query, retrieved, rerank_topk=None)
        retrieved = soft_trim_context(retrieved, context_token_limit)

    context_text, cited = attach_citations(retrieved, req.topic)

    # `source` se conserva aunque ya sólo haya una ruta: las filas históricas de
    # `retrieval.jsonl` llevan "chat" (o ninguna, que el comparador cuenta como
    # chat) y quitarlo ahora haría indistinguible el tráfico posterior al
    # rip-out del anterior en un fichero que no se reescribe.
    telemetry_log(
        {
            "source": "retrieve",
            "query": req.query,
            "original_language": meta.get("original_language"),
            "translated_query": (
                meta.get("original_query")
                if meta.get("original_language") != "en"
                else None
            ),
            "topic": req.topic,
            "mode": meta.get("mode"),
            "generative": req.generative,
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

    return {
        "context": context_text,
        "citations": _build_citations(retrieved, req.topic),
        "meta": {
            "topic": req.topic,
            "mode": meta.get("mode"),
            "generative": req.generative,
            "num_chunks": len(retrieved),
            "final_topk": meta.get("final_topk"),
            "original_language": meta.get("original_language"),
            "context_token_limit": context_token_limit,
        },
    }


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
    # Un tema mal escrito aquí medía **0.000 en silencio**, y 0.000 se lee como
    # "el retrieval es malísimo" en vez de como "te has equivocado de etiqueta".
    # Es exactamente la trampa contra la que avisa el cierre de PLAN.md: la
    # reparación de Chemistry midió 0.000 porque el golden set no tocaba los
    # documentos cambiados, y nadie lo vio hasta mucho después. Un banco que
    # miente sin avisar es peor que uno que no existe, porque se usa para
    # decidir.
    #
    # Misma separación por dueño que `/retrieve` (punto 8), y por el mismo
    # motivo: los dos fallos tienen dueños distintos.
    #
    # La validación va **antes de recuperar nada**: una tirada completa son
    # minutos y descubrir al final que el tema estaba mal escrito no ayuda a
    # nadie. Se comprueban los temas distintos, no un caso por caso, para no
    # pagar N viajes a Qdrant.
    unknown = sorted({c.topic for c in cases if c.topic not in TOPIC_LABELS})
    if unknown:
        logger.error(
            f"[/v1/eval/offline] Temas desconocidos {unknown}; válidos: {TOPIC_LABELS}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_topic",
                "topics": unknown,
                "valid_topics": TOPIC_LABELS,
            },
        )

    # Tema válido pero sin colección: no es culpa de quien llama, así que no es
    # un 400. Pero medirá 0.000 y eso hay que decirlo en la propia respuesta —
    # un log no lo ve quien lee la tabla de resultados.
    # Los temas distintos primero: un set por comprensión sobre `cases` dedupe
    # los resultados pero **no las llamadas**, y serían 157 viajes para 10 temas.
    missing_collections = sorted(
        t for t in sorted({c.topic for c in cases}) if not collection_exists(t)
    )
    topic_warnings = []
    for topic in missing_collections:
        logger.error(
            f"[/v1/eval/offline] Tema '{topic}' sin colección Qdrant; medirá 0.000"
        )
        topic_warnings.append(
            f"tema '{topic}': sin colección Qdrant; sus casos miden 0.000 "
            f"y el resultado no describe al retrieval"
        )

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
        # Los de tema van delante: si un tema no tiene colección, los avisos por
        # caso que vengan detrás son consecuencia de eso y no causas distintas.
        "warnings": topic_warnings + _eval_warnings(rows, file_aliases),
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
