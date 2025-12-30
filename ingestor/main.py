import faulthandler
import gc
import glob
import hashlib
import json
import logging
import os
import signal
import sys
from chunk import pdf_to_chunks_with_enhanced_validation, set_heartbeat_callback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ============================================================
# CRÍTICO: Forzar salida en segfault
# Las librerías CUDA/C pueden capturar SIGSEGV pero no salir, dejando
# el contenedor en estado zombie. Esto asegura el reinicio del contenedor.
# ============================================================

# Habilitar faulthandler para mejores diagnósticos de fallos
# Esto debe hacerse ANTES de cualquier import CUDA/PyTorch en módulos hijos
faulthandler.enable(file=sys.stderr, all_threads=True)


def _force_exit_on_signal(signum: int, frame) -> None:
    """Forzar salida inmediata en señales fatales para disparar reinicio de contenedor."""
    signal_name = signal.Signals(signum).name
    print(
        f"\n[FATAL] Caught {signal_name} - forcing exit to trigger container restart",
        file=sys.stderr,
        flush=True,
    )
    # Use os._exit() to bypass Python cleanup which might hang
    os._exit(128 + signum)


# Instalar manejadores de señales para señales fatales
# Nota: Estos ejecutan ANTES de los manejadores de librería C, así que tienen prioridad
for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT):
    try:
        signal.signal(sig, _force_exit_on_signal)
    except (OSError, ValueError):
        pass  # Algunas señales no pueden ser capturadas en ciertos contextos
from docling_client import DoclingClient, DoclingCrashLimitExceeded
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from settings import *
from whoosh import index
from whoosh.fields import ID, NUMERIC, TEXT, Schema

# ============================================================
# CRÍTICO: Configurar directorio de caché de modelos ANTES de imports
# Usar solo HF_HOME (TRANSFORMERS_CACHE está deprecado en v5)
# ============================================================
MODEL_CACHE_DIR = "/models_cache"
os.environ["HF_HOME"] = MODEL_CACHE_DIR

# Crear directorio de caché si no existe
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ============================================================
# DISPOSITIVO DE EMBEDDINGS: Todo en GPU con float16
# - Modelo en GPU con float16 (~650MB vs 1.3GB en float32)
# - Inferencia en GPU
# ============================================================
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DTYPE = "float16"

# Re-registrar manejadores de señales DESPUÉS de inicialización PyTorch/CUDA
# Las librerías CUDA pueden sobrescribir manejadores de señales Python durante init
for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT):
    try:
        signal.signal(sig, _force_exit_on_signal)
    except (OSError, ValueError):
        pass

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"[CONFIG] Embedding device: {EMBEDDING_DEVICE}")
logger.info(f"[CACHE] Model cache directory: {MODEL_CACHE_DIR}")
logger.info(f"[CACHE] HF_HOME: {os.environ['HF_HOME']}")

client = QdrantClient(url=QDRANT_URL)

# Inicialización de docling (extracción local - no se necesita servicio externo)
ENABLE_DOCLING = os.getenv("ENABLE_DOCLING", "true").lower() == "true"

docling_client = DoclingClient(enable_fallback=True) if ENABLE_DOCLING else None

logger.info(
    f"[CONFIG] Docling extraction: {'ENABLED (local)' if ENABLE_DOCLING else 'DISABLED'}"
)


# ============================================================
# HEARTBEAT: Para que healthcheck detecte contenedores atascados
# ============================================================
HEARTBEAT_FILE = "/tmp/ingestor_heartbeat"

# Tamaño Mega-batch para codificar textos muy grandes y prevenir GPU OOM
# Limpia la cache de GPU entre lotes
ENCODING_MEGA_BATCH_SIZE = 5000


def update_heartbeat(current_file: str = "") -> None:
    """Actualizar archivo de heartbeat con timestamp actual y archivo siendo procesado."""
    try:
        import time

        with open(HEARTBEAT_FILE, "w") as f:
            f.write(f"{time.time()}\n{current_file}\n")
    except Exception:
        pass  # No dejar que problemas de heartbeat afecten el procesamiento


# ============================================================
# WATCHDOG: Forzar salida si el proceso se bloquea (ej. corrupción de glibc)
# ============================================================
WATCHDOG_TIMEOUT = 450  # 7.5 minutes - same as healthcheck threshold
WATCHDOG_CHECK_INTERVAL = 60  # Revisar cada minuto


def _watchdog_thread() -> None:
    """Thread en segundo plano que monitorea heartbeat y fuerza salida en bloqueo.

    Esto captura casos donde el proceso se cuelga sin levantar una señal,
    como errores de 'corrupted double-linked list' de glibc.
    """
    import threading
    import time

    logger.info("[WATCHDOG] Started - monitoring for stalled processing")

    while True:
        time.sleep(WATCHDOG_CHECK_INTERVAL)

        try:
            if os.path.exists(HEARTBEAT_FILE):
                with open(HEARTBEAT_FILE, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_heartbeat = float(lines[0].strip())
                        current_file = lines[1].strip() if len(lines) > 1 else "unknown"
                        age = time.time() - last_heartbeat

                        if age > WATCHDOG_TIMEOUT:
                            logger.error(
                                f"[WATCHDOG] Heartbeat stale for {age:.0f}s (limit: {WATCHDOG_TIMEOUT}s)"
                            )
                            logger.error(
                                f"[WATCHDOG] Last file being processed: {current_file}"
                            )
                            logger.error(
                                "[WATCHDOG] Forcing exit to trigger container restart"
                            )
                            sys.stdout.flush()
                            sys.stderr.flush()
                            os._exit(1)  # Force immediate exit
        except Exception as e:
            # No dejar que errores de watchdog afecten la operación normal
            logger.debug(f"[WATCHDOG] Error checking heartbeat: {e}")


def start_watchdog() -> None:
    """Iniciar el thread de watchdog como daemon."""
    import threading

    watchdog = threading.Thread(target=_watchdog_thread, daemon=True, name="watchdog")
    watchdog.start()


# ============================================================
# SEGUIMIENTO DE ESTADO: Rastrear archivos ya procesados
# ============================================================
STATE_FILE = "/whoosh/.processing_state.json"


class ProcessingState:
    """Gestiona el estado de archivos procesados"""

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Carga estado anterior o crea uno nuevo"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                logger.info(
                    f"[STATE] Loaded existing state with {len(state.get('processed', {}))} processed files"
                )
                return state
            except Exception as e:
                logger.warning(f"[STATE] Failed to load state: {e}, creating new")
                return self._create_empty_state()
        else:
            logger.info("[STATE] No previous state found, creating new")
            return self._create_empty_state()

    def _create_empty_state(self) -> dict:
        """Crea estructura vacía de estado"""
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "last_scan": None,
            "processed": {},
            "failed": {},
        }

    def _save_state(self):
        """Guarda estado a archivo"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"[STATE] Failed to save state: {e}")

    def get_file_hash(self, file_path: str) -> str:
        """Calcula hash MD5 del archivo"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"[STATE] Failed to calculate hash for {file_path}: {e}")
            return None

    def is_already_processed(self, file_path: str) -> bool:
        """Verifica si un archivo ya fue procesado"""
        file_path = str(file_path)

        if file_path not in self.state["processed"]:
            return False

        file_info = self.state["processed"][file_path]
        if file_info.get("status") == "failed":
            logger.info(
                f"[STATE] Retrying previously failed file: {Path(file_path).name}"
            )
            return False

        current_hash = self.get_file_hash(file_path)
        stored_hash = file_info.get("hash")

        if current_hash and stored_hash and current_hash != stored_hash:
            logger.info(
                f"[STATE] File changed (hash mismatch), reprocessing: {Path(file_path).name}"
            )
            return False

        logger.info(f"[STATE] Skipping already processed: {Path(file_path).name}")
        return True

    def mark_as_processed(self, file_path: str, topic: str):
        """Marca archivo como procesado exitosamente"""
        file_path = str(file_path)
        self.state["processed"][file_path] = {
            "hash": self.get_file_hash(file_path),
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "status": "success",
        }
        self._save_state()
        logger.info(f"[STATE] Marked as processed: {Path(file_path).name}")

    def mark_as_failed(self, file_path: str, error: str):
        """Marca archivo como fallido"""
        file_path = str(file_path)
        self.state["processed"][file_path] = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(error)[:200],
        }
        self.state["failed"][file_path] = {
            "error": str(error)[:500],
            "timestamp": datetime.now().isoformat(),
        }
        self._save_state()
        logger.warning(f"[STATE] Marked as failed: {Path(file_path).name}")

    def update_scan_time(self):
        """Actualiza timestamp de último scan"""
        self.state["last_scan"] = datetime.now().isoformat()
        self._save_state()

    def get_stats(self) -> dict:
        """Retorna estadísticas del estado"""
        processed = self.state.get("processed", {})
        successful = sum(1 for v in processed.values() if v.get("status") == "success")
        failed = sum(1 for v in processed.values() if v.get("status") == "failed")

        return {
            "total_processed": len(processed),
            "successful": successful,
            "failed": failed,
            "last_scan": self.state.get("last_scan"),
        }

    def reset(self):
        """Reinicia el estado"""
        logger.warning("[STATE] Resetting processing state - will rescan all files")
        self.state = self._create_empty_state()
        self._save_state()


# Instancia global de estado
state = ProcessingState()


# ============================================================
# VALIDACIÓN DE VECTORES: Asegurar vectores válidos para Qdrant
# ============================================================
def validate_and_fix_vectors(vecs, dims):
    """
    Valida y corrige vectores para Qdrant
    - Elimina NaN, Inf
    - Asegura tipo correcto (list of floats)
    - Asegura dimensión correcta
    """
    if isinstance(vecs, torch.Tensor):
        vecs = vecs.float().cpu().numpy()

    if isinstance(vecs, np.ndarray):
        vecs = vecs.tolist()

    if not isinstance(vecs, list):
        raise ValueError(f"Vectors must be list, got {type(vecs)}")

    valid_vecs = []
    invalid_count = 0

    for i, vec in enumerate(vecs):
        # Convertir a lista si es necesario
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        elif isinstance(vec, torch.Tensor):
            vec = vec.float().cpu().numpy().tolist()

        # Verificar dimensión
        if len(vec) != dims:
            logger.warning(
                f"Vector {i} tiene dimensión incorrecta: {len(vec)} != {dims}"
            )
            invalid_count += 1
            # Rellenar o truncar
            if len(vec) < dims:
                vec = vec + [0.0] * (dims - len(vec))
            else:
                vec = vec[:dims]

        # Verificar y corregir valores inválidos
        valid_vec = []
        has_invalid = False
        for val in vec:
            if isinstance(val, (list, np.ndarray)):
                val = float(val[0]) if len(val) > 0 else 0.0

            # Convertir a float
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
                has_invalid = True

            # Verificar NaN/Inf
            if not np.isfinite(val):
                val = 0.0
                has_invalid = True

            valid_vec.append(val)

        if has_invalid:
            invalid_count += 1
            logger.warning(
                f"Vector {i} contained invalid values (NaN/Inf), replaced with 0.0"
            )

        valid_vecs.append(valid_vec)

    if invalid_count > 0:
        logger.warning(f"Fixed {invalid_count}/{len(vecs)} invalid vectors")

    return valid_vecs


# ============================================================
# CACHÉ DE MODELOS: Carga modelos con caché persistente
# ============================================================
class ModelCache:
    """Gestiona caché de modelos SentenceTransformer"""

    def __init__(self):
        self.models = {}
        self.cache_info_file = os.path.join(MODEL_CACHE_DIR, ".model_cache_info.json")
        self.use_gpu = torch.cuda.is_available()
        self.gpu_failed = False  # Rastrea si GPU falló
        logger.info("[CACHE] Initializing model cache")

    def get_model(self, model_name: str, device: str = "cpu") -> SentenceTransformer:
        """Obtiene modelo del caché o lo descarga"""
        # Si GPU falló antes, usar CPU directamente
        if self.gpu_failed and device == "cuda":
            logger.warning("[CACHE] GPU failed previously, using CPU")
            device = "cpu"

        cache_key = f"{model_name}_{device}_{EMBEDDING_DTYPE}"

        if cache_key in self.models:
            logger.info(f"[CACHE] Using cached model: {model_name}")
            return self.models[cache_key]

        logger.info(f"[CACHE] Loading model (may download if first time): {model_name}")
        logger.info(f"[CACHE] Cache directory: {MODEL_CACHE_DIR}")
        logger.info(f"[CACHE] Device: {device}")
        logger.info(f"[CACHE] Dtype: {EMBEDDING_DTYPE}")

        start_time = datetime.now().timestamp()
        try:
            # HF_HOME already set in environment
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )

            # Convertir a float16 SOLO si es GPU y no ha fallado antes
            if (
                device == "cuda"
                and EMBEDDING_DTYPE == "float16"
                and not self.gpu_failed
            ):
                try:
                    # Test simple para verificar que float16 funciona
                    test_tensor = torch.randn(1, 10).half().to(device)
                    _ = test_tensor * 2  # Operación simple

                    model = model.half()
                    logger.info(f"[CACHE] Model converted to float16")
                except Exception as e:
                    logger.warning(f"[CACHE] float16 conversion failed: {e}")
                    logger.warning(f"[CACHE] Keeping model in float32")

            elapsed = datetime.now().timestamp() - start_time
            logger.info(f"[CACHE] Model loaded in {elapsed:.2f}s")

            # Cache en memoria
            self.models[cache_key] = model

            return model
        except Exception as e:
            logger.error(
                f"[CACHE] Failed to load model {model_name}: {e}", exc_info=True
            )
            raise

    def encode_with_gpu(
        self, model: SentenceTransformer, texts: list, batch_size: int = 32
    ):
        """Codifica textos con fallback automático GPU/CPU.

        Para conjuntos de texto grandes (>ENCODING_MEGA_BATCH_SIZE), procesa en mega-batches
        con limpieza de caché GPU entre batches para prevenir OOM.
        """
        device = str(model.device)
        total_texts = len(texts)
        logger.info(f"[CACHE] Codificando {total_texts} textos en {device}")

        # Para conjuntos de texto grandes, usar procesamiento por mega-batches para prevenir OOM de GPU
        use_mega_batching = total_texts > ENCODING_MEGA_BATCH_SIZE

        if use_mega_batching:
            total_mega_batches = (
                total_texts + ENCODING_MEGA_BATCH_SIZE - 1
            ) // ENCODING_MEGA_BATCH_SIZE
            logger.info(
                f"[CACHE] Conjunto de texto grande detectado, procesando en {total_mega_batches} mega-batches de {ENCODING_MEGA_BATCH_SIZE}"
            )

        try:
            if use_mega_batching:
                all_vecs = []
                for mega_batch_num, mega_start in enumerate(
                    range(0, total_texts, ENCODING_MEGA_BATCH_SIZE), 1
                ):
                    mega_end = min(mega_start + ENCODING_MEGA_BATCH_SIZE, total_texts)
                    batch_texts = texts[mega_start:mega_end]

                    update_heartbeat(f"encoding_batch_{mega_start}-{mega_end}")
                    logger.info(
                        f"[CACHE] Mega-batch {mega_batch_num}/{total_mega_batches}: "
                        f"codificando textos {mega_start + 1}-{mega_end}"
                    )

                    # Codificar este mega-batch
                    vecs = model.encode(
                        batch_texts,
                        normalize_embeddings=True,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        convert_to_tensor=True,
                    )

                    # Convertir a numpy inmediatamente para liberar memoria GPU
                    if torch.is_tensor(vecs):
                        vecs = vecs.float().cpu().numpy()

                    all_vecs.append(vecs)

                    # Limpiar caché GPU entre mega-batches
                    if "cuda" in device.lower() and mega_batch_num < total_mega_batches:
                        torch.cuda.empty_cache()
                        gc.collect()

                # Concatenar todos los resultados
                return np.concatenate(all_vecs, axis=0)
            else:
                # Codificación por batch único original para conjuntos de texto más pequeños
                vecs = model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                )

                # Convertir a array numpy float32
                if torch.is_tensor(vecs):
                    vecs = vecs.float().cpu().numpy()

                return vecs

        except RuntimeError as e:
            error_msg = str(e)

            # Si es error de CUDA, intentar fallback a CPU
            if "CUDA" in error_msg or "cuda" in error_msg:
                logger.error(f"[CACHE] Codificación GPU falló: {e}")
                logger.warning("[CACHE] Haciendo fallback a codificación en CPU...")

                # Limpiar estado de GPU antes del fallback
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass

                # Marcar GPU como fallida
                self.gpu_failed = True

                try:
                    # Mover modelo a CPU
                    model = model.cpu()
                    if hasattr(model, "half"):
                        model = model.float()  # Volver a float32

                    # Reintentar en CPU con mega-batching para conjuntos grandes
                    if use_mega_batching:
                        all_vecs = []
                        for mega_batch_num, mega_start in enumerate(
                            range(0, total_texts, ENCODING_MEGA_BATCH_SIZE), 1
                        ):
                            mega_end = min(
                                mega_start + ENCODING_MEGA_BATCH_SIZE, total_texts
                            )
                            batch_texts = texts[mega_start:mega_end]

                            update_heartbeat(
                                f"cpu_encoding_batch_{mega_start}-{mega_end}"
                            )
                            logger.info(
                                f"[CACHE] CPU mega-batch {mega_batch_num}/{total_mega_batches}: "
                                f"encoding texts {mega_start + 1}-{mega_end}"
                            )

                            vecs = model.encode(
                                batch_texts,
                                normalize_embeddings=True,
                                batch_size=batch_size,
                                show_progress_bar=True,
                                convert_to_tensor=True,
                            )

                            if torch.is_tensor(vecs):
                                vecs = vecs.float().cpu().numpy()

                            all_vecs.append(vecs)
                            gc.collect()

                        logger.info("[CACHE] Successfully encoded on CPU")
                        return np.concatenate(all_vecs, axis=0)
                    else:
                        vecs = model.encode(
                            texts,
                            normalize_embeddings=True,
                            batch_size=batch_size,
                            show_progress_bar=True,
                            convert_to_tensor=True,
                        )

                        if torch.is_tensor(vecs):
                            vecs = vecs.float().cpu().numpy()

                        logger.info("[CACHE] Successfully encoded on CPU")
                        return vecs

                except Exception as cpu_error:
                    logger.error(f"[CACHE] CPU fallback also failed: {cpu_error}")
                    raise
            else:
                # Otro tipo de error, propagar
                raise


# Instancia global de caché de modelos
model_cache = ModelCache()


def topic_collection(topic: str) -> str:
    return f"rag_{topic.lower()}"


def ensure_qdrant(topic: str, d: int):
    coll = topic_collection(topic)

    if not client.collection_exists(collection_name=coll):
        logger.info(f"[QDRANT] Creating collection '{coll}' with dimension {d}")
        client.create_collection(
            collection_name=coll,
            vectors_config=models.VectorParams(size=d, distance=models.Distance.COSINE),
        )
    else:
        # Verifica si la dimensión coincide
        try:
            collection_info = client.get_collection(collection_name=coll)
            existing_dim = collection_info.config.params.vectors.size

            if existing_dim != d:
                logger.warning(
                    f"[QDRANT] Dimension mismatch for '{coll}': existing={existing_dim}, new={d}"
                )
                logger.warning(f"[QDRANT] Recreating collection with dimension {d}")
                client.delete_collection(collection_name=coll)
                client.create_collection(
                    collection_name=coll,
                    vectors_config=models.VectorParams(
                        size=d, distance=models.Distance.COSINE
                    ),
                )
                logger.info(
                    f"[QDRANT] Collection '{coll}' recreated with dimension {d}"
                )
            else:
                logger.info(
                    f"[QDRANT] Collection '{coll}' exists with correct dimension {d}"
                )
        except Exception as e:
            logger.error(f"[QDRANT] Error checking collection '{coll}': {e}")
            raise


def ensure_whoosh(topic: str):
    path = os.path.join(BM25_BASE_DIR, topic)
    os.makedirs(path, exist_ok=True)
    if not index.exists_in(path):
        logger.info(f"[WHOOSH] Creating index at {path}")
        schema = Schema(
            file_path=ID(stored=True),
            page=NUMERIC(stored=True),
            chunk_id=NUMERIC(stored=True),
            text=TEXT(stored=True),
            chunk_type=TEXT(stored=True),
            source=TEXT(stored=True),
        )
        index.create_in(path, schema)
    else:
        logger.info(f"[WHOOSH] Index at {path} already exists")


def process_docling_elements(
    elements: list[dict[str, Any]], pdf_path: str, total_pages: int = None
) -> list[dict[str, Any]]:
    """
    Convierte elementos Docling al formato de chunks
    Mantiene compatibilidad con el pipeline existente

    Args:
        elements: Lista de elementos del servicio Docling
        pdf_path: Ruta al archivo PDF
        total_pages: Número total de páginas en PDF (para validación)
    """
    chunks = []

    for i, elem in enumerate(elements):
        # Validar y acotar número de página
        page = elem.get("page", 1)
        if not isinstance(page, int):
            try:
                page = int(page)
            except (ValueError, TypeError):
                logger.warning(
                    f"[DOCLING] Invalid page type in element {i}: {page}, using 1"
                )
                page = 1

        # Acotar número de página al rango válido
        if total_pages:
            if page < 1 or page > total_pages:
                original_page = page
                page = max(1, min(page, total_pages))
                logger.warning(
                    f"[DOCLING] Invalid page {original_page} in element {i}, clamped to {page} "
                    f"(doc has {total_pages} pages)"
                )
        else:
            # If no total_pages, at least ensure page >= 1
            if page < 1:
                logger.warning(f"[DOCLING] Invalid page {page} in element {i}, using 1")
                page = 1

        chunk = {
            "text": elem["text"],
            "type": elem["type"],
            "page": page,  # Use validated page number
            "chunk_id": i,
            "file_path": pdf_path,
            "source": "docling",
            "bbox": elem.get("bbox"),
            "metadata": elem.get("metadata", {}),
        }

        # Para tablas e imágenes, todavía puedes aplicar análisis LLaVA
        # si es necesario (tu lógica existente)

        chunks.append(chunk)

    return chunks


def index_pdf(topic: str, pdf_path: str):
    """Indexar un archivo PDF a Qdrant y Whoosh"""

    if state.is_already_processed(pdf_path):
        return True

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting indexing: {Path(pdf_path).name}")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

    # ============================================================
    # EXTRAER TOTAL DE PÁGINAS: Para validación de número de página
    # ============================================================
    total_pages = None
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        logger.info(f"PDF has {total_pages} pages")
    except Exception as e:
        logger.warning(f"Could not determine page count: {e}")

    # Configuración del dispositivo de embeddings
    device = EMBEDDING_DEVICE
    logger.info(f"Device: {device}")

    if device == "cuda":
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        except Exception as e:
            logger.warning(f"CUDA info error: {e}")

    # ============================================================
    # USAR CACHÉ DE MODELOS: Obtener modelo desde caché
    # ============================================================
    embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
    logger.info(f"Loading embedding model: {embed_name}")

    try:
        model = model_cache.get_model(embed_name, device=device)
        dims = model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {dims}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load embedding model: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    ensure_qdrant(topic, dims)
    ensure_whoosh(topic)

    logger.info(f"Extracting chunks from PDF...")

    if docling_client:
        try:
            logger.info(f"[EXTRACTION] Trying Docling for {Path(pdf_path).name}")

            # Definir función de fallback
            def fallback_extraction(path):
                return pdf_to_chunks_with_enhanced_validation(str(path))

            # Extraer con Docling (con fallback automático)
            raw_elements = docling_client.extract_pdf_sync(
                Path(pdf_path), fallback_func=fallback_extraction
            )

            # Convertir a chunks con validación de páginas
            chunks = process_docling_elements(raw_elements, pdf_path, total_pages)

            logger.info(f"[DOCLING] ✓ Processed into {len(chunks)} chunks")

        except DoclingCrashLimitExceeded as e:
            # El archivo ha fallado docling demasiadas veces - usar unstructured (preserva números de página)
            logger.warning(f"[DOCLING] {e}")
            logger.info(
                "[DOCLING] Using unstructured extraction (page-accurate fallback)"
            )
            chunks = pdf_to_chunks_with_enhanced_validation(pdf_path)

        except Exception as e:
            logger.error(f"[DOCLING] Failed: {e}", exc_info=True)
            logger.info("[DOCLING] Using unstructured extraction (fallback)")
            chunks = pdf_to_chunks_with_enhanced_validation(pdf_path)
    else:
        # Docling deshabilitado, usar método original
        logger.info(f"[EXTRACTION] Usando método original (Docling deshabilitado)")
        try:
            chunks = pdf_to_chunks_with_enhanced_validation(pdf_path)

        except Exception as e:
            logger.error(f"[ERROR] Failed to extract text from PDF: {e}", exc_info=True)
            state.mark_as_failed(pdf_path, str(e))
            return False

    logger.info(f"[OK] Extracted {len(chunks)} chunks")
    text_count = sum(1 for c in chunks if c.get("type") == "text")
    table_count = sum(1 for c in chunks if c.get("type") == "table")
    image_count = sum(1 for c in chunks if c.get("type") == "image")
    logger.info(f"  - Text: {text_count}")
    logger.info(f"  - Tables: {table_count}")
    logger.info(f"  - Images: {image_count}")

    texts = [c["text"] for c in chunks]

    logger.info(f"Encoding {len(texts)} chunks...")
    try:
        # Usa inferencia en GPU si está disponible
        # ✅ IMPORTANTE: intfloat/multilingual-e5-large-instruct requiere prefijos específicos
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)

        if "e5" in embed_name.lower():
            # Usa prefijo de documento para indexación
            texts_to_encode = [
                f"Represent this document for retrieval: {text}" for text in texts
            ]
            logger.info("[E5] Usando prefijo 'Represent this document for retrieval:'")
        else:
            texts_to_encode = texts
            logger.info(f"[EMBED] No prefix needed for {embed_name}")

        vecs = model_cache.encode_with_gpu(
            model,
            texts_to_encode,  # ← CON PREFIXES
            batch_size=32,
        )

        # CRÍTICO: Validar y corregir vectores antes de enviar a Qdrant
        vecs = validate_and_fix_vectors(vecs, dims)

        # DEBUG: Verificar dimensión y tipo
        if len(vecs) > 0:
            logger.info(f"[OK] Codificados {len(vecs)} vectores")
            logger.info(f"[DEBUG] Tipo de vector: {type(vecs)}")
            logger.info(f"[DEBUG] Tipo del primer vector: {type(vecs[0])}")
            logger.info(f"[DEBUG] Dimensión del vector: {len(vecs[0])}")
            logger.info(f"[DEBUG] Tipo del primer valor: {type(vecs[0][0])}")
        else:
            logger.warning("[DEBUG] Vector vacío")

    except Exception as e:
        logger.error(f"[ERROR] Failed to encode chunks: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    payloads = []
    for idx, c in enumerate(chunks):
        # Validar número de página contra los límites del PDF
        page = c.get("page", 1)

        # Convertir a int si es necesario
        if not isinstance(page, int):
            try:
                page = int(page)
            except (ValueError, TypeError):
                logger.warning(
                    f"[MAIN] Tipo de página inválido {type(page)} en chunk {idx}, usando 1"
                )
                page = 1

        # Validar rango de página [1, total_pages]
        if total_pages:
            if page < 1 or page > total_pages:
                original_page = page
                page = max(1, min(page, total_pages))
                logger.warning(
                    f"[MAIN] Page {original_page} out of range [1-{total_pages}] in chunk {idx}, "
                    f"clamped to {page}"
                )
        else:
            # If no total_pages, at least ensure page >= 1
            if page < 1:
                logger.warning(f"[MAIN] Invalid page {page} in chunk {idx}, using 1")
                page = 1

        payloads.append(
            {
                "file_path": pdf_path,
                "page": page,
                "chunk_id": idx,
                "text": c["text"],
                "chunk_type": c.get("type", "text"),
                "source": c.get("source", "unknown"),
                "content_id": c.get("image_id") or c.get("table_id") or str(idx),
            }
        )

    total_chunks = len(vecs)
    logger.info(
        f"Upserting {total_chunks} vectors to Qdrant in batches of {QDRANT_BATCH_SIZE}..."
    )

    try:
        for batch_start in range(0, total_chunks, QDRANT_BATCH_SIZE):
            batch_end = min(batch_start + QDRANT_BATCH_SIZE, total_chunks)
            batch_ids = [
                abs(hash(f"{pdf_path}:{i}")) % (2**31)
                for i in range(batch_start, batch_end)
            ]
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            # Crear lista de PointStruct válidos
            points = [
                models.PointStruct(
                    id=batch_ids[i],
                    vector=batch_vecs[i],
                    payload=batch_payloads[i],
                )
                for i in range(len(batch_vecs))
            ]

            # Upsert moderno compatible con Qdrant >= 1.9
            client.upsert(
                collection_name=topic_collection(topic),
                points=points,
                wait=True,
            )
            batch_num = batch_start // QDRANT_BATCH_SIZE + 1
            total_batches = (total_chunks + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE
            logger.info(f"  [QDRANT] Batch {batch_num}/{total_batches}")

        logger.info(f"[OK] All vectors uploaded to Qdrant")
    except Exception as e:
        logger.error(f"[ERROR] Failed to upsert to Qdrant: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    logger.info(f"Indexing {len(chunks)} chunks to Whoosh (BM25)...")
    try:
        idx = index.open_dir(os.path.join(BM25_BASE_DIR, topic))
        writer = idx.writer(limitmb=512, procs=0, multisegment=True)

        for i, c in enumerate(chunks):
            # Validar número de página para indexación BM25 (misma validación que Qdrant)
            page = c.get("page", 1)

            # Convertir a int si es necesario
            if not isinstance(page, int):
                try:
                    page = int(page)
                except (ValueError, TypeError):
                    logger.warning(
                        f"[MAIN] Tipo de página inválido {type(page)} en chunk {i} para BM25, usando 1"
                    )
                    page = 1

            # Validar rango de página [1, total_pages]
            if total_pages:
                if page < 1 or page > total_pages:
                    original_page = page
                    page = max(1, min(page, total_pages))
                    logger.warning(
                        f"[MAIN] BM25: Page {original_page} out of range [1-{total_pages}] in chunk {i}, "
                        f"clamped to {page}"
                    )
            else:
                # If no total_pages, at least ensure page >= 1
                if page < 1:
                    logger.warning(
                        f"[MAIN] BM25: Invalid page {page} in chunk {i}, using 1"
                    )
                    page = 1

            writer.update_document(
                file_path=pdf_path,
                page=page,
                chunk_id=i,
                text=c["text"],
                chunk_type=c.get("type", "text"),
                source=c.get("source", "unknown"),
            )

        writer.commit()
        logger.info(f"[OK] All chunks indexed to Whoosh")
    except Exception as e:
        logger.error(f"[ERROR] Failed to index to Whoosh: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    logger.info(f"{'=' * 60}")
    logger.info(f"[SUCCESS] {Path(pdf_path).name}")
    logger.info(f"  - Total Chunks: {len(chunks)}")
    logger.info(f"  - Vectors: {len(vecs)}")
    logger.info(f"  - Topic: {topic}")
    logger.info(f"  - Collection: {topic_collection(topic)}")
    logger.info(f"{'=' * 60}\n")

    state.mark_as_processed(pdf_path, topic)
    return True


def initial_scan():
    """Escanear todos los directorios de topic e indexar PDFs"""
    # Configurar callback de heartbeat para operaciones largas en chunk.py
    set_heartbeat_callback(update_heartbeat)

    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO SCAN INICIAL")
    logger.info("=" * 60)
    logger.info(f"TOPIC_BASE_DIR: {TOPIC_BASE_DIR}")
    logger.info(f"BM25_BASE_DIR: {BM25_BASE_DIR}")
    logger.info(f"QDRANT_URL: {QDRANT_URL}")

    stats = state.get_stats()
    logger.info(f"\n[STATE] Previous processing state:")
    logger.info(f"  - Total files processed: {stats['total_processed']}")
    logger.info(f"  - Successful: {stats['successful']}")
    logger.info(f"  - Failed: {stats['failed']}")
    if stats["last_scan"]:
        logger.info(f"  - Last scan: {stats['last_scan']}")

    logger.info(f"\nPyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception:
            pass

    logger.info(f"\nTopics to scan: {', '.join(TOPIC_LABELS)}")
    logger.info("=" * 60 + "\n")

    pdf_count = 0
    skipped_count = 0
    error_count = 0
    start_time = datetime.now().timestamp()

    for t in TOPIC_LABELS:
        tdir = os.path.join(TOPIC_BASE_DIR, t)
        logger.info(f"Scanning topic directory: {tdir}")
        os.makedirs(tdir, exist_ok=True)

        pdfs = glob.glob(os.path.join(tdir, "*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {t}")

        for pdf in pdfs:
            abs_pdf = os.path.abspath(pdf)

            if state.is_already_processed(abs_pdf):
                skipped_count += 1
                continue

            pdf_count += 1
            # Actualizar heartbeat antes de cada PDF - permite que healthcheck detecte contenedores atascados
            update_heartbeat(os.path.basename(abs_pdf))
            try:
                success = index_pdf(t, abs_pdf)
                if not success:
                    error_count += 1
            except Exception as e:
                logger.error(
                    f"[ERROR] Unexpected error processing {abs_pdf}: {e}",
                    exc_info=True,
                )
                logger.warning(f"[SKIP] Continuing with next file...")
                state.mark_as_failed(abs_pdf, str(e))
                error_count += 1

    elapsed_time = datetime.now().timestamp() - start_time
    state.update_scan_time()

    logger.info("\n" + "=" * 60)
    logger.info("SCAN INICIAL COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"Nuevos PDFs procesados: {pdf_count}")
    logger.info(f"PDFs omitidos (ya procesados): {skipped_count}")
    logger.info(f"Errores: {error_count}")
    logger.info(f"Tiempo transcurrido: {elapsed_time:.2f} segundos")
    if pdf_count > 0:
        logger.info(
            f"Tiempo promedio por nuevo PDF: {elapsed_time / pdf_count:.2f} segundos"
        )
    logger.info("=" * 60 + "\n")


def delete_pdf_from_indexes(topic: str, pdf_path: str):
    """
    Borra un PDF específico de Qdrant y Whoosh
    """
    pdf_path = str(pdf_path)
    logger.info(f"\n{'=' * 60}")
    logger.info("DELETING PDF FROM INDEXES")
    logger.info(f"PDF: {Path(pdf_path).name}")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

    # ============================================================
    # BORRAR DE QDRANT
    # ============================================================
    coll = topic_collection(topic)

    try:
        logger.info(
            f"Consultando colección Qdrant '{coll}' para puntos con file_path={pdf_path}"
        )

        # Buscar TODOS los puntos con este file_path
        points_result = client.scroll(
            collection_name=coll,
            limit=10000,  # Asumir máx 10k puntos por archivo
            with_payload=True,
        )

        points = points_result[0]  # scroll retorna (points, next_page_offset)

        point_ids_to_delete = []
        for point in points:
            if point.payload.get("file_path") == pdf_path:
                point_ids_to_delete.append(point.id)

        if point_ids_to_delete:
            logger.info(f"Found {len(point_ids_to_delete)} points in Qdrant")
            client.delete(
                collection_name=coll,
                points_selector=point_ids_to_delete,
            )
            logger.info(f"✅ Deleted {len(point_ids_to_delete)} points from Qdrant")
        else:
            logger.warning(f"No points found in Qdrant for {Path(pdf_path).name}")

    except Exception as e:
        logger.error(f"Error deleting from Qdrant: {e}", exc_info=True)
        return False

    # ============================================================
    # BORRAR DE WHOOSH
    # ============================================================
    try:
        logger.info(f"Consultando índice Whoosh para file_path={pdf_path}")

        idx_path = os.path.join(BM25_BASE_DIR, topic)
        idx = index.open_dir(idx_path)
        writer = idx.writer()

        # Borrar todos los documentos con este file_path
        deleted_count = writer.delete_by_term("file_path", pdf_path)

        writer.commit()

        logger.info(f"✅ Deleted {deleted_count} documents from Whoosh")

    except Exception as e:
        logger.error(f"Error deleting from Whoosh: {e}", exc_info=True)
        return False

    # ============================================================
    # ACTUALIZAR ESTADO
    # ============================================================
    try:
        state.state["processed"].pop(pdf_path, None)
        state._save_state()
        logger.info(f"✅ Reset processing state for {Path(pdf_path).name}")
    except Exception as e:
        logger.error(f"Error updating state: {e}")

    logger.info(f"{'=' * 60}")
    logger.info("[SUCCESS] PDF deleted from all indexes")
    logger.info(f"{'=' * 60}\n")

    return True


def delete_all_files_from_topic(topic: str):
    """
    Borra TODOS los archivos de un topic específico de Qdrant y Whoosh
    Reinicia completamente el topic para poder re-indexar desde cero
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("DELETING ALL FILES FROM TOPIC")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

    # ============================================================
    # VERIFICAR QUE EL TOPIC EXISTA
    # ============================================================
    if topic not in TOPIC_LABELS:
        logger.error(
            f"Topic '{topic}' no encontrado en los topics configurados: {TOPIC_LABELS}"
        )
        return False

    coll = topic_collection(topic)

    # ============================================================
    # BORRAR COLECCIÓN COMPLETA DE QDRANT
    # ============================================================
    try:
        if client.collection_exists(collection_name=coll):
            # Obtener información antes de borrar
            collection_info = client.get_collection(collection_name=coll)
            points_count = collection_info.points_count
            logger.info(
                f"Encontrados {points_count} puntos en colección Qdrant '{coll}'"
            )

            # Borrar colección completa
            client.delete_collection(collection_name=coll)
            logger.info(f"✅ Borrada colección completa Qdrant '{coll}'")
        else:
            logger.warning(f"Colección Qdrant '{coll}' no existe")
            points_count = 0

    except Exception as e:
        logger.error(f"Error deleting Qdrant collection '{coll}': {e}", exc_info=True)
        return False

    # ============================================================
    # BORRAR ÍNDICE COMPLETO DE WHOOSH
    # ============================================================
    try:
        idx_path = os.path.join(BM25_BASE_DIR, topic)
        if os.path.exists(idx_path):
            # Contar documentos antes de borrar
            try:
                idx = index.open_dir(idx_path)
                with idx.searcher() as searcher:
                    doc_count = searcher.doc_count()
                logger.info(
                    f"Encontrados {doc_count} documentos en índice Whoosh '{idx_path}'"
                )
            except Exception:
                doc_count = 0
                logger.warning(
                    f"No se pudieron contar documentos en índice Whoosh '{idx_path}'"
                )

            # Borrar carpeta completa del índice
            import shutil

            shutil.rmtree(idx_path)
            logger.info(f"✅ Borrado índice completo Whoosh '{idx_path}'")
        else:
            logger.warning(f"Índice Whoosh '{idx_path}' no existe")
            doc_count = 0

    except Exception as e:
        logger.error(f"Error deleting Whoosh index '{idx_path}': {e}", exc_info=True)
        return False

    # ============================================================
    # LIMPIAR ESTADO DE PROCESAMIENTO PARA ESTE TOPIC
    # ============================================================
    try:
        files_to_remove = []
        for file_path, file_info in state.state["processed"].items():
            if file_info.get("topic") == topic:
                files_to_remove.append(file_path)

        # Remover archivos procesados de este topic
        removed_count = 0
        for file_path in files_to_remove:
            state.state["processed"].pop(file_path, None)
            removed_count += 1

        # También limpiar archivos fallidos de este topic
        failed_to_remove = []
        for file_path in list(state.state["failed"].keys()):
            # Verificar si la ruta del archivo pertenece al directorio de este topic
            topic_dir = os.path.join(TOPIC_BASE_DIR, topic)
            if (
                topic_dir in file_path
            ):  # Verifica que el archivo esté en el directorio del topic
                state.state["failed"].pop(file_path, None)
                failed_to_remove.append(file_path)

        state._save_state()
        logger.info(f"✅ Removidos {removed_count} archivos procesados del estado")
        if failed_to_remove:
            logger.info(
                f"✅ Removidos {len(failed_to_remove)} archivos fallidos del estado"
            )

    except Exception as e:
        logger.error(f"Error updating processing state: {e}", exc_info=True)
        return False

    # ============================================================
    # RECONSTRUIR ESTRUCTURA VACÍA (para que esté listo para re-indexar)
    # ============================================================
    try:
        # Recrear carpeta de Whoosh vacía
        ensure_whoosh(topic)

        # Obtener dimensión del modelo de embedding para este topic
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
        try:
            # Cargar modelo temporalmente para obtener dimensión
            temp_model = model_cache.get_model(embed_name, device="cpu")
            dims = temp_model.get_sentence_embedding_dimension()
            logger.info(f"Dimensión de embedding para topic '{topic}': {dims}")

            # Recrear colección de Qdrant vacía
            ensure_qdrant(topic, dims)
            logger.info(f"✅ Recreada colección vacía Qdrant '{coll}'")
        except Exception as e:
            logger.warning(f"No se pudo recrear colección Qdrant: {e}")
            logger.warning(
                f"Se creará automáticamente cuando se indexe el primer archivo"
            )

    except Exception as e:
        logger.error(f"Error recreating empty structures: {e}", exc_info=True)

    logger.info(f"{'=' * 60}")
    logger.info("[SUCCESS] All files deleted from topic")
    logger.info(f"  - Topic: {topic}")
    logger.info(f"  - Qdrant points: {points_count}")
    logger.info(f"  - Whoosh documents: {doc_count}")
    logger.info(f"  - Processed files removed: {removed_count}")
    logger.info(
        f"  - Failed files removed: {len(failed_to_remove) if 'failed_to_remove' in locals() else 0}"
    )
    logger.info(f"  - Empty structures recreated: Ready for re-indexing")
    logger.info(f"{'=' * 60}\n")

    return True


# ============================================================
# CLI: Para ejecutar manualmente
# ============================================================
"""
Comandos disponibles:

1. Borrar un archivo específico:
docker exec -it ingestor python /app/main.py delete Electricidad "/topics/Electricidad/archivo.pdf"
Ejemplo:
docker exec -it ingestor python /app/main.py delete Electricidad "/topics/Electricidad/Normas de Construcción de cuadros de automatización.pdf"

2. Borrar TODOS los archivos de un topic:
docker exec -it ingestor python /app/main.py delete-topic Programming
Ejemplo:
docker exec -it ingestor python /app/main.py delete-topic Chemistry

3. Escanear e indexar todo (default):
docker exec -it ingestor python /app/main.py
"""
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "delete":
            # Uso: python main.py delete "Electricidad" "/topics/Electricidad/archivo.pdf"
            if len(sys.argv) < 4:
                print("Usage: python main.py delete <topic> <pdf_path>")
                print(
                    "Example: python main.py delete Electricidad /topics/Electricidad/Step7.pdf"
                )
                sys.exit(1)

            topic = sys.argv[2]
            pdf_path = sys.argv[3]

            success = delete_pdf_from_indexes(topic, pdf_path)
            sys.exit(0 if success else 1)

        elif command == "delete-topic":
            # Uso: python main.py delete-topic "Electricidad"
            if len(sys.argv) < 3:
                print("Usage: python main.py delete-topic <topic>")
                print("Available topics:", ", ".join(TOPIC_LABELS))
                print("Example: python main.py delete-topic Chemistry")
                sys.exit(1)

            topic = sys.argv[2]

            print(
                f"⚠️  ADVERTENCIA: Esto borrará TODOS los archivos del topic '{topic}'"
            )
            print(f"   Esta acción no se puede deshacer!")
            print(f"   Topics disponibles: {', '.join(TOPIC_LABELS)}")

            # Pedir confirmación
            """
            try:
                confirm = input(f"Type 'DELETE {topic.upper()}' to confirm: ").strip()
                if confirm != f"DELETE {topic.upper()}":
                    print("❌ Operation cancelled")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Operation cancelled")
                sys.exit(1)
            """
            success = delete_all_files_from_topic(topic)
            sys.exit(0 if success else 1)

        else:
            print(f"Comando desconocido: {command}")
            print("Comandos disponibles:")
            print("  delete <topic> <pdf_path>     - Borrar un PDF específico")
            print(
                "  delete-topic <topic>          - Borrar TODOS los archivos de un topic"
            )
            print(
                "  (sin comando)                 - Escanear e indexar todos los topics"
            )
            sys.exit(1)

    else:
        # Default: escanear e indexar
        # Iniciar watchdog para detectar procesos atascados (corrupción glibc, etc.)
        start_watchdog()
        initial_scan()
