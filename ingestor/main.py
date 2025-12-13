import glob
import hashlib
import json
import logging
import os
from chunk import pdf_to_chunks_with_enhanced_validation
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from docling_client import DoclingClient
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from settings import *
from whoosh import index
from whoosh.fields import ID, NUMERIC, TEXT, Schema

# ============================================================
# CRITICAL: Configure model cache directory BEFORE imports
# Use only HF_HOME (TRANSFORMERS_CACHE is deprecated in v5)
# ============================================================
MODEL_CACHE_DIR = "/models_cache"
os.environ["HF_HOME"] = MODEL_CACHE_DIR

# Create cache directory if it doesn't exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ============================================================
# EMBEDDINGS DEVICE: Todo en GPU con float16
# - Modelo en GPU con float16 (~650MB vs 1.3GB en float32)
# - Inference en GPU
# - Total: LLaVA 16.7GB + embeddings 650MB = ~17.4GB (cabe)
# ============================================================
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DTYPE = "float16"

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"[CONFIG] Embedding device: {EMBEDDING_DEVICE}")
logger.info(f"[CACHE] Model cache directory: {MODEL_CACHE_DIR}")
logger.info(f"[CACHE] HF_HOME: {os.environ['HF_HOME']}")

client = QdrantClient(url=QDRANT_URL)

# docling initialiation
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling-service:8003")
ENABLE_DOCLING = os.getenv("ENABLE_DOCLING", "true").lower() == "true"

docling_client = (
    DoclingClient(docling_url=DOCLING_URL, enable_fallback=True)
    if ENABLE_DOCLING
    else None
)

logger.info(
    f"[CONFIG] Docling extraction: {'ENABLED' if ENABLE_DOCLING else 'DISABLED'}"
)


# ============================================================
# STATE TRACKING: Rastrear archivos ya procesados
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
# VECTOR VALIDATION: Asegurar vectores válidos para Qdrant
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
            logger.warning(f"Vector {i} has wrong dimension: {len(vec)} != {dims}")
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
# MODEL CACHE: Carga modelos con caché persistente
# ============================================================
class ModelCache:
    """Gestiona caché de modelos SentenceTransformer"""

    def __init__(self):
        self.models = {}
        self.cache_info_file = os.path.join(MODEL_CACHE_DIR, ".model_cache_info.json")
        self.use_gpu = torch.cuda.is_available()
        self.gpu_failed = False  # Track si GPU falló
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
        """Encode textos con GPU/CPU fallback automático"""
        device = str(model.device)
        logger.info(f"[CACHE] Encoding {len(texts)} texts on {device}")

        try:
            # Intenta encoding en dispositivo actual
            vecs = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
            )

            # Convierte a float32 numpy array
            if torch.is_tensor(vecs):
                vecs = vecs.float().cpu().numpy()

            return vecs

        except RuntimeError as e:
            error_msg = str(e)

            # Si es error de CUDA, intentar fallback a CPU
            if "CUDA" in error_msg or "cuda" in error_msg:
                logger.error(f"[CACHE] GPU encoding failed: {e}")
                logger.warning("[CACHE] Falling back to CPU encoding...")

                # Marcar GPU como fallida
                self.gpu_failed = True

                try:
                    # Mover modelo a CPU
                    model = model.cpu()
                    if hasattr(model, "half"):
                        model = model.float()  # Volver a float32

                    # Reintentar en CPU
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
    Convert Docling elements to your chunk format
    Preserves compatibility with existing pipeline

    Args:
        elements: List of elements from Docling service
        pdf_path: Path to PDF file
        total_pages: Total number of pages in PDF (for validation)
    """
    chunks = []

    for i, elem in enumerate(elements):
        # Validate and clamp page number
        page = elem.get("page", 1)
        if not isinstance(page, int):
            try:
                page = int(page)
            except (ValueError, TypeError):
                logger.warning(
                    f"[DOCLING] Invalid page type in element {i}: {page}, using 1"
                )
                page = 1

        # Clamp page number to valid range
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

        # For tables and images, you can still apply LLaVA analysis
        # if needed (your existing logic)

        chunks.append(chunk)

    return chunks


def index_pdf(topic: str, pdf_path: str, vllm_url: str = None, cache_db: str = None):
    """Index a single PDF file to both Qdrant and Whoosh"""

    if state.is_already_processed(pdf_path):
        return True

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting indexing: {Path(pdf_path).name}")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

    # ============================================================
    # EXTRACT TOTAL PAGES: For page number validation
    # ============================================================
    total_pages = None
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        logger.info(f"PDF has {total_pages} pages")
    except Exception as e:
        logger.warning(f"Could not determine page count: {e}")

    # Use CPU for embeddings (vLLM-LLaVA uses GPU)
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
    # USE MODEL CACHE: Obtener modelo desde caché
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
    vllm_url = vllm_url or VLLM_URL
    cache_db = cache_db or LLAVA_CACHE_DB

    if docling_client:
        try:
            logger.info(f"[EXTRACTION] Trying Docling for {Path(pdf_path).name}")

            # Define fallback function
            def fallback_extraction(path):
                return pdf_to_chunks_with_enhanced_validation(
                    str(path), vllm_url=vllm_url, cache_db=cache_db
                )

            # Extract with Docling (with automatic fallback)
            raw_elements = docling_client.extract_pdf_sync(
                Path(pdf_path), fallback_func=fallback_extraction
            )

            # Convert to chunks with page validation
            chunks = process_docling_elements(raw_elements, pdf_path, total_pages)

            logger.info(f"[DOCLING] ✓ Processed into {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"[DOCLING] Failed completely: {e}", exc_info=True)
            logger.info("[DOCLING] Using original extraction method")

            # Final fallback to original method
            chunks = pdf_to_chunks_with_enhanced_validation(
                pdf_path, vllm_url=vllm_url, cache_db=cache_db
            )
    else:
        # Docling disabled, use original method
        logger.info(f"[EXTRACTION] Using original method (Docling disabled)")
        try:
            chunks = pdf_to_chunks_with_enhanced_validation(
                pdf_path, vllm_url=vllm_url, cache_db=cache_db
            )

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
        # Usa GPU inference si está disponible
        # ✅ IMPORTANTE: intfloat/multilingual-e5-large-instruct requiere prefixes específicos
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)

        if "e5" in embed_name.lower():
            # Usa prefix de documento para indexación
            texts_to_encode = [
                f"Represent this document for retrieval: {text}" for text in texts
            ]
            logger.info("[E5] Using 'Represent this document for retrieval:' prefix")
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

        # DEBUG: Verifica dimensión y tipo
        if len(vecs) > 0:
            logger.info(f"[OK] Encoded {len(vecs)} vectors")
            logger.info(f"[DEBUG] Vector type: {type(vecs)}")
            logger.info(f"[DEBUG] First vector type: {type(vecs[0])}")
            logger.info(f"[DEBUG] Vector dimension: {len(vecs[0])}")
            logger.info(f"[DEBUG] First value type: {type(vecs[0][0])}")
        else:
            logger.warning("[DEBUG] Empty vector")

    except Exception as e:
        logger.error(f"[ERROR] Failed to encode chunks: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    payloads = []
    for idx, c in enumerate(chunks):
        # Validate page number against PDF boundaries
        page = c.get("page", 1)

        # Convert to int if needed
        if not isinstance(page, int):
            try:
                page = int(page)
            except (ValueError, TypeError):
                logger.warning(
                    f"[MAIN] Invalid page type {type(page)} in chunk {idx}, using 1"
                )
                page = 1

        # Validate page range [1, total_pages]
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
            # Validate page number for BM25 indexing (same validation as Qdrant)
            page = c.get("page", 1)

            # Convert to int if needed
            if not isinstance(page, int):
                try:
                    page = int(page)
                except (ValueError, TypeError):
                    logger.warning(
                        f"[MAIN] Invalid page type {type(page)} in chunk {i} for BM25, using 1"
                    )
                    page = 1

            # Validate page range [1, total_pages]
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
    """Scan all topic directories and index PDFs"""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING INITIAL SCAN")
    logger.info("=" * 60)
    logger.info(f"TOPIC_BASE_DIR: {TOPIC_BASE_DIR}")
    logger.info(f"BM25_BASE_DIR: {BM25_BASE_DIR}")
    logger.info(f"QDRANT_URL: {QDRANT_URL}")

    vllm_url = os.getenv("VLLM_URL", "http://vllm-llava:8000")
    cache_db = os.getenv("LLAVA_CACHE_DB", "/tmp/llava_cache/llava_cache.db")
    logger.info(f"VLLM_URL: {vllm_url}")
    logger.info(f"LLAVA_CACHE_DB: {cache_db}")

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
            try:
                success = index_pdf(t, abs_pdf, vllm_url=vllm_url, cache_db=cache_db)
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
    logger.info("INITIAL SCAN COMPLETED")
    logger.info("=" * 60)
    logger.info(f"New PDFs processed: {pdf_count}")
    logger.info(f"PDFs skipped (already processed): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    if pdf_count > 0:
        logger.info(f"Average time per new PDF: {elapsed_time / pdf_count:.2f} seconds")
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
            f"Querying Qdrant collection '{coll}' for points with file_path={pdf_path}"
        )

        # Busca TODOS los puntos con este file_path
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
        logger.info(f"Querying Whoosh index for file_path={pdf_path}")

        idx_path = os.path.join(BM25_BASE_DIR, topic)
        idx = index.open_dir(idx_path)
        writer = idx.writer()

        # Borra todos los documentos con este file_path
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
        logger.error(f"Topic '{topic}' not found in configured topics: {TOPIC_LABELS}")
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
            logger.info(f"Found {points_count} points in Qdrant collection '{coll}'")

            # Borrar colección completa
            client.delete_collection(collection_name=coll)
            logger.info(f"✅ Deleted entire Qdrant collection '{coll}'")
        else:
            logger.warning(f"Qdrant collection '{coll}' does not exist")
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
                logger.info(f"Found {doc_count} documents in Whoosh index '{idx_path}'")
            except Exception:
                doc_count = 0
                logger.warning(
                    f"Could not count documents in Whoosh index '{idx_path}'"
                )

            # Borrar carpeta completa del índice
            import shutil

            shutil.rmtree(idx_path)
            logger.info(f"✅ Deleted entire Whoosh index '{idx_path}'")
        else:
            logger.warning(f"Whoosh index '{idx_path}' does not exist")
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
            # Check if the file path belongs to this topic's directory
            topic_dir = os.path.join(TOPIC_BASE_DIR, topic)
            if (
                topic_dir in file_path
            ):  # Verifica que el archivo esté en el directorio del topic
                state.state["failed"].pop(file_path, None)
                failed_to_remove.append(file_path)

        state._save_state()
        logger.info(f"✅ Removed {removed_count} processed files from state")
        if failed_to_remove:
            logger.info(f"✅ Removed {len(failed_to_remove)} failed files from state")

    except Exception as e:
        logger.error(f"Error updating processing state: {e}", exc_info=True)
        return False

    # ============================================================
    # RECOSNCTRUIR ESTRUCTURA VACÍA (para que esté listo para re-indexar)
    # ============================================================
    try:
        # Recrear carpeta de Whoosh vacía
        ensure_whoosh(topic)

        # Obtener dimensión del embedding model para este topic
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
        try:
            # Cargar modelo temporalmente para obtener dimensión
            temp_model = model_cache.get_model(embed_name, device="cpu")
            dims = temp_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension for topic '{topic}': {dims}")

            # Recrear colección de Qdrant vacía
            ensure_qdrant(topic, dims)
            logger.info(f"✅ Recreated empty Qdrant collection '{coll}'")
        except Exception as e:
            logger.warning(f"Could not recreate Qdrant collection: {e}")
            logger.warning(f"Will be created automatically when first file is indexed")

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

            print(f"⚠️  WARNING: This will delete ALL files from topic '{topic}'")
            print(f"   This action cannot be undone!")
            print(f"   Available topics: {', '.join(TOPIC_LABELS)}")

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
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  delete <topic> <pdf_path>     - Delete a specific PDF")
            print("  delete-topic <topic>          - Delete ALL files from topic")
            print("  (no command)                  - Scan and index all topics")
            sys.exit(1)

    else:
        # Default: scan y indexa
        initial_scan()
