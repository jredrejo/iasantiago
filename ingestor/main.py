import os, glob
import torch
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from settings import *
from chunk import pdf_to_chunks
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
import time

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

        start_time = time.time()
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

            elapsed = time.time() - start_time
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


def index_pdf(topic: str, pdf_path: str, vllm_url: str = None, cache_db: str = None):
    """Index a single PDF file to both Qdrant and Whoosh"""

    if state.is_already_processed(pdf_path):
        return True

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting indexing: {Path(pdf_path).name}")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

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

    try:
        logger.info(f"Extracting chunks from PDF...")
        vllm_url = vllm_url or os.getenv("VLLM_URL", "http://vllm-llava:8000")
        cache_db = cache_db or os.getenv(
            "LLAVA_CACHE_DB", "/tmp/llava_cache/llava_cache.db"
        )

        chunks = pdf_to_chunks(pdf_path, vllm_url=vllm_url, cache_db=cache_db)
        logger.info(f"[OK] Extracted {len(chunks)} chunks")

        text_count = sum(1 for c in chunks if c.get("type") == "text")
        table_count = sum(1 for c in chunks if c.get("type") == "table")
        image_count = sum(1 for c in chunks if c.get("type") == "image")
        logger.info(f"  - Text: {text_count}")
        logger.info(f"  - Tables: {table_count}")
        logger.info(f"  - Images: {image_count}")

    except Exception as e:
        logger.error(f"[ERROR] Failed to extract text from PDF: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    texts = [c["text"] for c in chunks]

    logger.info(f"Encoding {len(texts)} chunks...")
    try:
        # Usa GPU inference si está disponible
        vecs = model_cache.encode_with_gpu(
            model,
            texts,
            batch_size=32,
        )

        # Asegúrate de que sean listas Python puras (no tensores)
        if torch.is_tensor(vecs):
            vecs = vecs.float().cpu().numpy().tolist()
        elif isinstance(vecs, type(torch.tensor([]))):  # numpy array
            vecs = vecs.tolist()
        elif not isinstance(vecs, list):
            vecs = vecs.tolist()

        # DEBUG: Verifica dimensión y tipo
        logger.info(f"[OK] Encoded {len(vecs)} vectors")
        logger.info(f"[DEBUG] Vector type: {type(vecs)}")
        logger.info(f"[DEBUG] First vector type: {type(vecs[0])}")
        logger.info(f"[DEBUG] Vector dimension: {len(vecs[0])}")

    except Exception as e:
        logger.error(f"[ERROR] Failed to encode chunks: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    payloads = []
    for idx, c in enumerate(chunks):
        payloads.append(
            {
                "file_path": pdf_path,
                "page": c.get("page", 1),
                "chunk_id": idx,
                "text": c["text"],
                "chunk_type": c.get("type", "text"),
                "source": c.get("source", "unknown"),
                "content_id": c.get("image_id") or c.get("table_id") or str(idx),
            }
        )

    QDRANT_BATCH_SIZE = 100
    total_chunks = len(vecs)
    logger.info(
        f"Upserting {total_chunks} vectors to Qdrant in batches of {QDRANT_BATCH_SIZE}..."
    )

    try:
        for batch_start in range(0, total_chunks, QDRANT_BATCH_SIZE):
            batch_end = min(batch_start + QDRANT_BATCH_SIZE, total_chunks)
            batch_ids = list(range(batch_start + 1, batch_end + 1))
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            client.upsert(
                collection_name=topic_collection(topic),
                points=models.Batch(
                    ids=batch_ids, vectors=batch_vecs, payloads=batch_payloads
                ),
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
            writer.update_document(
                file_path=pdf_path,
                page=c.get("page", 1),
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
        except:
            pass

    logger.info(f"\nTopics to scan: {', '.join(TOPIC_LABELS)}")
    logger.info("=" * 60 + "\n")

    pdf_count = 0
    skipped_count = 0
    error_count = 0
    start_time = time.time()

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

    elapsed_time = time.time() - start_time
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


if __name__ == "__main__":
    initial_scan()
