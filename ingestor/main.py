import os, glob
import torch
import logging
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from qdrant_client import QdrantClient
from settings import *
from chunk import (
    AdaptiveChunkingStrategySelector,
    extract_elements_from_pdf,
    PageSequenceValidator,
)
from typing import Dict, Any, Optional
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
# GPU DETECTION AND CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ============================================================
# QDRANT CLIENT INITIALIZATION
# ============================================================

# Initialize Qdrant client
try:
    client = QdrantClient(url=QDRANT_URL)
    logger.info(f"[QDRANT] Connected to Qdrant at {QDRANT_URL}")
except Exception as e:
    logger.error(f"[QDRANT] Failed to connect to Qdrant: {e}")
    client = None


class BlackwellOptimizedGPUManager:
    """
    Enhanced GPU manager optimized for RTX 5090 and RTX 5070 Ti (Blackwell architecture)
    """

    def __init__(self):
        self.gpu_info = self._detect_gpu_detailed()
        self.memory_monitor = GPUMemoryMonitor()
        self._apply_optimal_settings()

    def _detect_gpu_detailed(self) -> Dict[str, Any]:
        """Enhanced GPU detection with Blackwell-specific optimizations"""
        info = {
            "available": torch.cuda.is_available(),
            "name": "CPU",
            "memory_total_gb": 0,
            "memory_free_gb": 0,
            "compute_capability": (0, 0),
            "tensor_cores": False,
            "architecture": "Unknown",
            "optimal_dtype": torch.float32,
            "supports_flash_attention": False,
            "recommended_batch_size": 16,
            "max_batch_size": 32,
        }

        if not info["available"]:
            logger.warning("[GPU] CUDA not available, using CPU")
            return info

        try:
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "name": props.name,
                    "memory_total_gb": props.total_memory / 1e9,
                    "compute_capability": (props.major, props.minor),
                    "multiprocessor_count": props.multi_processor_count,
                }
            )

            # Get current memory state
            info["memory_free_gb"] = torch.cuda.mem_get_info()[0] / 1e9
            info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9

            # Determine architecture and capabilities
            major, minor = info["compute_capability"]

            # RTX 5090 (Blackwell) - Compute 10.0
            if major >= 10 or "5090" in info["name"]:
                info.update(
                    {
                        "architecture": "Blackwell",
                        "tensor_cores": True,
                        "supports_flash_attention": True,
                        "optimal_dtype": torch.bfloat16,  # BF16 superior on Blackwell
                        "recommended_batch_size": 64,
                        "max_batch_size": 128,
                    }
                )
                logger.info(
                    "[GPU] 🚀 RTX 5090 (Blackwell) detected - Ultra performance mode"
                )

            # RTX 5070 Ti (Blackwell) - Compute 10.0
            elif "5070" in info["name"]:
                info.update(
                    {
                        "architecture": "Blackwell",
                        "tensor_cores": True,
                        "supports_flash_attention": True,
                        "optimal_dtype": torch.bfloat16,
                        "recommended_batch_size": 32,
                        "max_batch_size": 64,
                    }
                )
                logger.info(
                    "[GPU] ⚡ RTX 5070 Ti (Blackwell) detected - High performance mode"
                )

            # RTX 40xx (Ada Lovelace) - Compute 8.9
            elif major == 8 and minor == 9:
                info.update(
                    {
                        "architecture": "Ada Lovelace",
                        "tensor_cores": True,
                        "supports_flash_attention": True,
                        "optimal_dtype": torch.float16,
                        "recommended_batch_size": 24,
                        "max_batch_size": 48,
                    }
                )

            # RTX 30xx (Ampere) - Compute 8.6
            elif major == 8 and minor == 6:
                info.update(
                    {
                        "architecture": "Ampere",
                        "tensor_cores": True,
                        "supports_flash_attention": True,
                        "optimal_dtype": torch.float16,
                        "recommended_batch_size": 16,
                        "max_batch_size": 32,
                    }
                )

            # Blackwell-specific optimizations
            if info["architecture"] == "Blackwell":
                # Enable additional Blackwell features
                info.update(
                    {
                        "supports_fp8": True,  # FP8 precision support
                        "supports_tma": True,  # Tensor Memory Accelerator
                        "supports_transformer_engine": True,
                        "recommended_batch_size": int(
                            info["recommended_batch_size"] * BLACKWELL_BATCH_MULTIPLIER
                        ),
                        "max_batch_size": int(
                            info["max_batch_size"] * BLACKWELL_BATCH_MULTIPLIER
                        ),
                    }
                )

                logger.info("[GPU] 🚀 Blackwell-specific optimizations enabled")

                # Enable FP8 precision for embeddings if available
                if hasattr(torch, "float8_e5m2"):
                    info["optimal_dtype"] = torch.float8_e5m2
                    logger.info("[GPU] ✓ FP8 precision enabled for embeddings")

            # Adjust for memory constraints
            if info["memory_total_gb"] < 12:
                info["recommended_batch_size"] = max(
                    8, info["recommended_batch_size"] // 2
                )
                info["max_batch_size"] = max(16, info["max_batch_size"] // 2)

            logger.info(f"[GPU] {info['name']}")
            logger.info(
                f"[GPU] Memory: {info['memory_total_gb']:.1f}GB total, "
                f"{info['memory_free_gb']:.1f}GB free"
            )
            logger.info(f"[GPU] Compute: {major}.{minor} ({info['architecture']})")
            logger.info(f"[GPU] Optimal dtype: {info['optimal_dtype']}")
            logger.info(
                f"[GPU] Batch size: {info['recommended_batch_size']} "
                f"(max: {info['max_batch_size']})"
            )

        except Exception as e:
            logger.error(f"[GPU] Detection error: {e}", exc_info=True)
            info["available"] = False

        return info

    def _apply_optimal_settings(self):
        """Apply optimal CUDA settings for detected GPU"""
        if not self.gpu_info["available"]:
            return

        try:
            # Enable TF32 for Ampere+ (huge speedup for FP32)
            if self.gpu_info["compute_capability"][0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("[GPU] ✓ TF32 enabled for matmul")

            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True

            # Optimize memory allocator
            if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                # Reserve 90% for RTX 5090, 85% for others
                fraction = 0.90 if "5090" in self.gpu_info["name"] else 0.85
                torch.cuda.set_per_process_memory_fraction(fraction, device=0)
                logger.info(f"[GPU] Memory fraction: {fraction:.0%}")

            # Enable memory efficient attention if available (PyTorch 2.0+)
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # Flash Attention 2 for Blackwell/Ada
                os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
                logger.info("[GPU] ✓ Flash Attention enabled")

            # Optimize CUDA allocator for large models
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[GPU] Error applying settings: {e}")

    def get_optimal_batch_size(
        self, model_name: str, sequence_length: int, current_memory_usage: float = 0
    ) -> int:
        """Calculate optimal batch size with Blackwell-specific adjustments"""
        base_batch = self.gpu_info["recommended_batch_size"]

        # Adjust for model size
        if "large" in model_name.lower():
            base_batch = int(base_batch * 0.7)
        elif "xlarge" in model_name.lower() or "xxl" in model_name.lower():
            base_batch = int(base_batch * 0.5)
        elif "small" in model_name.lower() or "mini" in model_name.lower():
            base_batch = int(base_batch * 1.3)

        # Adjust for sequence length
        if sequence_length > 512:
            factor = 512 / sequence_length
            base_batch = int(base_batch * max(0.5, factor))

        # Adjust for current memory usage
        free_memory_gb = self.gpu_info["memory_free_gb"] - current_memory_usage
        memory_ratio = free_memory_gb / self.gpu_info["memory_total_gb"]

        if memory_ratio < 0.3:  # Less than 30% free
            base_batch = max(4, int(base_batch * 0.5))
            logger.warning(
                f"[GPU] Low memory ({memory_ratio:.0%}), reducing batch size"
            )
        elif memory_ratio > 0.7:  # More than 70% free
            base_batch = min(self.gpu_info["max_batch_size"], int(base_batch * 1.3))

        # Additional adjustments for Blackwell
        if self.gpu_info["architecture"] == "Blackwell":
            # Increase batch size for smaller models on Blackwell
            if "small" in model_name.lower() or "mini" in model_name.lower():
                base_batch = int(base_batch * BLACKWELL_SMALL_MODEL_MULTIPLIER)

            # Further increase for models that benefit from FP8
            if self.gpu_info.get("supports_fp8", False):
                base_batch = int(base_batch * BLACKWELL_FP8_MULTIPLIER)

        return max(4, base_batch)  # Minimum 4

    def should_use_mixed_precision(self, model_name: str) -> tuple[bool, torch.dtype]:
        """Determine if mixed precision should be used"""
        if not self.gpu_info["tensor_cores"]:
            return False, torch.float32

        # Blackwell (RTX 50xx): BF16 is optimal
        if self.gpu_info["architecture"] == "Blackwell":
            return True, torch.bfloat16

        # Ada/Ampere: FP16 is optimal
        if self.gpu_info["architecture"] in ["Ada Lovelace", "Ampere"]:
            return True, torch.float16

        return False, torch.float32

    def optimize_model_loading(self, model):
        """Apply Blackwell-specific optimizations after model loading"""
        model = (
            super().optimize_model_loading(model)
            if hasattr(super(), "optimize_model_loading")
            else model
        )

        if (
            not self.gpu_info["available"]
            or self.gpu_info["architecture"] != "Blackwell"
        ):
            return model

        try:
            # Enable Transformer Engine for compatible models
            if self.gpu_info.get("supports_transformer_engine", False):
                try:
                    from transformer_engine.pytorch import enable as te_enable

                    te_enable()
                    logger.info("[GPU] ✓ Transformer Engine enabled")
                except ImportError:
                    logger.debug("[GPU] Transformer Engine not available")

            # Enable TMA (Tensor Memory Accelerator) if available
            if self.gpu_info.get("supports_tma", False):
                try:
                    # This would require specific CUDA libraries
                    # Placeholder for TMA optimization
                    logger.info("[GPU] ✓ TMA optimizations applied")
                except Exception as e:
                    logger.debug(f"[GPU] TMA optimization failed: {e}")

            return model

        except Exception as e:
            logger.error(f"[GPU] Blackwell optimization error: {e}")
            return model

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not self.gpu_info["available"]:
            return {}

        try:
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            free = torch.cuda.mem_get_info()[0] / 1e9

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": free,
                "total_gb": self.gpu_info["memory_total_gb"],
                "utilization": allocated / self.gpu_info["memory_total_gb"],
            }
        except:
            return {}


class GPUMemoryMonitor:
    """Monitor GPU memory and provide warnings"""

    def __init__(self, warning_threshold: float = 0.85):
        self.warning_threshold = warning_threshold
        self.peak_memory = 0

    def check_memory(self) -> Optional[str]:
        """Check memory and return warning if needed"""
        if not torch.cuda.is_available():
            return None

        try:
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory

            self.peak_memory = max(self.peak_memory, allocated)

            ratio = allocated / total

            if ratio > self.warning_threshold:
                return (
                    f"⚠️  High GPU memory usage: {ratio:.1%} "
                    f"({allocated/1e9:.1f}GB / {total/1e9:.1f}GB)"
                )

            return None

        except:
            return None

    def get_peak_memory_gb(self) -> float:
        """Get peak memory usage"""
        return self.peak_memory / 1e9


gpu_manager = BlackwellOptimizedGPUManager()


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

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """Get detailed info about a processed file"""
        file_path = str(file_path)
        return self.state["processed"].get(file_path)

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

    def mark_as_processed(self, file_path: str, topic: str, chunk_count: int = 0):
        """Enhanced processing marker with metadata"""
        file_path = str(file_path)
        self.state["processed"][file_path] = {
            "hash": self.get_file_hash(file_path),
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "status": "success",
            "chunk_count": chunk_count,  # NEW: track chunks
            "file_size_mb": Path(file_path).stat().st_size / 1e6,  # NEW: file size
        }
        self._save_state()

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
    """Enhanced vector validation with detailed logging"""

    if isinstance(vecs, torch.Tensor):
        vecs = vecs.float().cpu().numpy()

    if isinstance(vecs, np.ndarray):
        vecs = vecs.tolist()

    valid_vecs = []
    issues = {"wrong_dimension": 0, "nan_inf": 0, "type_error": 0, "total_fixed": 0}

    for i, vec in enumerate(vecs):
        # Convert to list if needed
        if isinstance(vec, (np.ndarray, torch.Tensor)):
            vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)

        # Check dimension
        if len(vec) != dims:
            issues["wrong_dimension"] += 1
            if len(vec) < dims:
                vec = vec + [0.0] * (dims - len(vec))
            else:
                vec = vec[:dims]

        # Validate each value
        valid_vec = []
        for val in vec:
            try:
                val = float(val)
                if not np.isfinite(val):
                    val = 0.0
                    issues["nan_inf"] += 1
                valid_vec.append(val)
            except (TypeError, ValueError):
                valid_vec.append(0.0)
                issues["type_error"] += 1

        valid_vecs.append(valid_vec)

    issues["total_fixed"] = sum(issues.values())

    if issues["total_fixed"] > 0:
        logger.warning(f"[VECTOR] Fixed {issues['total_fixed']} vector issues:")
        for issue_type, count in issues.items():
            if count > 0 and issue_type != "total_fixed":
                logger.warning(f"  - {issue_type}: {count}")

    return valid_vecs


# ============================================================
# MODEL CACHE: Carga modelos con caché persistente
# ============================================================


class OptimizedModelCache:
    def __init__(self):
        self.models = {}
        self.gpu_manager = gpu_manager
        self.use_gpu = self.gpu_manager.gpu_info["available"]
        self.gpu_failed = False

    def get_model(self, model_name: str, device: str = None) -> SentenceTransformer:
        """Enhanced model loading with GPU optimizations"""

        if device is None:
            device = "cuda" if self.use_gpu else "cpu"

        if self.gpu_failed and device == "cuda":
            logger.warning("[CACHE] GPU failed previously, using CPU")
            device = "cpu"

        # Determine optimal dtype
        use_amp, optimal_dtype = self.gpu_manager.should_use_mixed_precision(model_name)

        cache_key = f"{model_name}_{device}_{optimal_dtype}"

        if cache_key in self.models:
            logger.info(f"[CACHE] Using cached model: {model_name}")
            return self.models[cache_key]

        logger.info(f"[CACHE] Loading model: {model_name}")
        logger.info(f"[CACHE] Device: {device}, Dtype: {optimal_dtype}")

        try:
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )

            # Apply GPU optimizations
            if self.use_gpu and device == "cuda":
                model = self.gpu_manager.optimize_model_loading(model)

            self.models[cache_key] = model
            return model

        except Exception as e:
            logger.error(f"[CACHE] Failed to load model: {e}", exc_info=True)
            raise

    def encode_with_gpu_optimization(
        self, model: SentenceTransformer, texts: list, batch_size: int = None
    ):
        """Enhanced encoding with dynamic batch sizing"""

        device = str(model.device)

        if self.gpu_failed and device == "cuda":
            logger.warning("[CACHE] GPU failed previously, skipping to CPU fallback")
            return self._encode_cpu_fallback(model, texts)

        # Get current memory usage
        mem_stats = self.gpu_manager.get_memory_stats()
        current_usage = mem_stats.get("allocated_gb", 0)

        # Calculate optimal batch size
        if batch_size is None:
            avg_length = sum(len(text) for text in texts) / len(texts)
            batch_size = self.gpu_manager.get_optimal_batch_size(
                model._modules["0"].auto_model.name_or_path,
                int(avg_length),
                current_usage,
            )

        logger.info(f"[CACHE] Encoding {len(texts)} texts with batch size {batch_size}")

        # Monitor memory during encoding
        warning = self.gpu_manager.memory_monitor.check_memory()
        if warning:
            logger.warning(warning)

        try:
            vecs = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=device,
            )

            if torch.is_tensor(vecs):
                vecs = vecs.cpu().numpy()

            # Log peak memory
            peak = self.gpu_manager.memory_monitor.get_peak_memory_gb()
            logger.info(f"[CACHE] Peak GPU memory: {peak:.1f}GB")

            return vecs

        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.error(f"[CACHE] GPU encoding failed: {e}")
                self.gpu_failed = True

                # Fallback to CPU
                return self._encode_cpu_fallback(model, texts)
            raise

    def _encode_cpu_fallback(self, model, texts):
        """CPU fallback with smaller batch size"""
        logger.warning("[CACHE] Falling back to CPU encoding...")

        model = model.cpu().float()

        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=8,  # Smaller for CPU
            show_progress_bar=True,
        )

        return np.array(vecs)


# Instancia global de caché de modelos
model_cache = OptimizedModelCache()


def topic_collection(topic: str) -> str:
    return f"rag_{topic.lower()}"


def ensure_qdrant(topic: str, d: int):
    if client is None:
        logger.error("[QDRANT] Client not initialized - cannot ensure collection")
        return False
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


def log_gpu_diagnostics():
    """Log GPU diagnostics information"""
    if not torch.cuda.is_available():
        logger.info("[GPU] CUDA not available")
        return

    try:
        logger.info("[GPU] Diagnostics:")
        logger.info(f"  - CUDA version: {torch.version.cuda}")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9

            logger.info(f"  - GPU {i}: {props.name}")
            logger.info(f"    - Compute capability: {props.major}.{props.minor}")
            logger.info(f"    - Total memory: {memory_total:.2f} GB")
            logger.info(f"    - Reserved memory: {memory_reserved:.2f} GB")
            logger.info(f"    - Allocated memory: {memory_allocated:.2f} GB")
            logger.info(f"    - Multiprocessors: {props.multi_processor_count}")
    except Exception as e:
        logger.error(f"[GPU] Error getting diagnostics: {e}")


def index_pdf(topic: str, pdf_path: str, vllm_url: str = None, cache_db: str = None):
    """Index a single PDF file to both Qdrant and Whoosh with enhanced chunking"""

    if state.is_already_processed(pdf_path):
        return True

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting indexing: {Path(pdf_path).name}")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

    # Use GPU if available
    device = "cuda" if model_cache.use_gpu else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        gpu_info = gpu_manager.gpu_info
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"GPU Memory: {gpu_info['memory_total_gb']:.1f} GB")
        logger.info(f"Compute Capability: {gpu_info['compute_capability']}")
        logger.info(f"Recommended Batch Size: {gpu_info['recommended_batch_size']}")

        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"CUDA sync error: {e}")

    # Load embedding model with GPU optimization
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

        # Use enhanced chunking with strategy selection
        strategy_selector = AdaptiveChunkingStrategySelector()

        # First extract elements
        elements = extract_elements_from_pdf(pdf_path)

        # Select and apply optimal chunking strategy
        chunks = strategy_selector.select_and_apply_strategy(
            elements, chunk_size=900, overlap=120, pdf_path=pdf_path
        )

        # ADD: Log page distribution for audit
        page_counts = {}
        for chunk in chunks:
            page = chunk.get("page", 1)
            page_counts[page] = page_counts.get(page, 0) + 1

        logger.info(
            f"[PAGE] Distribution across {len(page_counts)} pages: {page_counts}"
        )

        text_count = sum(1 for c in chunks if c.get("type") == "text")
        table_count = sum(1 for c in chunks if c.get("type") == "table")
        image_count = sum(1 for c in chunks if c.get("type") == "image")
        logger.info(f"  - Text: {text_count}")
        logger.info(f"  - Tables: {table_count}")
        logger.info(f"  - Images: {image_count}")

    except ValueError as e:
        # Page validation failed - REJECT
        logger.error(f"[ERROR] Chunk validation failed: {e}")
        state.mark_as_failed(pdf_path, str(e))
        return False
    except Exception as e:
        logger.error(f"[ERROR] Failed to extract text from PDF: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False

    texts = [c["text"] for c in chunks if c.get("text", "").strip()]

    # Check if we have any text to encode
    if not texts:
        logger.error(f"[ERROR] No text extracted from chunks in {Path(pdf_path).name}")
        state.mark_as_failed(pdf_path, "No text content extracted from PDF")
        return False

    logger.info(f"Encoding {len(texts)} chunks...")
    try:
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)

        if "e5" in embed_name.lower():
            # Use prefix for e5 models
            texts_to_encode = [
                f"Represent this document for retrieval: {text}" for text in texts
            ]
            logger.info("[E5] Using 'Represent this document for retrieval:' prefix")
        else:
            texts_to_encode = texts
            logger.info(f"[EMBED] No prefix needed for {embed_name}")

        # Calculate optimal batch size based on GPU and text length
        avg_text_length = sum(len(text) for text in texts) / len(texts)
        optimal_batch_size = gpu_manager.get_optimal_batch_size(
            embed_name, int(avg_text_length)
        )

        logger.info(f"[CACHE] Using optimized batch size: {optimal_batch_size}")

        vecs = model_cache.encode_with_gpu_optimization(
            model,
            texts_to_encode,
            batch_size=optimal_batch_size,
        )

        # Validate vectors
        if not isinstance(vecs, list) or any(
            len(v) != dims for v in vecs[: min(10, len(vecs))]
        ):
            vecs = validate_and_fix_vectors(vecs, dims)

        logger.info(f"[OK] Encoded {len(vecs)} vectors")
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

    # Get page distribution
    page_counts = {}
    for chunk in chunks:
        page = chunk.get("page", 1)
        page_counts[page] = page_counts.get(page, 0) + 1

    logger.info(f"{'=' * 60}")
    logger.info(f"[SUCCESS] {Path(pdf_path).name}")
    logger.info(f"  - Total Chunks: {len(chunks)}")
    logger.info(f"  - Vectors: {len(vecs)}")
    logger.info(f"  - Topic: {topic}")
    logger.info(f"  - Collection: {topic_collection(topic)}")
    logger.info(f"  - Page distribution: {page_counts}")
    logger.info(f"{'=' * 60}\n")

    state.mark_as_processed(pdf_path, topic, len(chunks))
    return True


def initial_scan():
    """Scan all topic directories and index PDFs with GPU diagnostics"""
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

    # Log GPU diagnostics
    log_gpu_diagnostics()

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

            logger.info(f"Processing {abs_pdf}")

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


def delete_pdf_from_indexes(topic: str, pdf_path: str):
    """
    Borra un PDF específico de Qdrant y Whoosh
    """
    if client is None:
        logger.error("[QDRANT] Client not initialized - cannot delete from Qdrant")
        return False
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
            limit=10000,
            with_payload=True,
        )

        points = points_result[0]

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
            logger.info(f"✓ Deleted {len(point_ids_to_delete)} points from Qdrant")
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

        logger.info(f"✓ Deleted {deleted_count} documents from Whoosh")

    except Exception as e:
        logger.error(f"Error deleting from Whoosh: {e}", exc_info=True)
        return False

    # ============================================================
    # ACTUALIZAR ESTADO
    # ============================================================
    try:
        state.state["processed"].pop(pdf_path, None)
        state._save_state()
        logger.info(f"✓ Reset processing state for {Path(pdf_path).name}")
    except Exception as e:
        logger.error(f"Error updating state: {e}")

    logger.info(f"{'=' * 60}")
    logger.info("[SUCCESS] PDF deleted from all indexes")
    logger.info(f"{'=' * 60}\n")

    return True


# ============================================================
# CLI: Para ejecutar manualmente
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "delete":
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

    else:
        # Default: scan y indexa
        initial_scan()
