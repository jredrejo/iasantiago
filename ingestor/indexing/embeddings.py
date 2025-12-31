"""
Servicio de embeddings para generación de vectores.

Proporciona embedding de texto acelerado por GPU con respaldo
automático a CPU y procesamiento de mega-lotes para documentos grandes.
"""

import gc
import logging
import os
from datetime import datetime
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Configuración
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models_cache")
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DTYPE = "float16"
ENCODING_MEGA_BATCH_SIZE = 5000


def validate_and_fix_vectors(
    vecs: Union[torch.Tensor, np.ndarray, List],
    dims: int,
) -> List[List[float]]:
    """
    Valida y corrige vectores para Qdrant.

    - Elimina valores NaN, Inf
    - Asegura tipo correcto (lista de floats)
    - Asegura dimensión correcta

    Args:
        vecs: Vectores a validar (tensor, ndarray, o lista)
        dims: Dimensión esperada

    Returns:
        Lista de listas de vectores validados
    """
    if isinstance(vecs, torch.Tensor):
        vecs = vecs.float().cpu().numpy()

    if isinstance(vecs, np.ndarray):
        vecs = vecs.tolist()

    if not isinstance(vecs, list):
        raise ValueError(f"Los vectores deben ser lista, recibido {type(vecs)}")

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
                f"Vector {i} contenía valores inválidos (NaN/Inf), reemplazados con 0.0"
            )

        valid_vecs.append(valid_vec)

    if invalid_count > 0:
        logger.warning(f"Corregidos {invalid_count}/{len(vecs)} vectores inválidos")

    return valid_vecs


class EmbeddingService:
    """
    Servicio de embedding acelerado por GPU con caché y respaldo.

    Características:
    - Caché de modelos para uso repetido
    - Respaldo automático GPU/CPU
    - Procesamiento de mega-lotes para documentos grandes
    - Optimización float16 para eficiencia de memoria GPU
    """

    def __init__(
        self,
        cache_dir: str = MODEL_CACHE_DIR,
        default_device: str = EMBEDDING_DEVICE,
    ):
        self.cache_dir = cache_dir
        self.default_device = default_device
        self._models: dict = {}
        self._gpu_failed = False

        # Asegurar que existe el directorio de caché
        os.makedirs(cache_dir, exist_ok=True)

        # Establecer variable de entorno HF_HOME
        os.environ["HF_HOME"] = cache_dir

        logger.info(
            f"[EMBED] Inicializado con dispositivo={default_device}, caché={cache_dir}"
        )

    @property
    def device(self) -> str:
        """Dispositivo actual (respeta estado de fallo GPU)."""
        if self._gpu_failed:
            return "cpu"
        return self.default_device

    def get_model(
        self,
        model_name: str,
        device: Optional[str] = None,
    ) -> SentenceTransformer:
        """
        Obtiene modelo de caché o lo carga.

        Args:
            model_name: Nombre del modelo HuggingFace
            device: Dispositivo a cargar (por defecto: self.device)

        Returns:
            Modelo SentenceTransformer cargado
        """
        if device is None:
            device = self.device

        # Si GPU falló antes, usar CPU
        if self._gpu_failed and device == "cuda":
            logger.warning("[EMBED] GPU falló previamente, usando CPU")
            device = "cpu"

        cache_key = f"{model_name}_{device}_{EMBEDDING_DTYPE}"

        if cache_key in self._models:
            logger.info(f"[EMBED] Usando modelo cacheado: {model_name}")
            return self._models[cache_key]

        logger.info(f"[EMBED] Cargando modelo: {model_name}")
        logger.info(f"[EMBED] Dispositivo: {device}, Dtype: {EMBEDDING_DTYPE}")

        start_time = datetime.now().timestamp()

        try:
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )

            # Convertir a float16 si es GPU y no falló
            if (
                device == "cuda"
                and EMBEDDING_DTYPE == "float16"
                and not self._gpu_failed
            ):
                try:
                    # Prueba simple para verificar que float16 funciona
                    test_tensor = torch.randn(1, 10).half().to(device)
                    _ = test_tensor * 2

                    model = model.half()
                    logger.info("[EMBED] Modelo convertido a float16")
                except Exception as e:
                    logger.warning(f"[EMBED] Conversión float16 falló: {e}")
                    logger.warning("[EMBED] Manteniendo modelo en float32")

            elapsed = datetime.now().timestamp() - start_time
            logger.info(f"[EMBED] Modelo cargado en {elapsed:.2f}s")

            self._models[cache_key] = model
            return model

        except Exception as e:
            logger.error(f"[EMBED] Error al cargar modelo {model_name}: {e}")
            raise

    def encode(
        self,
        model: SentenceTransformer,
        texts: List[str],
        batch_size: int = 32,
        heartbeat_callback: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """
        Codifica textos con respaldo automático GPU/CPU.

        Para conjuntos grandes de texto (>ENCODING_MEGA_BATCH_SIZE), procesa en
        mega-lotes con limpieza de caché GPU entre lotes para prevenir OOM.

        Args:
            model: Modelo SentenceTransformer
            texts: Textos a codificar
            batch_size: Tamaño de lote para codificación
            heartbeat_callback: Callback opcional para actualizaciones de heartbeat

        Returns:
            Array numpy de embeddings
        """
        device = str(model.device)
        total_texts = len(texts)
        logger.info(f"[EMBED] Codificando {total_texts} textos en {device}")

        use_mega_batching = total_texts > ENCODING_MEGA_BATCH_SIZE

        if use_mega_batching:
            total_mega_batches = (
                total_texts + ENCODING_MEGA_BATCH_SIZE - 1
            ) // ENCODING_MEGA_BATCH_SIZE
            logger.info(
                f"[EMBED] Conjunto grande de texto, procesando en {total_mega_batches} "
                f"mega-lotes de {ENCODING_MEGA_BATCH_SIZE}"
            )

        try:
            if use_mega_batching:
                return self._encode_mega_batches(
                    model,
                    texts,
                    batch_size,
                    device,
                    total_mega_batches,
                    heartbeat_callback,
                )
            else:
                return self._encode_single(model, texts, batch_size)

        except RuntimeError as e:
            error_msg = str(e)

            if "CUDA" in error_msg or "cuda" in error_msg:
                logger.error(f"[EMBED] Codificación GPU falló: {e}")
                logger.warning("[EMBED] Recurriendo a CPU...")

                # Limpiar estado GPU
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass

                self._gpu_failed = True

                # Reintentar en CPU
                return self._encode_cpu_fallback(
                    model,
                    texts,
                    batch_size,
                    use_mega_batching,
                    total_mega_batches if use_mega_batching else 0,
                    heartbeat_callback,
                )
            else:
                raise

    def _encode_single(
        self,
        model: SentenceTransformer,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        """Codifica un solo lote de textos."""
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        if torch.is_tensor(vecs):
            vecs = vecs.float().cpu().numpy()

        return vecs

    def _encode_mega_batches(
        self,
        model: SentenceTransformer,
        texts: List[str],
        batch_size: int,
        device: str,
        total_mega_batches: int,
        heartbeat_callback: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """Codifica textos en mega-lotes."""
        all_vecs = []
        total_texts = len(texts)

        for mega_batch_num, mega_start in enumerate(
            range(0, total_texts, ENCODING_MEGA_BATCH_SIZE), 1
        ):
            mega_end = min(mega_start + ENCODING_MEGA_BATCH_SIZE, total_texts)
            batch_texts = texts[mega_start:mega_end]

            if heartbeat_callback:
                heartbeat_callback(f"encoding_batch_{mega_start}-{mega_end}")

            logger.info(
                f"[EMBED] Mega-lote {mega_batch_num}/{total_mega_batches}: "
                f"codificando textos {mega_start + 1}-{mega_end}"
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

            # Limpiar caché GPU entre mega-lotes
            if "cuda" in device.lower() and mega_batch_num < total_mega_batches:
                torch.cuda.empty_cache()
                gc.collect()

        return np.concatenate(all_vecs, axis=0)

    def _encode_cpu_fallback(
        self,
        model: SentenceTransformer,
        texts: List[str],
        batch_size: int,
        use_mega_batching: bool,
        total_mega_batches: int,
        heartbeat_callback: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """Respaldo a codificación CPU."""
        try:
            # Mover modelo a CPU
            model = model.cpu()
            if hasattr(model, "half"):
                model = model.float()  # Volver a float32

            if use_mega_batching:
                all_vecs = []
                total_texts = len(texts)

                for mega_batch_num, mega_start in enumerate(
                    range(0, total_texts, ENCODING_MEGA_BATCH_SIZE), 1
                ):
                    mega_end = min(mega_start + ENCODING_MEGA_BATCH_SIZE, total_texts)
                    batch_texts = texts[mega_start:mega_end]

                    if heartbeat_callback:
                        heartbeat_callback(
                            f"cpu_encoding_batch_{mega_start}-{mega_end}"
                        )

                    logger.info(
                        f"[EMBED] Mega-lote CPU {mega_batch_num}/{total_mega_batches}: "
                        f"codificando textos {mega_start + 1}-{mega_end}"
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

                logger.info("[EMBED] Codificación en CPU exitosa")
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

                logger.info("[EMBED] Codificación en CPU exitosa")
                return vecs

        except Exception as cpu_error:
            logger.error(f"[EMBED] Respaldo CPU también falló: {cpu_error}")
            raise

    def get_dimension(self, model: SentenceTransformer) -> int:
        """Obtiene dimensión de embedding para un modelo."""
        return model.get_sentence_embedding_dimension()

    def clear_cache(self) -> None:
        """Limpia caché de modelos y memoria GPU."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[EMBED] Caché de modelos limpiada")


# Instancia global
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Obtiene o crea el servicio de embedding global."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
