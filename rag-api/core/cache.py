# Archivo: rag-api/core/cache.py
# Descripción: Caché unificado para modelos de ML (embedders, rerankers, traductores)

import asyncio
import logging
import os
from typing import Dict, Optional, Any

from transformers import AutoTokenizer

from config.settings import VLLM_MODEL

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Caché centralizado para todos los modelos de ML.

    Proporciona acceso singleton a:
    - Embedders (SentenceTransformer) por tema
    - Reranker (CrossEncoder)
    - Traductores (MarianMT)
    - Tokenizador (transformers - correcto para el modelo)

    Uso:
        embedder = ModelCache.get_embedder("Chemistry")
        reranker = ModelCache.get_reranker()
        count = ModelCache.count_tokens("texto")
    """

    _embedders: Dict[str, Any] = {}
    _reranker: Optional[Any] = None
    _tokenizer: Optional[Any] = None  # transformers tokenizer
    _lock = asyncio.Lock()

    @classmethod
    def _get_device(cls) -> str:
        """Dispositivo de los embedders, unificado con RERANK_DEVICE (rerank.py).

        Se lee de EMBEDDING_DEVICE ('cuda' o 'cpu'). Por defecto 'cpu': rag-api
        comparte la GPU con vLLM (gpu_mem_util 0.95) y no queda VRAM, el mismo
        motivo por el que RERANK_DEVICE=cpu. Ponerlo a 'cuda' sólo si vLLM está
        parado o hay VRAM libre confirmada.
        """
        return os.getenv("EMBEDDING_DEVICE", "cpu")

    @classmethod
    def get_tokenizer(cls) -> Any:
        """Obtiene el tokenizador correcto para el modelo (singleton)"""
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL)
            logger.debug(f"Tokenizador {VLLM_MODEL} inicializado")
        return cls._tokenizer

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Cuenta tokens usando el tokenizador correcto del modelo"""
        tokenizer = cls.get_tokenizer()
        return len(tokenizer.encode(text))

    @classmethod
    def get_embedder(cls, topic: str, embed_config: Dict[str, str], default_model: str):
        """
        Obtiene o carga un embedder para un tema específico.

        Args:
            topic: Nombre del tema (ej: "Chemistry")
            embed_config: Diccionario de modelos por tema (EMBED_PER_TOPIC)
            default_model: Modelo por defecto si el tema no tiene uno específico

        Returns:
            SentenceTransformer configurado para el tema
        """
        from sentence_transformers import SentenceTransformer
        from config.settings import get_model_revision

        model_name = embed_config.get(topic, default_model)

        if model_name not in cls._embedders:
            try:
                device = cls._get_device()
                revision = get_model_revision(model_name)
                logger.info(
                    f"Cargando embedder: {model_name} en {device} "
                    f"(revision={revision or 'default'})"
                )

                embedder = SentenceTransformer(
                    model_name,
                    revision=revision,
                    trust_remote_code=True,
                    device=device,
                )
                cls._embedders[model_name] = embedder
                logger.info(f"Embedder {model_name} cargado exitosamente")

            except Exception as e:
                logger.error(
                    f"Error al cargar embedder {model_name}: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"No se pudo cargar el modelo de embedding {model_name}: {e}"
                )

        return cls._embedders[model_name]

    @classmethod
    def get_reranker(cls, model_name: str):
        """
        Obtiene o carga el reranker (singleton).

        Args:
            model_name: Nombre del modelo de reranking

        Returns:
            CrossEncoderReranker configurado
        """
        from rerank import CrossEncoderReranker

        if cls._reranker is None:
            logger.info(f"Cargando reranker: {model_name}")
            cls._reranker = CrossEncoderReranker(model_name)
            logger.info(f"Reranker {model_name} cargado exitosamente")

        return cls._reranker

    @classmethod
    def clear_cache(cls, cache_type: Optional[str] = None):
        """
        Limpia la caché de modelos.

        Args:
            cache_type: Tipo de caché a limpiar ('embedders', 'reranker', 'all')
                       Si es None o 'all', limpia todo.
        """
        if cache_type in (None, "all", "embedders"):
            cls._embedders.clear()
            logger.info("Caché de embedders limpiada")

        if cache_type in (None, "all", "reranker"):
            cls._reranker = None
            logger.info("Caché de reranker limpiada")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Retorna estadísticas de uso de la caché"""
        return {
            "embedders_cargados": len(cls._embedders),
            "reranker_cargado": 1 if cls._reranker else 0,
        }
