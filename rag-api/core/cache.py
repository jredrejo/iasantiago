# Archivo: rag-api/core/cache.py
# Descripción: Caché unificado para modelos de ML (embedders, rerankers, traductores)

import asyncio
import logging
import os
from typing import Dict, Optional, Tuple, Any

import tiktoken

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Caché centralizado para todos los modelos de ML.

    Proporciona acceso singleton a:
    - Embedders (SentenceTransformer) por tema
    - Reranker (CrossEncoder)
    - Traductores (MarianMT)
    - Tokenizador (tiktoken)

    Uso:
        embedder = ModelCache.get_embedder("Chemistry")
        reranker = ModelCache.get_reranker()
        count = ModelCache.count_tokens("texto")
    """

    _embedders: Dict[str, Any] = {}
    _reranker: Optional[Any] = None
    _translators: Dict[str, Tuple[Any, Any, str]] = {}
    _tokenizer: Optional[tiktoken.Encoding] = None
    _lock = asyncio.Lock()

    @classmethod
    def _get_device(cls) -> str:
        """Determina el dispositivo disponible (CUDA o CPU)"""
        return "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"

    @classmethod
    def get_tokenizer(cls) -> tiktoken.Encoding:
        """Obtiene el tokenizador tiktoken (singleton)"""
        if cls._tokenizer is None:
            cls._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.debug("Tokenizador tiktoken inicializado")
        return cls._tokenizer

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Cuenta tokens en un texto usando tiktoken"""
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

        model_name = embed_config.get(topic, default_model)

        if model_name not in cls._embedders:
            try:
                device = cls._get_device()
                logger.info(f"Cargando embedder: {model_name} en {device}")

                embedder = SentenceTransformer(
                    model_name,
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
    def get_translator(
        cls, source_lang: str, target_lang: str = "en"
    ) -> Optional[Tuple[Any, Any, str]]:
        """
        Obtiene o carga un modelo de traducción.

        Args:
            source_lang: Código de idioma origen (ej: 'es')
            target_lang: Código de idioma destino (default: 'en')

        Returns:
            Tupla (tokenizer, model, device) o None si falla
        """
        from transformers import MarianMTModel, MarianTokenizer

        cache_key = f"{source_lang}-{target_lang}"

        if cache_key in cls._translators:
            return cls._translators[cache_key]

        try:
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            logger.info(f"Cargando traductor: {model_name}")

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            device = cls._get_device()
            model = model.to(device)

            cls._translators[cache_key] = (tokenizer, model, device)
            logger.info(f"Traductor {model_name} cargado en {device}")

            return tokenizer, model, device

        except Exception as e:
            logger.error(f"Error al cargar traductor {source_lang}->{target_lang}: {e}")
            return None

    @classmethod
    def clear_cache(cls, cache_type: Optional[str] = None):
        """
        Limpia la caché de modelos.

        Args:
            cache_type: Tipo de caché a limpiar ('embedders', 'reranker', 'translators', 'all')
                       Si es None o 'all', limpia todo.
        """
        if cache_type in (None, "all", "embedders"):
            cls._embedders.clear()
            logger.info("Caché de embedders limpiada")

        if cache_type in (None, "all", "reranker"):
            cls._reranker = None
            logger.info("Caché de reranker limpiada")

        if cache_type in (None, "all", "translators"):
            cls._translators.clear()
            logger.info("Caché de traductores limpiada")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Retorna estadísticas de uso de la caché"""
        return {
            "embedders_cargados": len(cls._embedders),
            "reranker_cargado": 1 if cls._reranker else 0,
            "traductores_cargados": len(cls._translators),
        }


class VLLMState:
    """
    Estado global para tracking del modelo vLLM activo.

    Mantiene sincronización entre requests cuando hay cambios de modelo.
    """

    _current_model: Optional[str] = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_current_model(cls) -> Optional[str]:
        """Obtiene el modelo actualmente cargado"""
        async with cls._lock:
            return cls._current_model

    @classmethod
    async def set_current_model(cls, model: str):
        """Establece el modelo actual"""
        async with cls._lock:
            cls._current_model = model
            logger.info(f"Modelo vLLM actual establecido: {model}")

    @classmethod
    async def check_model_change(cls, requested_model: str) -> bool:
        """
        Verifica si hay un cambio de modelo pendiente.

        Args:
            requested_model: Modelo solicitado

        Returns:
            True si el modelo cambió, False si es el mismo
        """
        async with cls._lock:
            if cls._current_model is None:
                cls._current_model = requested_model
                logger.info(f"Modelo inicial establecido: {requested_model}")
                return False

            if cls._current_model != requested_model:
                logger.warning(
                    f"Cambio de modelo detectado: {cls._current_model} -> {requested_model}"
                )
                return True

            return False
