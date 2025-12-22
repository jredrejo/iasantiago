"""
Traducción de queries para recuperación RAG multilingüe.
Traduce queries no ingleses a inglés antes de la recuperación.
"""

from typing import Tuple
import logging
from transformers import MarianMTModel, MarianTokenizer
import os

logger = logging.getLogger(__name__)

# Caché de modelos de traducción
_translator_cache = {}

# Idiomas soportados y sus códigos
SUPPORTED_LANGS = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}


def detect_language(text: str) -> str:
    """
    Detección simple de idioma usando la librería langdetect.
    Usa 'en' como fallback si la detección falla.
    """
    try:
        from langdetect import detect

        lang = detect(text)
        return lang
    except ImportError:
        logger.warning("langdetect not installed, skipping language detection")
        return "en"
    except Exception as e:
        logger.debug(f"Language detection failed: {e}, assuming English")
        return "en"


def get_translator(source_lang: str, target_lang: str = "en"):
    """
    Carga o recupera un modelo de traducción desde caché.
    Usa modelos Helsinki-NLP/opus-mt para traducción rápida y ligera.
    """
    cache_key = f"{source_lang}-{target_lang}"

    if cache_key in _translator_cache:
        return _translator_cache[cache_key]

    try:
        # Helsinki-NLP models use format: opus-mt-{src}-{tgt}
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        logger.info(f"Loading translator: {model_name}")

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Determine device
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        model = model.to(device)

        _translator_cache[cache_key] = (tokenizer, model, device)
        logger.info(f"✓ Translator {model_name} loaded on {device}")

        return tokenizer, model, device

    except Exception as e:
        logger.error(f"Failed to load translator {source_lang}→{target_lang}: {e}")
        return None


def translate_query(
    query: str, source_lang: str = None, target_lang: str = "en"
) -> Tuple[str, str]:
    """
    Traduce un query del idioma origen al idioma destino.

    Args:
        query: El texto del query a traducir
        source_lang: Código de idioma origen (ej: 'es', 'fr').
                    Si es None, se autodetectará.
        target_lang: Código de idioma destino (default: 'en')

    Returns:
        (translated_query, source_language)
    """
    # Autodetectar idioma si no se proporciona
    if source_lang is None:
        source_lang = detect_language(query)

    # Si ya está en el idioma destino, retornar tal cual
    if source_lang == target_lang:
        return query, source_lang

    # Saltar traducción para queries en inglés
    if source_lang == "en":
        return query, source_lang

    # Intentar cargar y usar el traductor
    translator_info = get_translator(source_lang, target_lang)

    if translator_info is None:
        logger.warning(
            f"Traducción no disponible para {source_lang}→{target_lang}, "
            f"usando query original"
        )
        return query, source_lang

    tokenizer, model, device = translator_info

    try:
        # Tokenizar y traducir
        inputs = tokenizer(query, return_tensors="pt", padding=True).to(device)
        translated_ids = model.generate(**inputs)
        translated = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

        logger.info(
            f"🌐 Query traducido ({source_lang}→en): {query[:60]}... → {translated[:60]}..."
        )

        return translated, source_lang

    except Exception as e:
        logger.error(f"Traducción falló: {e}, usando query original")
        return query, source_lang


def should_translate(query: str) -> bool:
    """
    Verifica si el query debe traducirse.
    Retorna True si el query no está en inglés.
    """
    try:
        from langdetect import detect

        lang = detect(query)
        return lang != "en"
    except Exception:
        return False
