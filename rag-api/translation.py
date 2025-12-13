"""
Query translation for cross-lingual RAG retrieval.
Translates non-English queries to English before retrieval.
"""

from typing import Tuple
import logging
from transformers import MarianMTModel, MarianTokenizer
import os

logger = logging.getLogger(__name__)

# Cache for translation models
_translator_cache = {}

# Supported languages and their codes
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
    Simple language detection using langdetect library.
    Falls back to 'en' if detection fails.
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
    Load or retrieve a cached translation model.
    Uses Helsinki-NLP/opus-mt models for fast, lightweight translation.
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
    Translate query from source language to target language.

    Args:
        query: The query text to translate
        source_lang: Source language code (e.g., 'es', 'fr').
                    If None, will auto-detect.
        target_lang: Target language code (default: 'en')

    Returns:
        (translated_query, source_language)
    """
    # Auto-detect language if not provided
    if source_lang is None:
        source_lang = detect_language(query)

    # If already in target language, return as-is
    if source_lang == target_lang:
        return query, source_lang

    # Skip translation for English queries
    if source_lang == "en":
        return query, source_lang

    # Try to load and use translator
    translator_info = get_translator(source_lang, target_lang)

    if translator_info is None:
        logger.warning(
            f"Translation unavailable for {source_lang}→{target_lang}, "
            f"using original query"
        )
        return query, source_lang

    tokenizer, model, device = translator_info

    try:
        # Tokenize and translate
        inputs = tokenizer(query, return_tensors="pt", padding=True).to(device)
        translated_ids = model.generate(**inputs)
        translated = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

        logger.info(
            f"🌐 Translated query ({source_lang}→en): {query[:60]}... → {translated[:60]}..."
        )

        return translated, source_lang

    except Exception as e:
        logger.error(f"Translation failed: {e}, using original query")
        return query, source_lang


def should_translate(query: str) -> bool:
    """
    Check if query should be translated.
    Returns True if query is not in English.
    """
    try:
        from langdetect import detect

        lang = detect(query)
        return lang != "en"
    except Exception:
        return False
