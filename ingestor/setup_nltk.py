#!/usr/bin/env python3
"""
setup_nltk.py - Descarga y configura datos de NLTK
Ejecuta esto en el contenedor para pre-descargar los datos necesarios
"""

import nltk
import ssl
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# SSL Fix para descargas de NLTK
# ============================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def setup_nltk_data():
    """Descarga todos los datos necesarios de NLTK"""

    logger.info("Setting up NLTK data...")

    # Datos necesarios para Unstructured
    required_data = [
        "punkt",
        "punkt_tab",  # Required for Spanish tokenizer
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "universal_tagset",
        "wordnet",
        "omw-1.4",
    ]

    failed = []

    for data_name in required_data:
        try:
            logger.info(f"Downloading: {data_name}...")
            nltk.download(data_name, quiet=False)
            logger.info(f"✓ {data_name} downloaded")
        except Exception as e:
            logger.error(f"✗ Failed to download {data_name}: {e}")
            failed.append(data_name)

    # Verificar que los datos existan
    logger.info("\nVerifying NLTK data...")
    try:
        # Test English punkt tokenizer
        from nltk.tokenize import sent_tokenize

        test_text_en = "This is a test. It has two sentences."
        result_en = sent_tokenize(test_text_en)
        logger.info(
            f"✓ English Punkt tokenizer works: {len(result_en)} sentences detected"
        )

        # Test Spanish punkt tokenizer (requires punkt_tab)
        test_text_es = "Esto es una prueba. Tiene dos oraciones."
        result_es = sent_tokenize(test_text_es, language="spanish")
        logger.info(
            f"✓ Spanish Punkt tokenizer works: {len(result_es)} sentences detected"
        )

        # Test POS tagger
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize("This is a test")
        tagged = pos_tag(tokens)
        logger.info(f"✓ POS tagger works: {tagged}")

    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        failed.append("verification")

    if failed:
        logger.warning(f"\nFailed downloads: {', '.join(failed)}")
        logger.warning("This may cause issues during PDF extraction")
        return False
    else:
        logger.info("\n" + "=" * 60)
        logger.info("✅ All NLTK data downloaded successfully")
        logger.info("=" * 60)
        return True


if __name__ == "__main__":
    success = setup_nltk_data()
    sys.exit(0 if success else 1)
