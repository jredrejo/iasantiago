#!/usr/bin/env python3
"""
download_easyocr_models.py

Descarga previa de modelos EasyOCR para evitar descargas en tiempo de ejecución.
Ejecuta este script una vez antes de iniciar el ingestor para asegurar que los modelos estén cacheados.

Uso:
    python download_easyocr_models.py
    python download_easyocr_models.py --force  # Re-descargar aunque los modelos existan
    python download_easyocr_models.py --dir /ruta/custom  # Directorio de modelos personalizado
"""

import argparse
import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_ssl_context():
    """Configura contexto SSL para manejar problemas de certificados"""
    try:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        logger.info(
            "[SSL] Configurado contexto SSL no verificado para descargas de modelos"
        )
    except Exception as e:
        logger.warning(f"[SSL] No se pudo configurar el contexto SSL: {e}")


def download_easyocr_models(model_dir: str = None, force: bool = False) -> bool:
    """
    Descarga modelos de EasyOCR (detección + reconocimiento español/inglés).

    Args:
        model_dir: Directorio de modelos personalizado (por defecto: ~/.EasyOCR)
        force: Forzar re-descarga aunque los modelos existan

    Returns:
        True si fue exitoso, False en caso contrario
    """
    import urllib.request

    try:
        import easyocr
    except ImportError:
        logger.error(
            "[EASYOCR] EasyOCR no está instalado. Instalar con: pip install easyocr"
        )
        return False

    if model_dir:
        model_path = Path(model_dir)
    else:
        model_path = Path.home() / ".EasyOCR"

    # Crear directorio de modelos
    model_subdir = model_path / "model"
    model_subdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[EASYOCR] Directorio de modelos: {model_path}")

    # URLs desde el ModelHub de EasyOCR (https://www.jaided.ai/easyocr/modelhub/)
    models = {
        "craft_mlt_25k.pth": {
            "url": "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/craft_mlt_25k.pth",
            "size_mb": 79.3,
            "desc": "Modelo de detección de texto",
        },
        "latin_g2.pth": {
            "url": "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/latin_g2.pth",
            "size_mb": 14.7,
            "desc": "Modelo de reconocimiento latín (español)",
        },
        "english_g2.pth": {
            "url": "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/english_g2.pth",
            "size_mb": 14.4,
            "desc": "Modelo de reconocimiento inglés",
        },
    }

    # Verificar si los modelos ya existen
    craft_path = model_subdir / "craft_mlt_25k.pth"
    latin_path = model_subdir / "latin_g2.pth"
    english_path = model_subdir / "english_g2.pth"

    models_exist = craft_path.exists() and (
        latin_path.exists() or english_path.exists()
    )

    if models_exist and not force:
        logger.info("[EASYOCR] Los modelos ya existen. Usa --force para re-descargar.")
        logger.info("[EASYOCR] Tamaños de modelos:")
        logger.info(f"  - craft_mlt_25k.pth: {craft_path.stat().st_size / 1e6:.1f} MB")
        if latin_path.exists():
            logger.info(f"  - latin_g2.pth: {latin_path.stat().st_size / 1e6:.1f} MB")
        if english_path.exists():
            logger.info(
                f"  - english_g2.pth: {english_path.stat().st_size / 1e6:.1f} MB"
            )
        return True

    if models_exist and force:
        logger.info(
            "[EASYOCR] Los modelos existen pero se especificó --force, re-descargando..."
        )

    logger.info("[EASYOCR] Iniciando descarga directa desde HuggingFace...")
    logger.info(
        "[EASYOCR] Esto puede tomar 2-5 minutos dependiendo de la velocidad de red."
    )

    # Descargar cada modelo
    for model_name, model_info in models.items():
        dest_path = model_subdir / model_name

        if dest_path.exists() and not force:
            logger.info(f"[EASYOCR] {model_name} ya existe, omitiendo...")
            continue

        logger.info(
            f"[EASYOCR] Descargando {model_name} ({model_info['desc']}, ~{model_info['size_mb']}MB)..."
        )
        logger.info(f"[EASYOCR] URL: {model_info['url']}")

        try:
            # Descargar con barra de progreso
            def reporthook(count, block_size, total_size):
                percent = (
                    int(count * block_size * 100 / total_size) if total_size > 0 else 0
                )
                mb_downloaded = (count * block_size) / 1e6
                mb_total = total_size / 1e6
                if percent % 10 == 0:  # Log cada 10%
                    logger.info(
                        f"[EASYOCR]   Progreso: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                    )

            urllib.request.urlretrieve(model_info["url"], dest_path, reporthook)

            # Verificar tamaño
            actual_size = dest_path.stat().st_size
            expected_size = model_info["size_mb"] * 1e6

            if actual_size < expected_size * 0.9:  # Permitir 10% de margen
                logger.warning(
                    f"[EASYOCR] {model_name} parece estar incompleto "
                    f"({actual_size / 1e6:.1f}MB vs esperado ~{model_info['size_mb']}MB)"
                )
                dest_path.unlink()
                return False

            logger.info(
                f"[EASYOCR] {model_name} descargado correctamente ({actual_size / 1e6:.1f} MB)"
            )

        except Exception as e:
            logger.error(f"[EASYOCR] Error descargando {model_name}: {e}")
            # Limpiar archivo parcial si existe
            if dest_path.exists():
                dest_path.unlink()
            return False

    # Verificación final
    craft_exists = craft_path.exists()
    latin_exists = latin_path.exists()
    english_exists = english_path.exists()

    if not (craft_exists and (latin_exists or english_exists)):
        logger.error("[EASYOCR] Verificación de descarga fallida!")
        logger.error(
            f"[EASYOCR] craft_mlt_25k.pth: {'OK' if craft_exists else 'FALTANTE'}"
        )
        logger.error(f"[EASYOCR] latin_g2.pth: {'OK' if latin_exists else 'FALTANTE'}")
        logger.error(
            f"[EASYOCR] english_g2.pth: {'OK' if english_exists else 'FALTANTE'}"
        )
        return False

    logger.info("[EASYOCR] Descarga completa!")
    logger.info("[EASYOCR] Tamaños de modelos:")
    logger.info(f"  - craft_mlt_25k.pth: {craft_path.stat().st_size / 1e6:.1f} MB")
    if latin_exists:
        logger.info(f"  - latin_g2.pth: {latin_path.stat().st_size / 1e6:.1f} MB")
    if english_exists:
        logger.info(f"  - english_g2.pth: {english_path.stat().st_size / 1e6:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Descarga previa de modelos EasyOCR para el ingestor"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar re-descarga aunque los modelos ya existan",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directorio de modelos personalizado (por defecto: ~/.EasyOCR)",
    )

    args = parser.parse_args()

    # Configurar SSL para descargas
    setup_ssl_context()

    # Descargar modelos
    success = download_easyocr_models(model_dir=args.dir, force=args.force)

    if success:
        logger.info("[EASYOCR] Listo! Ahora puedes iniciar el ingestor.")
        return 0
    else:
        logger.error("[EASYOCR] Falló la descarga de modelos.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
