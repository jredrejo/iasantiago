# ingestor/docling_extractor.py
"""
Módulo de extracción de PDF con Docling (portado de docling-service)
Proporciona extracción de PDF acelerada por GPU con fallback a PyPDF
Incluye detección de fallos para omitir archivos problemáticos después de fallos repetidos
"""

import copy
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf
import torch
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)

# ============================================================
# CACHÉ DE EXTRACCIÓN
# ============================================================

_extraction_cache: Dict[str, List[Dict[str, Any]]] = {}
CACHE_DIR = (
    Path("/cache/docling")
    if os.path.exists("/cache")
    else Path(tempfile.gettempdir()) / "docling_cache"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "extraction_cache.json"

_tables_disabled_after_crash = False

# ============================================================
# DETECCIÓN DE FALLOS - Rastrea archivos que causan segfaults
# ============================================================

CRASH_STATE_FILE = CACHE_DIR / "crash_state.json"
MAX_CRASHES_BEFORE_SKIP = 1  # Después de 1 fallo, usar fallback de unstructured

_crash_state: Dict[str, int] = {}  # nombre_archivo -> contador de fallos


def _load_crash_state():
    """Carga el estado de fallos desde disco."""
    global _crash_state
    try:
        if CRASH_STATE_FILE.exists():
            with open(CRASH_STATE_FILE, "r") as f:
                _crash_state = json.load(f)
            logger.info(
                f"[DOCLING] Estado de fallos cargado: {len(_crash_state)} archivos rastreados"
            )
    except Exception as e:
        logger.warning(f"[DOCLING] Error al cargar estado de fallos: {e}")
        _crash_state = {}


def _save_crash_state():
    """Guarda el estado de fallos a disco."""
    try:
        with open(CRASH_STATE_FILE, "w") as f:
            json.dump(_crash_state, f)
    except Exception as e:
        logger.warning(f"[DOCLING] Error al guardar estado de fallos: {e}")


def mark_file_processing(filename: str):
    """Marca un archivo como actualmente en proceso (llamar antes de la extracción docling)."""
    global _crash_state
    # Incrementar contador de fallos - se decrementará en éxito
    _crash_state[filename] = _crash_state.get(filename, 0) + 1
    _save_crash_state()
    logger.debug(
        f"[DOCLING] Archivo {filename} marcado como procesando (contador: {_crash_state[filename]})"
    )


def mark_file_success(filename: str):
    """Marca un archivo como procesado exitosamente (llamar después de una extracción exitosa)."""
    global _crash_state
    # Reiniciar contador de fallos en éxito
    if filename in _crash_state:
        del _crash_state[filename]
        _save_crash_state()
        logger.debug(f"[DOCLING] Estado de fallos limpiado para {filename}")


class DoclingCrashLimitExceeded(Exception):
    """Se raise cuando un archivo ha fallado demasiadas veces y debe usar extracción alternativa."""

    pass


def should_skip_docling(filename: str) -> bool:
    """Verifica si un archivo ha fallado demasiadas veces y debe usar fallback."""
    count = _crash_state.get(filename, 0)
    if count >= MAX_CRASHES_BEFORE_SKIP:
        logger.warning(
            f"[DOCLING] El archivo {filename} ha fallado {count} veces - omitiendo docling"
        )
        return True
    return False


def _load_cache():
    global _extraction_cache
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                _extraction_cache = json.load(f)
            logger.info(
                f"[DOCLING] Cargadas {len(_extraction_cache)} extracciones en caché"
            )
    except Exception as e:
        logger.warning(f"[DOCLING] Error al cargar caché: {e}")


def _save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_extraction_cache, f)
    except Exception as e:
        logger.warning(f"[DOCLING] Error al guardar caché: {e}")


def _get_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ============================================================
# CONFIGURACIÓN DE GPU
# ============================================================

GPU_AVAILABLE = False


def setup_gpu() -> bool:
    global GPU_AVAILABLE
    if not torch.cuda.is_available():
        logger.warning("[DOCLING] CUDA no disponible, ejecutando en CPU")
        GPU_AVAILABLE = False
        return False

    gpu_count = torch.cuda.device_count()
    logger.info(f"[DOCLING] Encontrados {gpu_count} dispositivos CUDA")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"[DOCLING] Dispositivo {i}: {props.name}")
        logger.info(f"[DOCLING]   - Memoria total: {props.total_memory / 1e9:.2f} GB")

    memory_fraction = float(os.getenv("DOCLING_GPU_MEMORY_FRACTION", "0.30"))
    logger.info(f"[DOCLING] Fracción de memoria configurada: {memory_fraction:.2%}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for i in range(gpu_count):
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=i)

    GPU_AVAILABLE = True
    return True


# ============================================================
# INICIALIZACIÓN DE DOCLING
# ============================================================


def get_docling_converter():
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Deshabilitado - los PDFs tienen capa de texto
        pipeline_options.do_table_structure = not _tables_disabled_after_crash
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        logger.info("[DOCLING] Opciones del pipeline:")
        logger.info(f"  - do_ocr: {pipeline_options.do_ocr}")
        logger.info(f"  - do_table_structure: {pipeline_options.do_table_structure}")

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            )
        }

        return DocumentConverter(format_options=format_options)

    except Exception as e:
        logger.error(f"[DOCLING] Error al inicializar DocumentConverter: {e}")
        raise


def validate_pdf(pdf_path: Path) -> tuple:
    """
    Valida el PDF y retorna (es_valido, mensaje_de_error)
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)

            if num_pages == 0:
                return False, "El PDF no tiene páginas"

            logger.info(f"[DOCLING] Validación de PDF: {num_pages} páginas")

            if reader.is_encrypted:
                logger.warning("[DOCLING] El PDF está encriptado")
                return False, "El PDF está encriptado"

            # Verificar tamaños de página (primeras 5 páginas)
            for page_num, page in enumerate(reader.pages[:5], 1):
                try:
                    mediabox = page.mediabox
                    width = float(mediabox.width)
                    height = float(mediabox.height)

                    if width <= 0 or height <= 0:
                        return False, f"Página {page_num} tiene dimensiones inválidas"
                except Exception as e:
                    return (
                        False,
                        f"No se puede determinar el tamaño de página para página {page_num}",
                    )

            return True, None

    except Exception as e:
        logger.error(f"[DOCLING] Validación de PDF falló: {e}")
        return False, str(e)


# ============================================================
# FALLBACK: Extracción basada en PyPDF
# ============================================================


def extract_with_pypdf_fallback(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracción de fallback usando PyPDF cuando Docling falla.
    """
    logger.warning("[DOCLING] Usando extracción de fallback PyPDF")

    elements = []

    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"[DOCLING] Extrayendo de {num_pages} páginas vía PyPDF")

            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()

                    if not text or len(text.strip()) < 10:
                        continue

                    paragraphs = [
                        p.strip()
                        for p in text.split("\n\n")
                        if p.strip() and len(p.strip()) > 30
                    ]

                    if not paragraphs:
                        paragraphs = [text.strip()]

                    for para in paragraphs:
                        elements.append(
                            {
                                "type": "text",
                                "text": para,
                                "page": page_num + 1,
                                "bbox": None,
                                "metadata": {
                                    "source": "pypdf_fallback",
                                    "extraction_method": "pypdf",
                                },
                            }
                        )

                except Exception as e:
                    logger.error(f"[DOCLING] Error en página {page_num + 1}: {e}")
                    continue

            logger.info(f"[DOCLING] PyPDF extrajo {len(elements)} elementos")
            return elements

    except Exception as e:
        logger.error(f"[DOCLING] Extracción PyPDF falló: {e}", exc_info=True)
        raise


# ============================================================
# LÓGICA PRINCIPAL DE EXTRACCIÓN
# ============================================================


def extract_elements_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extrae elementos estructurados del PDF usando Docling acelerado por GPU.
    Usa fallback a PyPDF si Docling falla o el archivo ha fallado demasiadas veces.
    """
    global _extraction_cache

    filename = pdf_path.name

    # Verificar caché primero
    file_hash = _get_file_hash(pdf_path)
    if file_hash in _extraction_cache:
        logger.info(f"[DOCLING] Cache hit para {filename}")
        return copy.deepcopy(_extraction_cache[file_hash])

    # Verificar si este archivo ha fallado demasiadas veces - raise exception para activar fallback de unstructured
    if should_skip_docling(filename):
        raise DoclingCrashLimitExceeded(
            f"El archivo {filename} ha fallado {_crash_state.get(filename, 0)} veces - usar extracción alternativa"
        )

    file_size_mb = pdf_path.stat().st_size / 1e6
    logger.info(f"[DOCLING] Procesando: {filename}")
    logger.info(f"[DOCLING] Tamaño de archivo: {file_size_mb:.2f} MB")

    # Validar PDF primero
    is_valid, error_msg = validate_pdf(pdf_path)
    if not is_valid:
        logger.error(f"[DOCLING] Validación de PDF falló: {error_msg}")
        return extract_with_pypdf_fallback(pdf_path)

    # Marcar archivo como procesando (detección de fallos)
    mark_file_processing(filename)

    # Limpieza de memoria GPU
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"[DOCLING] Memoria GPU antes: {mem_before:.2f} GB")

    start_time = time.time()
    converter = None

    try:
        converter = get_docling_converter()
        logger.info("[DOCLING] Iniciando conversión...")
        result = converter.convert(str(pdf_path))
        logger.info("[DOCLING] Conversión completada")

        if not hasattr(result, "document"):
            raise ValueError("Resultado de Docling inválido - falta documento")

        doc = result.document
        elements = []

        # MÉTODO 0: export_to_markdown por página (NÚMEROS DE PÁGINA PRECISOS)
        if hasattr(doc, "export_to_markdown"):
            logger.info(
                "[DOCLING] Usando export_to_markdown() con extracción por página"
            )
            try:
                num_pages = len(doc.pages) if hasattr(doc, "pages") else 0

                if num_pages > 0:
                    logger.info(f"[DOCLING] Extrayendo markdown de {num_pages} páginas")

                    for page_num in range(1, num_pages + 1):
                        try:
                            page_md = doc.export_to_markdown(page_no=page_num)

                            if not page_md or not page_md.strip():
                                continue

                            paragraphs = [
                                p.strip()
                                for p in page_md.split("\n\n")
                                if p.strip() and len(p.strip()) > 30
                            ]

                            for para in paragraphs:
                                elements.append(
                                    {
                                        "type": "text",
                                        "text": para,
                                        "page": page_num,
                                        "bbox": None,
                                        "metadata": {
                                            "docling_type": "markdown_paragraph",
                                            "source": (
                                                "docling_gpu"
                                                if GPU_AVAILABLE
                                                else "docling_cpu"
                                            ),
                                            "method": "export_to_markdown_per_page",
                                            "page_source": "page_iteration",
                                        },
                                    }
                                )
                        except Exception as page_err:
                            logger.warning(
                                f"[DOCLING] Página {page_num} falló: {page_err}"
                            )
                            continue

                    if elements:
                        logger.info(
                            f"[DOCLING] Extraídos {len(elements)} elementos con páginas precisas"
                        )
                else:
                    # Fallback: estimar páginas
                    logger.warning(
                        "[DOCLING] Sin conteo de páginas, usando páginas estimadas"
                    )
                    markdown = doc.export_to_markdown()

                    if markdown.strip() and len(markdown) > 50:
                        paragraphs = [
                            p.strip()
                            for p in markdown.split("\n\n")
                            if p.strip() and len(p.strip()) > 30
                        ]

                        for para_idx, para in enumerate(paragraphs):
                            estimated_page = 1 + (para_idx // 5)
                            elements.append(
                                {
                                    "type": "text",
                                    "text": para,
                                    "page": estimated_page,
                                    "bbox": None,
                                    "metadata": {
                                        "docling_type": "markdown_paragraph",
                                        "source": (
                                            "docling_gpu"
                                            if GPU_AVAILABLE
                                            else "docling_cpu"
                                        ),
                                        "method": "export_to_markdown",
                                        "page_source": "estimated_from_order",
                                    },
                                }
                            )
            except Exception as e:
                logger.warning(f"[DOCLING] Export markdown falló: {e}")

        # MÉTODO 1: export_to_dict para números de página precisos (fallback)
        if not elements and hasattr(doc, "export_to_dict"):
            logger.info("[DOCLING] Usando export_to_dict() fallback")
            try:
                doc_dict = doc.export_to_dict()

                if "body" in doc_dict:
                    for item in doc_dict["body"]:
                        if not isinstance(item, dict):
                            continue

                        text = item.get("text", "").strip()
                        if not text or len(text) < 30:
                            continue

                        page = 1
                        if (
                            "prov" in item
                            and isinstance(item["prov"], list)
                            and len(item["prov"]) > 0
                        ):
                            prov = item["prov"][0]
                            if isinstance(prov, dict) and "page" in prov:
                                page = prov["page"]
                            elif hasattr(prov, "page"):
                                page = prov.page

                        elements.append(
                            {
                                "type": "text",
                                "text": text,
                                "page": page,
                                "bbox": None,
                                "metadata": {
                                    "docling_type": item.get("type", "text"),
                                    "source": (
                                        "docling_gpu"
                                        if GPU_AVAILABLE
                                        else "docling_cpu"
                                    ),
                                    "method": "export_to_dict",
                                    "page_source": "pdf_provenance",
                                },
                            }
                        )

                    logger.info(
                        f"[DOCLING] export_to_dict extrajo {len(elements)} elementos"
                    )
            except Exception as e:
                logger.warning(f"[DOCLING] Export dict falló: {e}")

        # Si no hay elementos, usar fallback PyPDF
        if not elements:
            logger.warning(
                "[DOCLING] No se extrajeron elementos - usando fallback PyPDF"
            )
            return extract_with_pypdf_fallback(pdf_path)

        elapsed = time.time() - start_time
        logger.info(f"[DOCLING] Extraídos {len(elements)} elementos en {elapsed:.2f}s")

        # Cachear el resultado
        _extraction_cache[file_hash] = copy.deepcopy(elements)
        _save_cache()

        # Marcar como exitoso - limpiar contador de fallos
        mark_file_success(filename)

        return elements

    except Exception as e:
        logger.error(f"[DOCLING] Extracción falló: {e}", exc_info=True)
        # No marcar éxito - contador de fallos permanece incrementado
        return extract_with_pypdf_fallback(pdf_path)

    finally:
        # Limpieza
        if converter:
            del converter

        if GPU_AVAILABLE:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Inicializar GPU y cargar estado al importar módulo
setup_gpu()
_load_cache()
_load_crash_state()
