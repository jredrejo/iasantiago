# ingestor/docling_extractor.py
"""
Docling PDF extraction module (ported from docling-service)
Provides GPU-accelerated PDF extraction with fallback to PyPDF
Includes crash detection to skip problematic files after repeated failures
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
# EXTRACTION CACHE
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
# CRASH DETECTION - Track files that cause segfaults
# ============================================================

CRASH_STATE_FILE = CACHE_DIR / "crash_state.json"
MAX_CRASHES_BEFORE_SKIP = 1  # After 1 crash, use unstructured fallback

_crash_state: Dict[str, int] = {}  # filename -> crash count


def _load_crash_state():
    """Load crash state from disk."""
    global _crash_state
    try:
        if CRASH_STATE_FILE.exists():
            with open(CRASH_STATE_FILE, "r") as f:
                _crash_state = json.load(f)
            logger.info(
                f"[DOCLING] Loaded crash state: {len(_crash_state)} files tracked"
            )
    except Exception as e:
        logger.warning(f"[DOCLING] Failed to load crash state: {e}")
        _crash_state = {}


def _save_crash_state():
    """Save crash state to disk."""
    try:
        with open(CRASH_STATE_FILE, "w") as f:
            json.dump(_crash_state, f)
    except Exception as e:
        logger.warning(f"[DOCLING] Failed to save crash state: {e}")


def mark_file_processing(filename: str):
    """Mark a file as currently being processed (call before docling extraction)."""
    global _crash_state
    # Increment crash count - will be decremented on success
    _crash_state[filename] = _crash_state.get(filename, 0) + 1
    _save_crash_state()
    logger.debug(
        f"[DOCLING] Marked {filename} as processing (count: {_crash_state[filename]})"
    )


def mark_file_success(filename: str):
    """Mark a file as successfully processed (call after successful extraction)."""
    global _crash_state
    # Reset crash count on success
    if filename in _crash_state:
        del _crash_state[filename]
        _save_crash_state()
        logger.debug(f"[DOCLING] Cleared crash state for {filename}")


class DoclingCrashLimitExceeded(Exception):
    """Raised when a file has crashed too many times and should use alternative extraction."""

    pass


def should_skip_docling(filename: str) -> bool:
    """Check if a file has crashed too many times and should use fallback."""
    count = _crash_state.get(filename, 0)
    if count >= MAX_CRASHES_BEFORE_SKIP:
        logger.warning(
            f"[DOCLING] File {filename} has crashed {count} times - skipping docling"
        )
        return True
    return False


def _load_cache():
    global _extraction_cache
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                _extraction_cache = json.load(f)
            logger.info(f"[DOCLING] Loaded {len(_extraction_cache)} cached extractions")
    except Exception as e:
        logger.warning(f"[DOCLING] Failed to load cache: {e}")


def _save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_extraction_cache, f)
    except Exception as e:
        logger.warning(f"[DOCLING] Failed to save cache: {e}")


def _get_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ============================================================
# GPU CONFIGURATION
# ============================================================

GPU_AVAILABLE = False


def setup_gpu() -> bool:
    global GPU_AVAILABLE
    if not torch.cuda.is_available():
        logger.warning("[DOCLING] CUDA not available, running on CPU")
        GPU_AVAILABLE = False
        return False

    gpu_count = torch.cuda.device_count()
    logger.info(f"[DOCLING] Found {gpu_count} CUDA device(s)")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"[DOCLING] Device {i}: {props.name}")
        logger.info(f"[DOCLING]   - Total memory: {props.total_memory / 1e9:.2f} GB")

    memory_fraction = float(os.getenv("DOCLING_GPU_MEMORY_FRACTION", "0.30"))
    logger.info(f"[DOCLING] Setting memory fraction: {memory_fraction:.2%}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for i in range(gpu_count):
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=i)

    GPU_AVAILABLE = True
    return True


# ============================================================
# DOCLING INITIALIZATION
# ============================================================


def get_docling_converter():
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Disabled - PDFs have text layer
        pipeline_options.do_table_structure = not _tables_disabled_after_crash
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        logger.info("[DOCLING] Pipeline options:")
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
        logger.error(f"[DOCLING] Error initializing DocumentConverter: {e}")
        raise


def validate_pdf(pdf_path: Path) -> tuple:
    """
    Validate PDF and return (is_valid, error_message)
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)

            if num_pages == 0:
                return False, "PDF has no pages"

            logger.info(f"[DOCLING] PDF validation: {num_pages} pages")

            if reader.is_encrypted:
                logger.warning("[DOCLING] PDF is encrypted")
                return False, "PDF is encrypted"

            # Check page sizes (first 5 pages)
            for page_num, page in enumerate(reader.pages[:5], 1):
                try:
                    mediabox = page.mediabox
                    width = float(mediabox.width)
                    height = float(mediabox.height)

                    if width <= 0 or height <= 0:
                        return False, f"Page {page_num} has invalid dimensions"
                except Exception as e:
                    return False, f"Cannot determine page size for page {page_num}"

            return True, None

    except Exception as e:
        logger.error(f"[DOCLING] PDF validation failed: {e}")
        return False, str(e)


# ============================================================
# FALLBACK: PyPDF-based extraction
# ============================================================


def extract_with_pypdf_fallback(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Fallback extraction using PyPDF when Docling fails.
    """
    logger.warning("[DOCLING] Using PyPDF fallback extraction")

    elements = []

    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"[DOCLING] Extracting from {num_pages} pages via PyPDF")

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
                    logger.error(f"[DOCLING] Error on page {page_num + 1}: {e}")
                    continue

            logger.info(f"[DOCLING] PyPDF extracted {len(elements)} elements")
            return elements

    except Exception as e:
        logger.error(f"[DOCLING] PyPDF extraction failed: {e}", exc_info=True)
        raise


# ============================================================
# MAIN EXTRACTION LOGIC
# ============================================================


def extract_elements_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract structured elements from PDF using GPU-accelerated Docling.
    Falls back to PyPDF if Docling fails or file has crashed too many times.
    """
    global _extraction_cache

    filename = pdf_path.name

    # Check cache first
    file_hash = _get_file_hash(pdf_path)
    if file_hash in _extraction_cache:
        logger.info(f"[DOCLING] Cache hit for {filename}")
        return copy.deepcopy(_extraction_cache[file_hash])

    # Check if this file has crashed too many times - raise exception to trigger unstructured fallback
    if should_skip_docling(filename):
        raise DoclingCrashLimitExceeded(
            f"File {filename} has crashed {_crash_state.get(filename, 0)} times - use alternative extraction"
        )

    file_size_mb = pdf_path.stat().st_size / 1e6
    logger.info(f"[DOCLING] Processing: {filename}")
    logger.info(f"[DOCLING] File size: {file_size_mb:.2f} MB")

    # Validate PDF first
    is_valid, error_msg = validate_pdf(pdf_path)
    if not is_valid:
        logger.error(f"[DOCLING] PDF validation failed: {error_msg}")
        return extract_with_pypdf_fallback(pdf_path)

    # Mark file as processing (crash detection)
    mark_file_processing(filename)

    # GPU memory cleanup
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"[DOCLING] GPU memory before: {mem_before:.2f} GB")

    start_time = time.time()
    converter = None

    try:
        converter = get_docling_converter()
        logger.info("[DOCLING] Starting conversion...")
        result = converter.convert(str(pdf_path))
        logger.info("[DOCLING] Conversion complete")

        if not hasattr(result, "document"):
            raise ValueError("Invalid Docling result - missing document")

        doc = result.document
        elements = []

        # METHOD 0: export_to_markdown per page (PRECISE page numbers)
        if hasattr(doc, "export_to_markdown"):
            logger.info("[DOCLING] Using export_to_markdown() with per-page extraction")
            try:
                num_pages = len(doc.pages) if hasattr(doc, "pages") else 0

                if num_pages > 0:
                    logger.info(f"[DOCLING] Extracting markdown from {num_pages} pages")

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
                                f"[DOCLING] Page {page_num} failed: {page_err}"
                            )
                            continue

                    if elements:
                        logger.info(
                            f"[DOCLING] Extracted {len(elements)} elements with precise pages"
                        )
                else:
                    # Fallback: estimate pages
                    logger.warning("[DOCLING] No page count, using estimated pages")
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
                logger.warning(f"[DOCLING] Markdown export failed: {e}")

        # METHOD 1: export_to_dict for precise page numbers (fallback)
        if not elements and hasattr(doc, "export_to_dict"):
            logger.info("[DOCLING] Using export_to_dict() fallback")
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
                        f"[DOCLING] export_to_dict extracted {len(elements)} elements"
                    )
            except Exception as e:
                logger.warning(f"[DOCLING] Dict export failed: {e}")

        # If no elements, use PyPDF fallback
        if not elements:
            logger.warning("[DOCLING] No elements extracted - using PyPDF fallback")
            return extract_with_pypdf_fallback(pdf_path)

        elapsed = time.time() - start_time
        logger.info(f"[DOCLING] Extracted {len(elements)} elements in {elapsed:.2f}s")

        # Cache the result
        _extraction_cache[file_hash] = copy.deepcopy(elements)
        _save_cache()

        # Mark as successful - clear crash counter
        mark_file_success(filename)

        return elements

    except Exception as e:
        logger.error(f"[DOCLING] Extraction failed: {e}", exc_info=True)
        # Don't mark success - crash counter remains incremented
        return extract_with_pypdf_fallback(pdf_path)

    finally:
        # Cleanup
        if converter:
            del converter

        if GPU_AVAILABLE:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Initialize GPU and load state on module import
setup_gpu()
_load_cache()
_load_crash_state()
