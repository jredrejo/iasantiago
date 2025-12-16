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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Docling PDF Extraction Service (GPU)")

# Extraction cache
_extraction_cache: Dict[str, List[Dict[str, Any]]] = {}
CACHE_DIR = (
    Path("/cache/docling")
    if os.path.exists("/cache")
    else Path(tempfile.gettempdir()) / "docling_cache"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "extraction_cache.json"
CRASH_STATE_FILE = CACHE_DIR / "last_extraction_state.txt"

_last_crashed = False
_extraction_count = 0
_tables_disabled_after_crash = False


def _load_cache():
    global _extraction_cache
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                _extraction_cache = json.load(f)
            logger.info(f"[CACHE] Loaded {len(_extraction_cache)} cached extractions")
    except Exception as e:
        logger.warning(f"[CACHE] Failed to load cache: {e}")


def _save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_extraction_cache, f)
    except Exception as e:
        logger.warning(f"[CACHE] Failed to save cache: {e}")


def _get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _detect_crash():
    global _last_crashed, _extraction_count, _tables_disabled_after_crash
    try:
        if CRASH_STATE_FILE.exists():
            content = CRASH_STATE_FILE.read_text().strip()
            if content == "IN_PROGRESS":
                logger.error(
                    "[CRASH] Previous extraction crashed - disabling table detection for next request"
                )
                _last_crashed = True
                _tables_disabled_after_crash = True
                _extraction_count = 0
                return True
    except Exception as e:
        logger.warning(f"[CRASH] Failed to check crash state: {e}")
    return False


def _mark_extraction_start():
    try:
        CRASH_STATE_FILE.write_text("IN_PROGRESS")
    except Exception as e:
        logger.warning(f"[CRASH] Failed to write state file: {e}")


def _mark_extraction_success():
    global _extraction_count, _tables_disabled_after_crash
    try:
        CRASH_STATE_FILE.write_text("SUCCESS")
        _extraction_count += 1
        # Re-enable table detection after first successful extraction post-crash
        if _tables_disabled_after_crash and _extraction_count >= 1:
            logger.info(
                f"[CRASH] Re-enabling table detection after {_extraction_count} successful extraction"
            )
            _tables_disabled_after_crash = False
    except Exception as e:
        logger.warning(f"[CRASH] Failed to mark success: {e}")


# ============================================================
# GPU CONFIGURATION
# ============================================================


def setup_gpu():
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available, running on CPU")
        return False

    gpu_count = torch.cuda.device_count()
    logger.info(f"[GPU] Found {gpu_count} CUDA device(s)")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"[GPU] Device {i}: {props.name}")
        logger.info(f"[GPU]   - Total memory: {props.total_memory / 1e9:.2f} GB")
        logger.info(f"[GPU]   - Compute capability: {props.major}.{props.minor}")

    memory_fraction = float(os.getenv("DOCLING_GPU_MEMORY_FRACTION", "0.30"))
    logger.info(f"[GPU] Setting memory fraction: {memory_fraction:.2%}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for i in range(gpu_count):
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=i)

    return True


GPU_AVAILABLE = setup_gpu()


# ============================================================
# DOCLING INITIALIZATION
# ============================================================


def get_docling_converter():
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = not _tables_disabled_after_crash

        # ⭐ FIX: Disable reading order detection to avoid page size issues
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        logger.info("[DOCLING] Pipeline options:")
        logger.info(f"  - do_ocr: {pipeline_options.do_ocr}")
        logger.info(f"  - do_table_structure: {pipeline_options.do_table_structure}")
        logger.info("  - generate_page_images: False (disabled for stability)")

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            )
        }

        return DocumentConverter(format_options=format_options)

    except Exception as e:
        logger.error(f"Error initializing DocumentConverter: {e}")
        raise


def validate_pdf(pdf_path: Path) -> tuple[bool, Optional[str]]:
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

            # Check if encrypted
            if reader.is_encrypted:
                logger.warning("[DOCLING] PDF is encrypted")
                return False, "PDF is encrypted"

            # Check page sizes
            for page_num, page in enumerate(reader.pages[:5], 1):  # Check first 5 pages
                try:
                    mediabox = page.mediabox
                    width = float(mediabox.width)
                    height = float(mediabox.height)

                    if width <= 0 or height <= 0:
                        logger.error(
                            f"[DOCLING] Page {page_num} has invalid size: {width}x{height}"
                        )
                        return False, f"Page {page_num} has invalid dimensions"

                    logger.debug(f"[DOCLING] Page {page_num}: {width:.1f}x{height:.1f}")
                except Exception as e:
                    logger.error(
                        f"[DOCLING] Cannot get dimensions for page {page_num}: {e}"
                    )
                    return False, f"Cannot determine page size for page {page_num}"

            # Try to read first page text
            try:
                first_page = reader.pages[0]
                text = first_page.extract_text()
                logger.info(
                    f"[DOCLING] First page text preview: {text[:100] if text else '(empty)'}"
                )
            except Exception as e:
                logger.warning(f"[DOCLING] Could not extract text from first page: {e}")

            return True, None

    except Exception as e:
        logger.error(f"[DOCLING] PDF validation failed: {e}")
        return False, str(e)


# ============================================================
# FALLBACK: PyPDF-based extraction (when Docling fails)
# ============================================================


def extract_with_pypdf_fallback(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Fallback extraction using PyPDF when Docling fails.
    Simple but reliable - extracts text page by page.
    """
    logger.warning("[FALLBACK] Using PyPDF for extraction (Docling failed)")

    elements = []

    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"[FALLBACK] Extracting from {num_pages} pages")

            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()

                    if not text or len(text.strip()) < 10:
                        logger.debug(
                            f"[FALLBACK] Page {page_num + 1}: Empty or too short"
                        )
                        continue

                    # Split into paragraphs (basic)
                    paragraphs = [
                        p.strip()
                        for p in text.split("\n\n")
                        if p.strip() and len(p.strip()) > 30
                    ]

                    if not paragraphs:
                        # If no double-newlines, treat whole page as one chunk
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

                    logger.debug(
                        f"[FALLBACK] Page {page_num + 1}: {len(paragraphs)} paragraphs"
                    )

                except Exception as e:
                    logger.error(f"[FALLBACK] Error on page {page_num + 1}: {e}")
                    continue

            logger.info(
                f"[FALLBACK] ✓ Extracted {len(elements)} elements from {num_pages} pages"
            )
            return elements

    except Exception as e:
        logger.error(f"[FALLBACK] PyPDF extraction failed: {e}", exc_info=True)
        raise


# ============================================================
# EXTRACTION LOGIC
# ============================================================


def extract_elements_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract structured elements from PDF using GPU-accelerated Docling.
    Falls back to PyPDF if Docling fails.
    """
    file_size_mb = pdf_path.stat().st_size / 1e6
    logger.info(f"[DOCLING] Processing: {pdf_path.name}")
    logger.info(f"[DOCLING] File size: {file_size_mb:.2f} MB")

    # ⭐ CRITICAL: Validate PDF first
    is_valid, error_msg = validate_pdf(pdf_path)
    if not is_valid:
        logger.error(f"[DOCLING] PDF validation failed: {error_msg}")
        logger.warning("[DOCLING] Falling back to PyPDF extraction")
        return extract_with_pypdf_fallback(pdf_path)

    # Pre-extraction memory cleanup
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"[DOCLING] GPU memory before: {mem_before:.2f} GB")

    start_time = time.time()

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
                # Get total pages from document
                num_pages = len(doc.pages) if hasattr(doc, "pages") else 0

                if num_pages > 0:
                    logger.info(f"[DOCLING] Extracting markdown from {num_pages} pages")

                    for page_num in range(1, num_pages + 1):
                        try:
                            # Export markdown for this specific page
                            page_md = doc.export_to_markdown(page_no=page_num)

                            if not page_md or not page_md.strip():
                                continue

                            # Split into paragraphs
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
                                        "page": page_num,  # ← PRECISE page number
                                        "bbox": None,
                                        "metadata": {
                                            "docling_type": "markdown_paragraph",
                                            "source": (
                                                "docling_gpu"
                                                if GPU_AVAILABLE
                                                else "docling_cpu"
                                            ),
                                            "method": "export_to_markdown_per_page",
                                            "page_source": "page_iteration",  # ← Precise!
                                        },
                                    }
                                )
                        except Exception as page_err:
                            logger.warning(f"[DOCLING] Page {page_num} markdown export failed: {page_err}")
                            continue

                    if elements:
                        logger.info(f"[DOCLING] Extracted {len(elements)} elements with precise page numbers")
                else:
                    # Fallback: no page count available, use old method with estimation
                    logger.warning("[DOCLING] No page count available, using estimated pages")
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

                        logger.info(f"[DOCLING] Extracted {len(elements)} elements (estimated pages)")
            except Exception as e:
                logger.warning(f"[DOCLING] Markdown export failed: {e}")

        # METHOD 1: Try export_to_dict for PRECISE page numbers
        if not elements and hasattr(doc, "export_to_dict"):
            logger.info("[DOCLING] Using export_to_dict() for precise page tracking")
            try:
                doc_dict = doc.export_to_dict()

                if "body" in doc_dict:
                    body_items = doc_dict["body"]
                    logger.info(f"[DOCLING] Found {len(body_items)} body items")

                    for item in body_items:
                        if not isinstance(item, dict):
                            continue

                        text = item.get("text", "").strip()
                        if not text or len(text) < 30:
                            continue

                        # ⭐ EXTRACT PRECISE PAGE NUMBER from provenance
                        page = 1  # default
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

                        elem_type = item.get("type", "text")

                        elements.append(
                            {
                                "type": "text",
                                "text": text,
                                "page": page,  # ← PRECISE page number from PDF
                                "bbox": None,
                                "metadata": {
                                    "docling_type": elem_type,
                                    "source": (
                                        "docling_gpu"
                                        if GPU_AVAILABLE
                                        else "docling_cpu"
                                    ),
                                    "method": "export_to_dict",
                                    "page_source": "pdf_provenance",  # ← From PDF structure
                                },
                            }
                        )

                    logger.info(
                        f"[DOCLING] Extracted {len(elements)} elements with precise pages"
                    )
            except Exception as e:
                logger.warning(f"[DOCLING] Dict export failed: {e}")

        # METHOD 2: Try direct body access
        if not elements and hasattr(doc, "body"):
            logger.info("[DOCLING] Method 2: Using direct body access")
            try:
                body = doc.body
                logger.info(f"[DOCLING] Body type: {type(body)}")

                # Handle GroupItem
                if hasattr(body, "children"):
                    logger.info("[DOCLING] Body is GroupItem with children")
                    body_items = list(body.children)
                elif hasattr(body, "__iter__"):
                    logger.info("[DOCLING] Body is iterable")
                    body_items = list(body)
                else:
                    logger.error(f"[DOCLING] Body is not iterable: {dir(body)}")
                    body_items = []

                logger.info(f"[DOCLING] Found {len(body_items)} body items")

                def extract_recursive(items, depth=0):
                    extracted = []
                    for item in items:
                        # Handle nested GroupItems
                        if hasattr(item, "children"):
                            logger.debug(f"[DOCLING] Nested GroupItem at depth {depth}")
                            extracted.extend(
                                extract_recursive(list(item.children), depth + 1)
                            )
                            continue

                        # Extract text
                        text = str(item.text) if hasattr(item, "text") else str(item)
                        if not text.strip() or len(text.strip()) < 30:
                            continue

                        # Get page with multiple fallback methods
                        page = 1
                        if hasattr(item, "prov") and item.prov:
                            if isinstance(item.prov, list) and len(item.prov) > 0:
                                prov = item.prov[0]
                                if isinstance(prov, dict):
                                    page = prov.get("page", 1)
                                elif hasattr(prov, "page"):
                                    page = prov.page

                        # Get type
                        elem_type = getattr(item, "label", "text")

                        extracted.append(
                            {
                                "type": "text",
                                "text": text.strip(),
                                "page": page,
                                "bbox": None,
                                "metadata": {
                                    "docling_type": elem_type,
                                    "source": (
                                        "docling_gpu"
                                        if GPU_AVAILABLE
                                        else "docling_cpu"
                                    ),
                                    "method": "direct_body_access",
                                    "page_source": "pdf_provenance",
                                },
                            }
                        )
                    return extracted

                elements = extract_recursive(body_items)
                logger.info(f"[DOCLING] Method 2 extracted {len(elements)} elements")

            except Exception as e:
                logger.warning(f"[DOCLING] Method 2 failed: {e}", exc_info=True)

        # METHOD 3: Try pages iteration
        if not elements and hasattr(doc, "pages"):
            logger.info("[DOCLING] Method 3: Using pages iteration")
            try:
                pages = doc.pages
                logger.info(f"[DOCLING] Document has {len(pages)} pages")

                for page_num, page in enumerate(pages, start=1):
                    # Try different text extraction methods
                    page_text = None

                    if hasattr(page, "export_to_text"):
                        page_text = page.export_to_text()
                    elif hasattr(page, "text"):
                        page_text = page.text
                    else:
                        page_text = str(page)

                    if page_text and page_text.strip() and len(page_text.strip()) > 30:
                        # Split page text into paragraphs for better chunking
                        paragraphs = [
                            p.strip()
                            for p in page_text.split("\n\n")
                            if p.strip() and len(p.strip()) > 30
                        ]

                        if not paragraphs:
                            paragraphs = [page_text.strip()]

                        for para in paragraphs:
                            elements.append(
                                {
                                    "type": "text",
                                    "text": para,
                                    "page": page_num,  # ← PRECISE page number from iteration
                                    "bbox": None,
                                    "metadata": {
                                        "docling_type": "page_text",
                                        "source": (
                                            "docling_gpu"
                                            if GPU_AVAILABLE
                                            else "docling_cpu"
                                        ),
                                        "method": "pages_iteration",
                                        "page_source": "page_iteration",
                                    },
                                }
                            )

                logger.info(
                    f"[DOCLING] Method 3 extracted {len(elements)} elements from {len(pages)} pages"
                )
            except Exception as e:
                logger.warning(f"[DOCLING] Method 3 failed: {e}")

        # If Docling extraction resulted in no elements, use fallback
        if not elements:
            logger.warning("[DOCLING] No elements extracted - using PyPDF fallback")
            return extract_with_pypdf_fallback(pdf_path)

        elapsed = time.time() - start_time

        if GPU_AVAILABLE:
            mem_after = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            mem_used = mem_peak - mem_before

            logger.info(
                f"[DOCLING] ✓ Extracted {len(elements)} elements in {elapsed:.2f}s "
                f"(GPU memory used: {mem_used:.2f} GB)"
            )
        else:
            logger.info(
                f"[DOCLING] ✓ Extracted {len(elements)} elements in {elapsed:.2f}s (CPU)"
            )

        return elements

    except Exception as e:
        logger.error(f"[DOCLING] ✗ Extraction failed: {e}", exc_info=True)
        logger.warning("[DOCLING] Falling back to PyPDF extraction")
        return extract_with_pypdf_fallback(pdf_path)

    finally:
        # Aggressive cleanup
        if GPU_AVAILABLE:
            try:
                del converter
            except:
                pass

            import gc

            gc.collect()

            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            mem_after = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[DOCLING] GPU memory after cleanup: {mem_after:.2f} GB")


# ============================================================
# API ENDPOINTS
# ============================================================


class ExtractedElement(BaseModel):
    type: str
    text: str
    page: int
    bbox: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ExtractionResponse(BaseModel):
    success: bool
    elements: List[ExtractedElement]
    stats: Dict[str, Any]
    error: Optional[str] = None


@app.get("/health")
async def health_check():
    try:
        gpu_info = {}
        if GPU_AVAILABLE:
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
            }
        else:
            gpu_info = {"gpu_available": False}

        return {"status": "healthy", **gpu_info}
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    file_hash = _get_file_hash(content)
    logger.info(f"[CACHE] Processing {file.filename} (hash: {file_hash[:16]}...)")

    # Check cache
    cached = False
    if file_hash in _extraction_cache:
        logger.info("[CACHE] ✓ Cache hit")
        elements = copy.deepcopy(_extraction_cache[file_hash])
        cached = True
    else:
        logger.info("[CACHE] ✗ Cache miss - extracting...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(content)

        try:
            _mark_extraction_start()
            elements = extract_elements_from_pdf(tmp_path)
            _mark_extraction_success()

            logger.info("[CACHE] Caching extraction")
            _extraction_cache[file_hash] = copy.deepcopy(elements)
            _save_cache()
            cached = False
        finally:
            try:
                tmp_path.unlink()
            except:
                pass

    try:
        stats = {
            "total_elements": len(elements),
            "by_type": {},
            "pages": set(),
            "gpu_used": GPU_AVAILABLE,
            "cached": cached,
        }

        for elem in elements:
            elem_type = elem["type"]
            stats["by_type"][elem_type] = stats["by_type"].get(elem_type, 0) + 1
            stats["pages"].add(elem["page"])

        stats["pages"] = len(stats["pages"])

        return ExtractionResponse(
            success=True,
            elements=[ExtractedElement(**e) for e in elements],
            stats=stats,
        )

    except Exception as e:
        logger.error(f"[API] Extraction failed: {e}", exc_info=True)
        return ExtractionResponse(
            success=False, elements=[], stats={"gpu_used": GPU_AVAILABLE}, error=str(e)
        )


@app.get("/gpu/stats")
async def gpu_stats():
    if not GPU_AVAILABLE:
        return {"gpu_available": False}

    return {
        "gpu_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


@app.post("/gpu/clear_cache")
async def clear_gpu_cache():
    if not GPU_AVAILABLE:
        return {"message": "GPU not available"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "message": "GPU cache cleared",
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
    }


@app.get("/")
async def root():
    return {
        "service": "Docling PDF Extraction Service",
        "version": "1.0.0 (GPU-enabled with PyPDF fallback)",
        "gpu_available": GPU_AVAILABLE,
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Docling PDF Extraction Service Starting")
    logger.info("=" * 60)

    _load_cache()
    _detect_crash()

    logger.info("✓ Service ready")
    logger.info("=" * 60)
