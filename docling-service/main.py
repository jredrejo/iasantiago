# docling-service/main.py (GPU-enabled)

import logging
import os
import tempfile
import time
import torch
import pypdf
import hashlib
import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Docling PDF Extraction Service (GPU)")

# Extraction cache: file_hash -> extraction results
_extraction_cache: Dict[str, List[Dict[str, Any]]] = {}
CACHE_DIR = (
    Path("/cache/docling")
    if os.path.exists("/cache")
    else Path(tempfile.gettempdir()) / "docling_cache"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "extraction_cache.json"
CRASH_STATE_FILE = CACHE_DIR / "last_extraction_state.txt"
EXTRACTION_COUNT_FILE = CACHE_DIR / "extraction_count.txt"

# Track if we crashed and need to disable tables temporarily
_last_crashed = False
_extraction_count = 0
_tables_disabled_after_crash = False


def _load_cache():
    """Load extraction cache from disk at startup"""
    global _extraction_cache
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                _extraction_cache = json.load(f)
            logger.info(f"[CACHE] Loaded {len(_extraction_cache)} cached extractions")
    except Exception as e:
        logger.warning(f"[CACHE] Failed to load cache: {e}")


def _save_cache():
    """Save extraction cache to disk"""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_extraction_cache, f)
    except Exception as e:
        logger.warning(f"[CACHE] Failed to save cache: {e}")


def _get_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(file_bytes).hexdigest()


def _detect_crash():
    """Detect if the service crashed during last extraction"""
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
    """Mark that extraction is starting (to detect crashes)"""
    try:
        CRASH_STATE_FILE.write_text("IN_PROGRESS")
    except Exception as e:
        logger.warning(f"[CRASH] Failed to write state file: {e}")


def _mark_extraction_success():
    """Mark that extraction completed successfully"""
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
    """Configure GPU settings for optimal Docling performance"""

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

    # Set memory fraction for Docling (don't use all GPU)
    memory_fraction = float(os.getenv("DOCLING_GPU_MEMORY_FRACTION", "0.15"))
    logger.info(f"[GPU] Setting memory fraction: {memory_fraction:.2%}")

    # This is a soft limit - PyTorch will allocate as needed up to this
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

    return True


# Initialize GPU on startup
GPU_AVAILABLE = setup_gpu()


# ============================================================
# DOCLING INITIALIZATION (GPU-aware)
# ============================================================

_docling_converter = None
_initialization_error = None


def get_docling_converter():
    try:
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        # Table structure detection: disable if we just recovered from a crash
        pipeline_options.do_table_structure = not _tables_disabled_after_crash
        logger.info("[DOCLING] Pipeline options:")
        logger.info(f"  - do_ocr: {pipeline_options.do_ocr}")
        logger.info(f"  - do_table_structure: {pipeline_options.do_table_structure}")
        if _tables_disabled_after_crash:
            logger.warning(
                "[DOCLING] ⚠️  Table detection DISABLED - service recovered from crash"
            )

        # Set up format options for PDF
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,  # GPU-accelerated backend
            )
        }

        # Initialize converter
        converter = DocumentConverter(format_options=format_options)

        return converter

    except Exception as e:
        print(f"Error initializing DocumentConverter: {e}")
        raise


def validate_pdf(pdf_path: Path) -> bool:
    """Validate PDF can be read"""
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"[DOCLING] PDF validation: {num_pages} pages")

            # Check if encrypted
            if reader.is_encrypted:
                logger.warning(f"[DOCLING] PDF is encrypted")
                return False

            # Try to read first page
            first_page = reader.pages[0]
            text = first_page.extract_text()
            logger.info(f"[DOCLING] First page text preview: {text[:100]}")

            return num_pages > 0
    except Exception as e:
        logger.error(f"[DOCLING] PDF validation failed: {e}")
        return False


# ============================================================
# EXTRACTION LOGIC (GPU-accelerated)
# ============================================================


def extract_elements_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract structured elements from PDF using GPU-accelerated Docling
    """
    converter = get_docling_converter()

    file_size_mb = pdf_path.stat().st_size / 1e6
    logger.info(f"[DOCLING] Processing: {pdf_path.name}")
    logger.info(f"[DOCLING] File size: {file_size_mb:.2f} MB")
    logger.info(f"[DOCLING] File exists: {pdf_path.exists()}")

    # Check if file is too large for available GPU memory
    if GPU_AVAILABLE and file_size_mb > 500:
        logger.warning(
            f"[DOCLING] ⚠️  Large file detected ({file_size_mb:.1f}MB) - may exceed GPU memory"
        )

    start_time = time.time()

    if GPU_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9
        mem_available = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            - torch.cuda.memory_allocated() / 1e9
        )
        logger.info(f"[DOCLING] GPU memory available: {mem_available:.2f} GB")

    try:
        # Convert PDF (GPU-accelerated table detection and layout analysis)
        logger.info(f"[DOCLING] Starting conversion...")
        result = converter.convert(str(pdf_path))
        logger.info(f"[DOCLING] Conversion complete")

        # ============================================================
        # DEBUG: Inspect result structure
        # ============================================================
        logger.info(f"[DOCLING] Result type: {type(result)}")
        logger.info(f"[DOCLING] Has document: {hasattr(result, 'document')}")

        if not hasattr(result, "document"):
            logger.error(f"[DOCLING] Result missing 'document' attribute")
            logger.error(f"[DOCLING] Available attributes: {dir(result)}")
            raise ValueError("Invalid Docling result - missing document")

        doc = result.document
        logger.info(f"[DOCLING] Document type: {type(doc)}")
        logger.info(f"[DOCLING] Document attributes: {dir(doc)}")

        # ============================================================
        # Try multiple extraction methods
        # ============================================================
        elements = []

        # METHOD 0: PREFERRED - Use export_to_markdown for proper chunking
        # This gives us full structured text that we can chunk properly
        if hasattr(doc, "export_to_markdown"):
            logger.info(f"[DOCLING] Method 0 (PREFERRED): Using export_to_markdown()")
            try:
                markdown = doc.export_to_markdown()
                logger.info(f"[DOCLING] Markdown length: {len(markdown)} chars")

                if markdown.strip() and len(markdown) > 50:
                    # Split by double newlines to get logical paragraphs
                    paragraphs = [
                        p.strip()
                        for p in markdown.split("\n\n")
                        if p.strip() and len(p.strip()) > 30
                    ]
                    logger.info(
                        f"[DOCLING] Found {len(paragraphs)} paragraphs from markdown"
                    )

                    for para_idx, para in enumerate(paragraphs):
                        # Try to extract page number from context
                        # Docling markdown doesn't preserve page info directly,
                        # so we estimate based on paragraph position
                        estimated_page = 1 + (para_idx // 5)  # Rough estimate

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
                                },
                            }
                        )

                    logger.info(
                        f"[DOCLING] Method 0 extracted {len(elements)} elements from markdown"
                    )
            except Exception as e:
                logger.warning(f"[DOCLING] Method 0 (markdown) failed: {e}")

        # METHOD 1: Fallback - Try export_to_dict (extracts structural elements)
        # NOTE: This only gets top-level sections, not fine-grained chunks
        if not elements and hasattr(doc, "export_to_dict"):
            logger.info(f"[DOCLING] Method 1 (FALLBACK): Using export_to_dict()")
            try:
                doc_dict = doc.export_to_dict()
                logger.info(f"[DOCLING] Dict keys: {doc_dict.keys()}")

                # Parse the dict structure
                if "body" in doc_dict:
                    body_items = doc_dict["body"]
                    logger.info(
                        f"[DOCLING] Found {len(body_items)} items in body (WARNING: coarse-grained)"
                    )

                    for idx, item in enumerate(body_items):
                        if idx < 3:
                            logger.debug(
                                f"[DOCLING] Item {idx}: {item.keys() if isinstance(item, dict) else type(item)}"
                            )

                        text = (
                            item.get("text", "")
                            if isinstance(item, dict)
                            else str(item)
                        )
                        if not text.strip() or len(text.strip()) < 30:
                            continue

                        page = (
                            item.get("prov", [{}])[0].get("page", 1)
                            if isinstance(item, dict)
                            else 1
                        )
                        elem_type = (
                            item.get("type", "text")
                            if isinstance(item, dict)
                            else "text"
                        )

                        elements.append(
                            {
                                "type": map_docling_type(elem_type),
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
                                    "method": "export_to_dict",
                                    "warning": "coarse-grained_sections_only",
                                },
                            }
                        )

                    logger.warning(
                        f"[DOCLING] Method 1 extracted only {len(elements)} coarse elements (sections). "
                        f"Consider using markdown export for finer granularity."
                    )
            except Exception as e:
                logger.warning(f"[DOCLING] Method 1 failed: {e}")

        # METHOD 3: Try direct body access
        if not elements and hasattr(doc, "body"):
            logger.info(f"[DOCLING] Method 3: Using direct body access")
            try:
                body = doc.body
                logger.info(f"[DOCLING] Body type: {type(body)}")

                # Handle GroupItem
                if hasattr(body, "children"):
                    logger.info(f"[DOCLING] Body is GroupItem with children")
                    body_items = list(body.children)
                elif hasattr(body, "__iter__"):
                    logger.info(f"[DOCLING] Body is iterable")
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
                        if not text.strip():
                            continue

                        # Get page
                        page = 1
                        if hasattr(item, "prov") and item.prov:
                            if isinstance(item.prov, list) and len(item.prov) > 0:
                                page = getattr(item.prov[0], "page", 1)

                        # Get type
                        elem_type = getattr(item, "label", "text")

                        extracted.append(
                            {
                                "type": map_docling_type(elem_type),
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
                                },
                            }
                        )
                    return extracted

                elements = extract_recursive(body_items)
                logger.info(f"[DOCLING] Method 3 extracted {len(elements)} elements")

            except Exception as e:
                logger.warning(f"[DOCLING] Method 3 failed: {e}", exc_info=True)

        # METHOD 4: Try pages iteration
        if not elements and hasattr(doc, "pages"):
            logger.info(f"[DOCLING] Method 4: Using pages iteration")
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

                    if page_text and page_text.strip():
                        elements.append(
                            {
                                "type": "text",
                                "text": page_text.strip(),
                                "page": page_num,
                                "bbox": None,
                                "metadata": {
                                    "docling_type": "page_text",
                                    "source": (
                                        "docling_gpu"
                                        if GPU_AVAILABLE
                                        else "docling_cpu"
                                    ),
                                },
                            }
                        )

                logger.info(f"[DOCLING] Method 4 extracted {len(elements)} elements")
            except Exception as e:
                logger.warning(f"[DOCLING] Method 4 failed: {e}")

        elapsed = time.time() - start_time

        # ============================================================
        # CRITICAL: If still no elements, raise error
        # ============================================================
        if not elements:
            logger.error(f"[DOCLING] All extraction methods failed - got 0 elements")
            logger.error(f"[DOCLING] Document attributes: {dir(doc)}")

            # Try one last method: raw text
            try:
                raw_text = str(doc)
                logger.error(f"[DOCLING] Raw document text length: {len(raw_text)}")
                if len(raw_text) > 100:
                    logger.error(f"[DOCLING] Sample: {raw_text[:200]}")
            except:
                pass

            raise ValueError(
                f"Failed to extract any elements from PDF after trying all methods"
            )

        # Log performance metrics
        if GPU_AVAILABLE:
            mem_after = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            mem_used = mem_peak - mem_before

            logger.info(
                f"[DOCLING] ✓ Extracted {len(elements)} elements in {elapsed:.2f}s "
                f"(GPU memory used: {mem_used:.2f} GB, peak: {mem_peak:.2f} GB)"
            )
        else:
            logger.info(
                f"[DOCLING] ✓ Extracted {len(elements)} elements in {elapsed:.2f}s (CPU)"
            )

        return elements

    except Exception as e:
        logger.error(f"[DOCLING] ✗ Extraction failed: {e}", exc_info=True)
        raise

    finally:
        # Aggressively clear GPU memory after processing to prevent fragmentation
        if GPU_AVAILABLE:
            # Delete converter to release model from memory
            del converter
            # Force garbage collection
            import gc

            gc.collect()
            # Clear GPU cache and reset peak memory stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.debug("[DOCLING] GPU memory cleared after extraction")


def map_docling_type(docling_type: str) -> str:
    """Map Docling element types to ingestor's expected types"""
    mapping = {
        "title": "heading",
        "section-header": "heading",
        "subtitle": "heading",
        "paragraph": "text",
        "text": "text",
        "caption": "text",
        "list-item": "text",
        "list": "text",
        "table": "table",
        "picture": "image",
        "figure": "image",
        "footnote": "text",
        "page-header": "text",
        "page-footer": "text",
    }
    return mapping.get(docling_type, "text")


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
    """Health check with GPU status"""
    try:
        converter = get_docling_converter()

        gpu_info = {}
        if GPU_AVAILABLE:
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
            }
        else:
            gpu_info = {"gpu_available": False}

        return {
            "status": "healthy",
            "docling_initialized": converter is not None,
            **gpu_info,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_pdf(file: UploadFile = File(...)):
    """Extract structured elements from uploaded PDF (GPU-accelerated)"""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    content = await file.read()
    file_hash = _get_file_hash(content)
    logger.info(f"[CACHE] Processing {file.filename} (hash: {file_hash[:16]}...)")

    # Check cache first
    cached = False
    if file_hash in _extraction_cache:
        logger.info(f"[CACHE] ✓ Cache hit for {file.filename}")
        # Deep copy to avoid modifying cached data
        elements = copy.deepcopy(_extraction_cache[file_hash])
        cached = True
    else:
        logger.info(f"[CACHE] ✗ Cache miss for {file.filename} - extracting...")
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(content)

        try:
            # Mark extraction starting (for crash detection)
            _mark_extraction_start()

            # Extract elements (GPU-accelerated)
            elements = extract_elements_from_pdf(tmp_path)

            # Mark successful completion
            _mark_extraction_success()

            # Cache successful extraction (deep copy to store immutable copy)
            logger.info(f"[CACHE] Caching extraction for hash {file_hash[:16]}...")
            _extraction_cache[file_hash] = copy.deepcopy(elements)
            _save_cache()
            cached = False
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    try:

        # Calculate stats
        stats = {
            "total_elements": len(elements),
            "by_type": {},
            "pages": set(),
            "gpu_used": GPU_AVAILABLE,
        }

        for elem in elements:
            elem_type = elem["type"]
            stats["by_type"][elem_type] = stats["by_type"].get(elem_type, 0) + 1
            stats["pages"].add(elem["page"])

        stats["pages"] = len(stats["pages"])
        stats["cached"] = cached

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

    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.get("/gpu/stats")
async def gpu_stats():
    """Get current GPU statistics"""
    if not GPU_AVAILABLE:
        return {"gpu_available": False}

    return {
        "gpu_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


@app.post("/gpu/clear_cache")
async def clear_gpu_cache():
    """Manually clear GPU cache"""
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
        "version": "1.0.0 (GPU-enabled)",
        "gpu_available": GPU_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "extract": "/extract (POST)",
            "gpu_stats": "/gpu/stats",
            "clear_cache": "/gpu/clear_cache (POST)",
        },
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Docling PDF Extraction Service Starting (GPU-enabled)")
    logger.info("=" * 60)

    # Load cache from disk
    _load_cache()

    # Detect if service crashed during last extraction
    _detect_crash()

    try:
        get_docling_converter()
        logger.info("✓ Service ready")
    except Exception as e:
        logger.error(f"✗ Service initialization failed: {e}")

    logger.info("=" * 60)
