"""
chunk.py - Enhanced Version with Advanced Chunking and Page Validation
Extrae: PDF, DOCX, PPTX (sin XLSX)
Manejo defensivo de CUDA
Enhanced chunking with NLTK + Hierarchical with heading tracking + Context-aware with same embedding model
Enhanced page number validation and adaptive chunking strategies
"""

import logging
import os
import re
import sqlite3
import ssl
import torch
import urllib.request
import inspect
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import nltk
import numpy as np
import pdfplumber
import pypdf
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

# ============================================================
# PDF PAGE COUNT CACHE (avoid redundant PDF opens)
# ============================================================


@lru_cache(maxsize=256)
def get_pdf_total_pages(pdf_path: str) -> Optional[int]:
    """Return total page count for a PDF, cached by file path."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        logger.warning(
            f"[PDF] Could not get page count for {os.path.basename(str(pdf_path))}: {e}"
        )
        return None


# Fix SSL issues for model downloads
def setup_ssl_context():
    """Setup SSL context to handle certificate issues"""
    try:
        # Create unverified SSL context (use with caution)
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.info("[SSL] Configured unverified SSL context for model downloads")
    except Exception as e:
        logger.warning(f"[SSL] Could not configure SSL context: {e}")


# EasyOCR import (GPU-accelerated, PyTorch-compatible)
try:
    import easyocr

    EASYOCR_AVAILABLE = True
    logger.info("[EASYOCR] EasyOCR imported successfully")
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("[EASYOCR] EasyOCR not available, install with: pip install easyocr")

# Set Tesseract language for unstructured
os.environ["TESSERACT_LANG"] = "spa+eng"  # Spanish + English
os.environ["OCR_LANGUAGES"] = "spa+eng"  # Alternative env var
os.environ["UNSTRUCTURED_LANGUAGES"] = "spa,eng"  # Spanish primero, luego English
os.environ["UNSTRUCTURED_FALLBACK_LANGUAGE"] = "eng"  # English si no se puede Spanish


# Try to download punkt data with error handling
def ensure_nltk_data():
    """Ensure NLTK data is available with fallback options"""
    try:
        # Check if punkt is already available
        nltk.data.find("tokenizers/punkt")
        logger.info("[NLTK] punkt data already available")
    except LookupError:
        try:
            logger.info("[NLTK] Downloading punkt data...")
            nltk.download("punkt", quiet=True)
            logger.info("[NLTK] punkt data downloaded successfully")
        except Exception as e:
            logger.error(f"[NLTK] Failed to download punkt: {e}")
            logger.warning("[NLTK] Will use fallback sentence tokenization")
            return False
    return True


# Call this function at module import
nltk_available = ensure_nltk_data()
_cached_sent_tokenizer = None


def get_sent_tokenizer():
    """Get cached sentence tokenizer"""
    global _cached_sent_tokenizer
    if _cached_sent_tokenizer is None:
        try:
            from nltk.tokenize import sent_tokenize

            # Ensure data is available
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            # Create Spanish-first tokenizer
            def spanish_tokenize(text):
                # Try Spanish first, fallback to English
                try:
                    return sent_tokenize(text, language="spanish")
                except Exception:
                    return sent_tokenize(text, language="english")

            _cached_sent_tokenizer = spanish_tokenize
            logger.info("[NLTK] Spanish sentence tokenizer loaded")
        except Exception as e:
            logger.warning(f"[NLTK] Failed to load, using fallback: {e}")
            _cached_sent_tokenizer = ContextAwareChunker._fallback_sentence_split
    return _cached_sent_tokenizer


# ============================================================
# GPU CONFIG FOR UNSTRUCTURED
# ============================================================
if os.getenv("UNSTRUCTURED_ENABLE_CUDA", "true").lower() == "true":
    os.environ.pop("UNSTRUCTURED_DISABLE_CUDA", None)
    logger.info("[CONFIG] CUDA ENABLED for Unstructured layout detection")
else:
    os.environ["UNSTRUCTURED_DISABLE_CUDA"] = "1"
    logger.warning("[CONFIG] CUDA DISABLED for Unstructured")


class AdvancedPageBoundaryDetector:
    """
    Advanced page boundary detection using multiple techniques
    """

    def __init__(self):
        self.pdf_analyzer = None
        self.visual_boundaries = {}

    def detect_boundaries(self, pdf_path: str) -> Dict[int, Dict[str, float]]:
        """
        Detect page boundaries using multiple techniques:
        - Visual analysis of PDF structure
        - Text content markers
        - PDF metadata
        """
        boundaries = {}

        try:
            # Use pdfplumber for detailed page analysis
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    boundaries[page_num] = {
                        "height": page.height,
                        "width": page.width,
                        "bbox": page.bbox,
                        "text_top": self._get_text_top(page),
                        "text_bottom": self._get_text_bottom(page),
                    }
        except Exception as e:
            logger.error(f"Error detecting page boundaries: {e}")

        return boundaries

    def _get_text_top(self, page) -> float:
        """Get the topmost text coordinate on a page"""
        try:
            words = page.extract_words()
            if words:
                return min(word["top"] for word in words)
            return 0
        except Exception:
            return 0

    def _get_text_bottom(self, page) -> float:
        """Get the bottommost text coordinate on a page"""
        try:
            words = page.extract_words()
            if words:
                return max(word["bottom"] for word in words)
            return page.height
        except Exception:
            return page.height

    def assign_precise_page(self, elem, boundaries: Dict[int, Dict[str, float]]) -> int:
        """
        Assign page number with higher precision using boundary information
        """
        if not boundaries:
            return elem.get("page", 1)

        try:
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "coordinates"):
                coords = elem.metadata.coordinates
                if hasattr(coords, "points") and len(coords.points) > 0:
                    # Get the minimum y-coordinate of the element
                    y_coord = min(p[1] for p in coords.points)

                    # Find which page this coordinate falls into
                    cumulative_height = 0
                    for page_num in sorted(boundaries.keys()):
                        page_height = boundaries[page_num]["height"]
                        if (
                            cumulative_height
                            <= y_coord
                            < cumulative_height + page_height
                        ):
                            return page_num
                        cumulative_height += page_height
                    # If we get here, the element is beyond all known pages
                    return max(boundaries.keys())
        except Exception as e:
            logger.debug(f"Error assigning precise page: {e}")

        return elem.get("page", 1)


class ContextAwareChunker:
    """
    Advanced chunking that maintains context across page boundaries
    while preserving accurate page attribution
    """

    def __init__(
        self, chunk_size: int = 900, overlap: int = 120, min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.page_detector = AdvancedPageBoundaryDetector()
        self.sent_tokenize = get_sent_tokenizer()

        try:
            from nltk.tokenize import sent_tokenize

            self.sent_tokenize = sent_tokenize
        except ImportError:
            logger.warning("[CHUNK] NLTK not available, using fallback")
            self.sent_tokenize = self._fallback_sentence_split

    def chunk_with_context_preservation(
        self,
        elements: List[Dict[str, Any]],
        pdf_path: str = None,
        strategy: str = "adaptive",
    ) -> List[Dict[str, Any]]:
        """
        Chunk document while preserving context and page accuracy
        """
        boundaries = {}
        if (
            pdf_path
            and pdf_path.lower().endswith(".pdf")
            and not hasattr(self, "_cached_boundaries")
        ):
            boundaries = self.page_detector.detect_boundaries(pdf_path)
            self._cached_boundaries = boundaries
        elif hasattr(self, "_cached_boundaries"):
            boundaries = self._cached_boundaries

        page_groups = self._group_by_page_enhanced(elements, boundaries, pdf_path)

        all_chunks = []

        for page_num in sorted(page_groups.keys()):
            page_elements = page_groups[page_num]

            if page_num > 1 and page_num - 1 in page_groups:
                prev_page_elements = page_groups[page_num - 1]
                context_elements = self._select_context_elements(prev_page_elements)
                page_elements = context_elements + page_elements

            if strategy == "adaptive":
                page_chunks = self._adaptive_chunk_with_context(page_elements, page_num)
            elif strategy == "semantic":
                page_chunks = self._semantic_chunk_with_context(page_elements, page_num)
            else:
                page_chunks = self._simple_chunk_with_context(page_elements, page_num)

            all_chunks.extend(page_chunks)

        all_chunks = self._validate_and_fix_chunks(all_chunks, boundaries)

        return all_chunks

    def _group_by_page_enhanced(
        self,
        elements: List[Dict],
        boundaries: Dict[int, Dict[str, float]],
        pdf_path: str = None,
    ) -> Dict[int, List[Dict]]:
        """Group elements by page with enhanced page detection."""
        page_groups = defaultdict(list)

        for elem in elements:
            if boundaries:
                page = self.page_detector.assign_precise_page(elem, boundaries)
            else:
                fallback = elem.get("page", 1) if isinstance(elem, dict) else 1
                page = RobustPageExtractor.extract_page_number(
                    elem, pdf_path=pdf_path, fallback_page=fallback
                )

            if not isinstance(page, int) or page < 1:
                logger.warning(f"[CHUNK] Invalid page {page}, using page 1")
                page = 1

            page_groups[page].append(elem)

        return dict(page_groups)

    def _select_context_elements(self, prev_page_elements: List[Dict]) -> List[Dict]:
        """Select relevant elements from previous page for context"""
        headings = [
            e for e in prev_page_elements if e.get("type") in ["heading", "title"]
        ]
        text_elements = [e for e in prev_page_elements if e.get("type") == "text"]

        context = headings[:2]
        if len(text_elements) > 3:
            context.extend(text_elements[-3:])
        else:
            context.extend(text_elements)

        for elem in context:
            elem["is_context"] = True
            elem["source_page"] = elem.get("page", 1)

        return context

    # (the rest of the chunker code is unchanged from your file)
    # ... KEEPING all your chunking methods as-is ...

    def _semantic_chunk_with_context(
        self, elements: List[Dict], page_num: int
    ) -> List[Dict[str, Any]]:
        chunks = []

        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        if page_elements:
            text_content = "\n\n".join(
                e.get("text", "") for e in page_elements if e.get("type") == "text"
            )

            if text_content.strip():
                sentences = (
                    self.sent_tokenize(text_content)
                    if self.sent_tokenize
                    else text_content.split(". ")
                )

                current_chunk = {"sentences": [], "text": "", "char_count": 0}

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    potential_size = current_chunk["char_count"] + len(sentence) + 1

                    if potential_size <= self.chunk_size:
                        current_chunk["sentences"].append(sentence)
                        current_chunk["text"] = " ".join(current_chunk["sentences"])
                        current_chunk["char_count"] = len(current_chunk["text"])
                    else:
                        if current_chunk["char_count"] >= self.min_chunk_size:
                            full_text = (
                                self._prepare_context_text(context_elements)
                                + current_chunk["text"]
                                if not chunks
                                else current_chunk["text"]
                            )

                            chunks.append(
                                {
                                    "page": page_num,
                                    "text": full_text,
                                    "type": "text",
                                    "source": "semantic_with_context",
                                    "sentence_count": len(current_chunk["sentences"]),
                                    "has_context": bool(context_elements)
                                    and not chunks,
                                }
                            )

                        overlap_sentences = self._get_overlap_sentences(
                            current_chunk["sentences"]
                        )

                        current_chunk = {
                            "sentences": overlap_sentences + [sentence],
                            "text": " ".join(overlap_sentences + [sentence]),
                            "char_count": len(" ".join(overlap_sentences + [sentence])),
                        }

                if current_chunk["char_count"] >= self.min_chunk_size:
                    full_text = (
                        self._prepare_context_text(context_elements)
                        + current_chunk["text"]
                        if not chunks
                        else current_chunk["text"]
                    )

                    chunks.append(
                        {
                            "page": page_num,
                            "text": full_text,
                            "type": "text",
                            "source": "semantic_with_context",
                            "sentence_count": len(current_chunk["sentences"]),
                            "has_context": bool(context_elements) and not chunks,
                        }
                    )

            for elem in page_elements:
                if elem.get("type") in ["table", "image"]:
                    chunks.append(
                        {
                            "page": page_num,
                            "text": elem.get("text", ""),
                            "type": elem.get("type"),
                            "source": f"{elem.get('type')}_standalone",
                            "metadata": elem.get("metadata", {}),
                        }
                    )

        return chunks

    def _adaptive_chunk_with_context(
        self, elements: List[Dict], page_num: int
    ) -> List[Dict[str, Any]]:
        chunks = []

        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        text_elements = [e for e in page_elements if e.get("type") == "text"]
        table_elements = [e for e in page_elements if e.get("type") == "table"]
        image_elements = [e for e in page_elements if e.get("type") == "image"]

        if text_elements:
            text_content = "\n\n".join(e.get("text", "") for e in text_elements)
            text_chunks = self._semantic_split_text(
                text_content, page_num, context_elements
            )
            chunks.extend(text_chunks)

        for table in table_elements:
            table_text = table.get("text", "")
            if len(table_text) <= self.chunk_size * 1.5:
                chunks.append(
                    {
                        "page": page_num,
                        "text": table_text,
                        "type": "table",
                        "source": "table_standalone",
                        "metadata": table.get("metadata", {}),
                    }
                )
            else:
                table_chunks = self._split_large_table(table_text, page_num)
                chunks.extend(table_chunks)

        for image in image_elements:
            chunks.append(
                {
                    "page": page_num,
                    "text": image.get("text", ""),
                    "type": "image",
                    "source": "image_standalone",
                    "metadata": image.get("metadata", {}),
                }
            )

        return chunks

    def _semantic_split_text(
        self, text: str, page_num: int, context_elements: List[Dict] = None
    ) -> List[Dict]:
        if not text.strip():
            return []

        sentences = self.sent_tokenize(text) if self.sent_tokenize else text.split(". ")

        chunks = []
        current_chunk = {"sentences": [], "text": "", "char_count": 0}
        first_chunk = True

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_size = current_chunk["char_count"] + len(sentence) + 1

            if potential_size <= self.chunk_size:
                current_chunk["sentences"].append(sentence)
                current_chunk["text"] = " ".join(current_chunk["sentences"])
                current_chunk["char_count"] = len(current_chunk["text"])
            else:
                if current_chunk["char_count"] >= self.min_chunk_size:
                    full_text = (
                        self._prepare_context_text(context_elements)
                        + current_chunk["text"]
                        if first_chunk and context_elements
                        else current_chunk["text"]
                    )

                    chunks.append(
                        {
                            "page": page_num,
                            "text": full_text,
                            "type": "text",
                            "source": "semantic_with_context",
                            "sentence_count": len(current_chunk["sentences"]),
                            "has_context": bool(context_elements) and first_chunk,
                        }
                    )
                    first_chunk = False

                overlap_sentences = self._get_overlap_sentences(
                    current_chunk["sentences"]
                )

                current_chunk = {
                    "sentences": overlap_sentences + [sentence],
                    "text": " ".join(overlap_sentences + [sentence]),
                    "char_count": len(" ".join(overlap_sentences + [sentence])),
                }

        if current_chunk["char_count"] >= self.min_chunk_size:
            full_text = (
                self._prepare_context_text(context_elements) + current_chunk["text"]
                if first_chunk and context_elements
                else current_chunk["text"]
            )

            chunks.append(
                {
                    "page": page_num,
                    "text": full_text,
                    "type": "text",
                    "source": "semantic_with_context",
                    "sentence_count": len(current_chunk["sentences"]),
                    "has_context": bool(context_elements) and first_chunk,
                }
            )

        return chunks

    def _simple_chunk_with_context(
        self, elements: List[Dict], page_num: int
    ) -> List[Dict[str, Any]]:
        chunks = []

        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        all_text = "\n\n".join(
            e.get("text", "") for e in page_elements if e.get("text", "").strip()
        )

        if all_text.strip():
            start = 0
            first_chunk = True

            while start < len(all_text):
                end = min(start + self.chunk_size, len(all_text))
                chunk_text = all_text[start:end]

                if len(chunk_text) >= self.min_chunk_size:
                    full_text = (
                        self._prepare_context_text(context_elements) + chunk_text
                        if first_chunk and context_elements
                        else chunk_text
                    )

                    chunks.append(
                        {
                            "page": page_num,
                            "text": full_text,
                            "type": "text",
                            "source": "simple_with_context",
                            "has_context": bool(context_elements) and first_chunk,
                        }
                    )
                    first_chunk = False

                start = end - self.overlap if end < len(all_text) else end

        for elem in page_elements:
            if elem.get("type") in ["table", "image"]:
                chunks.append(
                    {
                        "page": page_num,
                        "text": elem.get("text", ""),
                        "type": elem.get("type"),
                        "source": f"{elem.get('type')}_standalone",
                        "metadata": elem.get("metadata", {}),
                    }
                )

        return chunks

    def _prepare_context_text(self, context_elements: List[Dict]) -> str:
        if not context_elements:
            return ""

        context_texts = []
        for elem in context_elements:
            elem_text = elem.get("text", "").strip()
            if elem_text:
                if elem.get("type") in ["heading", "title"]:
                    context_texts.append(f"## {elem_text}")
                else:
                    context_texts.append(elem_text)

        if context_texts:
            return "\n\n".join(context_texts) + "\n\n"

        return ""

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        overlap_sentences = []
        char_count = 0

        for sentence in reversed(sentences):
            if char_count + len(sentence) <= self.overlap:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence) + 1
            else:
                break

        return overlap_sentences

    def _split_large_table(self, table_text: str, page_num: int) -> List[Dict]:
        rows = table_text.split("\n")

        if not rows:
            return []

        header = rows[0] if rows else ""
        data_rows = rows[1:] if len(rows) > 1 else []

        chunks = []
        current_rows = [header]
        current_size = len(header)

        for row in data_rows:
            row_size = len(row)

            if current_size + row_size <= self.chunk_size * 1.5:
                current_rows.append(row)
                current_size += row_size
            else:
                if len(current_rows) > 1:
                    chunks.append(
                        {
                            "page": page_num,
                            "text": "\n".join(current_rows),
                            "type": "table",
                            "source": "table_split",
                            "is_continuation": len(chunks) > 0,
                        }
                    )

                current_rows = [header, row]
                current_size = len(header) + row_size

        if len(current_rows) > 1:
            chunks.append(
                {
                    "page": page_num,
                    "text": "\n".join(current_rows),
                    "type": "table",
                    "source": "table_split",
                    "is_continuation": len(chunks) > 0,
                }
            )

        return chunks

    def _validate_and_fix_chunks(
        self, chunks: List[Dict], boundaries: Dict[int, Dict[str, float]]
    ) -> List[Dict]:
        validated = []

        for i, chunk in enumerate(chunks):
            page = chunk.get("page", 1)
            if not isinstance(page, int) or page < 1:
                logger.error(f"[CHUNK] Invalid page in chunk {i}: {page}")
                chunk["page"] = 1

            if boundaries and page > max(boundaries.keys()):
                logger.warning(
                    f"[CHUNK] Page {page} exceeds expected max {max(boundaries.keys())}"
                )
                chunk["page"] = max(boundaries.keys())

            text = chunk.get("text", "").strip()
            if text and len(text) >= self.min_chunk_size:
                chunk["text"] = text
                chunk["chunk_id"] = i
                chunk["char_count"] = len(text)
                validated.append(chunk)
            elif text:
                logger.debug(f"[CHUNK] Skipping small chunk {i}: {len(text)} chars")

        logger.info(f"[CHUNK] Validated {len(validated)}/{len(chunks)} chunks")

        return validated

    @staticmethod
    def _fallback_sentence_split(text: str) -> List[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁ-Ú])", text)
        return [s.strip() for s in sentences if s.strip()]


class PageSequenceValidator:
    """Validate and fix page number sequences"""

    @staticmethod
    def validate_and_fix(chunks: list, total_pages: int = None) -> tuple[list, list]:
        """
        Validate page sequence and attempt to fix issues
        Returns: (fixed_chunks, issues_found)
        """
        if not chunks:
            return chunks, []

        issues = []

        # Extract all pages
        pages = [c.get("page", 1) for c in chunks]

        # Detect issues
        issues.extend(PageSequenceValidator._detect_invalid_pages(pages))
        issues.extend(PageSequenceValidator._detect_large_gaps(pages))
        issues.extend(PageSequenceValidator._detect_out_of_order(pages))
        if total_pages:
            issues.extend(
                PageSequenceValidator._detect_page_overflow(pages, total_pages)
            )

        # Attempt fixes if issues found
        if issues:
            logger.warning(f"[PAGE] Found {len(issues)} issues, attempting fixes...")
            chunks = PageSequenceValidator._fix_page_numbers(chunks, pages, total_pages)

        return chunks, issues

    @staticmethod
    def _detect_invalid_pages(pages: list) -> list:
        """Detect invalid page numbers"""
        issues = []
        for i, page in enumerate(pages):
            if not isinstance(page, int) or page < 1:
                issues.append(f"Chunk {i}: invalid page number {page}")
            elif page > 50000:  # Unrealistic page number
                issues.append(f"Chunk {i}: suspicious page number {page}")
        return issues

    @staticmethod
    def _detect_large_gaps(pages: list) -> list:
        """Detect large gaps in page sequence"""
        issues = []
        if not pages:
            return issues

        sorted_unique = sorted(set(p for p in pages if isinstance(p, int) and p > 0))

        for i in range(1, len(sorted_unique)):
            gap = sorted_unique[i] - sorted_unique[i - 1]
            if gap > 10:  # Large gap
                issues.append(
                    f"Large gap: {gap} pages between {sorted_unique[i - 1]} and {sorted_unique[i]}"
                )

        return issues

    @staticmethod
    def _detect_out_of_order(pages: list) -> list:
        """Detect out-of-order pages"""
        issues = []

        for i in range(1, len(pages)):
            if pages[i] < pages[i - 1]:
                issues.append(
                    f"Out of order: chunk {i - 1} page {pages[i - 1]} -> chunk {i} page {pages[i]}"
                )

        return issues

    @staticmethod
    def _detect_page_overflow(pages: list, total_pages: int) -> list:
        """Detect pages that exceed the total page count"""
        issues = []
        for i, page in enumerate(pages):
            if isinstance(page, int) and page > total_pages:
                issues.append(
                    f"Chunk {i}: page {page} exceeds total pages {total_pages}"
                )
        return issues

    @staticmethod
    def _fix_page_numbers(chunks: list, pages: list, total_pages: int = None) -> list:
        """Attempt to fix page number issues"""
        fixed_chunks = chunks.copy()

        # Fix 1: Replace invalid pages with sequential numbers
        last_valid_page = 1
        for i, chunk in enumerate(fixed_chunks):
            page = chunk.get("page", 1)

            if not isinstance(page, int) or page < 1 or page > 50000:
                # Use sequential numbering
                chunk["page"] = last_valid_page
                logger.debug(f"[FIX] Chunk {i}: invalid page -> {last_valid_page}")
            else:
                last_valid_page = page

        # Fix 2: Apply total_pages constraint
        if total_pages:
            for i, chunk in enumerate(fixed_chunks):
                page = chunk.get("page", 1)
                if isinstance(page, int) and page > total_pages:
                    chunk["page"] = total_pages
                    logger.debug(
                        f"[FIX] Chunk {i}: page {page} -> {total_pages} (clamped)"
                    )

        # Fix 3: Smooth out large jumps
        for i in range(1, len(fixed_chunks)):
            prev_page = fixed_chunks[i - 1].get("page", 1)
            curr_page = fixed_chunks[i].get("page", 1)

            if curr_page - prev_page > 20:  # Suspicious jump
                # Assume sequential
                fixed_chunks[i]["page"] = prev_page + 1
                logger.debug(
                    f"[FIX] Chunk {i}: large jump {prev_page}->{curr_page}, "
                    f"using {prev_page + 1}"
                )

        return fixed_chunks


class RobustPageExtractor:
    """Multi-strategy page number extraction with validation"""

    _cache = {}  # element cache
    _pdf_layout_cache: Dict[str, List[float]] = {}  # pdf_path -> page heights cache

    @staticmethod
    def extract_page_number(elem, pdf_path: str = None, fallback_page: int = 1) -> int:
        """
        Extract page number with multiple validation strategies
        Priority order:
        1. Element metadata (page_number)
        2. Element coordinates (for PDFs)
        3. Sequential inference from previous elements
        4. Text content analysis (page markers)
        5. Fallback to default
        """
        elem_id = id(elem)
        if elem_id in RobustPageExtractor._cache:
            return RobustPageExtractor._cache[elem_id]

        page = RobustPageExtractor._extract_from_metadata(elem)
        if page is not None:
            RobustPageExtractor._cache[elem_id] = page
            return page

        if pdf_path and pdf_path.lower().endswith(".pdf"):
            page = RobustPageExtractor._extract_from_coordinates(elem, pdf_path)
            if page is not None:
                RobustPageExtractor._cache[elem_id] = page
                return page

        page = RobustPageExtractor._extract_from_text_content(elem)
        if page is not None:
            RobustPageExtractor._cache[elem_id] = page
            return page

        page = RobustPageExtractor._infer_from_position(elem)
        if page is not None:
            RobustPageExtractor._cache[elem_id] = page
            return page

        logger.warning(
            f"[PAGE] Could not extract page for {elem.__class__.__name__}, "
            f"using fallback: {fallback_page}"
        )
        RobustPageExtractor._cache[elem_id] = fallback_page
        return fallback_page

    @staticmethod
    def _extract_from_metadata(elem) -> Optional[int]:
        try:
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                page = elem.metadata.page_number
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

            if hasattr(elem, "metadata") and hasattr(elem.metadata, "to_dict"):
                meta_dict = elem.metadata.to_dict()
                if "page_number" in meta_dict:
                    page = meta_dict["page_number"]
                    if isinstance(page, (int, float)) and page > 0:
                        return int(page)

            if hasattr(elem, "page"):
                page = elem.page
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"[PAGE] Metadata extraction failed: {e}")

        return None

    @staticmethod
    def _extract_from_coordinates(elem, pdf_path: str = None) -> Optional[int]:
        """Extract page from element coordinates (uses cached PDF layout for speed)."""
        try:
            if not hasattr(elem, "metadata") or not hasattr(
                elem.metadata, "coordinates"
            ):
                return None

            coords = elem.metadata.coordinates
            if not hasattr(coords, "points") or not coords.points:
                return None

            y_coords = [p[1] for p in coords.points if len(p) >= 2]
            if not y_coords:
                return None

            min_y = min(y_coords)

            heights: Optional[List[float]] = None
            if pdf_path:
                heights = RobustPageExtractor._pdf_layout_cache.get(pdf_path)
                if heights is None:
                    try:
                        with pdfplumber.open(pdf_path) as pdf:
                            heights = [float(p.height) for p in pdf.pages]
                        RobustPageExtractor._pdf_layout_cache[pdf_path] = heights
                    except Exception as e:
                        logger.debug(f"[PAGE] PDF layout cache fill failed: {e}")
                        heights = None

            if heights:
                cumulative = 0.0
                for i, h in enumerate(heights):
                    if cumulative <= min_y < cumulative + h:
                        estimated_page = i + 1
                        break
                    cumulative += h
                else:
                    estimated_page = len(heights)
            else:
                estimated_page = max(1, int(min_y / 842) + 1)

            if estimated_page > 10000:
                logger.warning(f"[PAGE] Suspicious page from coords: {estimated_page}")
                return None

            logger.debug(f"[PAGE] Estimated from coordinates: {estimated_page}")
            return estimated_page

        except Exception as e:
            logger.debug(f"[PAGE] Coordinate extraction failed: {e}")
            return None

    @staticmethod
    def _extract_from_text_content(elem) -> Optional[int]:
        try:
            if not hasattr(elem, "text"):
                return None

            text = elem.text.strip()
            if not text:
                return None

            patterns = [
                r"(?:page|página|p\.|pg\.?)\s*(\d+)",
                r"^\s*(\d+)\s*$",
                r"\[(\d+)\]",
            ]

            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    page_num = int(match.group(1))
                    if 1 <= page_num <= 10000:
                        logger.debug(f"[PAGE] Found in text: {page_num}")
                        return page_num

        except Exception as e:
            logger.debug(f"[PAGE] Text content extraction failed: {e}")

        return None

    @staticmethod
    def _infer_from_position(elem) -> Optional[int]:
        try:
            if hasattr(elem, "id") and isinstance(elem.id, str):
                match = re.search(r"page[_-]?(\d+)", elem.id, re.IGNORECASE)
                if match:
                    page_num = int(match.group(1))
                    if 1 <= page_num <= 10000:
                        logger.debug(f"[PAGE] Inferred from ID: {page_num}")
                        return page_num

        except Exception as e:
            logger.debug(f"[PAGE] Position inference failed: {e}")

        return None


def pdf_to_chunks_with_enhanced_validation(
    pdf_path: str,
    chunk_size: int = 900,
    overlap: int = 120,
    use_adaptive_chunking: bool = True,
    add_context: bool = True,
    embed_model=None,
    validate_pages: bool = True,
    detect_visual_boundaries: bool = True,
) -> List[Dict[str, Any]]:
    """
    Enhanced PDF to chunks conversion with improved page validation and chunking
    """
    pdf_path_p = Path(pdf_path)

    total_pages = get_pdf_total_pages(str(pdf_path_p))
    has_extractable_text = check_pdf_has_text(pdf_path_p)

    if has_extractable_text:
        logger.info(
            "PDF appears to have extractable text, attempting text extraction..."
        )
        chunks = extract_text_with_multiple_methods(pdf_path_p)
        if chunks:
            logger.info(f"Successfully extracted {len(chunks)} text chunks")
            return chunks
        else:
            logger.warning("Text extraction methods failed, falling back to OCR...")

    elements = extract_elements_best_effort(
        str(pdf_path_p),
        has_extractable_text=has_extractable_text,
        total_pages=total_pages,
    )

    if not elements:
        raise ValueError(f"No elements extracted from PDF: {pdf_path}")

    # Use ContextAwareChunker directly instead of AdaptiveChunkingStrategySelector
    chunker = ContextAwareChunker(chunk_size, overlap)
    chunks = chunker.chunk_with_context_preservation(
        elements,
        pdf_path=str(pdf_path_p) if detect_visual_boundaries else None,
        strategy="adaptive",
    )

    if validate_pages:
        # Use comprehensive page validation
        chunks, issues = PageSequenceValidator.validate_and_fix(chunks, total_pages)

        if issues and len(issues) > 30:  # Too many issues = reject file
            raise ValueError(
                f"Page validation failed: {len(issues)} critical issues: {issues[:10]}"
            )

    return chunks


def check_pdf_has_text(pdf_path: Union[Path, str]) -> bool:
    """
    Check if PDF likely has extractable text using multiple methods.
    Accepts Path or str for call-site compatibility.
    """
    try:
        pdf_path = Path(pdf_path)

        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            page_count = len(pdf_reader.pages)

            for i in range(min(10, page_count)):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    logger.info(f"pypdf detected text on page {i+1}")
                    return True

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:10]):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    logger.info(f"pdfplumber detected text on page {i+1}")
                    return True

        logger.info("No extractable text detected in PDF")
        return False
    except Exception as e:
        logger.warning(f"Error checking for extractable text: {e}")
        return True


def process_elements_to_chunks(elements: List[Any]) -> List[Dict[str, Any]]:
    """
    Process unstructured elements into chunks
    """
    chunks = []

    for element in elements:
        chunk = {
            "type": element.category if hasattr(element, "category") else "unknown",
            "text": str(element),
            "page": (
                element.metadata.page_number
                if hasattr(element, "metadata") and element.metadata
                else 1
            ),
        }

        if hasattr(element, "metadata") and element.metadata:
            if hasattr(element.metadata, "coordinates"):
                chunk["coordinates"] = element.metadata.coordinates
            if hasattr(element.metadata, "filename"):
                chunk["source_file"] = element.metadata.filename

        chunks.append(chunk)

    return chunks


def extract_text_with_multiple_methods(
    pdf_path: Union[Path, str]
) -> List[Dict[str, Any]]:
    """
    Hybrid extraction: pypdf for text (fast) + pdfplumber for tables.
    All methods preserve accurate page numbers.
    Fallback to unstructured if both fail.
    """
    pdf_path = Path(pdf_path)
    total_pages = get_pdf_total_pages(str(pdf_path))

    # 1. Hybrid approach: pypdf (text) + pdfplumber (tables)
    try:
        elements = []

        # 1a. Fast text extraction with pypdf
        logger.info("Trying pypdf (fastest) for text...")
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    elements.append(
                        {
                            "type": "text",
                            "text": text,
                            "page": i + 1,
                            "source": "pypdf",
                        }
                    )

        # 1b. Table extraction with pdfplumber (only if text was found)
        if elements:
            logger.info(
                f"pypdf extracted {len(elements)} text elements, now extracting tables..."
            )
            table_count = 0
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        try:
                            for table in page.extract_tables():
                                if table:
                                    processed_table = []
                                    for row in table:
                                        processed_row = [
                                            str(cell) if cell is not None else ""
                                            for cell in row
                                        ]
                                        processed_table.append(processed_row)

                                    table_text = "\n".join(
                                        ["\t".join(row) for row in processed_table]
                                    )
                                    if table_text.strip():
                                        elements.append(
                                            {
                                                "type": "table",
                                                "text": table_text,
                                                "page": i + 1,
                                                "source": "pdfplumber",
                                            }
                                        )
                                        table_count += 1
                        except Exception as table_error:
                            logger.warning(
                                f"Error extracting table from page {i+1}: {table_error}"
                            )
                            continue
            except Exception as e:
                logger.warning(f"pdfplumber table extraction failed: {e}")

            logger.info(
                f"Extracted {len(elements)} elements ({len(elements) - table_count} text, {table_count} tables)"
            )
            return elements

    except Exception as e:
        logger.warning(f"Hybrid pypdf+pdfplumber failed: {e}")

    # 2. Fallback: pdfplumber for both text and tables
    try:
        logger.info("Trying pdfplumber for text+tables...")
        with pdfplumber.open(pdf_path) as pdf:
            elements = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    elements.append(
                        {
                            "type": "text",
                            "text": text,
                            "page": i + 1,
                            "source": "pdfplumber",
                        }
                    )

                try:
                    for table in page.extract_tables():
                        if table:
                            processed_table = []
                            for row in table:
                                processed_row = [
                                    str(cell) if cell is not None else ""
                                    for cell in row
                                ]
                                processed_table.append(processed_row)

                            table_text = "\n".join(
                                ["\t".join(row) for row in processed_table]
                            )
                            elements.append(
                                {
                                    "type": "table",
                                    "text": table_text,
                                    "page": i + 1,
                                    "source": "pdfplumber",
                                }
                            )
                except Exception as table_error:
                    logger.warning(
                        f"Error extracting table from page {i+1}: {table_error}"
                    )
                    continue

            if elements:
                logger.info(f"pdfplumber extracted {len(elements)} elements")
                return elements
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    # 3. Last resort: unstructured (may use internal OCR)
    try:
        logger.info("Trying unstructured partition_pdf...")
        elements = fast_partition_pdf(
            str(pdf_path), strategy="auto", enable_tables=False, total_pages=total_pages
        )

        if elements and len(elements) > 0:
            text_content = "\n".join([str(e) for e in elements if hasattr(e, "text")])
            if len(text_content.strip()) > 100:
                logger.info(
                    f"unstructured partition_pdf extracted {len(elements)} elements"
                )
                return process_elements_to_chunks(elements)
    except Exception as e:
        logger.warning(f"unstructured partition_pdf failed: {e}")

    logger.warning("All text extraction methods failed")
    return []


class EasyOCRProcessor:
    """GPU-accelerated OCR using EasyOCR with PyTorch backend"""

    _reader_cache = None
    _initialization_attempted = False

    def __init__(self, use_gpu=True, gpu_id=0, model_storage_directory=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if not self.use_gpu and use_gpu:
            logger.warning("[EASYOCR] CUDA not available, falling back to CPU")

        if model_storage_directory:
            os.environ["EASYOCR_MODULE_PATH"] = str(model_storage_directory)
            self.model_dir = Path(model_storage_directory)
        else:
            self.model_dir = Path.home() / ".EasyOCR"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[EASYOCR] Model directory: {self.model_dir}")

        if EasyOCRProcessor._reader_cache is not None:
            logger.info("[EASYOCR] Using cached reader")
            self.reader = EasyOCRProcessor._reader_cache
            return

        if (
            EasyOCRProcessor._initialization_attempted
            and EasyOCRProcessor._reader_cache is None
        ):
            raise RuntimeError("[EASYOCR] Previous initialization failed, not retrying")

        try:
            EasyOCRProcessor._initialization_attempted = True

            setup_ssl_context()

            models_exist = self._check_models_exist()

            if not models_exist:
                logger.info("[EASYOCR] Models not found, will download on first use")
                logger.info(
                    "[EASYOCR] This may take 2-5 minutes depending on network speed"
                )
            else:
                logger.info("[EASYOCR] Using existing models")

            logger.info("[EASYOCR] Initializing reader...")
            self.reader = easyocr.Reader(
                ["es", "en"],
                gpu=self.use_gpu,
                model_storage_directory=str(self.model_dir),
                download_enabled=True,
                detector=True,
                recognizer=True,
                verbose=False,
            )

            EasyOCRProcessor._reader_cache = self.reader

            device = "GPU" if self.use_gpu else "CPU"
            logger.info(f"[EASYOCR] ✓ Successfully initialized on {device}")

        except Exception as e:
            logger.error(f"[EASYOCR] Initialization failed: {e}")
            logger.info(
                "[EASYOCR] Try pre-downloading models manually (see instructions below)"
            )
            raise

    def _check_models_exist(self) -> bool:
        try:
            craft_path = self.model_dir / "model" / "craft_mlt_25k.pth"
            spanish_path = self.model_dir / "model" / "spanish_g2.pth"
            english_path = self.model_dir / "model" / "english_g2.pth"

            models_exist = craft_path.exists() and (
                spanish_path.exists() or english_path.exists()
            )

            if models_exist:
                logger.info(f"[EASYOCR] Found existing models in {self.model_dir}")
            else:
                logger.info(f"[EASYOCR] Models not found, will download")
                logger.info(f"[EASYOCR] Detection model: {craft_path.exists()}")
                logger.info(f"[EASYOCR] Spanish model: {spanish_path.exists()}")
                logger.info(f"[EASYOCR] English model: {english_path.exists()}")

            return models_exist
        except Exception as e:
            logger.warning(f"[EASYOCR] Could not check for existing models: {e}")
            return False

    def extract_text_from_image(self, image_path: str) -> str:
        try:
            results = self.reader.readtext(image_path)

            if not results:
                return ""

            text_lines = []
            for bbox, text, confidence in results:
                if confidence > 0.6:
                    text_lines.append(text.strip())

            return "\n".join(text_lines)

        except Exception as e:
            logger.error(f"[EASYOCR] Failed to process {image_path}: {e}")
            return ""

    def process_pdf_page_batch(self, pdf_path: str, page_nums: list) -> dict:
        from pdf2image import convert_from_path
        import tempfile as _tempfile

        logger.debug(
            f"[EASYOCR] Processing pages {page_nums} for {os.path.basename(pdf_path)}"
        )

        first_page = min(page_nums)
        last_page = max(page_nums)

        images = convert_from_path(
            pdf_path,
            first_page=first_page,
            last_page=last_page,
            dpi=300,
            thread_count=4,
        )

        results = {}
        temp_files = []

        try:
            for idx, image in enumerate(images):
                actual_page_num = first_page + idx
                if actual_page_num in page_nums:
                    tmp = _tempfile.NamedTemporaryFile(
                        suffix=f"_page_{actual_page_num}.png", delete=False
                    )
                    image.save(tmp.name)
                    temp_files.append((actual_page_num, tmp.name))

            for page_num, tmp_path in temp_files:
                text = self.extract_text_from_image(tmp_path)
                results[page_num] = text

        finally:
            for _, tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        return results


def _partition_pdf_compat(filename: str, **kwargs):
    """Call unstructured.partition.pdf.partition_pdf passing only supported kwargs."""
    from unstructured.partition.pdf import partition_pdf as _partition_pdf

    sig = inspect.signature(_partition_pdf)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return _partition_pdf(filename=filename, **filtered)


def _slice_pdf_to_temp(pdf_path: str, pages_1_indexed: List[int]) -> Tuple[str, int]:
    """Create a temporary PDF containing only the requested 1-indexed pages."""
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore

    pages = sorted(set(int(p) for p in pages_1_indexed))
    start_page = pages[0]

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for p in pages:
        writer.add_page(reader.pages[p - 1])

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    with open(tmp.name, "wb") as f:
        writer.write(f)
    return tmp.name, start_page


def fast_partition_pdf(
    pdf_path: str,
    strategy: str = "auto",
    *,
    pages: Optional[List[int]] = None,  # 1-indexed page numbers
    enable_tables: bool = False,  # tables are expensive; default OFF (speed)
    hi_res_model_name: Optional[str] = None,
    total_pages: Optional[int] = None,
) -> List[Any]:
    """
    Speed + page-accuracy optimized wrapper around unstructured.partition_pdf.

    Fixes:
      - supports page restriction (pages=...) to avoid reprocessing whole PDFs in batches
      - avoids table inference by default (enable_tables=False) for speed
      - filters kwargs to what the installed unstructured version supports
      - when slicing PDFs, remaps page_number to original numbering if needed
    """
    if total_pages is None:
        total_pages = get_pdf_total_pages(str(pdf_path))

    pages_desc = "all" if not pages else f"{min(pages)}-{max(pages)}"
    logger.info(
        f"[UNSTRUCTURED] strategy={strategy} pages={pages_desc} total_pages={total_pages or 'unknown'}"
    )

    kwargs: Dict[str, Any] = {
        "strategy": strategy,
        "languages": ["spa", "eng"],
        "extract_images_in_pdf": False,
        "chunking_strategy": None,
        "keep_extra_chars": False,
        "max_characters": 15000,
        "extract_tables": bool(enable_tables),
        "infer_table_structure": bool(enable_tables),
        "multipage_sections": True,
    }

    if strategy == "fast":
        kwargs["max_characters"] = 20000
        kwargs["extract_tables"] = False
        kwargs["infer_table_structure"] = False

    elif strategy == "hi_res":
        kwargs["max_characters"] = 10000
        kwargs["model_name"] = "yolox"
        kwargs["skip_infer_table_types"] = ["pdf", "jpg", "png"]
        if hi_res_model_name is not None:
            kwargs["hi_res_model_name"] = hi_res_model_name

    elif strategy == "ocr_only":
        kwargs.update(
            {
                "ocr_languages": "spa+eng",
                "ocr_mode": "entire_page",
                "max_characters": 12000,
                "ocr_kwargs": {"config": "--oem 3 --psm 6"},
            }
        )

    if not pages:
        return _partition_pdf_compat(str(pdf_path), **kwargs)

    from unstructured.partition.pdf import partition_pdf as _partition_pdf

    sig = inspect.signature(_partition_pdf)

    if "page_numbers" in sig.parameters:
        kwargs["page_numbers"] = pages
        return _partition_pdf_compat(str(pdf_path), **kwargs)

    if "pages" in sig.parameters:
        kwargs["pages"] = pages
        return _partition_pdf_compat(str(pdf_path), **kwargs)

    pages_sorted = sorted(set(int(p) for p in pages))
    runs: List[List[int]] = []
    run: List[int] = [pages_sorted[0]]
    for p in pages_sorted[1:]:
        if p == run[-1] + 1:
            run.append(p)
        else:
            runs.append(run)
            run = [p]
    runs.append(run)

    out: List[Any] = []
    has_starting = "starting_page_number" in sig.parameters

    for r in runs:
        tmp_path, start_page = _slice_pdf_to_temp(str(pdf_path), r)
        try:
            if has_starting:
                kwargs["starting_page_number"] = start_page
            elems = _partition_pdf_compat(tmp_path, **kwargs)

            if not has_starting:
                offset = start_page - 1
                for e in elems:
                    try:
                        if (
                            hasattr(e, "metadata")
                            and getattr(e, "metadata", None)
                            and hasattr(e.metadata, "page_number")
                        ):
                            pn = getattr(e.metadata, "page_number", None)
                            if isinstance(pn, (int, float)):
                                e.metadata.page_number = int(pn) + offset
                    except Exception:
                        pass

            out.extend(elems)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return out


def extract_elements_best_effort(
    pdf_path: str,
    *,
    has_extractable_text: Optional[bool] = None,
    total_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Unified extraction entrypoint (Fixes #1 and #5):
      - If the PDF has extractable text: run ONE hi_res pass (tables off) and accept if sufficient.
      - Otherwise: run OCR pipeline (EasyOCR -> Tesseract) once.
      - Validate page numbers once at the end.

    Priority: page accuracy first, then speed.
    """
    if total_pages is None:
        total_pages = get_pdf_total_pages(str(pdf_path))

    if has_extractable_text is None:
        has_extractable_text = check_pdf_has_text(pdf_path)

    if has_extractable_text:
        try:
            raw = fast_partition_pdf(
                str(pdf_path),
                strategy="hi_res",
                enable_tables=False,
                total_pages=total_pages,
            )

            processed: List[Dict[str, Any]] = []
            for elem in raw:
                d = elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
                if (
                    "page" not in d
                    and hasattr(elem, "metadata")
                    and getattr(elem, "metadata", None)
                ):
                    pn = getattr(elem.metadata, "page_number", None)
                    if pn:
                        d["page"] = pn
                d["page"] = int(d.get("page", 1) or 1)
                d.setdefault("type", d.get("category", "text"))
                d["source"] = "unstructured_hi_res"
                processed.append(d)

            text_chars = sum(len((p.get("text") or "").strip()) for p in processed)
            avg = text_chars / max(total_pages or 1, 1)

            if processed and text_chars > 500 and avg > 80:
                return validate_page_numbers_with_count(
                    processed, str(pdf_path), total_pages
                )

            logger.info("[EXTRACT] hi_res insufficient -> OCR pipeline")

        except Exception as e:
            logger.warning(f"[EXTRACT] hi_res failed -> OCR pipeline: {e}")

    return extract_elements_from_pdf_gpu(
        str(pdf_path),
        has_extractable_text=False,
        total_pages=total_pages,
    )


def batch_unstructured_processing(pdf_path: str) -> List[Any]:
    """
    Backwards-compatible entrypoint.

    Fixes:
      - removes per-batch trial-and-error (Fix #1)
      - avoids repeated PDF opens for page count (Fix #3)
      - avoids validating per batch (Fix #4)

    Now routes through extract_elements_best_effort().
    """
    return extract_elements_best_effort(str(pdf_path))


def extract_elements_from_pdf_gpu(
    pdf_path: str,
    has_extractable_text: bool = True,
    total_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    OCR extraction pipeline with accurate page numbers and minimal redundancy.

    Fixes:
      - uses cached page count (Fix #3)
      - avoids redundant strategy attempts (Fix #1)
      - validates page numbers once at the end (Fix #4)

    Behavior:
      - EasyOCR (GPU) if available
      - Fallback to Tesseract via unstructured ocr_only
    """
    if total_pages is None:
        total_pages = get_pdf_total_pages(str(pdf_path))

    num_pages = total_pages or 0
    elements: List[Dict[str, Any]] = []

    if EASYOCR_AVAILABLE:
        try:
            logger.info("[OCR] Using EasyOCR GPU pipeline")
            ocr_processor = EasyOCRProcessor(use_gpu=True, gpu_id=0)

            if num_pages > 0:
                BATCH_SIZE = 20
                for batch_start in range(1, num_pages + 1, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE - 1, num_pages)
                    page_nums = list(range(batch_start, batch_end + 1))

                    page_texts = ocr_processor.process_pdf_page_batch(
                        pdf_path, page_nums
                    )
                    for page_num, text in page_texts.items():
                        if text and text.strip():
                            elements.append(
                                {
                                    "text": text,
                                    "type": "text",
                                    "page": int(page_num),
                                    "source": "easyocr_gpu",
                                }
                            )
            else:
                logger.warning("[OCR] Unknown total pages; skipping EasyOCR batching")

            total_chars = sum(len(e.get("text", "")) for e in elements)
            if elements and total_chars > 300:
                return validate_page_numbers_with_count(
                    elements, str(pdf_path), total_pages
                )

            logger.info("[OCR] EasyOCR insufficient -> Tesseract fallback")
            elements = []

        except Exception as e:
            logger.warning(f"[OCR] EasyOCR failed -> Tesseract: {e}")
            elements = []

    try:
        logger.info("[OCR] Using Tesseract (unstructured ocr_only)")

        if num_pages > 0:
            BATCH_SIZE = 20
            for batch_start in range(1, num_pages + 1, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE - 1, num_pages)
                page_nums = list(range(batch_start, batch_end + 1))

                raw = fast_partition_pdf(
                    pdf_path,
                    strategy="ocr_only",
                    pages=page_nums,
                    enable_tables=False,
                    total_pages=total_pages,
                )

                for elem in raw:
                    d = (
                        elem.to_dict()
                        if hasattr(elem, "to_dict")
                        else {"text": str(elem)}
                    )
                    if (
                        "page" not in d
                        and hasattr(elem, "metadata")
                        and getattr(elem, "metadata", None)
                    ):
                        pn = getattr(elem.metadata, "page_number", None)
                        if pn:
                            d["page"] = pn
                    d["page"] = int(d.get("page", batch_start) or batch_start)
                    d.setdefault("type", d.get("category", "text"))
                    d["source"] = "tesseract_ocr"
                    elements.append(d)
        else:
            raw = fast_partition_pdf(
                pdf_path,
                strategy="ocr_only",
                enable_tables=False,
                total_pages=total_pages,
            )
            for elem in raw:
                d = elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
                if (
                    "page" not in d
                    and hasattr(elem, "metadata")
                    and getattr(elem, "metadata", None)
                ):
                    pn = getattr(elem.metadata, "page_number", None)
                    if pn:
                        d["page"] = pn
                d["page"] = int(d.get("page", 1) or 1)
                d.setdefault("type", d.get("category", "text"))
                d["source"] = "tesseract_ocr"
                elements.append(d)

        return validate_page_numbers_with_count(elements, str(pdf_path), total_pages)

    except Exception as e:
        logger.error(f"[OCR] Tesseract extraction failed: {e}")

    try:
        logger.info("[OCR] Last resort: unstructured auto (tables off)")
        raw = fast_partition_pdf(
            pdf_path, strategy="auto", enable_tables=False, total_pages=total_pages
        )
        processed: List[Dict[str, Any]] = []
        for elem in raw:
            d = elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
            if (
                "page" not in d
                and hasattr(elem, "metadata")
                and getattr(elem, "metadata", None)
            ):
                pn = getattr(elem.metadata, "page_number", None)
                if pn:
                    d["page"] = pn
            d["page"] = int(d.get("page", 1) or 1)
            d.setdefault("type", d.get("category", "text"))
            d["source"] = "basic_extraction"
            processed.append(d)
        return validate_page_numbers_with_count(processed, str(pdf_path), total_pages)
    except Exception as e:
        logger.error(f"[OCR] All extraction methods failed: {e}")
        return []


def validate_page_numbers(
    elements: List[Dict[str, Any]], pdf_path: str
) -> List[Dict[str, Any]]:
    """Legacy function - use validate_page_numbers_with_count for optimization"""
    return validate_page_numbers_with_count(elements, pdf_path, None)


def validate_page_numbers_with_count(
    elements: List[Dict[str, Any]], pdf_path: str, total_pages: int = None
) -> List[Dict[str, Any]]:
    """
    Page number validation and correction.

    Fix #4: This function must NOT open PDFs.
    Pass total_pages from get_pdf_total_pages() upstream when you want upper-bound clamping.
    """
    if not elements:
        return elements

    validated_elements: List[Dict[str, Any]] = []
    page_issues: List[str] = []

    for i, elem in enumerate(elements):
        page_num = elem.get("page", 1)

        try:
            page_num = int(page_num)
        except Exception:
            page_num = 1
            page_issues.append(f"Element {i}: Invalid page type, using 1")

        if page_num < 1:
            page_issues.append(f"Element {i}: Page {page_num} -> 1 (clamped)")
            page_num = 1

        if total_pages and page_num > total_pages:
            original_page = page_num
            page_num = total_pages
            page_issues.append(
                f"Element {i}: Page {original_page} -> {page_num} (clamped)"
            )

        elem["page"] = page_num
        validated_elements.append(elem)

    if page_issues:
        logger.warning(
            f"[PAGE] Fixed {len(page_issues)} page issues in {os.path.basename(str(pdf_path))}"
        )
        for issue in page_issues[:5]:
            logger.warning(f"[PAGE]   {issue}")
        if len(page_issues) > 5:
            logger.warning(f"[PAGE]   ... and {len(page_issues) - 5} more")

    return validated_elements


def extract_elements_from_pdf(
    pdf_path: str, num_processes: int = None
) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper.
    Uses the unified best-effort extractor so we don't maintain two parallel pipelines.
    """
    import multiprocessing

    if num_processes is None:
        cpu_count = multiprocessing.cpu_count()
        num_processes = max(1, int(cpu_count * 0.8))

    total_pages = get_pdf_total_pages(str(pdf_path))
    has_text = check_pdf_has_text(pdf_path)

    # If you want to actually use num_processes, extend fast_partition_pdf to accept n_jobs
    # and pass it through in extract_elements_best_effort. Otherwise keep as is.
    elements = extract_elements_best_effort(
        str(pdf_path),
        has_extractable_text=has_text,
        total_pages=total_pages,
    )
    return elements
