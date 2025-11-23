"""
chunk.py - Enhanced Version with Advanced Chunking and Page Validation
Extrae: PDF, DOCX, PPTX (sin XLSX)
Mantiene caché SQLite + LLaVA análisis
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
import numpy as np
import pdfplumber
import pypdf
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


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
                except:
                    return sent_tokenize(text, language="english")

            _cached_sent_tokenizer = spanish_tokenize
            logger.info("[NLTK] Spanish sentence tokenizer loaded")
        except Exception as e:
            logger.warning(f"[NLTK] Failed to load, using fallback: {e}")
            _cached_sent_tokenizer = ContextAwareChunker._fallback_sentence_split
    return _cached_sent_tokenizer


# ============================================================
# DISABLE CUDA FOR UNSTRUCTURED ONLY (prevent OOM with vLLM)
# ============================================================
os.environ["UNSTRUCTURED_DISABLE_CUDA"] = "false"
logger.warning("[CONFIG] CUDA DISABLED for Unstructured (vLLM-LLaVA uses GPU)")

# ============================================================
# CONFIGURATION: Deshabilitar LLaVA si es necesario
# ============================================================
DISABLE_LLAVA = os.getenv("DISABLE_LLAVA", "false").lower() == "true"
if DISABLE_LLAVA:
    logger.warning("[CONFIG] LLaVA analysis is DISABLED")
else:
    logger.info("[CONFIG] LLaVA analysis is ENABLED")


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
        except:
            return 0

    def _get_text_bottom(self, page) -> float:
        """Get the bottommost text coordinate on a page"""
        try:
            words = page.extract_words()
            if words:
                return max(word["bottom"] for word in words)
            return page.height
        except:
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
        # Use cached tokenizer (avoids repeated import overhead)
        self.sent_tokenize = get_sent_tokenizer()

        # Try to import NLTK for sentence tokenization
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
        # Detect page boundaries if PDF path is provided
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

        # Group elements by page with enhanced page detection
        page_groups = self._group_by_page_enhanced(elements, boundaries)

        # Process chunks with context preservation
        all_chunks = []

        for page_num in sorted(page_groups.keys()):
            page_elements = page_groups[page_num]

            # Add context from previous page if needed
            if page_num > 1 and page_num - 1 in page_groups:
                prev_page_elements = page_groups[page_num - 1]
                context_elements = self._select_context_elements(prev_page_elements)
                page_elements = context_elements + page_elements

            # Apply chunking strategy
            if strategy == "adaptive":
                page_chunks = self._adaptive_chunk_with_context(page_elements, page_num)
            elif strategy == "semantic":
                page_chunks = self._semantic_chunk_with_context(page_elements, page_num)
            else:
                page_chunks = self._simple_chunk_with_context(page_elements, page_num)

            all_chunks.extend(page_chunks)

        # Final validation to ensure page accuracy
        all_chunks = self._validate_and_fix_chunks(all_chunks, boundaries)

        return all_chunks

    def _group_by_page_enhanced(
        self, elements: List[Dict], boundaries: Dict[int, Dict[str, float]]
    ) -> Dict[int, List[Dict]]:
        """Group elements by page with enhanced page detection"""
        page_groups = defaultdict(list)

        for elem in elements:
            # Use enhanced page detection if boundaries are available
            if boundaries:
                page = self.page_detector.assign_precise_page(elem, boundaries)
            else:
                # Fall back to standard page extraction with PDF path
                page = RobustPageExtractor.extract_page_number(elem, pdf_path)

            # Validate page number
            if not isinstance(page, int) or page < 1:
                logger.warning(f"[CHUNK] Invalid page {page}, using page 1")
                page = 1

            page_groups[page].append(elem)

        return dict(page_groups)

    def _select_context_elements(self, prev_page_elements: List[Dict]) -> List[Dict]:
        """Select relevant elements from previous page for context"""
        # Prioritize headings and the last few text elements
        headings = [
            e for e in prev_page_elements if e.get("type") in ["heading", "title"]
        ]
        text_elements = [e for e in prev_page_elements if e.get("type") == "text"]

        # Take up to 2 headings and the last 3 text elements
        context = headings[:2]
        if len(text_elements) > 3:
            context.extend(text_elements[-3:])
        else:
            context.extend(text_elements)

        # Mark these as context elements
        for elem in context:
            elem["is_context"] = True
            elem["source_page"] = elem.get("page", 1)

        return context

    def _semantic_chunk_with_context(
        self, elements: List[Dict], page_num: int
    ) -> List[Dict[str, Any]]:
        """Semantic chunking with context preservation"""
        chunks = []

        # Separate context from actual page content
        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        # Process text with semantic chunking
        if page_elements:
            text_content = "\n\n".join(
                e.get("text", "") for e in page_elements if e.get("type") == "text"
            )

            if text_content.strip():
                # Tokenize into sentences
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

                    # Check if adding this sentence exceeds chunk size
                    potential_size = current_chunk["char_count"] + len(sentence) + 1

                    if potential_size <= self.chunk_size:
                        # Add to current chunk
                        current_chunk["sentences"].append(sentence)
                        current_chunk["text"] = " ".join(current_chunk["sentences"])
                        current_chunk["char_count"] = len(current_chunk["text"])
                    else:
                        # Save current chunk if it meets minimum size
                        if current_chunk["char_count"] >= self.min_chunk_size:
                            # Add context to the first chunk only
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

                        # Start new chunk with overlap
                        overlap_sentences = self._get_overlap_sentences(
                            current_chunk["sentences"]
                        )

                        current_chunk = {
                            "sentences": overlap_sentences + [sentence],
                            "text": " ".join(overlap_sentences + [sentence]),
                            "char_count": len(" ".join(overlap_sentences + [sentence])),
                        }

                # Add final chunk
                if current_chunk["char_count"] >= self.min_chunk_size:
                    # Add context to the first chunk only
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

            # Process non-text elements (tables, images)
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
        """Adaptive chunking based on content type and density"""
        chunks = []

        # Separate context from actual page content
        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        # Separate by element type
        text_elements = [e for e in page_elements if e.get("type") == "text"]
        table_elements = [e for e in page_elements if e.get("type") == "table"]
        image_elements = [e for e in page_elements if e.get("type") == "image"]

        # Process text with semantic chunking
        if text_elements:
            text_content = "\n\n".join(e.get("text", "") for e in text_elements)
            text_chunks = self._semantic_split_text(
                text_content, page_num, context_elements
            )
            chunks.extend(text_chunks)

        # Process tables (keep whole, but split if too large)
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
                # Split large table by rows
                table_chunks = self._split_large_table(table_text, page_num)
                chunks.extend(table_chunks)

        # Process images (keep whole)
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
        """Split text using sentence boundaries"""
        if not text.strip():
            return []

        # Tokenize into sentences
        sentences = self.sent_tokenize(text) if self.sent_tokenize else text.split(". ")

        chunks = []
        current_chunk = {"sentences": [], "text": "", "char_count": 0}
        first_chunk = True

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence exceeds chunk size
            potential_size = current_chunk["char_count"] + len(sentence) + 1

            if potential_size <= self.chunk_size:
                # Add to current chunk
                current_chunk["sentences"].append(sentence)
                current_chunk["text"] = " ".join(current_chunk["sentences"])
                current_chunk["char_count"] = len(current_chunk["text"])
            else:
                # Save current chunk if it meets minimum size
                if current_chunk["char_count"] >= self.min_chunk_size:
                    # Add context to the first chunk only
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

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk["sentences"]
                )

                current_chunk = {
                    "sentences": overlap_sentences + [sentence],
                    "text": " ".join(overlap_sentences + [sentence]),
                    "char_count": len(" ".join(overlap_sentences + [sentence])),
                }

        # Add final chunk
        if current_chunk["char_count"] >= self.min_chunk_size:
            # Add context to the first chunk only
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
        """Simple fixed-size chunking with context preservation"""
        chunks = []

        # Separate context from actual page content
        context_elements = [e for e in elements if e.get("is_context", False)]
        page_elements = [e for e in elements if not e.get("is_context", False)]

        # Combine all text
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
                    # Add context to the first chunk only
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

        # Process non-text elements
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
        """Prepare context text from context elements"""
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
        """Get sentences for overlap based on character count"""
        if not sentences:
            return []

        overlap_sentences = []
        char_count = 0

        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= self.overlap:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence) + 1
            else:
                break

        return overlap_sentences

    def _split_large_table(self, table_text: str, page_num: int) -> List[Dict]:
        """Split large table by rows while preserving header"""
        rows = table_text.split("\n")

        if not rows:
            return []

        # Assume first row is header
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
                # Save current chunk
                if len(current_rows) > 1:  # More than just header
                    chunks.append(
                        {
                            "page": page_num,
                            "text": "\n".join(current_rows),
                            "type": "table",
                            "source": "table_split",
                            "is_continuation": len(chunks) > 0,
                        }
                    )

                # Start new chunk with header
                current_rows = [header, row]
                current_size = len(header) + row_size

        # Add final chunk
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
        """Final validation of all chunks with enhanced page checking"""
        validated = []

        for i, chunk in enumerate(chunks):
            # Ensure page number is valid
            page = chunk.get("page", 1)
            if not isinstance(page, int) or page < 1:
                logger.error(f"[CHUNK] Invalid page in chunk {i}: {page}")
                chunk["page"] = 1

            # If boundaries are available, verify page is within expected range
            if boundaries and page > max(boundaries.keys()):
                logger.warning(
                    f"[CHUNK] Page {page} exceeds expected max {max(boundaries.keys())}"
                )
                chunk["page"] = max(boundaries.keys())

            # Ensure text exists and meets minimum size
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
        """Fallback sentence splitter when NLTK is not available"""
        import re

        # Split on period, exclamation, question mark followed by space and capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁ-Ú])", text)
        return [s.strip() for s in sentences if s.strip()]


class SemanticAwareChunker(ContextAwareChunker):
    """
    Enhanced chunking that considers semantic boundaries using sentence embeddings
    """

    def __init__(
        self, *args, embedding_model=None, similarity_threshold=0.75, **kwargs
    ):
        # Remove similarity_threshold from kwargs before passing to parent
        if "similarity_threshold" in kwargs:
            kwargs.pop("similarity_threshold")
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold  # Configurable

    def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """Find semantic boundaries using sentence similarity"""
        if not self.embedding_model or len(sentences) < 2:
            return []

        # Encode sentences in batches for efficiency
        embeddings = self.embedding_model.encode(
            sentences, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )

        # Calculate cosine similarity between consecutive sentences
        boundaries = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1])
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)

        return boundaries

    def _semantic_chunk_with_context(
        self, elements: List[Dict], page_num: int
    ) -> List[Dict[str, Any]]:
        """Enhanced semantic chunking with boundary detection"""
        # Extract text content
        text_content = "\n\n".join(
            e.get("text", "") for e in elements if e.get("type") == "text"
        )

        if not text_content.strip():
            return []

        # Tokenize into sentences
        sentences = (
            self.sent_tokenize(text_content)
            if self.sent_tokenize
            else text_content.split(". ")
        )

        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(sentences)

        # Create chunks based on semantic boundaries
        chunks = []
        start = 0

        for boundary in boundaries:
            # Create chunk from start to boundary
            chunk_sentences = sentences[start:boundary]
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    {
                        "page": page_num,
                        "text": chunk_text,
                        "type": "text",
                        "source": "semantic_boundary",
                        "sentence_count": len(chunk_sentences),
                    }
                )

            start = boundary

        # Add remaining sentences
        if start < len(sentences):
            chunk_sentences = sentences[start:]
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    {
                        "page": page_num,
                        "text": chunk_text,
                        "type": "text",
                        "source": "semantic_boundary",
                        "sentence_count": len(chunk_sentences),
                    }
                )

        # Process non-text elements
        for elem in elements:
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


class AdaptiveChunkingStrategySelector:
    """
    Automatically selects the best chunking strategy based on document characteristics
    """

    def __init__(self):
        self.strategies = {
            "semantic": None,  # Will use ContextAwareChunker
            "hierarchical": None,  # Will use ContextAwareChunker
            "adaptive": None,  # Will use ContextAwareChunker
            "semantic_boundary": None,  # NEW strategy
            "simple": None,  # Will use ContextAwareChunker
        }

    def analyze_document(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced document analysis with content type detection"""
        if not elements:
            return {"strategy": "simple", "reason": "empty_document"}

        # Count different element types
        type_counts = defaultdict(int)
        for elem in elements:
            type_counts[elem.get("type", "unknown")] += 1

        # Calculate text characteristics
        text_elements = [e for e in elements if e.get("type") == "text"]
        total_text = sum(len(e.get("text", "")) for e in text_elements)
        avg_text_length = total_text / len(text_elements) if text_elements else 0

        # Count headings
        heading_elements = [
            e for e in elements if e.get("type") in ["heading", "title"]
        ]

        # Analyze page distribution
        pages = set(e.get("page", 1) for e in elements)
        avg_elements_per_page = len(elements) / len(pages) if pages else 0

        # NEW: Analyze content complexity
        table_ratio = type_counts.get("table", 0) / len(elements) if elements else 0
        image_ratio = type_counts.get("image", 0) / len(elements) if elements else 0

        # Check if semantic boundaries would be beneficial
        if avg_text_length > 800 and len(heading_elements) < len(elements) * 0.05:
            return {
                "strategy": "semantic_boundary",
                "reason": "long_text_with_few_headings",
                "confidence": 0.8,
            }

        # Determine strategy with content type consideration
        if table_ratio > 0.3:  # Table-heavy document
            return {
                "strategy": "table_aware",
                "reason": "table_heavy_document",
                "confidence": 0.9,
                "table_ratio": table_ratio,
            }
        elif image_ratio > 0.3:  # Image-heavy document
            return {
                "strategy": "image_aware",
                "reason": "image_heavy_document",
                "confidence": 0.9,
                "image_ratio": image_ratio,
            }
        elif len(heading_elements) > len(elements) * 0.1:  # More than 10% headings
            return {
                "strategy": "hierarchical",
                "reason": "structured_document_with_headings",
                "confidence": 0.8,
            }
        elif avg_text_length > 1000:  # Long text elements
            return {
                "strategy": "semantic_boundary",
                "reason": "long_text_elements",
                "confidence": 0.7,
            }
        elif len(type_counts) > 2:  # Mixed content types
            return {
                "strategy": "adaptive",
                "reason": "mixed_content_types",
                "confidence": 0.9,
            }
        else:
            return {
                "strategy": "semantic_boundary",
                "reason": "default_choice",
                "confidence": 0.5,
            }

    def select_and_apply_strategy(
        self,
        elements: List[Dict[str, Any]],
        chunk_size: int = 900,
        overlap: int = 120,
        pdf_path: str = None,
    ) -> List[Dict[str, Any]]:
        analysis = self.analyze_document(elements)
        strategy = analysis["strategy"]

        logger.info(f"[CHUNK] Selected strategy: {strategy} ({analysis['reason']})")

        # Choose appropriate chunker
        if strategy == "semantic_boundary":
            # Import here to avoid circular imports
            from main import model_cache

            embed_model = model_cache.get_model(
                "intfloat/multilingual-e5-large-instruct"
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                embed_model = embed_model.to(device)
                logger.info(f"Model moved to {device.upper()}")
            except Exception as e:
                logger.error(f"Failed to move model to {device}: {e}")
                device = "cpu"  # Fallback to CPU
                embed_model = embed_model.to(device)

            chunker = SemanticAwareChunker(
                chunk_size,
                overlap,
                embedding_model=embed_model,
                similarity_threshold=0.75,
            )
        else:
            chunker = ContextAwareChunker(chunk_size, overlap)

        return chunker.chunk_with_context_preservation(elements, pdf_path, strategy)


class RobustPageExtractor:
    """Multi-strategy page number extraction with validation"""

    _cache = {}  # Add class-level cache

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
        # Create cache key
        elem_id = id(elem)
        if elem_id in RobustPageExtractor._cache:
            return RobustPageExtractor._cache[elem_id]

        # Strategy 1: Direct metadata extraction
        page = RobustPageExtractor._extract_from_metadata(elem)
        if page is not None:
            RobustPageExtractor._cache[elem_id] = page
            return page

        # Strategy 2: Coordinate-based extraction (PDF specific)
        if pdf_path and pdf_path.lower().endswith(".pdf"):
            page = RobustPageExtractor._extract_from_coordinates(elem, pdf_path)
            if page is not None:
                RobustPageExtractor._cache[elem_id] = page
                return page

        # Strategy 3: Text content analysis (look for page markers)
        page = RobustPageExtractor._extract_from_text_content(elem)
        if page is not None:
            RobustPageExtractor._cache[elem_id] = page
            return page

        # Strategy 4: Element position in document
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
        """Extract from element metadata"""
        try:
            # Method 1: Direct page_number attribute
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                page = elem.metadata.page_number
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

            # Method 2: Metadata dict
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "to_dict"):
                meta_dict = elem.metadata.to_dict()
                if "page_number" in meta_dict:
                    page = meta_dict["page_number"]
                    if isinstance(page, (int, float)) and page > 0:
                        return int(page)

            # Method 3: Direct page attribute
            if hasattr(elem, "page"):
                page = elem.page
                if isinstance(page, (int, float)) and page > 0:
                    return int(page)

        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"[PAGE] Metadata extraction failed: {e}")

        return None

    @staticmethod
    def _extract_from_coordinates(elem, pdf_path: str = None) -> Optional[int]:
        """Extract page from element coordinates"""
        try:
            if not hasattr(elem, "metadata") or not hasattr(
                elem.metadata, "coordinates"
            ):
                return None

            coords = elem.metadata.coordinates
            if not hasattr(coords, "points") or not coords.points:
                return None

            # Get all y-coordinates
            y_coords = [p[1] for p in coords.points if len(p) >= 2]
            if not y_coords:
                return None

            # Use minimum y-coordinate (top of element)
            min_y = min(y_coords)

            # Get actual page dimensions from the PDF if available
            if pdf_path:
                try:
                    import pdfplumber

                    # Try to get actual page heights for better accuracy
                    with pdfplumber.open(pdf_path) as pdf:
                        if len(pdf.pages) > 0:
                            # Calculate cumulative page heights to find the correct page
                            cumulative_height = 0
                            for i, page in enumerate(pdf.pages):
                                if (
                                    cumulative_height
                                    <= min_y
                                    < cumulative_height + page.height
                                ):
                                    estimated_page = i + 1  # Pages are 1-indexed
                                    logger.debug(
                                        f"[PAGE] Using actual PDF layout: page {estimated_page}"
                                    )
                                    break
                                cumulative_height += page.height
                            else:
                                # If y-coordinate exceeds all pages, use last page
                                estimated_page = len(pdf.pages)
                                logger.debug(
                                    f"[PAGE] Y-coordinate beyond pages, using page {estimated_page}"
                                )
                        else:
                            # Fallback to standard A4 height
                            estimated_page = max(1, int(min_y / 842) + 1)
                except Exception as e:
                    logger.debug(f"[PAGE] PDF analysis failed: {e}")
                    # Fallback to standard A4 height
                    estimated_page = max(1, int(min_y / 842) + 1)
            else:
                # No PDF path available, use standard A4 height
                estimated_page = max(1, int(min_y / 842) + 1)

            # Sanity check: page shouldn't be > 10000
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
        """Try to find page markers in text content"""
        try:
            if not hasattr(elem, "text"):
                return None

            text = elem.text.strip()
            if not text:
                return None

            # Look for common page markers
            # "Page 5", "Página 5", "P. 5", "pg 5", etc.
            patterns = [
                r"(?:page|página|p\.|pg\.?)\s*(\d+)",
                r"^\s*(\d+)\s*$",  # Just a number (risky)
                r"\[(\d+)\]",  # [5]
            ]

            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    page_num = int(match.group(1))
                    if 1 <= page_num <= 10000:  # Sanity check
                        logger.debug(f"[PAGE] Found in text: {page_num}")
                        return page_num

        except Exception as e:
            logger.debug(f"[PAGE] Text content extraction failed: {e}")

        return None

    @staticmethod
    def _infer_from_position(elem) -> Optional[int]:
        """Infer page from element's position in document"""
        try:
            # Check if element has an ID or index that could indicate position
            if hasattr(elem, "id") and isinstance(elem.id, str):
                # Some parsers include page info in IDs
                match = re.search(r"page[_-]?(\d+)", elem.id, re.IGNORECASE)
                if match:
                    page_num = int(match.group(1))
                    if 1 <= page_num <= 10000:
                        logger.debug(f"[PAGE] Inferred from ID: {page_num}")
                        return page_num

        except Exception as e:
            logger.debug(f"[PAGE] Position inference failed: {e}")

        return None


class PageSequenceValidator:
    """Validate and fix page number sequences"""

    @staticmethod
    def validate_and_fix(chunks: list) -> tuple[list, list]:
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

        # Attempt fixes if issues found
        if issues:
            logger.warning(f"[PAGE] Found {len(issues)} issues, attempting fixes...")
            chunks = PageSequenceValidator._fix_page_numbers(chunks, pages)

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
    def _fix_page_numbers(chunks: list, pages: list) -> list:
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

        # Fix 2: Smooth out large jumps
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


# ============================================================
# SQLITE CACHE MANAGER
# ============================================================


class SQLiteCacheManager:
    """Caché SQLite thread-safe para imágenes y tablas"""

    def __init__(self, cache_db: str = "/tmp/llava_cache/llava_cache.db"):
        self.cache_db = Path(cache_db)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Cache database: {self.cache_db}")

    def _init_db(self) -> None:
        """Inicializa tablas SQLite"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS image_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_hash TEXT UNIQUE NOT NULL,
                        description TEXT NOT NULL,
                        width INTEGER,
                        height INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER DEFAULT 1
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS table_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        table_hash TEXT UNIQUE NOT NULL,
                        analysis TEXT NOT NULL,
                        rows INTEGER,
                        cols INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER DEFAULT 1
                    )
                    """
                )

                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_image_hash ON image_cache(image_hash)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_table_hash ON table_cache(table_hash)"
                )

                conn.commit()
                logger.info("Cache database initialized")
        except Exception as e:
            logger.error(f"Error initializing cache DB: {e}")

    def _get_connection(self):
        """Retorna conexión SQLite thread-safe"""
        conn = sqlite3.connect(str(self.cache_db), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def load_image_cache(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Carga descripción en caché de imagen"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT description, width, height FROM image_cache WHERE image_hash = ?",
                    (image_hash,),
                )
                row = cursor.fetchone()

                if row:
                    cursor.execute(
                        "UPDATE image_cache SET accessed_at = CURRENT_TIMESTAMP, hit_count = hit_count + 1 WHERE image_hash = ?",
                        (image_hash,),
                    )
                    conn.commit()
                    logger.debug(f"Cache hit (imagen): {image_hash}")

                    return {
                        "description": row["description"],
                        "width": row["width"],
                        "height": row["height"],
                    }
        except Exception as e:
            logger.error(f"Error loading image cache: {e}")

        return None

    def load_table_cache(self, table_hash: str) -> Optional[Dict[str, Any]]:
        """Carga análisis en caché de tabla"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT analysis FROM table_cache WHERE table_hash = ?",
                    (table_hash,),
                )
                row = cursor.fetchone()

                if row:
                    cursor.execute(
                        "UPDATE table_cache SET accessed_at = CURRENT_TIMESTAMP, hit_count = hit_count + 1 WHERE table_hash = ?",
                        (table_hash,),
                    )
                    conn.commit()
                    logger.debug(f"Cache hit (tabla): {table_hash}")

                    return {"analysis": row["analysis"]}
        except Exception as e:
            logger.error(f"Error loading table cache: {e}")

        return None

    def save_image_cache(
        self, image_hash: str, description: str, width: int, height: int
    ) -> None:
        """Guarda descripción de imagen en caché"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO image_cache
                       (image_hash, description, width, height, accessed_at, hit_count)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 1)""",
                    (image_hash, description, width, height),
                )
                conn.commit()
                logger.debug(f"Cache saved (imagen): {image_hash}")
        except Exception as e:
            logger.error(f"Error saving image cache: {e}")

    def save_table_cache(
        self, table_hash: str, analysis: str, rows: int, cols: int
    ) -> None:
        """Guarda análisis de tabla en caché"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO table_cache
                       (table_hash, analysis, rows, cols, accessed_at, hit_count)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 1)""",
                    (table_hash, analysis, rows, cols),
                )
                conn.commit()
                logger.debug(f"Cache saved (tabla): {table_hash}")
        except Exception as e:
            logger.error(f"Error saving table cache: {e}")

<<<<<<< HEAD

# ============================================================
# SIMPLE EXTRACTOR
# ============================================================


class SimpleExtractor:
    """Extractor simple: Unstructured.io (sin CUDA)"""

    SUPPORTED_FORMATS = {
        ".pdf": "PDF",
        ".docx": "Word",
        ".doc": "Word",
        ".pptx": "PowerPoint",
        ".ppt": "PowerPoint",
        ".html": "HTML",
        ".htm": "HTML",
        ".md": "Markdown",
        ".txt": "Text",
        ".png": "Image",
        ".jpg": "Image",
        ".jpeg": "Image",
    }

    def __init__(
        self,
        vllm_url: str = "http://vllm-llava:8000",
        cache_db: str = "/tmp/llava_cache/llava_cache.db",
    ):
        self.vllm_url = vllm_url
        self.cache = SQLiteCacheManager(cache_db=cache_db)
        self.stats = {
            "text_chunks": 0,
            "tables_processed": 0,
            "tables_cached": 0,
            "images_processed": 0,
            "images_cached": 0,
        }

    def extract_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Extrae documento con Unstructured.io (usa OCR solo si es necesario)"""
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        logger.info(f"Extracting: {Path(file_path).name}")

        if not self.is_supported(ext):
            logger.warning(f"Format not supported: {ext}")
            return []

        try:
            elements = []
            ocr_used = False  # 🔹 Detecta si se usó OCR

            # ============================================================
            # PDF: modo híbrido (texto directo + OCR cuando hace falta)
            # ============================================================
            if ext == ".pdf":
                from unstructured.partition.pdf import partition_pdf

                # Primer intento: extracción directa
                elements = partition_pdf(
                    filename=file_path,
                    strategy="fast",  # texto directo (sin OCR)
                    infer_table_structure=True,
                    extract_image_block_types=["Image"],
                    extract_strategy="auto",
                    languages=["es", "en"],
                    split_pdf_pages=True,
                )

                # Si no hay texto significativo, repetir con OCR
                text_count = sum(
                    1 for e in elements if hasattr(e, "text") and e.text.strip()
                )
                if text_count == 0:
                    logger.warning(
                        "[PDF] Sin texto embebido detectado, aplicando OCR (hi_res)..."
                    )
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="hi_res",  # usa OCR cuando es necesario
                        infer_table_structure=True,
                        extract_image_block_types=["Image"],
                        ocr_languages="spa+eng",
                        extract_strategy="auto",
                        split_pdf_pages=True,
                    )
                    ocr_used = True

            # ============================================================
            # Otros formatos (Word, PowerPoint, HTML, etc.)
            # ============================================================
            elif ext in [".docx", ".doc"]:
                elements = partition_docx(file_path, infer_table_structure=False)
            elif ext in [".pptx", ".ppt"]:
                elements = partition_pptx(file_path, infer_table_structure=False)
            else:
                elements = partition(
                    file_path,
                    infer_table_structure=False,
                    languages=["es", "en"],
=======
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Image stats
                cursor.execute(
                    "SELECT COUNT(*) as count, SUM(hit_count) as hits FROM image_cache"
                )
                img_row = cursor.fetchone()

                # Table stats
                cursor.execute(
                    "SELECT COUNT(*) as count, SUM(hit_count) as hits FROM table_cache"
>>>>>>> chunking
                )
                table_row = cursor.fetchone()

<<<<<<< HEAD
            logger.info(f"Found {len(elements)} elements")
            if ocr_used:
                logger.info("[INFO] OCR activado para este documento.")
            else:
                logger.info("[INFO] Extracción directa sin OCR.")

            # Procesamiento
            chunks = self._process_elements(elements)
            self._log_stats()

            return chunks

=======
                return {
                    "images": {
                        "cached": img_row["count"] or 0,
                        "hits": img_row["hits"] or 0,
                    },
                    "tables": {
                        "cached": table_row["count"] or 0,
                        "hits": table_row["hits"] or 0,
                    },
                }
>>>>>>> chunking
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "images": {"cached": 0, "hits": 0},
                "tables": {"cached": 0, "hits": 0},
            }

    def get_top_cached(self, limit: int = 10) -> Dict[str, List[Dict]]:
        """Get top cached items"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Top images
                cursor.execute(
                    "SELECT image_hash, hit_count, created_at FROM image_cache ORDER BY hit_count DESC LIMIT ?",
                    (limit,),
                )
                top_images = [dict(row) for row in cursor.fetchall()]

                # Top tables
                cursor.execute(
                    "SELECT table_hash, hit_count, created_at FROM table_cache ORDER BY hit_count DESC LIMIT ?",
                    (limit,),
                )
                top_tables = [dict(row) for row in cursor.fetchall()]

                return {"top_images": top_images, "top_tables": top_tables}
        except Exception as e:
            logger.error(f"Error getting top cached items: {e}")
            return {"top_images": [], "top_tables": []}

    def clear_all_cache(self):
        """Clear all cache"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM image_cache")
                cursor.execute("DELETE FROM table_cache")
                conn.commit()
                logger.info("All cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def clear_old_cache(self, days: int = 30):
        """Clear cache older than specified days"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM image_cache WHERE created_at < datetime('now', '-{} days')".format(
                        days
                    )
                )
                cursor.execute(
                    "DELETE FROM table_cache WHERE created_at < datetime('now', '-{} days')".format(
                        days
                    )
                )
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted} old cache entries")
                return deleted
        except Exception as e:
            logger.error(f"Error clearing old cache: {e}")
            return 0


def pdf_to_chunks_with_enhanced_validation(
    pdf_path: str,
    chunk_size: int = 900,
    overlap: int = 120,
    vllm_url: str = None,
    cache_db: str = None,
    use_adaptive_chunking: bool = True,
    add_context: bool = True,
    embed_model=None,
    validate_pages: bool = True,
    detect_visual_boundaries: bool = True,
) -> List[Dict[str, Any]]:
    """
    Enhanced PDF to chunks conversion with improved page validation and chunking
    """

    # First, try to determine if this PDF likely has extractable text
    has_extractable_text = check_pdf_has_text(pdf_path)

    if has_extractable_text:
        logger.info(
            "PDF appears to have extractable text, attempting text extraction..."
        )
        chunks = extract_text_with_multiple_methods(pdf_path)
        if chunks:
            logger.info(f"Successfully extracted {len(chunks)} text chunks")
            return chunks
        else:
            logger.warning("Text extraction methods failed, falling back to OCR...")

    # If text extraction failed or PDF doesn't have extractable text, use OCR
    elements = extract_elements_from_pdf_gpu(
        pdf_path, has_extractable_text=has_extractable_text
    )

    if not elements:
        raise ValueError(f"No elements extracted from PDF: {pdf_path}")

    # Use adaptive chunking strategy selector
    strategy_selector = AdaptiveChunkingStrategySelector()

    # Select and apply optimal chunking strategy
    chunks = strategy_selector.select_and_apply_strategy(
        elements,
        chunk_size=chunk_size,
        overlap=overlap,
        pdf_path=pdf_path if detect_visual_boundaries else None,
    )

    # Validate page sequence if requested
    if validate_pages:
        chunks, issues = PageSequenceValidator.validate_and_fix(chunks)

        if issues and len(issues) > 30:  # Too many issues = reject file
            raise ValueError(
                f"Page validation failed: {len(issues)} critical issues: {issues[:10]}"
            )

    return chunks


def check_pdf_has_text(pdf_path: Path) -> bool:
    """
    Check if PDF likely has extractable text using multiple methods
    """
    try:
        # Method 1: Try with pypdf first (fastest check)
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            page_count = len(pdf_reader.pages)

            # Check first few pages for text
            for i in range(min(10, page_count)):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # If we find substantial text
                    logger.info(f"pypdf detected text on page {i+1}")
                    return True

        # Method 2: Try with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:10]):  # Check first 10 pages
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    logger.info(f"pdfplumber detected text on page {i+1}")
                    return True

        logger.info("No extractable text detected in PDF")
        return False
    except Exception as e:
        logger.warning(f"Error checking for extractable text: {e}")
        # Default to assuming text might be present
        return True


def extract_with_pdfminer(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text using pdfminer.six
    """
    try:
        from pdfminer.high_level import extract_text

        logger.info("Trying pdfminer.six...")

        text = extract_text(str(pdf_path))
        if text and len(text.strip()) > 100:
            logger.info(f"pdfminer.six extracted {len(text)} characters")
            return [
                {
                    "type": "text",
                    "text": text,
                    "page": 1,  # pdfminer doesn't easily provide page numbers
                    "source": "pdfminer",
                }
            ]
    except Exception as e:
        logger.warning(f"pdfminer.six failed: {e}")

    return []


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

        # Add additional metadata if available
        if hasattr(element, "metadata") and element.metadata:
            if hasattr(element.metadata, "coordinates"):
                chunk["coordinates"] = element.metadata.coordinates
            if hasattr(element.metadata, "filename"):
                chunk["source_file"] = element.metadata.filename

        chunks.append(chunk)

    return chunks


def extract_text_with_multiple_methods(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Try multiple text extraction methods in order of preference
    """
    # Method 1: Try with optimized unstructured partition_pdf
    try:
        logger.info("Trying optimized unstructured partition_pdf...")
        elements = fast_partition_pdf(str(pdf_path), strategy="auto")

        if elements and len(elements) > 0:
            # Check if we got meaningful text
            text_content = "\n".join([str(e) for e in elements if hasattr(e, "text")])
            if len(text_content.strip()) > 100:  # If we got substantial text
                logger.info(
                    f"unstructured partition_pdf extracted {len(elements)} elements"
                )
                return process_elements_to_chunks(elements)
    except Exception as e:
        logger.warning(f"unstructured partition_pdf failed: {e}")

    # Method 2: Try with pdfminer.six
    pdfminer_result = extract_with_pdfminer(pdf_path)
    if pdfminer_result:
        return pdfminer_result

    # Method 3: Try with pdfplumber directly (with improved error handling)
    try:
        logger.info("Trying pdfplumber directly...")
        with pdfplumber.open(pdf_path) as pdf:
            elements = []
            for i, page in enumerate(pdf.pages):
                # Extract text
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

                # Extract tables with improved error handling
                try:
                    for table in page.extract_tables():
                        if table:
                            # Handle None values in table cells
                            processed_table = []
                            for row in table:
                                # Convert all cells to strings, replace None with empty string
                                processed_row = [
                                    str(cell) if cell is not None else ""
                                    for cell in row
                                ]
                                processed_table.append(processed_row)

                            # Join the processed table into text
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

    # Method 4: Try with pypdf as last resort
    try:
        logger.info("Trying pypdf...")
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            elements = []

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

            if elements:
                logger.info(f"pypdf extracted {len(elements)} elements")
                return elements
    except Exception as e:
        logger.warning(f"pypdf failed: {e}")

    logger.warning("All text extraction methods failed")
    return []


class EasyOCRProcessor:
    """GPU-accelerated OCR using EasyOCR with PyTorch backend"""

    # Class-level cache for the reader to avoid re-initialization
    _reader_cache = None
    _initialization_attempted = False

    def __init__(self, use_gpu=True, gpu_id=0, model_storage_directory=None):
        """
        Initialize EasyOCR with GPU support and proper model handling

        Args:
            use_gpu: Enable GPU acceleration (recommended)
            gpu_id: GPU device ID (default: 0)
            model_storage_directory: Where to store EasyOCR models (default: ~/.EasyOCR/)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if not self.use_gpu and use_gpu:
            logger.warning("[EASYOCR] CUDA not available, falling back to CPU")

        # Set model storage directory
        if model_storage_directory:
            os.environ["EASYOCR_MODULE_PATH"] = str(model_storage_directory)
            self.model_dir = Path(model_storage_directory)
        else:
            self.model_dir = Path.home() / ".EasyOCR"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[EASYOCR] Model directory: {self.model_dir}")

        # Use cached reader if available
        if EasyOCRProcessor._reader_cache is not None:
            logger.info("[EASYOCR] Using cached reader")
            self.reader = EasyOCRProcessor._reader_cache
            return

        # Avoid repeated initialization attempts if it failed before
        if (
            EasyOCRProcessor._initialization_attempted
            and EasyOCRProcessor._reader_cache is None
        ):
            raise RuntimeError("[EASYOCR] Previous initialization failed, not retrying")

        try:
            EasyOCRProcessor._initialization_attempted = True

            # Setup SSL context before downloading models
            setup_ssl_context()

            # Check if models are already downloaded
            models_exist = self._check_models_exist()

            if not models_exist:
                logger.info("[EASYOCR] Models not found, will download on first use")
                logger.info(
                    "[EASYOCR] This may take 2-5 minutes depending on network speed"
                )
            else:
                logger.info("[EASYOCR] Using existing models")

            # Initialize EasyOCR reader with Spanish and English support
            logger.info("[EASYOCR] Initializing reader...")
            self.reader = easyocr.Reader(
                ["es", "en"],  # Spanish and English languages
                gpu=self.use_gpu,
                model_storage_directory=str(self.model_dir),
                download_enabled=True,  # Allow downloading models
                detector=True,
                recognizer=True,
                verbose=False,
            )

            # Cache the reader for future use
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
        """Check if required models are already downloaded"""
        try:
            # Check for detection model
            craft_path = self.model_dir / "model" / "craft_mlt_25k.pth"

            # Check for recognition models (Spanish and English)
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
        """Extract text from image using EasyOCR"""
        try:
            # Use EasyOCR to read the image
            results = self.reader.readtext(image_path)

            if not results:
                return ""

            # Combine all detected text with confidence filtering
            text_lines = []
            for bbox, text, confidence in results:
                # Only include high-confidence text (>60% confidence)
                if confidence > 0.6:
                    text_lines.append(text.strip())

            return "\n".join(text_lines)

        except Exception as e:
            logger.error(f"[EASYOCR] Failed to process {image_path}: {e}")
            return ""

    def process_pdf_page_batch(self, pdf_path: str, page_nums: list) -> dict:
        """
        Process multiple PDF pages in batch for better GPU utilization
        Returns: {page_num: text}
        """
        from pdf2image import convert_from_path
        import tempfile

        logger.debug(
            f"[EASYOCR] Processing pages {page_nums} for {os.path.basename(pdf_path)}"
        )

        # Convert multiple pages at once with exact page range
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
            # Save all images with page-aware naming
            for idx, image in enumerate(images):
                actual_page_num = first_page + idx
                if actual_page_num in page_nums:
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=f"_page_{actual_page_num}.png", delete=False
                    )
                    image.save(tmp.name)
                    temp_files.append((actual_page_num, tmp.name))

            # Process each image
            for page_num, tmp_path in temp_files:
                text = self.extract_text_from_image(tmp_path)
                results[page_num] = text

        finally:
            # Cleanup temp files
            for _, tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        return results


def manually_download_easyocr_models(model_dir: Path = None):
    """
    Manually download EasyOCR models to avoid SSL issues

    Usage:
        from chunk import manually_download_easyocr_models
        manually_download_easyocr_models()
    """

    if model_dir is None:
        model_dir = Path.home() / ".EasyOCR"

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model").mkdir(exist_ok=True)

    setup_ssl_context()

    models = {
        "craft_mlt_25k.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip",
        "spanish_g2.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/spanish_g2.zip",
        "english_g2.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
    }

    logger.info(f"[EASYOCR] Downloading models to {model_dir}")

    for model_name, url in models.items():
        model_path = model_dir / "model" / model_name

        if model_path.exists():
            logger.info(f"[EASYOCR] ✓ {model_name} already exists")
            continue

        try:
            logger.info(f"[EASYOCR] Downloading {model_name}...")

            # Download with progress
            zip_path = model_dir / f"{model_name}.zip"
            urllib.request.urlretrieve(url, zip_path)

            # Extract
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(model_dir / "model")

            # Cleanup zip
            zip_path.unlink()

            logger.info(f"[EASYOCR] ✓ Downloaded {model_name}")

        except Exception as e:
            logger.error(f"[EASYOCR] Failed to download {model_name}: {e}")
            logger.info(f"[EASYOCR] You can manually download from: {url}")

    logger.info("[EASYOCR] Model download complete")


# Alternative: Download models at import time if they don't exist
def ensure_easyocr_models():
    """Ensure EasyOCR models are available, download if needed"""
    if not EASYOCR_AVAILABLE:
        return False

    model_dir = Path.home() / ".EasyOCR"
    craft_path = model_dir / "model" / "craft_mlt_25k.pth"

    if not craft_path.exists():
        logger.info("[EASYOCR] Models not found, attempting download...")
        try:
            manually_download_easyocr_models(model_dir)
            return True
        except Exception as e:
            logger.error(f"[EASYOCR] Auto-download failed: {e}")
            return False
    return True


def fast_partition_pdf(pdf_path: str, strategy: str = "auto") -> List[Any]:
    """
    Optimized unstructured.partition_pdf with speed-focused parameters
    3-5x faster than default unstructured calls
    """
    # Speed-optimized kwargs for different strategies
    if strategy == "hi_res":
        kwargs = {
            "infer_table_structure": True,
            "extract_images_in_pdf": False,
            "extract_tables": True,
            "chunking_strategy": None,  # Skip internal chunking
            "max_characters": 10000,  # Larger chunks
            "languages": ["spa", "eng"],
            "strategy": "hi_res",
            "model_name": "yolox",  # Faster model
            "skip_infer_table_types": ["pdf", "jpg", "png"],
        }
    elif strategy == "ocr_only":
        kwargs = {
            "languages": ["spa", "eng"],
            "strategy": "ocr_only",
            "ocr_languages": "spa+eng",
            "ocr_mode": "entire_page",  # Faster than individual blocks
            "extract_images_in_pdf": False,
            "extract_tables": True,
            "infer_table_structure": True,
            "keep_extra_chars": False,  # Skip character cleanup
            "max_characters": 12000,  # Larger chunks for OCR
            "ocr_kwargs": {"config": "--oem 3 --psm 6"},  # Fast PSM mode
        }
    else:  # auto
        kwargs = {
            "languages": ["spa", "eng"],
            "strategy": "auto",
            "extract_images_in_pdf": False,  # Major speedup
            "extract_tables": True,
            "infer_table_structure": True,
            "max_characters": 15000,  # Large chunks
            "keep_extra_chars": False,  # Skip cleanup
        }

    return partition_pdf(filename=pdf_path, **kwargs)


def batch_unstructured_processing(pdf_path: str) -> List[Any]:
    """
    Process PDF with optimal strategy based on document size:
    - Small docs (≤25 pages): Use extract_elements_from_pdf_gpu() directly
    - Large docs (>25 pages): Process in batches with proper fallback hierarchy
      - If PDF has extractable text: try text extraction first, then OCR
      - If PDF has no extractable text: use OCR only (EasyOCR → Tesseract)
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    # For smaller documents, use the full GPU pipeline directly
    if total_pages <= 25:
        logger.info(
            f"[BATCH] Small doc ({total_pages} pages), using extract_elements_from_pdf_gpu()"
        )
        return extract_elements_from_pdf_gpu(pdf_path)

    # Check once if the PDF has extractable text to avoid wasteful text extraction attempts
    has_extractable_text = check_pdf_has_text(pdf_path)
    if has_extractable_text:
        logger.info(
            f"[BATCH] Large doc ({total_pages} pages) with extractable text, processing in batches with text+OCR hierarchy"
        )
    else:
        logger.info(
            f"[BATCH] Large doc ({total_pages} pages) with NO extractable text, processing in batches with OCR only"
        )

    all_elements = []
    BATCH_SIZE = 20  # Optimal batch size for memory/speed balance

    for batch_start in range(1, total_pages + 1, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE - 1, total_pages)
        page_nums = list(range(batch_start, batch_end + 1))

        logger.info(f"[BATCH] Processing pages {batch_start}-{batch_end}")

        batch_elements = []

        # Only attempt text extraction if the PDF actually has extractable text
        if has_extractable_text:
            try:
                logger.debug(
                    f"[BATCH] Attempting text extraction for pages {batch_start}-{batch_end}"
                )
                elements = fast_partition_pdf(
                    pdf_path, strategy="hi_res", pages=page_nums
                )

                text_elements = [
                    e
                    for e in elements
                    if hasattr(e, "category")
                    and e.category == "Text"
                    or (hasattr(e, "to_dict") and e.to_dict().get("type") == "text")
                ]

                total_text_length = sum(len(str(e)) for e in text_elements)
                avg_text_per_page = total_text_length / len(page_nums)

                # Check if we got sufficient text
                MIN_TOTAL_CHARS = 200
                MIN_AVG_CHARS_PER_PAGE = 50

                if (
                    text_elements
                    and total_text_length > MIN_TOTAL_CHARS
                    and avg_text_per_page > MIN_AVG_CHARS_PER_PAGE
                ):
                    logger.debug(
                        f"[BATCH] Text extraction successful: {total_text_length} chars"
                    )
                    batch_elements = [
                        (
                            elem.to_dict()
                            if hasattr(elem, "to_dict")
                            else {"text": str(elem)}
                        )
                        for elem in elements
                    ]
                    for elem in batch_elements:
                        elem["source"] = "batch_text_extraction"

            except Exception as text_error:
                logger.debug(f"[BATCH] Text extraction failed: {text_error}")

        # Try EasyOCR (either as fallback or primary method if no extractable text)
        if not batch_elements and EASYOCR_AVAILABLE:
            try:
                if has_extractable_text:
                    logger.info(
                        f"[BATCH] Using EasyOCR as fallback for pages {batch_start}-{batch_end}"
                    )
                else:
                    logger.info(
                        f"[BATCH] Using EasyOCR (OCR-only doc) for pages {batch_start}-{batch_end}"
                    )

                ocr_processor = EasyOCRProcessor(use_gpu=True, gpu_id=0)
                page_texts = ocr_processor.process_pdf_page_batch(pdf_path, page_nums)

                for page_num, text in page_texts.items():
                    if text.strip():
                        batch_elements.append(
                            {
                                "text": text,
                                "type": "text",
                                "page": page_num,
                                "source": "batch_easyocr_gpu",
                            }
                        )

                total_easyocr_text = sum(len(elem["text"]) for elem in batch_elements)
                logger.debug(
                    f"[BATCH] EasyOCR extracted {total_easyocr_text} chars from {len(batch_elements)} pages"
                )

            except Exception as easyocr_error:
                logger.warning(f"[BATCH] EasyOCR failed: {easyocr_error}")

        # If still no elements, fallback to Tesseract OCR
        if not batch_elements:
            try:
                logger.info(
                    f"[BATCH] Using Tesseract OCR for pages {batch_start}-{batch_end}"
                )
                elements = fast_partition_pdf(
                    pdf_path, strategy="ocr_only", pages=page_nums
                )

                batch_elements = []
                for elem in elements:
                    elem_dict = (
                        elem.to_dict()
                        if hasattr(elem, "to_dict")
                        else {"text": str(elem)}
                    )
                    elem_dict["page"] = elem_dict.get(
                        "page", batch_start
                    )  # Ensure correct page
                    elem_dict["source"] = "batch_tesseract_ocr"
                    batch_elements.append(elem_dict)

                logger.debug(
                    f"[BATCH] Tesseract processed {len(batch_elements)} elements"
                )

            except Exception as tess_error:
                logger.error(
                    f"[BATCH] Tesseract failed for pages {batch_start}-{batch_end}: {tess_error}"
                )
                continue  # Skip this batch

        # Validate page numbers for this batch
        batch_elements = validate_page_numbers_with_count(
            batch_elements, pdf_path, total_pages
        )
        all_elements.extend(batch_elements)

    logger.info(
        f"[BATCH] Completed processing: {len(all_elements)} total elements from {total_pages} pages"
    )
    return all_elements


def extract_elements_from_pdf_gpu(
    pdf_path: str, has_extractable_text: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract PDF elements using optimized OCR pipeline with accurate page numbering
    1. Get PDF page count once (optimization)
    2. Try standard text extraction (fastest)
    3. Try EasyOCR with GPU acceleration if insufficient text
    4. Fallback to optimized Tesseract processing
    5. Apply page validation once to final results
    """

    # Get PDF page count once to avoid redundant file opening
    total_pages = None
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        logger.debug(f"[PDF] Document has {total_pages} pages")
    except Exception as e:
        logger.warning(f"[PDF] Could not get page count: {e}")

    # First try standard text extraction (fastest)
    if has_extractable_text:
        try:
            logger.info(f"[OCR] Attempting fast text extraction...")
            elements = fast_partition_pdf(pdf_path, strategy="hi_res")

            text_elements = [
                e
                for e in elements
                if e.get("type") == "text" and e.get("text", "").strip()
            ]

            # Calculate total text length to decide if extraction was truly successful
            total_text_length = sum(len(e.get("text", "")) for e in text_elements)

            # Calculate average text per page to detect if we're just getting headers/footers
            avg_text_per_page = (
                total_text_length / total_pages if total_pages else total_text_length
            )

            logger.info(
                f"[OCR] Standard extraction: {len(text_elements)} elements, "
                f"{total_text_length} total chars, "
                f"{avg_text_per_page:.1f} avg chars/page"
            )

            # More stringent check: require both sufficient total text AND reasonable text per page
            # This prevents cases where we only extract headers/footers/metadata
            MIN_TOTAL_CHARS = 500
            MIN_AVG_CHARS_PER_PAGE = 100  # At least 100 chars per page on average

            if (
                text_elements
                and total_text_length > MIN_TOTAL_CHARS
                and avg_text_per_page > MIN_AVG_CHARS_PER_PAGE
            ):
                logger.info(
                    f"[OCR] Standard extraction successful with sufficient text"
                )
                processed_elements = [
                    elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
                    for elem in elements
                ]
                # Apply page validation once with known page count
                return validate_page_numbers_with_count(
                    processed_elements, pdf_path, total_pages
                )
            else:
                logger.info(
                    f"[OCR] Standard extraction insufficient: "
                    f"total_chars={total_text_length} (need >{MIN_TOTAL_CHARS}), "
                    f"avg_per_page={avg_text_per_page:.1f} (need >{MIN_AVG_CHARS_PER_PAGE}), "
                    f"proceeding to EasyOCR"
                )
        except Exception as first_error:
            logger.warning(
                f"[OCR] Hi-res extraction failed: {first_error}, proceeding to EasyOCR"
            )

    # Try EasyOCR with GPU acceleration (fast & accurate)
    if EASYOCR_AVAILABLE:
        logger.info(f"[OCR] Using EasyOCR GPU acceleration")
        try:
            # Initialize EasyOCR processor
            ocr_processor = EasyOCRProcessor(use_gpu=True, gpu_id=0)

            # Use cached page count instead of reopening PDF
            num_pages = total_pages or 1  # Fallback to 1 if we couldn't get page count
            logger.info(f"[EASYOCR] Processing {num_pages} pages")

            # Process in batches for better GPU utilization
            BATCH_SIZE = 8  # Process 8 pages at once on RTX 5090
            elements = []

            for batch_start in range(1, num_pages + 1, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_pages + 1)
                page_nums = list(range(batch_start, batch_end))

                logger.info(
                    f"[EASYOCR] Processing batch pages {batch_start}-{batch_end-1}/{num_pages}"
                )

                # Batch process with accurate page mapping
                page_texts = ocr_processor.process_pdf_page_batch(pdf_path, page_nums)

                for page_num, text in page_texts.items():
                    if text.strip():
                        # Validate page number before creating element
                        if 1 <= page_num <= num_pages:
                            elements.append(
                                {
                                    "text": text,
                                    "type": "text",
                                    "page": page_num,  # Ensure correct page number
                                    "source": "easyocr_gpu",
                                }
                            )
                        else:
                            logger.warning(
                                f"[EASYOCR] Invalid page number {page_num}, skipping"
                            )

            # Validate total elements and page coverage
            pages_processed = set(elem["page"] for elem in elements)
            missing_pages = set(range(1, num_pages + 1)) - pages_processed

            if missing_pages:
                logger.info(
                    f"[EASYOCR] No text extracted from pages: {sorted(missing_pages)}"
                )

            # Check if EasyOCR was successful
            total_easyocr_text = sum(len(elem["text"]) for elem in elements)
            logger.info(
                f"[EASYOCR] Extracted {len(elements)} elements from "
                f"{len(pages_processed)}/{num_pages} pages, "
                f"{total_easyocr_text} total chars"
            )

            # If EasyOCR got reasonable results, return them
            if elements and total_easyocr_text > 200:
                logger.info(f"[EASYOCR] Successfully extracted text, returning results")
                return validate_page_numbers_with_count(elements, pdf_path, total_pages)
            else:
                logger.warning(
                    f"[EASYOCR] Insufficient text extracted ({total_easyocr_text} chars), "
                    f"falling back to Tesseract"
                )

        except Exception as e:
            logger.warning(f"[EASYOCR] GPU processing failed: {e}")
            logger.info("[EASYOCR] Falling back to Tesseract")
    else:
        logger.warning("[EASYOCR] Not available, using Tesseract directly")

    # Fallback to optimized Tesseract processing
    logger.info(f"[OCR] Using optimized fast Tesseract processing")

    try:
        # Use cached page count for validation
        if total_pages:
            logger.info(f"[TESSERACT] Processing {total_pages} pages")
        else:
            logger.warning("[TESSERACT] No page count available, proceeding anyway")

        # Use optimized fast unstructured with Tesseract
        elements = fast_partition_pdf(pdf_path, strategy="ocr_only")

        # Convert elements and validate page numbers
        processed_elements = []
        pages_seen = set()

        for elem in elements:
            elem_dict = (
                elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
            )

            # Extract and validate page number
            page_num = elem_dict.get("page", 1)
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            else:
                # Try to extract page number from metadata
                page_num = RobustPageExtractor.extract_page_number(elem, pdf_path, 1)
                elem_dict["page"] = page_num

            # Validate page number is within expected range
            if total_pages and 1 <= page_num <= total_pages:
                elem_dict["source"] = "tesseract_ocr"
                processed_elements.append(elem_dict)
                pages_seen.add(page_num)
            else:
                logger.warning(
                    f"[TESSERACT] Invalid page {page_num}, clamping to valid range"
                )
                elem_dict["page"] = max(1, min(page_num, total_pages or 1))
                elem_dict["source"] = "tesseract_ocr"
                processed_elements.append(elem_dict)

        # Report page coverage
        if total_pages:
            missing_pages = set(range(1, total_pages + 1)) - pages_seen
            if missing_pages:
                logger.info(
                    f"[TESSERACT] No text from pages: {sorted(list(missing_pages))}"
                )

        logger.info(
            f"[TESSERACT] Extracted {len(processed_elements)} elements from "
            f"{len(pages_seen)}/{total_pages if total_pages else 'unknown'} pages"
        )
        # Apply comprehensive page validation to Tesseract results with cached page count
        return validate_page_numbers_with_count(
            processed_elements, pdf_path, total_pages
        )

    except Exception as e:
        logger.error(f"[OCR] Tesseract extraction failed: {e}")
        logger.info("[OCR] Trying last resort extraction")

        # Last resort: optimized text extraction with page validation
        try:
            elements = fast_partition_pdf(pdf_path, strategy="auto")

            processed_elements = []
            for elem in elements:
                elem_dict = (
                    elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
                )
                elem_dict["source"] = "basic_extraction"
                processed_elements.append(elem_dict)

            logger.info(
                f"[OCR] Basic extraction completed with {len(processed_elements)} elements"
            )
            # Apply comprehensive page validation even to fallback results with cached page count
            return validate_page_numbers_with_count(
                processed_elements, pdf_path, total_pages
            )

        except Exception as fallback_e:
            logger.error(f"[OCR] All extraction methods failed: {fallback_e}")
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
    Optimized comprehensive page number validation and correction
    Uses cached page count to avoid redundant PDF opening

    Args:
        elements: List of extracted elements with page metadata
        pdf_path: Path to PDF (for logging)
        total_pages: Pre-computed total page count (optimization)

    Returns:
        Validated elements with corrected page numbers
    """
    if not elements:
        return elements

    # Only open PDF if page count not provided
    if total_pages is None:
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        except Exception as e:
            logger.warning(
                f"[PAGE] Could not verify page count for {os.path.basename(pdf_path)}: {e}"
            )
            total_pages = None

    validated_elements = []
    page_issues = []

    for i, elem in enumerate(elements):
        page_num = elem.get("page", 1)

        # Convert to integer if needed
        if not isinstance(page_num, int):
            try:
                page_num = int(float(page_num))
            except (ValueError, TypeError):
                page_num = 1
                page_issues.append(f"Element {i}: Invalid page type, using 1")

        # Validate page range
        if total_pages and (page_num < 1 or page_num > total_pages):
            original_page = page_num
            page_num = max(1, min(page_num, total_pages))
            page_issues.append(
                f"Element {i}: Page {original_page} -> {page_num} (clamped)"
            )

        # Ensure correct page number
        elem["page"] = page_num
        validated_elements.append(elem)

    if page_issues:
        logger.warning(
            f"[PAGE] Fixed {len(page_issues)} page issues in {os.path.basename(pdf_path)}"
        )
        for issue in page_issues[:5]:  # Show first 5 issues
            logger.warning(f"[PAGE]   {issue}")
        if len(page_issues) > 5:
            logger.warning(f"[PAGE]   ... and {len(page_issues) - 5} more")

    return validated_elements


def extract_elements_from_pdf(
    pdf_path: str, num_processes: int = None
) -> List[Dict[str, Any]]:
    """
    Extract elements from PDF with parallel processing and GPU acceleration

    Args:
        pdf_path: Path to PDF file
        num_processes: Number of parallel processes (default: CPU count - 2)
    """
    import multiprocessing

    # Auto-detect optimal number of processes
    if num_processes is None:
        cpu_count = multiprocessing.cpu_count()
        # Use 80% of available cores, leaving some for system
        num_processes = max(1, int(cpu_count * 0.8))

    logger.info(f"[PDF] Using {num_processes} parallel processes for extraction")

    elements = []

    try:
        # First attempt with parallel hi_res strategy
        try:
            elements = partition_pdf(
                filename=pdf_path,
                infer_table_structure=True,
                languages=["spa", "eng"],
                strategy="hi_res",
                extract_images_in_pdf=True,
                extract_tables=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                # PARALLEL PROCESSING OPTIONS
                multipage_sections=True,  # Process pages independently
                n_jobs=num_processes,  # Number of parallel jobs
                # PERFORMANCE OPTIMIZATIONS
                pdf_infer_table_structure=True,
                strategy_kwargs={
                    "hi_res_model_name": "yolox",  # Faster model
                },
            )

            text_elements = [
                e
                for e in elements
                if e.get("type") == "text" and e.get("text", "").strip()
            ]
            if not text_elements:
                logger.warning("[PDF] No text found with hi_res, trying OCR...")
                raise ValueError("No text elements found")

        except Exception as first_error:
            logger.warning(f"[PDF] Hi-res extraction failed: {first_error}")

            # Fallback to parallel OCR
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    infer_table_structure=True,
                    languages=["spa", "eng"],
                    strategy="ocr_only",
                    extract_images_in_pdf=True,
                    extract_tables=True,
                    ocr_languages="spa+eng",
                    ocr_mode="entire_page",
                    # PARALLEL OCR
                    n_jobs=num_processes,
                    multipage_sections=True,
                )

                text_elements = [
                    e
                    for e in elements
                    if e.get("type") == "text" and e.get("text", "").strip()
                ]
                if not text_elements:
                    raise ValueError("OCR extraction failed")

            except Exception as second_error:
                logger.warning(f"[PDF] OCR extraction failed: {second_error}")

                # Last resort with parallel auto
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="auto",
                    n_jobs=num_processes,
                    multipage_sections=True,
                )

    except Exception as e:
        logger.error(f"[PDF] All extraction strategies failed: {e}")
        raise ValueError(f"PDF extraction failed: {e}")

    # Convert to dict format
    element_dicts = []
    for elem in elements:
        elem_dict = elem.to_dict() if hasattr(elem, "to_dict") else {"text": str(elem)}
        elem_dict["type"] = elem.category if hasattr(elem, "category") else "unknown"
        element_dicts.append(elem_dict)

    logger.info(
        f"[PDF] Extracted {len(element_dicts)} elements using {num_processes} processes"
    )
    return element_dicts
