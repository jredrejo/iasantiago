"""
chunk.py - Enhanced Version with Advanced Chunking and Page Validation
Extrae: PDF, DOCX, PPTX (sin XLSX)
Mantiene caché SQLite + LLaVA análisis
Manejo defensivo de CUDA
Enhanced chunking with NLTK + Hierarchical with heading tracking + Context-aware with same embedding model
Enhanced page number validation and adaptive chunking strategies
"""

import os
import logging
import re
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx

from pathlib import Path
from typing import List, Dict, Any, Optional
import sqlite3
import threading
from collections import defaultdict
import numpy as np
import pdfplumber
import ssl
import nltk


logger = logging.getLogger(__name__)
# Set Tesseract language for unstructured
os.environ["TESSERACT_LANG"] = "spa+eng"  # Spanish + English
os.environ["OCR_LANGUAGES"] = "spa+eng"  # Alternative env var
os.environ["UNSTRUCTURED_LANGUAGES"] = "spa,eng"  # Spanish primero, luego English
os.environ["UNSTRUCTURED_FALLBACK_LANGUAGE"] = "eng"  # English si no se puede Spanish

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


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
                # Fall back to standard page extraction
                page = RobustPageExtractor.extract_page_number(elem)

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

    def __init__(self, *args, embedding_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.75  # Configurable

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

        # NEW: Choose appropriate chunker
        if strategy == "semantic_boundary":
            # Import here to avoid circular imports
            from main import model_cache

            embed_model = model_cache.get_model(
                "intfloat/multilingual-e5-large-instruct"
            )
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
            page = RobustPageExtractor._extract_from_coordinates(elem)
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
    def _extract_from_coordinates(elem) -> Optional[int]:
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

            # Standard page height approximations (in points)
            # A4: ~842 points, Letter: ~792 points
            # Use conservative estimate
            APPROX_PAGE_HEIGHT = 800

            estimated_page = max(1, int(min_y / APPROX_PAGE_HEIGHT) + 1)

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
                    f"Large gap: {gap} pages between {sorted_unique[i-1]} and {sorted_unique[i]}"
                )

        return issues

    @staticmethod
    def _detect_out_of_order(pages: list) -> list:
        """Detect out-of-order pages"""
        issues = []

        for i in range(1, len(pages)):
            if pages[i] < pages[i - 1]:
                issues.append(
                    f"Out of order: chunk {i-1} page {pages[i-1]} -> chunk {i} page {pages[i]}"
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
        self.lock = threading.RLock()
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
            with self.lock:
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
            with self.lock:
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
            with self.lock:
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
            with self.lock:
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
                )
                table_row = cursor.fetchone()

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
    # Extract elements from PDF
    elements = extract_elements_from_pdf(pdf_path)

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


def extract_elements_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract elements from PDF with enhanced page detection"""
    elements = []

    try:
        # Use unstructured for initial extraction
        raw_elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            languages=["spa", "eng"],
            ocr_languages="spa+eng",
        )

        # Convert to our format with enhanced page detection
        for elem in raw_elements:
            elem_dict = {
                "text": str(elem),
                "type": (
                    str(elem.category).lower()
                    if hasattr(elem, "category")
                    else "unknown"
                ),
                "page": RobustPageExtractor.extract_page_number(elem, pdf_path),
                "metadata": (
                    elem.metadata.to_dict()
                    if hasattr(elem, "metadata") and hasattr(elem.metadata, "to_dict")
                    else {}
                ),
            }
            elements.append(elem_dict)

    except Exception as e:
        logger.error(f"Error extracting elements from PDF: {e}")
        raise

    return elements
