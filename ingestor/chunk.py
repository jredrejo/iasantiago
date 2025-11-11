"""
chunk.py  –  Hybrid extractor
Unstructured.io  (default, local)
MinerU micro-service  (remote, PDF-only, complexity-gated)
SQLite cache  (images / tables)
LLaVA analysis  (via vLLM)
"""

import os
import logging
import hashlib
import requests
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO
from PIL import Image
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CUDA / GPU  –  keep Unstructured OFF GPU (vLLM uses it)
# ------------------------------------------------------------------
os.environ["UNSTRUCTURED_DISABLE_CUDA"] = "false"
logger.warning("[CONFIG] CUDA disabled for Unstructured (vLLM-LLaVA uses GPU)")

# ------------------------------------------------------------------
# Feature flags
# ------------------------------------------------------------------
DISABLE_LLAVA = os.getenv("DISABLE_LLAVA", "false").lower() == "true"
USE_MINERU = os.getenv("USE_MINERU", "true").lower() == "true"
MINERU_URL = os.getenv("MINERU_SERVICE_URL", "http://mineru-extractor:8003")


# ------------------------------------------------------------------
# SQLite Cache Manager  (unchanged)
# ------------------------------------------------------------------
class SQLiteCacheManager:
    """Thread-safe SQLite cache for images / tables."""

    def __init__(self, cache_db: str = "/tmp/llava_cache/llava_cache.db"):
        self.cache_db = Path(cache_db)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._init_db()

    # ---------- internal helpers ----------
    def _init_db(self) -> None:
        with self.lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS image_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    width INTEGER, height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 1
                )"""
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS table_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_hash TEXT UNIQUE NOT NULL,
                    analysis TEXT NOT NULL,
                    rows INTEGER, cols INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 1
                )"""
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_hash ON image_cache(image_hash)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_table_hash ON table_cache(table_hash)"
            )
            conn.commit()
            conn.close()

    def _get_conn(self):
        import sqlite3

        return sqlite3.connect(str(self.cache_db), check_same_thread=False, timeout=10)

    # ---------- public API ----------
    def load_image_cache(self, image_hash: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT description,width,height FROM image_cache WHERE image_hash=?",
                (image_hash,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE image_cache SET accessed_at=CURRENT_TIMESTAMP,hit_count=hit_count+1 WHERE image_hash=?",
                    (image_hash,),
                )
                conn.commit()
                conn.close()
                logger.debug(f"Cache hit (image): {image_hash}")
                return {"description": row[0], "width": row[1], "height": row[2]}
            conn.close()
        return None

    def save_image_cache(
        self, image_hash: str, description: str, width: int, height: int
    ) -> None:
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO image_cache
                (image_hash,description,width,height,accessed_at,hit_count)
                VALUES (?,?,?,?,CURRENT_TIMESTAMP,1)""",
                (image_hash, description, width, height),
            )
            conn.commit()
            conn.close()
            logger.debug(f"Cache saved (image): {image_hash}")

    def load_table_cache(self, table_hash: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT analysis FROM table_cache WHERE table_hash=?", (table_hash,)
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE table_cache SET accessed_at=CURRENT_TIMESTAMP,hit_count=hit_count+1 WHERE table_hash=?",
                    (table_hash,),
                )
                conn.commit()
                conn.close()
                logger.debug(f"Cache hit (table): {table_hash}")
                return {"analysis": row[0]}
            conn.close()
        return None

    def save_table_cache(
        self, table_hash: str, analysis: str, rows: int, cols: int
    ) -> None:
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO table_cache
                (table_hash,analysis,rows,cols,accessed_at,hit_count)
                VALUES (?,?,?,?,CURRENT_TIMESTAMP,1)""",
                (table_hash, analysis, rows, cols),
            )
            conn.commit()
            conn.close()
            logger.debug(f"Cache saved (table): {table_hash}")


# ------------------------------------------------------------------
# Page-number helper  (robust)
# ------------------------------------------------------------------
def extract_page_number(
    element, element_idx: int, total_elements: int, file_path: str
) -> int:
    """Return validated page number (1-based)."""
    page = 1
    try:
        if hasattr(element, "metadata") and element.metadata:
            md = element.metadata
            if hasattr(md, "page_number") and md.page_number:
                page = int(md.page_number)
            elif hasattr(md, "page") and md.page:
                page = int(md.page)
            elif total_elements > 0 and hasattr(md, "coordinates"):
                # crude fallback: ~25 elements per page
                page = (element_idx // 25) + 1
                logger.debug(
                    f"Inferred page {page} from position for element {element_idx}"
                )
        # sanity bounds
        if page < 1 or page > 10000:
            logger.warning(
                f"Invalid page {page} in {Path(file_path).name}, defaulting to 1"
            )
            page = 1
    except (ValueError, TypeError) as e:
        logger.warning(f"Page parse error in {Path(file_path).name}: {e}")
        page = 1
    except Exception as e:
        logger.error(f"Unexpected page error in {Path(file_path).name}: {e}")
        page = 1
    return page


# ------------------------------------------------------------------
# Hybrid Extractor
# ------------------------------------------------------------------
class HybridExtractor:
    SUPPORTED_FORMATS = {
        ext: True
        for ext in [
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".html",
            ".htm",
            ".md",
            ".txt",
            ".png",
            ".jpg",
            ".jpeg",
        ]
    }

    def __init__(
        self,
        vllm_url: str = "http://vllm-llava:8000",
        cache_db: str = "/tmp/llava_cache/llava_cache.db",
    ):
        self.vllm_url = vllm_url
        self.cache = SQLiteCacheManager(cache_db=cache_db)
        self.mineru_url = MINERU_URL
        self.use_mineru = USE_MINERU
        self.stats = {
            "text_chunks": 0,
            "tables_processed": 0,
            "tables_cached": 0,
            "images_processed": 0,
            "images_cached": 0,
            "mineru_used": 0,
            "unstructured_used": 0,
        }

    # ---------- public entry ----------
    def extract_document(self, file_path: str) -> List[Dict[str, Any]]:
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        logger.info(f"Extracting: {Path(file_path).name}")

        if not self.is_supported(ext):
            logger.warning(f"Format not supported: {ext}")
            return []

        # PDF + MinerU enabled + complexity threshold
        if ext == ".pdf" and self.use_mineru:
            complexity = self._detect_pdf_complexity(file_path)
            threshold = float(os.getenv("MINERU_COMPLEXITY_THRESHOLD", "0.6"))
            logger.warning(f"Complexity of {complexity} with threshold of {threshold}")
            if complexity >= threshold:
                logger.info(
                    f"[EXTRACTOR] Complexity {complexity:.2f} >= {threshold} → MinerU service"
                )
                chunks = self._extract_via_mineru_service(file_path)
                self.stats["mineru_used"] += 1
                return self._post_process_chunks(chunks, file_path)

        # Default: local Unstructured
        logger.info("[EXTRACTOR] Using local Unstructured.io")
        chunks = self._extract_with_unstructured(file_path)
        self.stats["unstructured_used"] += 1
        return self._post_process_chunks(chunks, file_path)

    # ---------- helpers ----------
    def is_supported(self, ext: str) -> bool:
        return ext.lower() in self.SUPPORTED_FORMATS

    def _detect_pdf_complexity(self, file_path: str) -> float:
        """Return 0.0-1.0 complexity score (quick heuristic)."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

            score = 0.0
            score += min(num_pages / 100, 0.3)
            score += min(file_size_mb / 50, 0.2)

            # sample first 3 pages for image density
            sample = min(3, num_pages)
            images = 0
            for p in reader.pages[:sample]:
                if hasattr(p, "images"):
                    images += len(p.images)
            avg_img = images / sample if sample else 0
            score += min(avg_img / 5, 0.3)

            # low text → likely scanned
            text_len = sum(len(p.extract_text() or "") for p in reader.pages[:sample])
            if text_len < 1000:
                score += 0.2

            return min(score, 1.0)
        except Exception as e:
            logger.warning(f"Complexity detection failed: {e}")
            return 0.5

    # ---------- Unstructured local ----------
    def _extract_with_unstructured(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            elements = []
            ext = Path(file_path).suffix.lower()
            ocr_used = False

            if ext == ".pdf":
                elements = partition_pdf(
                    filename=file_path,
                    strategy="fast",
                    infer_table_structure=True,
                    extract_image_block_types=["Image"],
                    extract_strategy="auto",
                    languages=["es", "en"],
                    split_pdf_pages=True,
                )
                # OCR fallback if no text
                if (
                    sum(1 for e in elements if hasattr(e, "text") and e.text.strip())
                    == 0
                ):
                    logger.warning("[PDF] No embedded text, switching to OCR (hi_res)")
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="hi_res",
                        infer_table_structure=True,
                        extract_image_block_types=["Image"],
                        ocr_languages="spa+eng",
                        extract_strategy="auto",
                        split_pdf_pages=True,
                    )
                    ocr_used = True
            elif ext in [".docx", ".doc"]:
                elements = partition_docx(file_path, infer_table_structure=False)
            elif ext in [".pptx", ".ppt"]:
                elements = partition_pptx(file_path, infer_table_structure=False)
            else:
                elements = partition(
                    file_path, infer_table_structure=False, languages=["es", "en"]
                )

            logger.info(
                f"Unstructured found {len(elements)} elements (OCR: {ocr_used})"
            )
            return self._process_elements(elements, file_path)

        except Exception as e:
            logger.error(f"Unstructured extraction failed: {e}", exc_info=True)
            return []

    def _process_elements(self, elements, file_path: str) -> List[Dict[str, Any]]:
        """Convert Unstructured elements → chunks."""
        chunks = []
        total = len(elements)
        for idx, el in enumerate(elements):
            typ = el.__class__.__name__
            page = extract_page_number(el, idx, total, file_path)

            if typ in ["Text", "NarrativeText", "Title", "Heading", "Paragraph"]:
                if hasattr(el, "text") and el.text.strip():
                    chunks.append(
                        {
                            "page": page,
                            "text": el.text.strip(),
                            "type": "text",
                            "source": "unstructured",
                        }
                    )
                    self.stats["text_chunks"] += 1

            elif typ == "Table":
                chunk = self._process_table_element(el, page)
                if chunk:
                    chunks.append(chunk)

            elif typ in ["Image", "Picture"]:
                chunk = self._process_image_element(el, page)
                if chunk:
                    chunks.append(chunk)

        return chunks

    # ---------- MinerU remote ----------
    def _extract_via_mineru_service(self, file_path: str) -> List[Dict[str, Any]]:
        """Call MinerU micro-service; fallback to Unstructured on any error."""
        import requests

        logger.info(
            f"[MinerU-HTTP] POST {self.mineru_url}/extract  file={Path(file_path).name}"
        )
        try:
            rsp = requests.post(
                f"{self.mineru_url}/extract", json={"file_path": file_path}, timeout=120
            )
            if rsp.status_code != 200:
                logger.warning(f"[MinerU-HTTP] {rsp.status_code}  {rsp.text[:100]}")
                logger.warning("Falling back → Unstructured")
                return self._extract_with_unstructured(file_path)

            data = rsp.json()
            chunks = data["chunks"]
            logger.info(
                f"[MinerU-HTTP] received {len(chunks)} chunks  pages={data.get('page_count')}"
            )
            self.stats["mineru_used"] += 1

            # enrich tables / images with LLaVA (MinerU service returns raw text)
            for c in chunks:
                if c["type"] == "table":
                    c["text"] = self._analyze_table_with_llava(c["text"])
                    c["source"] = "mineru+llava"
                elif c["type"] == "image":
                    c["text"] = self._describe_image_with_llava(
                        Image.open(BytesIO(requests.get(c["url"]).content))
                        if c.get("url")
                        else Image.new("RGB", (100, 100))
                    )
                    c["source"] = "mineru+llava"
            return chunks

        except Exception as e:
            logger.error(f"[MinerU-HTTP] {e}")
            logger.warning("Falling back → Unstructured")
            return self._extract_with_unstructured(file_path)

    # ---------- table / image  (LLaVA) ----------
    def _process_table_element(self, element, page: int) -> Optional[Dict[str, Any]]:
        self.stats["tables_processed"] += 1
        try:
            text = element.text if hasattr(element, "text") else str(element)
            if not text.strip():
                return None
            h = hashlib.md5(text.encode()).hexdigest()[:16]
            cached = self.cache.load_table_cache(h)
            if cached:
                self.stats["tables_cached"] += 1
                analysis = cached["analysis"]
            else:
                analysis = self._analyze_table_with_llava(text)
                rows = len(text.split("\n"))
                cols = max(
                    len(row.split("\t")) for row in text.split("\n") if row.strip()
                )
                self.cache.save_table_cache(h, analysis, rows, cols)
            return {
                "page": page,
                "text": analysis,
                "type": "table",
                "source": "unstructured+llava",
            }
        except Exception as e:
            logger.error(f"Table processing error: {e}")
            return None

    def _process_image_element(self, element, page: int) -> Optional[Dict[str, Any]]:
        self.stats["images_processed"] += 1
        try:
            img = element.image if hasattr(element, "image") else None
            if not img:
                return None
            buf = BytesIO()
            img.save(buf, format="PNG")
            h = hashlib.md5(buf.getvalue()).hexdigest()[:16]
            cached = self.cache.load_image_cache(h)
            if cached:
                self.stats["images_cached"] += 1
                desc = cached["description"]
            else:
                desc = self._describe_image_with_llava(img)
                w, h = img.width, img.height
                self.cache.save_image_cache(h, desc, w, h)
            return {
                "page": page,
                "text": desc,
                "type": "image",
                "source": "unstructured+llava",
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

    # ---------- LLaVA calls ----------
    def _analyze_table_with_llava(self, table_text: str) -> str:
        if DISABLE_LLAVA:
            return table_text[:300]
        prompt = f"Analiza esta tabla:\n1. Que datos contiene?\n2. Columnas principales?\n3. Patrones importantes?\n\nTabla:\n{table_text[:800]}\n\nResponde brevemente."
        try:
            r = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLaVA table error: {e}")
        return table_text[:300]

    def _describe_image_with_llava(self, image: Image.Image) -> str:
        if DISABLE_LLAVA:
            return "[Image processed]"
        try:
            buf = BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            prompt = "Describe brevemente que ves en esta imagen (máximo 100 palabras)."
            r = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}"
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": 150,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLaVA image error: {e}")
        return "[Image processed]"

    # ---------- final chunking ----------
    def _post_process_chunks(self, chunks: List[Dict], file_path: str) -> List[Dict]:
        """Apply chunk-size split on text chunks only."""
        final = []
        chunk_size = 900
        overlap = 120
        for c in chunks:
            if c["type"] == "text" and len(c["text"]) > chunk_size:
                text = c["text"]
                start = 0
                while start < len(text):
                    end = min(len(text), start + chunk_size)
                    seg = text[start:end]
                    final.append({**c, "text": seg, "chunk_id": len(final)})
                    start = end - overlap if end - overlap > start else end
            else:
                final.append({**c, "chunk_id": len(final)})
        logger.info(f"Final chunk count: {len(final)}")
        self._log_stats()
        return final

    def _log_stats(self):
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Text chunks: {self.stats['text_chunks']}")
        logger.info(
            f"Tables: {self.stats['tables_processed']}  (cached: {self.stats['tables_cached']})"
        )
        logger.info(
            f"Images: {self.stats['images_processed']}  (cached: {self.stats['images_cached']})"
        )
        logger.info(
            f"Engines  –  MinerU: {self.stats['mineru_used']}  Unstructured: {self.stats['unstructured_used']}"
        )
        logger.info("=" * 60 + "\n")


# ------------------------------------------------------------------
# Public helper  (unchanged signature)
# ------------------------------------------------------------------
def pdf_to_chunks(
    path: str,
    chunk_size: int = 900,
    overlap: int = 120,
    vllm_url: str = "http://vllm-llava:8000",
    cache_db: str = "/tmp/llava_cache/llava_cache.db",
) -> List[Dict[str, Any]]:
    """Extract → chunk  (kept for backward compatibility)."""
    extractor = HybridExtractor(vllm_url=vllm_url, cache_db=cache_db)
    chunks = extractor.extract_document(path)
    # chunking already applied inside extractor
    return chunks
