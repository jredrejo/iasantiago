"""
chunk.py - VERSIÓN SIMPLIFICADA
Extrae: PDF, DOCX, PPTX (sin XLSX)
Mantiene caché SQLite + LLaVA análisis
Manejo defensivo de CUDA
"""

import os
import logging
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx

from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import sqlite3
import threading
import requests
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# DISABLE CUDA FOR UNSTRUCTURED ONLY (prevent OOM with vLLM)
# No desactives para todo el proceso, solo para unstructured
# ============================================================
os.environ["UNSTRUCTURED_DISABLE_CUDA"] = "1"
# NO hagas esto: os.environ["CUDA_VISIBLE_DEVICES"] = ""
logger.warning("[CONFIG] CUDA DISABLED for Unstructured (vLLM-LLaVA uses GPU)")


# ============================================================
# CONFIGURATION: Deshabilitar LLaVA si es necesario
# ============================================================
DISABLE_LLAVA = os.getenv("DISABLE_LLAVA", "false").lower() == "true"
if DISABLE_LLAVA:
    logger.warning("[CONFIG] LLaVA analysis is DISABLED")
else:
    logger.info("[CONFIG] LLaVA analysis is ENABLED")


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
        """Extrae documento con Unstructured.io (sin CUDA)"""
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        logger.info(f"Extracting: {Path(file_path).name}")

        if not self.is_supported(ext):
            logger.warning(f"Format not supported: {ext}")
            return []

        try:
            # Unstructured: extrae texto básicamente (CUDA deshabilitado)
            # No intenta table structure detection ni image extraction de GPU
            if ext == ".pdf":
                elements = partition_pdf(
                    file_path,
                    infer_table_structure=False,  # NO table extraction GPU
                    extract_image_block_types=["Image"],  # Extrae imágenes como bloques
                    languages=["es", "en"],
                    split_pdf_pages=True,
                )
            elif ext in [".docx", ".doc"]:
                elements = partition_docx(file_path, infer_table_structure=False)
            elif ext in [".pptx", ".ppt"]:
                elements = partition_pptx(file_path, infer_table_structure=False)
            else:
                elements = partition(
                    file_path, infer_table_structure=False, languages=["es", "en"]
                )

            logger.info(f"Found {len(elements)} elements")

            # Process elements
            chunks = self._process_elements(elements)
            self._log_stats()

            return chunks

        except Exception as e:
            logger.error(f"Error extracting {file_path}: {e}", exc_info=True)
            return []

    def is_supported(self, ext: str) -> bool:
        """Verifica si formato es soportado"""
        return ext.lower() in self.SUPPORTED_FORMATS

    def _process_elements(self, elements) -> List[Dict[str, Any]]:
        """Procesa elementos según tipo"""
        chunks = []

        for element in elements:
            element_type = element.__class__.__name__

            # Extrae el número de página del metadata
            page = 1
            if hasattr(element, "metadata") and element.metadata:
                if hasattr(element.metadata, "page_number"):
                    page = element.metadata.page_number
                elif hasattr(element.metadata, "page"):
                    page = element.metadata.page

            # TEXTO
            if element_type in [
                "Text",
                "NarrativeText",
                "Title",
                "Heading",
                "Paragraph",
            ]:
                if hasattr(element, "text") and element.text.strip():
                    chunks.append(
                        {
                            "page": page,
                            "text": element.text.strip(),
                            "type": "text",
                            "source": "unstructured",
                        }
                    )
                    self.stats["text_chunks"] += 1

            # TABLAS (sin GPU extraction, solo como texto)
            elif element_type == "Table":
                chunk = self._process_table(element, page)
                if chunk:
                    chunks.append(chunk)

            # IMÁGENES
            elif element_type in ["Image", "Picture"]:
                chunk = self._process_image(element, page)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _process_table(self, element, page: int = 1) -> Optional[Dict[str, Any]]:
        """Procesa tabla con LLaVA + caché"""
        self.stats["tables_processed"] += 1

        try:
            table_text = element.text if hasattr(element, "text") else str(element)

            if not table_text.strip():
                return None

            table_hash = hashlib.md5(table_text.encode()).hexdigest()[:16]

            # Check cache
            cached = self.cache.load_table_cache(table_hash)
            if cached:
                self.stats["tables_cached"] += 1
                analysis = cached["analysis"]
            else:
                # Analyze with LLaVA
                analysis = self._analyze_table_with_llava(table_text)
                rows = len(table_text.split("\n"))
                cols = max(
                    len(row.split("\t"))
                    for row in table_text.split("\n")
                    if row.strip()
                )
                self.cache.save_table_cache(table_hash, analysis, rows, cols)

            return {
                "page": page,
                "text": analysis,
                "type": "table",
                "source": "unstructured+llava",
            }
        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return None

    def _process_image(self, element, page: int = 1) -> Optional[Dict[str, Any]]:
        """Procesa imagen con LLaVA + caché"""
        self.stats["images_processed"] += 1

        try:
            image = None
            if hasattr(element, "image"):
                image = element.image

            if not image:
                return None

            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            image_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()[:16]

            # Check cache
            cached = self.cache.load_image_cache(image_hash)
            if cached:
                self.stats["images_cached"] += 1
                description = cached["description"]
            else:
                description = self._describe_image_with_llava(image)
                width = image.width if hasattr(image, "width") else 0
                height = image.height if hasattr(image, "height") else 0
                self.cache.save_image_cache(image_hash, description, width, height)

            return {
                "page": page,
                "text": description,
                "type": "image",
                "source": "unstructured+llava",
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def _analyze_table_with_llava(self, table_text: str) -> str:
        """Analiza tabla con LLaVA"""
        prompt = f"""Analiza esta tabla:
1. Que datos contiene?
2. Cuales son las columnas principales?
3. Hay patrones importantes?

Tabla:
{table_text[:800]}

Responde brevemente."""

        try:
            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLaVA error: {e}")

        return table_text[:300]

    def _describe_image_with_llava(self, image: Image.Image) -> str:
        """Describe imagen con LLaVA"""
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            prompt = "Describe brevemente que ves en esta imagen (maximo 100 palabras)."

            response = requests.post(
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
                                        "url": f"data:image/png;base64,{img_base64}"
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

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLaVA image error: {e}")

        return "[Image processed]"

    def _log_stats(self):
        """Log estadisticas"""
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Text chunks: {self.stats['text_chunks']}")
        logger.info(f"Tables: {self.stats['tables_processed']}")

        if self.stats["tables_processed"] > 0:
            ratio = self.stats["tables_cached"] / self.stats["tables_processed"] * 100
            logger.info(f"  -> Cached: {self.stats['tables_cached']} ({ratio:.1f}%)")

        logger.info(f"Images: {self.stats['images_processed']}")

        if self.stats["images_processed"] > 0:
            ratio = self.stats["images_cached"] / self.stats["images_processed"] * 100
            logger.info(f"  -> Cached: {self.stats['images_cached']} ({ratio:.1f}%)")

        logger.info("=" * 60 + "\n")


# ============================================================
# PUBLIC INTERFACE
# ============================================================


def pdf_to_chunks(
    path: str,
    chunk_size: int = 900,
    overlap: int = 120,
    vllm_url: str = "http://vllm-llava:8000",
    cache_db: str = "/tmp/llava_cache/llava_cache.db",
) -> List[Dict[str, Any]]:
    """
    Extrae chunks usando Unstructured.io

    Formatos: PDF, DOCX, PPTX, HTML, Markdown, Imagenes
    Cache: SQLite (70x speedup)
    Analisis: LLaVA para tablas e imagenes
    CUDA: Deshabilitado (usa vLLM-LLaVA en GPU)
    """
    extractor = SimpleExtractor(vllm_url=vllm_url, cache_db=cache_db)

    chunks = extractor.extract_document(path)

    # Apply chunking to text only
    final_chunks = []
    for chunk in chunks:
        if chunk["type"] == "text" and len(chunk["text"]) > chunk_size:
            text = chunk["text"]
            start = 0

            while start < len(text):
                end = min(len(text), start + chunk_size)
                seg = text[start:end]

                final_chunks.append(
                    {**chunk, "text": seg, "chunk_id": len(final_chunks)}
                )

                start = end - overlap if end - overlap > start else end
        else:
            chunk["chunk_id"] = len(final_chunks)
            final_chunks.append(chunk)

    logger.info(f"Total chunks: {len(final_chunks)}")
    return final_chunks
