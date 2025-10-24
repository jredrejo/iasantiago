"""
chunk_hybrid.py - EXTRACTOR HÍBRIDO
Usa Unstructured.io para formatos variados
Usa MinerU para PDFs complejos
Mantiene SQLite caché personalizado + LLaVA análisis
"""

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import sqlite3
import threading
import logging
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# SQLITE CACHE MANAGER (MANTENEMOS IGUAL)
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

                # Nueva tabla: métricas de extracción
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS extraction_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_hash TEXT UNIQUE NOT NULL,
                        file_path TEXT,
                        extraction_method TEXT,
                        complexity_score REAL,
                        num_tables INTEGER,
                        num_images INTEGER,
                        extraction_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_image_hash ON image_cache(image_hash)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_table_hash ON table_cache(table_hash)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_file_hash ON extraction_metrics(file_hash)"
                )

                conn.commit()
                logger.info("Cache database initialized (hybrid mode)")
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

    def save_extraction_metric(
        self,
        file_hash: str,
        file_path: str,
        method: str,
        complexity: float,
        num_tables: int,
        num_images: int,
        extraction_time: float,
    ) -> None:
        """Guarda métricas de extracción"""
        try:
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """INSERT OR REPLACE INTO extraction_metrics
                           (file_hash, file_path, extraction_method, complexity_score,
                            num_tables, num_images, extraction_time)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            file_hash,
                            file_path,
                            method,
                            complexity,
                            num_tables,
                            num_images,
                            extraction_time,
                        ),
                    )
                    conn.commit()
                    logger.debug(f"Extraction metric saved: {file_path} ({method})")
        except Exception as e:
            logger.error(f"Error saving extraction metric: {e}")


# ============================================================
# COMPLEXITY ANALYZER (NUEVO)
# ============================================================


class PDFComplexityAnalyzer:
    """Analiza la complejidad de un PDF para decidir qué extractor usar"""

    def __init__(self):
        self.complexity_threshold_mineru = 0.6  # 0.0-1.0

    def analyze(self, file_path: str) -> Tuple[float, str]:
        """
        Analiza complejidad del PDF

        Returns:
            (complexity_score: 0.0-1.0, recommendation: "unstructured" | "mineru" | "hybrid")
        """
        try:
            import PyPDF2

            score = 0.0
            details = []

            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)

            # Factor 1: Número de páginas
            if num_pages > 100:
                score += 0.1
                details.append(f"Muchas páginas: {num_pages}")
            elif num_pages > 50:
                score += 0.05

            # Factor 2: Analizar primera página
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    first_page = reader.pages[0]

                    # Contar objetos
                    text = first_page.extract_text() or ""
                    text_length = len(text)

                    # Heurística: PDFs con poco texto relativo tienen más gráficos/tablas
                    if num_pages > 0:
                        avg_text_per_page = text_length / num_pages

                        # PDFs científicos/técnicos tienen menos texto (más figuras)
                        if avg_text_per_page < 500:
                            score += 0.3
                            details.append(
                                f"Bajo texto por página: {avg_text_per_page:.0f} chars"
                            )
                        elif avg_text_per_page < 1000:
                            score += 0.15
                            details.append(
                                f"Texto medio: {avg_text_per_page:.0f} chars"
                            )
            except:
                pass

            # Factor 3: Detectar tablas (heurística)
            try:
                from pdfplumber import open as pdf_open

                with pdf_open(file_path) as pdf:
                    for page_num, page in enumerate(
                        pdf.pages[:3]
                    ):  # Check first 3 pages
                        tables = page.find_table()
                        if tables:
                            score += 0.2
                            details.append(f"Tabla detectada en página {page_num + 1}")
            except:
                pass

            # Normalizar score a 0.0-1.0
            score = min(score, 1.0)

            # Decidir recomendación
            if score >= self.complexity_threshold_mineru:
                recommendation = "mineru"
                logger.info(f"PDF complexity: {score:.2f} (COMPLEX) → Using MinerU")
            else:
                recommendation = "unstructured"
                logger.info(
                    f"PDF complexity: {score:.2f} (SIMPLE) → Using Unstructured.io"
                )

            if details:
                logger.debug(f"Complexity factors: {', '.join(details)}")

            return score, recommendation

        except Exception as e:
            logger.warning(
                f"Error analyzing PDF complexity: {e}, using default (unstructured)"
            )
            return 0.5, "unstructured"


# ============================================================
# UNSTRUCTURED EXTRACTOR (MANTENEMOS)
# ============================================================


class UnstructuredExtractor:
    """Extrae con Unstructured.io (generalista)"""

    def __init__(self, vllm_url: str, cache: SQLiteCacheManager):
        self.vllm_url = vllm_url
        self.cache = cache
        self.name = "unstructured"

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extrae documento con Unstructured.io"""
        logger.info(f"Unstructured.io: Extrayendo {Path(file_path).name}")

        try:
            ext = Path(file_path).suffix.lower()

            if ext == ".pdf":
                elements = partition_pdf(
                    file_path,
                    infer_table_structure=True,
                    extract_image_block_types=["Image", "Table"],
                    strategy="hi_res",
                )
            elif ext in [".docx", ".doc"]:
                elements = partition_docx(file_path, infer_table_structure=True)
            elif ext in [".pptx", ".ppt"]:
                elements = partition_pptx(file_path, infer_table_structure=True)
            elif ext in [".xlsx", ".xls"]:
                elements = partition_xlsx(file_path, infer_table_structure=True)
            else:
                elements = partition(file_path, infer_table_structure=True)

            logger.info(f"Unstructured.io: {len(elements)} elementos encontrados")
            return elements

        except Exception as e:
            logger.error(f"Error en Unstructured.io: {e}", exc_info=True)
            return []


# ============================================================
# MINERU EXTRACTOR (NUEVO)
# ============================================================


class MinerUExtractor:
    """Extrae PDFs complejos con MinerU"""

    def __init__(self, vllm_url: str, cache: SQLiteCacheManager):
        self.vllm_url = vllm_url
        self.cache = cache
        self.name = "mineru"
        self._mineru = None

    def _load_mineru(self):
        """Lazy load MinerU (puede ser pesado)"""
        if self._mineru is None:
            try:
                from mineru.pdf_extract import PDFExtractor

                self._mineru = PDFExtractor()
                logger.info("MinerU cargado exitosamente")
            except ImportError:
                logger.error("MinerU no instalado. Instala con: pip install mineru")
                return None
        return self._mineru

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extrae PDF con MinerU"""
        logger.info(f"MinerU: Extrayendo {Path(file_path).name}")

        try:
            mineru = self._load_mineru()
            if not mineru:
                logger.warning("MinerU no disponible, fallback a Unstructured.io")
                return None

            # MinerU retorna estructura compleja
            content = mineru.extract(file_path)

            # Convertir formato MinerU a nuestro formato
            elements = self._convert_mineru_to_elements(content)

            logger.info(f"MinerU: {len(elements)} elementos encontrados")
            return elements

        except Exception as e:
            logger.error(f"Error en MinerU: {e}", exc_info=True)
            logger.warning("Fallback a Unstructured.io")
            return None

    def _convert_mineru_to_elements(self, mineru_content) -> List[Dict]:
        """Convierte formato MinerU a formato compatible"""
        elements = []

        try:
            # MinerU retorna: {pages: [{blocks: [...]}]}
            if isinstance(mineru_content, dict) and "pages" in mineru_content:
                for page_num, page in enumerate(mineru_content["pages"], start=1):
                    if "blocks" in page:
                        for block in page["blocks"]:
                            block_type = block.get("type", "text")

                            if block_type == "text":
                                elements.append(
                                    {
                                        "type": "Text",
                                        "text": block.get("content", ""),
                                        "page": page_num,
                                    }
                                )
                            elif block_type == "table":
                                elements.append(
                                    {
                                        "type": "Table",
                                        "text": self._format_table(block),
                                        "page": page_num,
                                    }
                                )
                            elif block_type == "image":
                                elements.append(
                                    {
                                        "type": "Image",
                                        "text": block.get("content", ""),
                                        "page": page_num,
                                    }
                                )
        except Exception as e:
            logger.error(f"Error convirtiendo formato MinerU: {e}")

        return elements

    def _format_table(self, table_block) -> str:
        """Formatea tabla de MinerU a string"""
        try:
            if isinstance(table_block.get("content"), list):
                rows = table_block["content"]
                formatted = "\n".join(
                    " | ".join(str(cell) for cell in row) for row in rows
                )
                return formatted
        except:
            pass

        return str(table_block.get("content", ""))


# ============================================================
# HYBRID EXTRACTOR (CORE)
# ============================================================


class HybridExtractor:
    """
    Extractor híbrido que elige mejor extractor automáticamente
    - Unstructured.io: Formatos variados + PDFs simples
    - MinerU: PDFs complejos (papers, reportes)
    """

    def __init__(
        self,
        vllm_url: str = "http://vllm:8000",
        cache_db: str = "/tmp/llava_cache/llava_cache.db",
    ):
        self.vllm_url = vllm_url
        self.cache = SQLiteCacheManager(cache_db=cache_db)
        self.analyzer = PDFComplexityAnalyzer()

        self.unstructured = UnstructuredExtractor(vllm_url, self.cache)
        self.mineru = MinerUExtractor(vllm_url, self.cache)

        self.stats = {
            "text_chunks": 0,
            "tables_processed": 0,
            "tables_cached": 0,
            "images_processed": 0,
            "images_cached": 0,
            "extraction_method": None,
        }

    def extract_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Extrae documento eligiendo mejor método"""
        import time

        start_time = time.time()

        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        logger.info(f"Hybrid extraction: {Path(file_path).name}")

        # Calcular hash del archivo
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:16]

        # Decidir extractor
        if ext == ".pdf":
            complexity_score, recommendation = self.analyzer.analyze(file_path)

            if recommendation == "mineru":
                logger.info(f"Complejidad: {complexity_score:.2f} → MinerU")
                elements = self.mineru.extract(file_path)

                # Si MinerU falla, fallback a Unstructured
                if elements is None:
                    logger.warning("MinerU falló, fallback a Unstructured.io")
                    elements = self.unstructured.extract(file_path)
                    self.stats["extraction_method"] = "unstructured (fallback)"
                else:
                    self.stats["extraction_method"] = "mineru"
            else:
                logger.info(f"Complejidad: {complexity_score:.2f} → Unstructured.io")
                elements = self.unstructured.extract(file_path)
                self.stats["extraction_method"] = "unstructured"
        else:
            # No-PDF: Unstructured.io
            elements = self.unstructured.extract(file_path)
            self.stats["extraction_method"] = f"unstructured ({ext})"

        if not elements:
            logger.error(f"Extracción falló: {file_path}")
            return []

        # Procesar elementos
        chunks = self._process_elements(elements, file_path)

        # Guardar métricas
        extraction_time = time.time() - start_time
        num_tables = sum(1 for c in chunks if c["type"] == "table")
        num_images = sum(1 for c in chunks if c["type"] == "image")

        self.cache.save_extraction_metric(
            file_hash,
            file_path,
            self.stats["extraction_method"],
            complexity_score if ext == ".pdf" else 0.0,
            num_tables,
            num_images,
            extraction_time,
        )

        self._log_stats()
        return chunks

    def _process_elements(self, elements, file_path: str) -> List[Dict[str, Any]]:
        """Procesa elementos según tipo"""
        chunks = []

        for element in elements:
            element_type = (
                element.__class__.__name__
                if hasattr(element, "__class__")
                else element.get("type", "Text")
            )

            # TEXTO
            if element_type in [
                "Text",
                "NarrativeText",
                "Title",
                "Heading",
                "Paragraph",
            ]:
                text = (
                    element.text
                    if hasattr(element, "text")
                    else element.get("text", "")
                )
                if text and text.strip():
                    chunks.append(
                        {
                            "page": (
                                getattr(element.metadata, "page_number", 1)
                                if hasattr(element, "metadata")
                                else 1
                            ),
                            "text": text.strip(),
                            "type": "text",
                            "source": "hybrid",
                            "chunk_id": len(chunks),
                        }
                    )
                    self.stats["text_chunks"] += 1

            # TABLAS
            elif element_type == "Table":
                chunk = self._process_table(element)
                if chunk:
                    chunks.append(chunk)

            # IMÁGENES
            elif element_type in ["Image", "Picture"]:
                chunk = self._process_image(element)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _process_table(self, element) -> Optional[Dict[str, Any]]:
        """Procesa tabla con LLaVA + caché"""
        self.stats["tables_processed"] += 1

        try:
            table_text = element.text if hasattr(element, "text") else str(element)

            if not table_text.strip():
                return None

            table_hash = hashlib.md5(table_text.encode()).hexdigest()[:16]

            # Verificar caché
            cached = self.cache.load_table_cache(table_hash)
            if cached:
                self.stats["tables_cached"] += 1
                analysis = cached["analysis"]
            else:
                analysis = self._analyze_table_with_llava(table_text)
                rows = len(table_text.split("\n"))
                cols = max(
                    len(row.split("\t"))
                    for row in table_text.split("\n")
                    if row.strip()
                )
                self.cache.save_table_cache(table_hash, analysis, rows, cols)

            page = (
                getattr(element.metadata, "page_number", 1)
                if hasattr(element, "metadata")
                else 1
            )

            return {
                "page": page,
                "text": analysis,
                "type": "table",
                "source": "hybrid+llava",
                "table_id": table_hash,
                "chunk_id": len([]),
            }
        except Exception as e:
            logger.error(f"Error procesando tabla: {e}")
            return None

    def _process_image(self, element) -> Optional[Dict[str, Any]]:
        """Procesa imagen con LLaVA + caché"""
        self.stats["images_processed"] += 1

        try:
            image = None
            if hasattr(element, "image"):
                image = element.image
            elif hasattr(element, "image_base64"):
                import base64

                img_data = base64.b64decode(element.image_base64)
                image = Image.open(BytesIO(img_data))

            if not image:
                return None

            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            image_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()[:16]

            # Verificar caché
            cached = self.cache.load_image_cache(image_hash)
            if cached:
                self.stats["images_cached"] += 1
                description = cached["description"]
            else:
                description = self._describe_image_with_llava(image)
                width = image.width if hasattr(image, "width") else 0
                height = image.height if hasattr(image, "height") else 0
                self.cache.save_image_cache(image_hash, description, width, height)

            page = (
                getattr(element.metadata, "page_number", 1)
                if hasattr(element, "metadata")
                else 1
            )

            return {
                "page": page,
                "text": description,
                "type": "image",
                "source": "hybrid+llava",
                "image_id": image_hash,
                "chunk_id": len([]),
            }
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            return None

    def _analyze_table_with_llava(self, table_text: str) -> str:
        """Analiza tabla con LLaVA"""
        prompt = f"""Analiza esta tabla y extrae:
1. Qué datos contiene
2. Relaciones entre columnas
3. Insights o patrones importantes

Tabla:
{table_text[:1000]}

Responde de forma concisa."""

        try:
            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.3,
                },
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error con LLaVA (tabla): {e}")

        return table_text[:500]

    def _describe_image_with_llava(self, image: Image.Image) -> str:
        """Describe imagen con LLaVA"""
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            prompt = """Analiza esta imagen en detalle. Extrae:
1. Texto visible
2. Elementos visuales principales
3. Contexto relevante

Responde de forma concisa."""

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
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
                timeout=60,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error con LLaVA (imagen): {e}")

        return "[Imagen procesada]"

    def _log_stats(self):
        """Log estadísticas"""
        logger.info("\n" + "=" * 60)
        logger.info("ESTADÍSTICAS EXTRACCIÓN (HÍBRIDA)")
        logger.info("=" * 60)
        logger.info(f"Método: {self.stats['extraction_method']}")
        logger.info(f"Chunks de texto: {self.stats['text_chunks']}")
        logger.info(f"Tablas procesadas: {self.stats['tables_processed']}")

        if self.stats["tables_processed"] > 0:
            logger.info(f"  → Desde caché: {self.stats['tables_cached']}")
            ratio = self.stats["tables_cached"] / self.stats["tables_processed"] * 100
            logger.info(f"  → Ratio caché: {ratio:.1f}%")

        logger.info(f"Imágenes procesadas: {self.stats['images_processed']}")

        if self.stats["images_processed"] > 0:
            logger.info(f"  → Desde caché: {self.stats['images_cached']}")
            ratio = self.stats["images_cached"] / self.stats["images_processed"] * 100
            logger.info(f"  → Ratio caché: {ratio:.1f}%")

        logger.info("=" * 60 + "\n")


# ============================================================
# INTERFAZ PÚBLICA (COMPATIBLE)
# ============================================================


def pdf_to_chunks(
    path: str,
    chunk_size: int = 900,
    overlap: int = 120,
    vllm_url: str = "http://vllm:8000",
    cache_db: str = "/tmp/llava_cache/llava_cache.db",
) -> List[Dict[str, Any]]:
    """
    Extrae chunks usando extractor HÍBRIDO

    Automáticamente elige:
    - MinerU para PDFs complejos
    - Unstructured.io para PDFs simples + otros formatos

    Args:
        path: Ruta del documento
        chunk_size: Tamaño máximo chunks
        overlap: Solapamiento entre chunks
        vllm_url: URL de vLLM
        cache_db: Ruta de base de datos SQLite

    Returns:
        Lista de chunks con metadatos
    """
    extractor = HybridExtractor(vllm_url=vllm_url, cache_db=cache_db)

    chunks = extractor.extract_document(path)

    # Aplicar chunking solo a texto (opcional)
    final_chunks = []
    for chunk in chunks:
        if chunk["type"] == "text" and len(chunk["text"]) > chunk_size:
            # Dividir chunks de texto grandes
            text = chunk["text"]
            start = 0
            chunk_counter = 0

            while start < len(text):
                end = min(len(text), start + chunk_size)
                seg = text[start:end]

                final_chunks.append(
                    {
                        **chunk,
                        "text": seg,
                        "chunk_id": len(final_chunks),
                        "sub_chunk": chunk_counter,
                    }
                )

                start = end - overlap if end - overlap > start else end
                chunk_counter += 1
        else:
            chunk["chunk_id"] = len(final_chunks)
            final_chunks.append(chunk)

    logger.info(f"Total chunks generados: {len(final_chunks)}")
    return final_chunks
