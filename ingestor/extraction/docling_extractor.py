"""
Extracción de PDF basada en Docling con aceleración GPU.

Proporciona extracción de documentos de alta calidad usando la biblioteca Docling
con seguimiento de fallos y respaldo automático.
"""

import copy
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf
import torch
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from core.cache import ExtractionCache, get_file_hash_sha256
from core.gpu import get_gpu_manager
from core.heartbeat import call_heartbeat
from extraction.base import Element, ExtractionError

logger = logging.getLogger(__name__)

# Estado a nivel de módulo
_tables_disabled_after_crash = False
_extraction_cache: Optional[ExtractionCache] = None


class DoclingCrashLimitExceeded(Exception):
    """Se lanza cuando un archivo ha fallado demasiadas veces y debe usar extracción alternativa."""

    pass


class CrashStateManager:
    """Rastrea archivos que causan fallos para comportamiento de omitir-después-de-N-fallos."""

    def __init__(self, cache_dir: Path, max_crashes: int = 1):
        self.state_file = cache_dir / "crash_state.json"
        self.max_crashes = max_crashes
        self._state: Dict[str, int] = {}
        self._load()

    def _load(self):
        """Carga estado de fallos desde disco."""
        import json

        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    self._state = json.load(f)
                logger.info(
                    f"[DOCLING] Estado de fallos cargado: {len(self._state)} archivos rastreados"
                )
        except Exception as e:
            logger.warning(f"[DOCLING] Error al cargar estado de fallos: {e}")
            self._state = {}

    def _save(self):
        """Guarda estado de fallos en disco."""
        import json

        try:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f)
        except Exception as e:
            logger.warning(f"[DOCLING] Error al guardar estado de fallos: {e}")

    def should_skip(self, filename: str) -> bool:
        """Verifica si un archivo ha fallado demasiadas veces."""
        count = self._state.get(filename, 0)
        if count >= self.max_crashes:
            logger.warning(
                f"[DOCLING] Archivo {filename} ha fallado {count} veces - omitiendo docling"
            )
            return True
        return False

    def mark_processing(self, filename: str):
        """Marca un archivo como actualmente en proceso (incrementa contador de fallos)."""
        self._state[filename] = self._state.get(filename, 0) + 1
        self._save()

    def mark_success(self, filename: str):
        """Marca un archivo como procesado exitosamente (limpia contador de fallos)."""
        if filename in self._state:
            del self._state[filename]
            self._save()


def _get_extraction_cache() -> ExtractionCache:
    """Obtiene o crea la caché de extracción."""
    global _extraction_cache
    if _extraction_cache is None:
        _extraction_cache = ExtractionCache()
    return _extraction_cache


def _get_crash_state() -> CrashStateManager:
    """Obtiene el gestor de estado de fallos."""
    cache = _get_extraction_cache()
    return CrashStateManager(cache.cache_dir)


class DoclingExtractor:
    """
    Extractor de PDF basado en Docling con aceleración GPU.

    Características:
    - Conversión de documentos acelerada por GPU
    - Seguimiento de fallos para omitir archivos problemáticos
    - Caché de extracción para acceso repetido
    - Respaldo automático a PyPDF en caso de fallo
    """

    def __init__(self, enable_tables: bool = False, enable_ocr: bool = False):
        """
        Inicializa el extractor.

        Args:
            enable_tables: Habilitar detección de estructura de tablas (más lento)
            enable_ocr: Habilitar OCR para páginas escaneadas (más lento)
        """
        self.enable_tables = enable_tables and not _tables_disabled_after_crash
        self.enable_ocr = enable_ocr
        self._gpu_manager = get_gpu_manager()
        self._cache = _get_extraction_cache()
        self._crash_state = _get_crash_state()

    @property
    def name(self) -> str:
        return (
            "DoclingExtractor (GPU)"
            if self._gpu_manager.is_available
            else "DoclingExtractor (CPU)"
        )

    def can_handle(self, pdf_path: Path) -> bool:
        """Verifica si el archivo puede ser procesado (no ha fallado demasiadas veces)."""
        filename = Path(pdf_path).name
        return not self._crash_state.should_skip(filename)

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae elementos del PDF usando Docling.

        Returns:
            Lista de objetos Element extraídos

        Raises:
            DoclingCrashLimitExceeded: Si el archivo ha fallado demasiadas veces
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name

        # Verificar límite de fallos
        if self._crash_state.should_skip(filename):
            raise DoclingCrashLimitExceeded(
                f"Archivo {filename} ha fallado demasiadas veces - usar extracción alternativa"
            )

        # Verificar caché
        file_hash = get_file_hash_sha256(str(pdf_path))
        if file_hash:
            cached = self._cache.get(file_hash)
            if cached:
                logger.info(f"[DOCLING] Acierto de caché para {filename}")
                return [Element.from_dict(d) for d in cached]

        # Marcar como en proceso (para detección de fallos)
        self._crash_state.mark_processing(filename)

        # Registrar info del archivo
        file_size_mb = pdf_path.stat().st_size / 1e6
        logger.info(f"[DOCLING] Procesando: {filename}")
        logger.info(f"[DOCLING] Tamaño del archivo: {file_size_mb:.2f} MB")

        # Validar PDF
        is_valid, error_msg = self._validate_pdf(pdf_path)
        if not is_valid:
            logger.error(f"[DOCLING] Validación de PDF fallida: {error_msg}")
            return self._extract_pypdf_fallback(pdf_path)

        # Limpieza de GPU antes de extracción
        if self._gpu_manager.is_available:
            self._gpu_manager.clear_cache()
            self._gpu_manager.log_memory_usage("antes")

        start_time = time.time()
        converter = None

        try:
            converter = self._get_converter()
            logger.info("[DOCLING] Iniciando conversión...")

            call_heartbeat(f"docling_convert_{filename}")
            result = converter.convert(str(pdf_path))

            if not hasattr(result, "document"):
                raise ValueError("Resultado de Docling inválido - falta document")

            elements = self._extract_from_document(result.document, pdf_path)

            if not elements:
                logger.warning(
                    "[DOCLING] No se extrajeron elementos - usando respaldo PyPDF"
                )
                return self._extract_pypdf_fallback(pdf_path)

            elapsed = time.time() - start_time
            logger.info(
                f"[DOCLING] Extraídos {len(elements)} elementos en {elapsed:.2f}s"
            )

            # Cachear el resultado
            if file_hash:
                self._cache.put(file_hash, [e.to_dict() for e in elements])

            # Marcar éxito
            self._crash_state.mark_success(filename)

            return elements

        except Exception as e:
            logger.error(f"[DOCLING] Extracción fallida: {e}", exc_info=True)
            return self._extract_pypdf_fallback(pdf_path)

        finally:
            if converter:
                del converter

            if self._gpu_manager.is_available:
                gc.collect()
                self._gpu_manager.clear_cache()

    def _get_converter(self) -> DocumentConverter:
        """Obtiene DocumentConverter configurado."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.enable_ocr
        pipeline_options.do_table_structure = self.enable_tables
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        logger.info("[DOCLING] Opciones del pipeline:")
        logger.info(f"  - do_ocr: {pipeline_options.do_ocr}")
        logger.info(f"  - do_table_structure: {pipeline_options.do_table_structure}")

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseV4DocumentBackend,
            )
        }

        return DocumentConverter(format_options=format_options)

    def _extract_from_document(self, doc, pdf_path: Path) -> List[Element]:
        """Extrae elementos del objeto documento de Docling."""
        elements: List[Element] = []
        source = "docling_gpu" if self._gpu_manager.is_available else "docling_cpu"

        # Método 1: export_to_markdown por página (más preciso)
        if hasattr(doc, "export_to_markdown"):
            try:
                num_pages = len(doc.pages) if hasattr(doc, "pages") else 0

                if num_pages > 0:
                    logger.info(f"[DOCLING] Extrayendo markdown de {num_pages} páginas")

                    for page_num in range(1, num_pages + 1):
                        if page_num % 5 == 0:
                            call_heartbeat(f"docling_page_{page_num}")

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
                                    Element(
                                        text=para,
                                        type="text",
                                        page=page_num,
                                        source=source,
                                        metadata={
                                            "docling_type": "markdown_paragraph",
                                            "method": "export_to_markdown_per_page",
                                        },
                                    )
                                )
                        except Exception as e:
                            logger.warning(f"[DOCLING] Página {page_num} falló: {e}")

                    if elements:
                        logger.info(
                            f"[DOCLING] Extraídos {len(elements)} elementos con páginas precisas"
                        )
                        return elements

                else:
                    # Respaldo: estimar páginas
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
                                Element(
                                    text=para,
                                    type="text",
                                    page=estimated_page,
                                    source=source,
                                    metadata={
                                        "docling_type": "markdown_paragraph",
                                        "method": "export_to_markdown",
                                        "page_source": "estimated",
                                    },
                                )
                            )

                        if elements:
                            return elements

            except Exception as e:
                logger.warning(f"[DOCLING] Exportar markdown falló: {e}")

        # Método 2: export_to_dict (respaldo)
        if not elements and hasattr(doc, "export_to_dict"):
            logger.info("[DOCLING] Usando respaldo export_to_dict()")
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
                            and item["prov"]
                        ):
                            prov = item["prov"][0]
                            if isinstance(prov, dict) and "page" in prov:
                                page = prov["page"]
                            elif hasattr(prov, "page"):
                                page = prov.page

                        elements.append(
                            Element(
                                text=text,
                                type="text",
                                page=page,
                                source=source,
                                metadata={
                                    "docling_type": item.get("type", "text"),
                                    "method": "export_to_dict",
                                },
                            )
                        )

                    logger.info(
                        f"[DOCLING] export_to_dict extrajo {len(elements)} elementos"
                    )

            except Exception as e:
                logger.warning(f"[DOCLING] Exportar dict falló: {e}")

        return elements

    def _validate_pdf(self, pdf_path: Path) -> tuple:
        """Valida estructura del PDF."""
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                num_pages = len(reader.pages)

                if num_pages == 0:
                    return False, "PDF no tiene páginas"

                logger.info(f"[DOCLING] Validación PDF: {num_pages} páginas")

                if reader.is_encrypted:
                    logger.warning("[DOCLING] PDF está encriptado")
                    return False, "PDF está encriptado"

                # Verificar tamaños de página
                for page_num, page in enumerate(reader.pages[:5], 1):
                    try:
                        mediabox = page.mediabox
                        width = float(mediabox.width)
                        height = float(mediabox.height)

                        if width <= 0 or height <= 0:
                            return (
                                False,
                                f"Página {page_num} tiene dimensiones inválidas",
                            )
                    except Exception:
                        return (
                            False,
                            f"No se puede determinar tamaño de página {page_num}",
                        )

                return True, None

        except Exception as e:
            logger.error(f"[DOCLING] Validación de PDF fallida: {e}")
            return False, str(e)

    def _extract_pypdf_fallback(self, pdf_path: Path) -> List[Element]:
        """Extracción de respaldo usando PyPDF."""
        logger.warning("[DOCLING] Usando extracción de respaldo PyPDF")
        elements: List[Element] = []

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
                                Element(
                                    text=para,
                                    type="text",
                                    page=page_num + 1,
                                    source="pypdf_fallback",
                                )
                            )

                    except Exception as e:
                        logger.error(f"[DOCLING] Error en página {page_num + 1}: {e}")

                logger.info(f"[DOCLING] PyPDF extrajo {len(elements)} elementos")
                return elements

        except Exception as e:
            logger.error(f"[DOCLING] Extracción PyPDF fallida: {e}", exc_info=True)
            raise ExtractionError(f"Todos los métodos de extracción fallaron: {e}")
