"""
Extracción de PDF escaneado con el pipeline VLM de Docling (§7.2).

Sustituto candidato de la cadena EasyOCR/Tesseract para páginas sin capa de
texto. Usa `VlmPipeline` con `granite-docling-258M` (Apache-2, 258 M de
parámetros), que **ya viene en docling 2.65.0**: la nota del §7.2 que pedía
subir a 2.7x "para el pipeline VLM" es incorrecta, `GRANITEDOCLING_TRANSFORMERS`
está en `docling.datamodel.vlm_model_specs` de esta versión (verificado
2026-07-31 contra la imagen `iasantiago-rag-ingestor:latest`).

**Es aditivo y está apagado por defecto.** Nada se ha borrado: `OCRExtractor` y
`UnstructuredExtractor` siguen en la cadena y siguen siendo el camino real
mientras `VLM_EXTRACTOR_ENABLED` no se ponga a `true`. Borrar `unstructured` y
la cadena EasyOCR es decisión humana (§9) y necesita la comparación sobre
escaneados reales que produce `scripts/compare_vlm_vs_ocr.py`.

Por qué puede ganar a EasyOCR, y qué habría que comprobar antes de creérselo:

- Devuelve un `DoclingDocument`, así que la fragmentación pasa por
  `HybridChunker` con secciones y tablas en vez de aplanar a párrafos sueltos.
  EasyOCR sólo sabe producir `Element` planos.
- Quita el pin `opencv-python-headless==4.8.0.76` (2023, CVEs conocidas) que
  sólo existe por compatibilidad `cv2.dnn.DictValue` con easyocr.
- Pero es **mucho más lento por página** que EasyOCR: es un modelo generativo,
  no un detector. De ahí la guarda `VLM_MAX_PAGES`.

Los elementos salen con `source="docling_vlm"`, distinguible en `payload.source`
de Qdrant: la comprobación por contenido que exige `LESSONS.md` necesita poder
separar lo que produjo este extractor de lo que produjo `easyocr_gpu`.
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Optional

from core.config import DOCLING_CONVERT_MAX_SECONDS
from core.gpu import get_gpu_manager
from core.heartbeat import BackgroundHeartbeat, call_heartbeat
from extraction.base import Element, ExtractionError, check_pdf_has_text

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Apagado por defecto: encenderlo cambia lo que se indexa de cada escaneado y
# eso sólo entra con un delta de eval por delante (PLAN.md, cierre de fase).
VLM_EXTRACTOR_ENABLED = _env_flag("VLM_EXTRACTOR_ENABLED", False)

# Techo de páginas por fichero. El VLM genera doctags token a token: un manual
# escaneado de 400 páginas son horas de GPU para un resultado que EasyOCR da en
# minutos. Por encima del techo este extractor se aparta y deja pasar la cadena
# OCR de siempre — es también la "guarda previa por número de páginas" que pide
# el §6.8 para que un fichero grande no se coma la ventana del watchdog.
VLM_MAX_PAGES = int(os.getenv("VLM_MAX_PAGES", "200"))

# Sólo tiene sentido sobre páginas sin capa de texto. Si el PDF tiene texto, o
# lo cogió Docling normal o lo coge pypdf/pdfplumber, ambos más baratos y mejores.
VLM_ONLY_SCANNED = _env_flag("VLM_ONLY_SCANNED", True)


class VlmExtractor:
    """
    Extractor de escaneados con el pipeline VLM de Docling.

    Se construye siempre, pero `can_handle()` devuelve False mientras
    `VLM_EXTRACTOR_ENABLED` esté apagado: así el extractor entra en la cadena sin
    cambiar ninguna ejecución existente y se enciende por entorno para medirlo.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        max_pages: Optional[int] = None,
        only_scanned: Optional[bool] = None,
    ):
        self.enabled = VLM_EXTRACTOR_ENABLED if enabled is None else enabled
        self.max_pages = VLM_MAX_PAGES if max_pages is None else max_pages
        self.only_scanned = VLM_ONLY_SCANNED if only_scanned is None else only_scanned
        self._gpu_manager = get_gpu_manager()
        self._last_document: Optional[Any] = None
        # Un único converter para toda la ejecución (§7.3): construirlo carga los
        # 258 M de parámetros del VLM. Reconstruirlo por PDF fue exactamente el
        # coste que se quitó en `dabf8fb` para el converter de layout.
        self._converter: Optional[Any] = None

    @property
    def name(self) -> str:
        device = "GPU" if self._gpu_manager.is_available else "CPU"
        return f"VlmExtractor (granite-docling, {device})"

    @property
    def last_document(self) -> Optional[Any]:
        """DoclingDocument de la última conversión, o None si no lo hubo."""
        return self._last_document

    def can_handle(self, pdf_path: Path) -> bool:
        """Decide si merece la pena gastar el VLM en este fichero.

        Rechaza —dejando pasar la cadena OCR de siempre— si está apagado, si el
        PDF ya tiene capa de texto, o si excede `VLM_MAX_PAGES`.
        """
        if not self.enabled:
            return False

        pdf_path = Path(pdf_path)

        if self.only_scanned and check_pdf_has_text(pdf_path):
            logger.info(
                f"[VLM] {pdf_path.name} tiene capa de texto: no es trabajo para el VLM"
            )
            return False

        pages = self._page_count(pdf_path)
        if pages is not None and pages > self.max_pages:
            logger.info(
                f"[VLM] {pdf_path.name} tiene {pages} páginas (> VLM_MAX_PAGES="
                f"{self.max_pages}): se deja a la cadena OCR"
            )
            return False

        return True

    @staticmethod
    def _page_count(pdf_path: Path) -> Optional[int]:
        """Páginas del PDF, o None si no se puede saber.

        No se usa `core.cache.get_pdf_total_pages` para no acoplar la guarda a un
        módulo de caché; un fallo aquí no debe impedir la extracción, sólo hace
        que la guarda de páginas no se aplique.
        """
        try:
            import pypdf

            with open(pdf_path, "rb") as fh:
                return len(pypdf.PdfReader(fh).pages)
        except Exception as e:
            logger.warning(f"[VLM] No se pudo contar páginas de {pdf_path.name}: {e}")
            return None

    def _get_converter(self):
        """Construye el DocumentConverter VLM una sola vez y lo reutiliza."""
        if self._converter is not None:
            return self._converter

        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        device = (
            AcceleratorDevice.CUDA
            if self._gpu_manager.is_available
            else AcceleratorDevice.CPU
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=GRANITEDOCLING_TRANSFORMERS,
            accelerator_options=AcceleratorOptions(device=device),
        )
        # No se conservan imágenes: sólo se indexa texto y guardarlas multiplica
        # la memoria del documento sin que nada aguas abajo las lea.
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        logger.info(
            f"[VLM] Construyendo converter: {GRANITEDOCLING_TRANSFORMERS.repo_id} "
            f"en {device.value}"
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
        return self._converter

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae elementos del PDF con el pipeline VLM.

        Raises:
            ExtractionError: si el extractor está apagado o la conversión falla.
        """
        pdf_path = Path(pdf_path)
        self._last_document = None

        if not self.enabled:
            raise ExtractionError("VlmExtractor deshabilitado (VLM_EXTRACTOR_ENABLED)")

        logger.info(f"[VLM] Procesando: {pdf_path.name}")
        start = time.time()

        if self._gpu_manager.is_available:
            self._gpu_manager.clear_cache()

        try:
            converter = self._get_converter()

            # Igual que en DoclingExtractor: `convert()` es una sola llamada
            # bloqueante y aquí es aún más larga (una generación por página). Sin
            # el heartbeat de fondo el watchdog mataría conversiones sanas y les
            # cargaría un crash — el fallo del §6.8-bis, tal cual.
            with BackgroundHeartbeat(
                f"vlm_convert_{pdf_path.name}",
                interval=30.0,
                max_duration=DOCLING_CONVERT_MAX_SECONDS,
            ):
                result = converter.convert(str(pdf_path))

            if not hasattr(result, "document"):
                raise ExtractionError("Resultado del VLM inválido - falta document")

            self._last_document = result.document
            elements = self._elements_from_document(result.document)

            if not elements:
                raise ExtractionError(f"El VLM no extrajo texto de {pdf_path.name}")

            elapsed = time.time() - start
            pages = len(getattr(result.document, "pages", {}) or {}) or 1
            logger.info(
                f"[VLM] {len(elements)} elementos de {pages} páginas en "
                f"{elapsed:.1f}s ({elapsed / pages:.1f}s/página)"
            )
            return elements

        except ExtractionError:
            raise
        except Exception as e:
            logger.error(f"[VLM] Extracción fallida: {e}", exc_info=True)
            raise ExtractionError(f"VlmExtractor falló en {pdf_path.name}: {e}") from e

        finally:
            if self._gpu_manager.is_available:
                gc.collect()
                self._gpu_manager.clear_cache()

    def _elements_from_document(self, doc) -> List[Element]:
        """Aplana el DoclingDocument a Elements por página.

        El `DoclingDocument` completo se conserva en `last_document` para que el
        pipeline fragmente con HybridChunker; estos Element son el camino de
        respaldo y lo que ven los extractores que no tienen estructura.
        """
        elements: List[Element] = []
        num_pages = len(getattr(doc, "pages", {}) or {})

        if not num_pages or not hasattr(doc, "export_to_markdown"):
            text = (doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else "").strip()
            if text:
                elements.append(
                    Element(text=text, type="text", page=1, source="docling_vlm")
                )
            return elements

        for page_num in range(1, num_pages + 1):
            if page_num % 5 == 0:
                call_heartbeat(f"vlm_page_{page_num}")
            try:
                text = doc.export_to_markdown(page_no=page_num).strip()
            except Exception as e:
                logger.warning(f"[VLM] Página {page_num} no exportable: {e}")
                continue
            if text:
                elements.append(
                    Element(
                        text=text,
                        type="text",
                        page=page_num,
                        source="docling_vlm",
                    )
                )

        return elements
