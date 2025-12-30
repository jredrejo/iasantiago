"""
Extracción basada en OCR usando EasyOCR y Tesseract.

Proporciona OCR acelerado por GPU para PDFs escaneados con EasyOCR,
con respaldo Tesseract.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch

from core.cache import get_pdf_total_pages
from core.config import setup_ssl_context
from core.heartbeat import call_heartbeat
from extraction.base import Element
from extraction.unstructured_extractor import UnstructuredExtractor

logger = logging.getLogger(__name__)

# Verificar disponibilidad de EasyOCR
try:
    import easyocr

    EASYOCR_AVAILABLE = True
    logger.info("[EASYOCR] EasyOCR importado exitosamente")
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("[EASYOCR] EasyOCR no disponible, instalar con: pip install easyocr")


class EasyOCRProcessor:
    """
    OCR acelerado por GPU usando EasyOCR con backend PyTorch.

    Características:
    - Inicialización perezosa (modelos cargados en primer uso)
    - Instancia de reader compartida entre llamadas
    - Respaldo automático GPU/CPU
    """

    _reader_cache = None
    _initialization_attempted = False

    def __init__(
        self,
        use_gpu: bool = True,
        gpu_id: int = 0,
        model_storage_directory: Optional[str] = None,
    ):
        """
        Inicializa el procesador OCR.

        Args:
            use_gpu: Si usar aceleración GPU
            gpu_id: ID del dispositivo GPU
            model_storage_directory: Directorio para almacenar modelos
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if not self.use_gpu and use_gpu:
            logger.warning("[EASYOCR] CUDA no disponible, usando respaldo CPU")

        if model_storage_directory:
            os.environ["EASYOCR_MODULE_PATH"] = str(model_storage_directory)
            self.model_dir = Path(model_storage_directory)
        else:
            self.model_dir = Path.home() / ".EasyOCR"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[EASYOCR] Directorio de modelos: {self.model_dir}")

        # Usar reader cacheado si está disponible
        if EasyOCRProcessor._reader_cache is not None:
            logger.info("[EASYOCR] Usando reader cacheado")
            self.reader = EasyOCRProcessor._reader_cache
            return

        # No reintentar si la inicialización anterior falló
        if (
            EasyOCRProcessor._initialization_attempted
            and EasyOCRProcessor._reader_cache is None
        ):
            raise RuntimeError("[EASYOCR] Inicialización anterior falló, no reintentando")

        self._initialize_reader()

    def _initialize_reader(self):
        """Inicializa el reader de EasyOCR."""
        EasyOCRProcessor._initialization_attempted = True

        setup_ssl_context()
        models_exist = self._check_models_exist()

        if not models_exist:
            logger.info("[EASYOCR] Modelos no encontrados, se descargarán en primer uso")
            logger.info(
                "[EASYOCR] Esto puede tomar 2-5 minutos dependiendo de la velocidad de red"
            )
            call_heartbeat("easyocr_model_download")
        else:
            logger.info("[EASYOCR] Usando modelos existentes")

        logger.info("[EASYOCR] Inicializando reader...")
        call_heartbeat("easyocr_init")

        self.reader = easyocr.Reader(
            ["es", "en"],
            gpu=self.use_gpu,
            model_storage_directory=str(self.model_dir),
            download_enabled=True,
            detector=True,
            recognizer=True,
            verbose=False,
        )

        call_heartbeat("easyocr_init_done")
        EasyOCRProcessor._reader_cache = self.reader

        device = "GPU" if self.use_gpu else "CPU"
        logger.info(f"[EASYOCR] Inicializado exitosamente en {device}")

    def _check_models_exist(self) -> bool:
        """Verifica si los modelos de EasyOCR ya están descargados."""
        try:
            craft_path = self.model_dir / "model" / "craft_mlt_25k.pth"
            spanish_path = self.model_dir / "model" / "spanish_g2.pth"
            english_path = self.model_dir / "model" / "english_g2.pth"

            models_exist = craft_path.exists() and (
                spanish_path.exists() or english_path.exists()
            )

            if models_exist:
                logger.info(f"[EASYOCR] Encontrados modelos existentes en {self.model_dir}")
            else:
                logger.info("[EASYOCR] Modelos no encontrados, se descargarán")

            return models_exist
        except Exception as e:
            logger.warning(f"[EASYOCR] No se pudo verificar modelos existentes: {e}")
            return False

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extrae texto de un archivo de imagen.

        Args:
            image_path: Ruta al archivo de imagen

        Returns:
            Texto extraído como cadena
        """
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
            logger.error(f"[EASYOCR] Falló al procesar {image_path}: {e}")
            return ""

    def process_pdf_page_batch(
        self, pdf_path: str, page_nums: List[int]
    ) -> Dict[int, str]:
        """
        Procesa múltiples páginas de PDF con OCR.

        Args:
            pdf_path: Ruta al archivo PDF
            page_nums: Lista de números de página (indexados desde 1)

        Returns:
            Diccionario mapeando números de página a texto extraído
        """
        from pdf2image import convert_from_path

        logger.debug(
            f"[EASYOCR] Procesando páginas {page_nums} para {os.path.basename(pdf_path)}"
        )

        first_page = min(page_nums)
        last_page = max(page_nums)

        call_heartbeat(f"ocr_batch_{first_page}-{last_page}")
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
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=f"_page_{actual_page_num}.png", delete=False
                    )
                    image.save(tmp.name)
                    temp_files.append((actual_page_num, tmp.name))

            for page_num, tmp_path in temp_files:
                call_heartbeat(f"ocr_page_{page_num}")
                text = self.extract_text_from_image(tmp_path)
                results[page_num] = text

        finally:
            for _, tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        return results


class OCRExtractor:
    """
    Extractor OCR con soporte EasyOCR (GPU) y Tesseract (CPU).

    Intenta EasyOCR primero para aceleración GPU, recurre a Tesseract.
    """

    def __init__(self, use_gpu: bool = True, batch_size: int = 20):
        """
        Inicializa el extractor OCR.

        Args:
            use_gpu: Si intentar aceleración GPU
            batch_size: Páginas por lote para procesamiento
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._tesseract_extractor = UnstructuredExtractor(
            strategy="ocr_only",
            enable_tables=False,
        )

    @property
    def name(self) -> str:
        if EASYOCR_AVAILABLE and self.use_gpu:
            return "OCRExtractor (EasyOCR GPU + Tesseract fallback)"
        return "OCRExtractor (Tesseract)"

    def can_handle(self, pdf_path: Path) -> bool:
        """Siempre retorna True - OCR puede manejar cualquier PDF."""
        return True

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae texto del PDF usando OCR.

        Intenta EasyOCR primero, recurre a Tesseract.
        """
        pdf_path = Path(pdf_path)
        total_pages = get_pdf_total_pages(str(pdf_path))

        # Intentar EasyOCR primero
        if EASYOCR_AVAILABLE and self.use_gpu:
            elements = self._extract_easyocr(pdf_path, total_pages)
            if elements and self._is_sufficient(elements):
                return elements
            logger.info("[OCR] EasyOCR insuficiente -> respaldo Tesseract")

        # Respaldo Tesseract
        elements = self._extract_tesseract(pdf_path, total_pages)
        if elements:
            return elements

        # Último recurso: unstructured auto
        logger.info("[OCR] Último recurso: unstructured auto")
        auto_extractor = UnstructuredExtractor(strategy="auto", enable_tables=False)
        return auto_extractor.extract(pdf_path)

    def _extract_easyocr(
        self, pdf_path: Path, total_pages: Optional[int]
    ) -> List[Element]:
        """Extrae usando pipeline EasyOCR GPU."""
        try:
            logger.info("[OCR] Usando pipeline EasyOCR GPU")
            ocr_processor = EasyOCRProcessor(use_gpu=True, gpu_id=0)
            elements: List[Element] = []

            num_pages = total_pages or 0
            if num_pages <= 0:
                logger.warning("[OCR] Total de páginas desconocido; omitiendo lotes EasyOCR")
                return []

            for batch_start in range(1, num_pages + 1, self.batch_size):
                batch_end = min(batch_start + self.batch_size - 1, num_pages)
                page_nums = list(range(batch_start, batch_end + 1))
                call_heartbeat(f"easyocr_batch_{batch_start}-{batch_end}")

                page_texts = ocr_processor.process_pdf_page_batch(
                    str(pdf_path), page_nums
                )
                for page_num, text in page_texts.items():
                    if text and text.strip():
                        elements.append(
                            Element(
                                text=text,
                                type="text",
                                page=int(page_num),
                                source="easyocr_gpu",
                            )
                        )

            return elements

        except Exception as e:
            logger.warning(f"[OCR] EasyOCR falló: {e}")
            return []

    def _extract_tesseract(
        self, pdf_path: Path, total_pages: Optional[int]
    ) -> List[Element]:
        """Extrae usando Tesseract vía unstructured."""
        try:
            logger.info("[OCR] Usando Tesseract (unstructured ocr_only)")
            call_heartbeat("tesseract_ocr")

            num_pages = total_pages or 0
            if num_pages <= 0:
                # Procesar PDF completo
                return self._tesseract_extractor.extract(pdf_path)

            # Procesar en lotes
            elements: List[Element] = []
            for batch_start in range(1, num_pages + 1, self.batch_size):
                batch_end = min(batch_start + self.batch_size - 1, num_pages)
                page_nums = list(range(batch_start, batch_end + 1))
                call_heartbeat(f"tesseract_batch_{batch_start}-{batch_end}")

                batch_elements = self._tesseract_extractor.extract_pages(
                    pdf_path, page_nums, total_pages
                )
                elements.extend(batch_elements)

            return elements

        except Exception as e:
            logger.error(f"[OCR] Extracción Tesseract falló: {e}")
            return []

    def _is_sufficient(self, elements: List[Element]) -> bool:
        """Verifica si los resultados de extracción son suficientes."""
        total_chars = sum(len(e.text) for e in elements)
        return elements and total_chars > 300
