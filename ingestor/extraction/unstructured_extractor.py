"""
Wrapper de partition_pdf de Unstructured para extracción de documentos.

Proporciona extracción consciente del layout usando la biblioteca unstructured
con estrategias configurables (fast, hi_res, ocr_only).
"""

import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.cache import get_pdf_total_pages
from extraction.base import Element

logger = logging.getLogger(__name__)


def _partition_pdf_compat(filename: str, **kwargs) -> List[Any]:
    """
    Llama a unstructured.partition.pdf.partition_pdf con solo kwargs soportados.

    Filtra kwargs para coincidir con la firma de la versión instalada.
    """
    from unstructured.partition.pdf import partition_pdf as _partition_pdf

    sig = inspect.signature(_partition_pdf)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return _partition_pdf(filename=filename, **filtered)


def _slice_pdf_to_temp(pdf_path: str, pages_1_indexed: List[int]) -> tuple:
    """
    Crea un PDF temporal conteniendo solo las páginas solicitadas (indexadas desde 1).

    Returns:
        Tupla de (ruta_archivo_temp, número_página_inicio)
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
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


class UnstructuredExtractor:
    """
    Extractor usando biblioteca unstructured para extracción consciente del layout.

    Soporta múltiples estrategias:
    - fast: Extracción rápida sin análisis de layout
    - hi_res: Alta resolución con detección de layout
    - ocr_only: Extracción basada en OCR para documentos escaneados
    - auto: Selección automática de estrategia
    """

    def __init__(
        self,
        strategy: str = "auto",
        enable_tables: bool = False,
        hi_res_model_name: Optional[str] = None,
    ):
        """
        Inicializa el extractor.

        Args:
            strategy: Estrategia de extracción ('fast', 'hi_res', 'ocr_only', 'auto')
            enable_tables: Si extraer tablas (más lento)
            hi_res_model_name: Nombre del modelo para estrategia hi_res
        """
        self.strategy = strategy
        self.enable_tables = enable_tables
        self.hi_res_model_name = hi_res_model_name

    @property
    def name(self) -> str:
        return f"UnstructuredExtractor ({self.strategy})"

    def can_handle(self, pdf_path: Path) -> bool:
        """Siempre retorna True - este es un extractor seguro de respaldo."""
        return True

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae elementos del PDF usando biblioteca unstructured.
        """
        pdf_path = Path(pdf_path)
        total_pages = get_pdf_total_pages(str(pdf_path))

        raw_elements = self._partition(str(pdf_path), total_pages=total_pages)
        return self._convert_elements(raw_elements)

    def extract_pages(
        self, pdf_path: Path, pages: List[int], total_pages: Optional[int] = None
    ) -> List[Element]:
        """
        Extrae páginas específicas del PDF.

        Args:
            pdf_path: Ruta al archivo PDF
            pages: Lista de números de página (indexados desde 1) a extraer
            total_pages: Conteo total de páginas opcional
        """
        if total_pages is None:
            total_pages = get_pdf_total_pages(str(pdf_path))

        raw_elements = self._partition(
            str(pdf_path), pages=pages, total_pages=total_pages
        )
        return self._convert_elements(raw_elements)

    def _partition(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        total_pages: Optional[int] = None,
    ) -> List[Any]:
        """
        Lógica principal de partición con soporte de filtrado de páginas.
        """
        if total_pages is None:
            total_pages = get_pdf_total_pages(str(pdf_path))

        pages_desc = "all" if not pages else f"{min(pages)}-{max(pages)}"
        logger.info(
            f"[UNSTRUCTURED] strategy={self.strategy} pages={pages_desc} "
            f"total_pages={total_pages or 'desconocido'}"
        )

        kwargs = self._build_kwargs()

        # Si no hay filtro de páginas, procesar PDF completo
        if not pages:
            return _partition_pdf_compat(str(pdf_path), **kwargs)

        # Intentar filtrado nativo de páginas primero
        from unstructured.partition.pdf import partition_pdf as _partition_pdf

        sig = inspect.signature(_partition_pdf)

        if "page_numbers" in sig.parameters:
            kwargs["page_numbers"] = pages
            return _partition_pdf_compat(str(pdf_path), **kwargs)

        if "pages" in sig.parameters:
            kwargs["pages"] = pages
            return _partition_pdf_compat(str(pdf_path), **kwargs)

        # Respaldo: dividir PDF en rangos y procesar por separado
        return self._partition_with_slicing(pdf_path, pages, kwargs, sig)

    def _build_kwargs(self) -> Dict[str, Any]:
        """Construye kwargs para partition_pdf basado en estrategia."""
        kwargs: Dict[str, Any] = {
            "strategy": self.strategy,
            "languages": ["spa", "eng"],
            "extract_images_in_pdf": False,
            "chunking_strategy": None,
            "keep_extra_chars": False,
            "max_characters": 15000,
            "extract_tables": bool(self.enable_tables),
            "infer_table_structure": bool(self.enable_tables),
            "multipage_sections": True,
        }

        if self.strategy == "fast":
            kwargs["max_characters"] = 20000
            kwargs["extract_tables"] = False
            kwargs["infer_table_structure"] = False

        elif self.strategy == "hi_res":
            kwargs["max_characters"] = 10000
            kwargs["model_name"] = "yolox"
            kwargs["skip_infer_table_types"] = ["pdf", "jpg", "png"]
            if self.hi_res_model_name is not None:
                kwargs["hi_res_model_name"] = self.hi_res_model_name

        elif self.strategy == "ocr_only":
            kwargs.update(
                {
                    "ocr_languages": "spa+eng",
                    "ocr_mode": "entire_page",
                    "max_characters": 12000,
                    "ocr_kwargs": {"config": "--oem 3 --psm 6"},
                }
            )

        return kwargs

    def _partition_with_slicing(
        self,
        pdf_path: str,
        pages: List[int],
        kwargs: Dict[str, Any],
        sig: inspect.Signature,
    ) -> List[Any]:
        """
        Particiona dividiendo PDF en rangos contiguos.
        Usado cuando unstructured no soporta filtrado de páginas nativamente.
        """
        pages_sorted = sorted(set(int(p) for p in pages))

        # Agrupar en rangos contiguos
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

                # Remapear números de página si es necesario
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

    def _convert_elements(self, raw_elements: List[Any]) -> List[Element]:
        """Convierte elementos de unstructured a nuestra dataclass Element."""
        elements: List[Element] = []

        for elem in raw_elements:
            # Obtener dict del elemento
            if hasattr(elem, "to_dict"):
                d = elem.to_dict()
            else:
                d = {"text": str(elem)}

            # Extraer número de página
            page = d.get("page", 1)
            if page is None and hasattr(elem, "metadata") and elem.metadata:
                pn = getattr(elem.metadata, "page_number", None)
                if pn:
                    page = pn
            page = int(page or 1)

            # Extraer tipo
            elem_type = d.get("type") or d.get("category", "text")

            elements.append(
                Element(
                    text=str(d.get("text", "")),
                    type=str(elem_type),
                    page=page,
                    source=f"unstructured_{self.strategy}",
                    metadata=d.get("metadata", {}),
                )
            )

        return elements


def process_elements_to_chunks(elements: List[Any]) -> List[Dict[str, Any]]:
    """
    Función legacy para compatibilidad hacia atrás.
    Convierte elementos de unstructured a formato dict.
    """
    result: List[Dict[str, Any]] = []
    for elem in elements:
        if hasattr(elem, "to_dict"):
            d = elem.to_dict()
        else:
            d = {"text": str(elem)}

        if "page" not in d and hasattr(elem, "metadata") and elem.metadata:
            pn = getattr(elem.metadata, "page_number", None)
            if pn:
                d["page"] = pn

        d["page"] = int(d.get("page", 1) or 1)
        d.setdefault("type", d.get("category", "text"))
        result.append(d)

    return result
