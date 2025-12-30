"""
Extracción de texto usando pypdf y pdfplumber.

Proporciona extracción rápida de texto para PDFs con capas de texto embebidas,
con extracción de tablas vía pdfplumber.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pdfplumber
import pypdf

from core.cache import get_pdf_total_pages
from core.config import LARGE_PDF_BATCH_SIZE
from core.heartbeat import call_heartbeat
from extraction.base import Element, ExtractionError

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extractor usando pypdf para texto y pdfplumber para tablas.

    Este es el método de extracción más rápido para PDFs con texto embebido.
    Usa enfoque híbrido: pypdf (rápido) para texto + pdfplumber para tablas.
    """

    @property
    def name(self) -> str:
        return "TextExtractor (pypdf+pdfplumber)"

    def can_handle(self, pdf_path: Path) -> bool:
        """Siempre retorna True - este es un extractor seguro de respaldo."""
        return True

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae texto y tablas del PDF.

        Usa procesamiento por lotes para PDFs grandes (>1000 páginas) para prevenir bloqueos.
        """
        pdf_path = Path(pdf_path)
        total_pages = get_pdf_total_pages(str(pdf_path))
        logger.info(f"Páginas: {total_pages}")

        # Intentar enfoque híbrido primero
        elements = self._extract_hybrid(pdf_path, total_pages)
        if elements:
            return elements

        # Respaldo solo con pdfplumber
        logger.info("a por el  pdfplumber")
        elements = self._extract_pdfplumber(pdf_path, total_pages)
        if elements:
            return elements

        logger.info("No HAY ELEMENTOS")
        return []

    def _extract_hybrid(
        self, pdf_path: Path, total_pages: Optional[int]
    ) -> List[Element]:
        """
        Extracción híbrida: pypdf para texto (rápido) + pdfplumber para tablas.
        """
        is_large_pdf = total_pages is not None and total_pages > LARGE_PDF_BATCH_SIZE
        elements: List[Element] = []

        try:
            # Extraer texto con pypdf
            if is_large_pdf:
                elements = self._extract_pypdf_batched(pdf_path, total_pages)
            else:
                elements = self._extract_pypdf(pdf_path)

            if not elements:
                return []

            # Agregar tablas con pdfplumber
            table_count = self._extract_tables(pdf_path, total_pages, elements)

            logger.info(
                f"Extraídos {len(elements)} elementos "
                f"({len(elements) - table_count} texto, {table_count} tablas)"
            )
            return elements

        except Exception as e:
            logger.warning(f"Híbrido pypdf+pdfplumber falló: {e}")
            return []

    def _extract_pypdf(self, pdf_path: Path) -> List[Element]:
        """Extrae texto del PDF usando pypdf."""
        elements: List[Element] = []
        logger.info("Intentando pypdf (más rápido) para texto...")

        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    elements.append(
                        Element(
                            text=text,
                            type="text",
                            page=i + 1,
                            source="pypdf",
                        )
                    )
        return elements

    def _extract_pypdf_batched(self, pdf_path: Path, total_pages: int) -> List[Element]:
        """Extrae texto de PDF grande en lotes usando pypdf."""
        elements: List[Element] = []
        total_batches = (total_pages + LARGE_PDF_BATCH_SIZE - 1) // LARGE_PDF_BATCH_SIZE

        logger.info(
            f"PDF grande detectado ({total_pages} páginas), "
            f"procesando con pypdf en {total_batches} lotes..."
        )

        for batch_num, batch_start in enumerate(
            range(0, total_pages, LARGE_PDF_BATCH_SIZE), 1
        ):
            batch_end = min(batch_start + LARGE_PDF_BATCH_SIZE, total_pages)
            call_heartbeat(f"pypdf_batch_{batch_start + 1}-{batch_end}")
            logger.info(
                f"Lote pypdf {batch_num}/{total_batches}: "
                f"páginas {batch_start + 1}-{batch_end}"
            )

            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for i in range(batch_start, batch_end):
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        elements.append(
                            Element(
                                text=text,
                                type="text",
                                page=i + 1,
                                source="pypdf",
                            )
                        )
        return elements

    def _extract_tables(
        self, pdf_path: Path, total_pages: Optional[int], elements: List[Element]
    ) -> int:
        """Extrae tablas usando pdfplumber y las agrega a la lista de elementos."""
        is_large_pdf = total_pages is not None and total_pages > LARGE_PDF_BATCH_SIZE
        table_count = 0

        logger.info(
            f"pypdf extrajo {len(elements)} elementos de texto, ahora extrayendo tablas..."
        )

        try:
            if is_large_pdf:
                table_count = self._extract_tables_batched(
                    pdf_path, total_pages, elements
                )
            else:
                table_count = self._extract_tables_simple(pdf_path, elements)
        except Exception as e:
            logger.warning(f"Extracción de tablas pdfplumber falló: {e}")

        return table_count

    def _extract_tables_simple(self, pdf_path: Path, elements: List[Element]) -> int:
        """Extrae tablas del PDF sin lotes."""
        table_count = 0
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                table_count += self._process_page_tables(page, i + 1, elements)
        return table_count

    def _extract_tables_batched(
        self, pdf_path: Path, total_pages: int, elements: List[Element]
    ) -> int:
        """Extrae tablas de PDF grande en lotes."""
        table_count = 0
        total_batches = (total_pages + LARGE_PDF_BATCH_SIZE - 1) // LARGE_PDF_BATCH_SIZE

        for batch_num, batch_start in enumerate(
            range(0, total_pages, LARGE_PDF_BATCH_SIZE), 1
        ):
            batch_end = min(batch_start + LARGE_PDF_BATCH_SIZE, total_pages)
            call_heartbeat(f"pdfplumber_tables_batch_{batch_start + 1}-{batch_end}")
            logger.info(
                f"Lote tablas pdfplumber {batch_num}/{total_batches}: "
                f"páginas {batch_start + 1}-{batch_end}"
            )

            with pdfplumber.open(pdf_path) as pdf:
                for i in range(batch_start, batch_end):
                    page = pdf.pages[i]
                    table_count += self._process_page_tables(page, i + 1, elements)

        return table_count

    def _process_page_tables(self, page, page_num: int, elements: List[Element]) -> int:
        """Procesa tablas de una sola página y las agrega a elementos."""
        table_count = 0
        try:
            for table in page.extract_tables():
                if table:
                    table_text = self._format_table(table)
                    if table_text.strip():
                        elements.append(
                            Element(
                                text=table_text,
                                type="table",
                                page=page_num,
                                source="pdfplumber",
                            )
                        )
                        table_count += 1
        except Exception as e:
            logger.warning(f"Error extrayendo tabla de página {page_num}: {e}")
        return table_count

    def _format_table(self, table: list) -> str:
        """Formatea una tabla como texto separado por tabulaciones."""
        processed_rows = []
        for row in table:
            processed_row = [str(cell) if cell is not None else "" for cell in row]
            processed_rows.append("\t".join(processed_row))
        return "\n".join(processed_rows)

    def _extract_pdfplumber(
        self, pdf_path: Path, total_pages: Optional[int]
    ) -> List[Element]:
        """Extracción de respaldo usando pdfplumber para texto y tablas."""
        is_large_pdf = total_pages is not None and total_pages > LARGE_PDF_BATCH_SIZE
        elements: List[Element] = []

        try:
            if is_large_pdf:
                elements = self._extract_pdfplumber_batched(pdf_path, total_pages)
            else:
                elements = self._extract_pdfplumber_simple(pdf_path)

            if elements:
                logger.info(f"pdfplumber extrajo {len(elements)} elementos")
        except Exception as e:
            logger.warning(f"pdfplumber falló: {e}")

        return elements

    def _extract_pdfplumber_simple(self, pdf_path: Path) -> List[Element]:
        """Extrae texto y tablas del PDF usando pdfplumber."""
        elements: List[Element] = []
        logger.info("Intentando pdfplumber para texto+tablas...")

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extraer texto
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    elements.append(
                        Element(
                            text=text,
                            type="text",
                            page=i + 1,
                            source="pdfplumber",
                        )
                    )

                # Extraer tablas
                self._process_page_tables(page, i + 1, elements)

        return elements

    def _extract_pdfplumber_batched(
        self, pdf_path: Path, total_pages: int
    ) -> List[Element]:
        """Extrae de PDF grande en lotes usando pdfplumber."""
        elements: List[Element] = []
        total_batches = (total_pages + LARGE_PDF_BATCH_SIZE - 1) // LARGE_PDF_BATCH_SIZE

        logger.info(
            f"Intentando respaldo pdfplumber para PDF grande ({total_pages} páginas) "
            f"en {total_batches} lotes..."
        )

        for batch_num, batch_start in enumerate(
            range(0, total_pages, LARGE_PDF_BATCH_SIZE), 1
        ):
            batch_end = min(batch_start + LARGE_PDF_BATCH_SIZE, total_pages)
            call_heartbeat(f"pdfplumber_fallback_batch_{batch_start + 1}-{batch_end}")
            logger.info(
                f"Lote respaldo pdfplumber {batch_num}/{total_batches}: "
                f"páginas {batch_start + 1}-{batch_end}"
            )

            with pdfplumber.open(pdf_path) as pdf:
                for i in range(batch_start, batch_end):
                    page = pdf.pages[i]

                    # Extraer texto
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        elements.append(
                            Element(
                                text=text,
                                type="text",
                                page=i + 1,
                                source="pdfplumber",
                            )
                        )

                    # Extraer tablas
                    self._process_page_tables(page, i + 1, elements)

        return elements
