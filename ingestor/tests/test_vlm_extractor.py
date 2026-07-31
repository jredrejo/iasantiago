"""
Tests del VlmExtractor (§7.2): las guardas que deciden si llega a correr.

Lo que se prueba aquí no es la calidad del VLM —eso lo mide
`scripts/compare_vlm_vs_ocr.py` sobre escaneados reales— sino que el extractor
sea **inerte por defecto** y que sus dos guardas se apliquen. Es la propiedad de
la que depende que añadirlo a la cadena no cambie ninguna ejecución existente.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extraction.base import ExtractionError
from extraction.vlm_extractor import VlmExtractor


@pytest.fixture
def pdf(tmp_path) -> Path:
    p = tmp_path / "escaneado.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    return p


# --- (A) Apagado por defecto ---------------------------------------------
#
# La cadena de extracción lo construye siempre. Si `can_handle` no dijera que no
# mientras está apagado, encenderlo dejaría de ser una decisión de entorno.


def test_deshabilitado_no_maneja_nada(pdf):
    assert VlmExtractor(enabled=False).can_handle(pdf) is False


def test_deshabilitado_extract_lanza(pdf):
    with pytest.raises(ExtractionError, match="deshabilitado"):
        VlmExtractor(enabled=False).extract(pdf)


def test_deshabilitado_no_construye_el_converter(pdf):
    """Construir el converter carga 258 M de parámetros: no debe ocurrir jamás
    por el mero hecho de estar en la cadena."""
    ext = VlmExtractor(enabled=False)
    with patch.object(VlmExtractor, "_get_converter") as get_converter:
        ext.can_handle(pdf)
        with pytest.raises(ExtractionError):
            ext.extract(pdf)
    get_converter.assert_not_called()


# --- (B) Guarda de capa de texto -----------------------------------------
#
# El VLM es para páginas sin texto. Un PDF con capa de texto lo resuelven
# docling normal o pypdf, más baratos y mejores.


@pytest.mark.parametrize(
    "has_text,only_scanned,esperado",
    [
        (True, True, False),  # tiene texto y sólo-escaneados: se aparta
        (False, True, True),  # escaneado de verdad: le toca
        (True, False, True),  # guarda desactivada: acepta igualmente
    ],
)
def test_guarda_capa_de_texto(pdf, has_text, only_scanned, esperado):
    ext = VlmExtractor(enabled=True, only_scanned=only_scanned, max_pages=1000)
    with patch(
        "extraction.vlm_extractor.check_pdf_has_text", return_value=has_text
    ), patch.object(VlmExtractor, "_page_count", return_value=10):
        assert ext.can_handle(pdf) is esperado


# --- (C) Guarda de número de páginas -------------------------------------
#
# El VLM genera doctags token a token. Un escaneado de cientos de páginas son
# horas de GPU; por encima del techo se deja a EasyOCR. Es además la guarda
# previa por páginas que pide el §6.8 para no comerse la ventana del watchdog.


@pytest.mark.parametrize(
    "pages,max_pages,esperado",
    [
        (10, 200, True),
        (200, 200, True),  # el techo es inclusivo
        (201, 200, False),
        (None, 200, True),  # páginas desconocidas: no se puede aplicar la guarda
    ],
)
def test_guarda_max_paginas(pdf, pages, max_pages, esperado):
    ext = VlmExtractor(enabled=True, only_scanned=False, max_pages=max_pages)
    with patch.object(VlmExtractor, "_page_count", return_value=pages):
        assert ext.can_handle(pdf) is esperado


def test_page_count_no_revienta_con_pdf_ilegible(pdf):
    """Un fallo contando páginas desactiva la guarda, no la extracción."""
    assert VlmExtractor(enabled=True)._page_count(pdf) is None


# --- (D) Marcado de procedencia ------------------------------------------
#
# `LESSONS.md`: la comprobación de una tirada se hace por contenido
# (`payload.source`), no por su resumen. Si el VLM marcase sus elementos igual
# que easyocr, no habría forma de saber cuál de los dos produjo el índice.


def _doc(pages, markdown_por_pagina):
    doc = MagicMock()
    doc.pages = {i: MagicMock() for i in range(1, pages + 1)}
    doc.export_to_markdown.side_effect = lambda page_no=None: markdown_por_pagina.get(
        page_no, ""
    )
    return doc


def test_elementos_marcados_docling_vlm():
    ext = VlmExtractor(enabled=True)
    doc = _doc(3, {1: "Título y párrafo", 2: "  ", 3: "Otra página"})

    elements = ext._elements_from_document(doc)

    assert [e.source for e in elements] == ["docling_vlm", "docling_vlm"]
    # La página vacía no genera Element, y las que sí lo hacen conservan su número.
    assert [e.page for e in elements] == [1, 3]


def test_pagina_que_falla_no_aborta_el_documento():
    ext = VlmExtractor(enabled=True)
    doc = MagicMock()
    doc.pages = {1: MagicMock(), 2: MagicMock()}

    def export(page_no=None):
        if page_no == 1:
            raise RuntimeError("doctags corruptos")
        return "página buena"

    doc.export_to_markdown.side_effect = export

    elements = ext._elements_from_document(doc)

    assert len(elements) == 1
    assert elements[0].page == 2
