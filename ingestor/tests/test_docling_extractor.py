"""
Tests de los dos puntos donde docling perdía documentos enteros.

Ambos casos salen de la tirada del 2026-07-30 (PLAN.md §6.8-bis), donde 69
ficheros —los manuales de Omron y KUKA y los apuntes de Química— acabaron en
`pypdf_fallback` teniendo docling perfectamente disponible.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extraction.docling_extractor import DoclingExtractor


@pytest.fixture
def extractor():
    return DoclingExtractor()


# --- (A) PDFs "encriptados" que en realidad se abren solos ----------------
#
# Los manuales de fabricante traen contraseña de propietario (prohibir copiar o
# imprimir) y contraseña de usuario VACÍA. pypdf marca is_encrypted=True y el
# validador los rechazaba en bloque; decrypt("") los abre sin más.


def _reader(encrypted, decrypt_result, pages=10):
    reader = MagicMock()
    reader.is_encrypted = encrypted
    reader.decrypt.return_value = decrypt_result
    page = MagicMock()
    page.mediabox.width = 595.0
    page.mediabox.height = 842.0
    reader.pages = [page] * pages
    return reader


@pytest.mark.parametrize("decrypt_result", [1, 2])
def test_pdf_solo_restringido_por_permisos_es_valido(extractor, decrypt_result):
    """decrypt("") != 0 => el PDF se ha abierto, docling puede con él."""
    with patch("pypdf.PdfReader", return_value=_reader(True, decrypt_result)):
        with patch("builtins.open", MagicMock()):
            ok, motivo = extractor._validate_pdf(Path("/topics/x/manual_omron.pdf"))
    assert ok is True
    assert motivo is None


def test_pdf_con_contrasena_de_usuario_se_rechaza(extractor):
    """decrypt("") == 0 es el único caso que de verdad necesita contraseña."""
    with patch("pypdf.PdfReader", return_value=_reader(True, 0)):
        with patch("builtins.open", MagicMock()):
            ok, motivo = extractor._validate_pdf(Path("/topics/x/protegido.pdf"))
    assert ok is False
    assert "contraseña" in motivo


def test_pdf_sin_cifrar_no_llama_a_decrypt(extractor):
    reader = _reader(False, 0)
    with patch("pypdf.PdfReader", return_value=reader):
        with patch("builtins.open", MagicMock()):
            ok, _ = extractor._validate_pdf(Path("/topics/x/normal.pdf"))
    assert ok is True
    reader.decrypt.assert_not_called()


# --- (B) el respaldo export_to_dict ---------------------------------------
#
# El texto NO cuelga de doc_dict["body"] (ahí sólo está el nodo raíz). Iterar
# ese dict recorre sus CLAVES, así que el respaldo devolvía siempre 0
# elementos: era código muerto que parecía una red de seguridad.


class _FakeDoc:
    """DoclingDocument mínimo con la forma real de export_to_dict()."""

    def __init__(self, doc_dict, pages=0, markdown=""):
        self._dict = doc_dict
        self._markdown = markdown
        self.pages = {n: object() for n in range(1, pages + 1)}

    def export_to_dict(self):
        return self._dict

    def export_to_markdown(self, page_no=None, **kwargs):
        return self._markdown


DOC_DICT = {
    # Así es como docling_core serializa el cuerpo: un nodo, no una lista.
    "body": {"self_ref": "#/body", "children": [], "label": "unspecified"},
    "texts": [
        {
            "label": "paragraph",
            "text": "El controlador NX102 admite hasta cuatro puertos EtherCAT "
            "y se configura desde Sysmac Studio.",
            "prov": [{"page_no": 7}],
        },
        {"label": "paragraph", "text": "corto", "prov": [{"page_no": 8}]},
    ],
    "tables": [
        {
            "label": "table",
            "text": "Modelo | Puertos | Tension\nNX102 | 4 | 24 Vcc\nNX1P2 | 2 | 24 Vcc",
            "prov": [{"page_no": 9}],
        }
    ],
}


def test_export_to_dict_lee_texts_y_tables(extractor):
    elements = extractor._extract_from_document(_FakeDoc(DOC_DICT), Path("/x/a.pdf"))
    assert len(elements) == 2, "debe coger el párrafo largo y la tabla"
    assert {e.page for e in elements} == {7, 9}
    assert {e.type for e in elements} == {"text", "table"}
    assert any("NX102" in e.text for e in elements)


def test_export_to_dict_ignora_el_nodo_body(extractor):
    """Con body poblado pero texts/tables vacíos no debe inventarse nada."""
    doc = _FakeDoc(
        {"body": {"self_ref": "#/body", "children": [{"$ref": "#/texts/0"}]},
         "texts": [], "tables": []}
    )
    assert extractor._extract_from_document(doc, Path("/x/a.pdf")) == []


def test_prov_con_nombre_antiguo_page(extractor):
    doc = _FakeDoc(
        {
            "body": {},
            "texts": [{"label": "paragraph", "text": "x" * 60, "prov": [{"page": 3}]}],
            "tables": [],
        }
    )
    assert extractor._extract_from_document(doc, Path("/x/a.pdf"))[0].page == 3


def test_sin_prov_cae_a_pagina_1(extractor):
    doc = _FakeDoc(
        {"body": {}, "texts": [{"label": "paragraph", "text": "y" * 60}], "tables": []}
    )
    assert extractor._extract_from_document(doc, Path("/x/a.pdf"))[0].page == 1


# --- (C) documento con páginas pero sin 'prov' ----------------------------
#
# export_to_markdown(page_no=N) filtra por provenance: si los items no la
# traen, devuelve vacío para TODAS las páginas. Antes se daba el documento por
# perdido sin probar el documento entero.


def test_paginas_vacias_reintenta_el_documento_completo(extractor):
    texto = "Este párrafo tiene longitud más que suficiente para superar el umbral."
    doc = _FakeDoc({"body": {}, "texts": [], "tables": []}, pages=5, markdown=texto)
    doc.export_to_markdown = lambda page_no=None, **kw: "" if page_no else texto

    elements = extractor._extract_from_document(doc, Path("/x/a.pdf"))
    assert elements, "debe recuperar el texto del documento completo"
    assert texto in elements[0].text
