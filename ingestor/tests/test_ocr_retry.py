"""
Tests del reintento por OCR cuando la fragmentación devuelve cero.

El caso que motiva esto (PLAN.md, punto 2): un escaneado cuya única capa de
texto es la cabecera repetida de un descargador pasa el `_is_sufficient` del
pipeline —que cuenta el mobiliario de página como contenido—, así que **no llega
nunca al OCR**; luego `detect_boilerplate` le quita esa cabecera al fragmentar y
el documento se queda en cero fragmentos.

Se arregla en la ruta de error, no en el criterio de aceptación: cambiar
`_is_sufficient` movería la aceptación de los 562 ficheros del corpus y obligaría
a remedir las líneas base. Por eso lo que se prueba aquí es sobre todo **cuándo
NO se reintenta**: el camino feliz tiene que quedar bit a bit como estaba.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunking.token_chunker import Chunk
from extraction.base import Element, ExtractionResult
from extraction.pipeline import ExtractionPipeline


# --- (A) La vía OCR forzada del pipeline ----------------------------------


@pytest.fixture
def pdf(tmp_path) -> Path:
    p = tmp_path / "escaneado.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    return p


def test_extract_document_ocr_valida_paginas_y_marca_procedencia(pdf):
    """Devuelve un ExtractionResult equivalente al de la cadena normal: páginas
    ajustadas al rango real y sin estructura (el OCR no produce ninguna)."""
    elementos = [
        Element(text="una", type="text", page=0, source="easyocr_gpu"),
        Element(text="dos", type="text", page=99, source="easyocr_gpu"),
    ]

    with patch("extraction.ocr_extractor.OCRExtractor") as OCR, patch(
        "extraction.pipeline.get_pdf_total_pages", return_value=3
    ):
        OCR.return_value.extract.return_value = elementos
        OCR.return_value.name = "OCRExtractor (Tesseract)"
        result = ExtractionPipeline().extract_document_ocr(pdf)

    assert [e.page for e in result.elements] == [1, 3]
    assert result.docling_document is None
    assert result.has_structure is False
    # `payload.source` es lo que permite comprobar por contenido que un fichero
    # entró por esta vía y no por la cadena normal (LESSONS.md).
    assert [e.source for e in result.elements] == ["easyocr_gpu", "easyocr_gpu"]


def test_extract_document_ocr_no_toca_la_cadena_de_respaldo(pdf):
    """No debe construir los extractores por defecto: cargar Docling son
    segundos de GPU y aquí ya sabemos que la cadena falló."""
    pipeline = ExtractionPipeline()

    with patch("extraction.ocr_extractor.OCRExtractor") as OCR, patch(
        "extraction.pipeline.get_pdf_total_pages", return_value=1
    ), patch.object(ExtractionPipeline, "_create_default_extractors") as crear:
        OCR.return_value.extract.return_value = []
        OCR.return_value.name = "OCRExtractor (Tesseract)"
        pipeline.extract_document_ocr(pdf)

    crear.assert_not_called()


# --- (B) La decisión de reintentar en index_pdf ---------------------------
#
# `main` inicializa servicios al importarse (estado, embeddings, Qdrant), así que
# se importa una vez y se sustituyen sus globales en cada prueba.


@pytest.fixture(scope="module")
def main_mod():
    import main

    return main


def _chunk(text="contenido", source="easyocr_gpu"):
    return Chunk(text=text, page=1, type="text", source=source)


@pytest.fixture
def entorno(main_mod, tmp_path):
    """Deja index_pdf con todo lo caro sustituido: sólo queda su lógica."""
    pdf = tmp_path / "libro.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    pipeline = MagicMock()
    state = MagicMock()
    state.is_already_processed.return_value = False
    qdrant = MagicMock()
    embeddings = MagicMock()
    embeddings.get_dimension.return_value = 8
    embeddings.encode.return_value = [[0.0] * 8]

    with patch.object(main_mod, "extraction_pipeline", pipeline), patch.object(
        main_mod, "state", state
    ), patch.object(main_mod, "qdrant_service", qdrant), patch.object(
        main_mod, "embedding_service", embeddings
    ), patch.object(
        main_mod, "ensure_qdrant"
    ), patch.object(
        main_mod, "validate_and_fix_vectors", side_effect=lambda v, d: v
    ), patch.object(
        main_mod, "get_pdf_total_pages", return_value=10
    ):
        yield {
            "pdf": str(pdf),
            "pipeline": pipeline,
            "state": state,
            "qdrant": qdrant,
        }


def test_no_reintenta_cuando_hay_chunks(main_mod, entorno):
    """El camino feliz no debe tocar el OCR ni una vez."""
    entorno["pipeline"].extract_document.return_value = ExtractionResult(
        elements=[Element("hola", "text", 1, "docling")], extractor="DoclingExtractor"
    )

    with patch.object(main_mod, "build_chunks", return_value=[_chunk()]):
        assert main_mod.index_pdf("electricidad", entorno["pdf"]) is True

    entorno["pipeline"].extract_document_ocr.assert_not_called()
    entorno["state"].mark_as_processed.assert_called_once()


def test_reintenta_por_ocr_y_recupera_el_documento(main_mod, entorno):
    """Cero fragmentos con una extracción de texto: se reintenta y se indexa."""
    entorno["pipeline"].extract_document.return_value = ExtractionResult(
        elements=[Element("cabecera del descargador", "text", 1, "pypdf")],
        extractor="TextExtractor (pypdf + pdfplumber)",
    )
    entorno["pipeline"].extract_document_ocr.return_value = ExtractionResult(
        elements=[Element("texto real de la página", "text", 1, "easyocr_gpu")],
        extractor="OCRExtractor (Tesseract)",
    )

    with patch.object(main_mod, "build_chunks", side_effect=[[], [_chunk()]]):
        assert main_mod.index_pdf("electricidad", entorno["pdf"]) is True

    entorno["pipeline"].extract_document_ocr.assert_called_once()
    entorno["state"].mark_as_processed.assert_called_once()
    entorno["qdrant"].upsert_vectors.assert_called_once()
    # El payload lleva la procedencia del OCR: es lo que se puede comprobar
    # después contra Qdrant sin fiarse del resumen de la tirada.
    payloads = entorno["qdrant"].upsert_vectors.call_args[0][2]
    assert payloads[0]["source"] == "easyocr_gpu"


def test_no_reintenta_si_la_extraccion_ya_venia_del_ocr(main_mod, entorno):
    """Si el OCR ya corrió y aun así no hubo fragmentos, repetirlo es quemar
    GPU para llegar al mismo sitio."""
    entorno["pipeline"].extract_document.return_value = ExtractionResult(
        elements=[Element("ruido", "text", 1, "easyocr_gpu")],
        extractor="OCRExtractor (EasyOCR GPU + Tesseract fallback)",
    )

    with patch.object(main_mod, "build_chunks", return_value=[]):
        assert main_mod.index_pdf("electricidad", entorno["pdf"]) is False

    entorno["pipeline"].extract_document_ocr.assert_not_called()
    entorno["state"].mark_as_failed.assert_called_once()


def test_si_el_reintento_falla_se_registra_el_fallo_original(main_mod, entorno):
    """El fallo que hay que dejar en el estado es el de los cero fragmentos, no
    el del reintento: es el que describe qué le pasa al fichero."""
    entorno["pipeline"].extract_document.return_value = ExtractionResult(
        elements=[Element("cabecera", "text", 1, "pypdf")], extractor="TextExtractor"
    )
    entorno["pipeline"].extract_document_ocr.side_effect = RuntimeError(
        "CUDA out of memory"
    )

    with patch.object(main_mod, "build_chunks", return_value=[]):
        assert main_mod.index_pdf("electricidad", entorno["pdf"]) is False

    motivo = entorno["state"].mark_as_failed.call_args[0][1]
    assert "No se produjo ningún chunk" in motivo
    assert "CUDA" not in motivo


def test_el_reintento_no_deja_a_medias_el_indice(main_mod, entorno):
    """Un documento que sólo se salva por el reintento se escribe igual que
    cualquier otro: un borrado previo y un único upsert."""
    entorno["pipeline"].extract_document.return_value = ExtractionResult(
        elements=[Element("cabecera", "text", 1, "pypdf")], extractor="TextExtractor"
    )
    entorno["pipeline"].extract_document_ocr.return_value = ExtractionResult(
        elements=[Element("texto", "text", 1, "easyocr_gpu")],
        extractor="OCRExtractor (Tesseract)",
    )

    with patch.object(main_mod, "build_chunks", side_effect=[[], [_chunk()]]):
        main_mod.index_pdf("electricidad", entorno["pdf"])

    entorno["qdrant"].delete_by_file.assert_called_once()
    assert entorno["qdrant"].upsert_vectors.call_count == 1
