"""
Pruebas de la fragmentación por tokens.

Se centran en las garantías que motivaron la Fase 1: ningún fragmento debe
superar el presupuesto de tokens del modelo (antes se truncaba en silencio al
embeber), no deben mezclarse páginas, y hay que suprimir cabeceras/pies
repetidos y duplicados.

Requieren el tokenizador de `thenlper/gte-large` en la caché de modelos.
"""

import pytest

from chunking.token_chunker import (
    MIN_CHUNK_CHARS,
    Chunk,
    TokenCounter,
    _split_by_tokens,
    chunk_elements,
    dedupe_chunks,
    detect_boilerplate,
)
from extraction.base import Element

MODEL = "thenlper/gte-large"
MAX_TOKENS = 512


@pytest.fixture(scope="module")
def counter():
    c = TokenCounter(MODEL)
    if c.tokenizer is None:
        pytest.skip("tokenizador no disponible en la caché de modelos")
    return c


def _element(text, page=1):
    return Element(text=text, type="text", page=page, source="test")


def _long_text(n_sentences=300):
    return " ".join(f"Esta es la frase número {i} del documento." for i in range(n_sentences))


# --------------------------------------------------------------------------
# Presupuesto de tokens
# --------------------------------------------------------------------------


def test_ningun_fragmento_supera_el_presupuesto(counter):
    """La garantía central: nada por encima de 512 llega al embedder."""
    elements = [_element(_long_text() * 3)]
    chunks = chunk_elements(elements, MODEL, max_tokens=MAX_TOKENS)

    assert chunks
    assert all(counter.count(c.text) <= MAX_TOKENS for c in chunks)


def test_la_particion_no_pierde_contenido(counter):
    """Partir un texto largo no debe descartar la mayor parte del texto."""
    text = _long_text()
    pieces = _split_by_tokens(text, counter, MAX_TOKENS)

    original_words = len(text.split())
    kept_words = sum(len(p.split()) for p in pieces)
    assert kept_words >= 0.95 * original_words


def test_texto_sin_puntuacion_tambien_se_parte(counter):
    """Una 'frase' única gigantesca (tablas, texto sin puntos) debe partirse."""
    text = " ".join(f"palabra{i}" for i in range(4000))
    pieces = _split_by_tokens(text, counter, MAX_TOKENS)

    assert len(pieces) > 1
    assert all(counter.count(p) <= MAX_TOKENS for p in pieces)


# --------------------------------------------------------------------------
# Agrupación y páginas
# --------------------------------------------------------------------------


def test_agrupa_elementos_pequenos_de_la_misma_pagina():
    """Muchos fragmentos diminutos de una página deben agruparse."""
    elements = [_element(f"Frase corta número {i} de la misma página.") for i in range(20)]
    chunks = chunk_elements(elements, MODEL, max_tokens=MAX_TOKENS)

    assert len(chunks) < 20


def test_no_mezcla_paginas_distintas():
    """Un fragmento no debe abarcar dos páginas: rompería la cita."""
    elements = [
        _element("Contenido de la página. " + "Texto suficientemente largo. " * 2, page=p)
        for p in range(1, 6)
    ]
    chunks = chunk_elements(elements, MODEL, max_tokens=MAX_TOKENS)

    assert [c.page for c in chunks] == [1, 2, 3, 4, 5]


def test_descarta_fragmentos_por_debajo_del_minimo():
    """Restos de maquetación ('Pág. 7') no merecen un vector propio."""
    assert chunk_elements([_element("Pág. 7")], MODEL, max_tokens=MAX_TOKENS) == []
    assert len("Pág. 7") < MIN_CHUNK_CHARS


# --------------------------------------------------------------------------
# Cabeceras/pies y duplicados
# --------------------------------------------------------------------------


def test_detecta_cabecera_repetida():
    elements = [
        _element(f"Manual de Usuario v3\nContenido propio de la página {p}", page=p)
        for p in range(1, 11)
    ]
    boilerplate = detect_boilerplate(elements)

    assert "manual de usuario v3" in boilerplate
    assert "contenido propio de la página 3" not in boilerplate


def test_documento_corto_no_activa_la_supresion():
    """Con pocas páginas no hay evidencia: suprimir borraría contenido real."""
    elements = [_element("Cabecera\nTexto", page=p) for p in (1, 2)]
    assert detect_boilerplate(elements) == set()


def test_la_cabecera_se_elimina_del_texto_final():
    elements = [
        _element(f"Manual de Usuario v3\nContenido único y suficientemente largo de la página {p}.", page=p)
        for p in range(1, 11)
    ]
    boilerplate = detect_boilerplate(elements)
    chunks = chunk_elements(elements, MODEL, max_tokens=MAX_TOKENS, boilerplate=boilerplate)

    assert chunks
    assert all("Manual de Usuario v3" not in c.text for c in chunks)


def test_crash_state_se_guarda_en_formato_compatible(tmp_path):
    """
    `crash_state.json` debe seguir siendo {archivo: int}.

    Vive en un volumen persistente compartido entre versiones del ingestor. Al
    escribir aquí el formato enriquecido, la versión anterior del código
    reventaba con "'>=' not supported between instances of 'dict' and 'int'" y
    dejaba de indexar el archivo afectado.
    """
    import json

    from extraction.docling_extractor import CrashStateManager

    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("x.pdf", reason="OOM de CUDA")

    raw = json.loads((tmp_path / "crash_state.json").read_text())
    assert raw == {"x.pdf": 1}, "el contador debe serializarse como entero"

    # El motivo se conserva, pero en un fichero aparte que el código antiguo ignora.
    reasons = json.loads((tmp_path / "crash_reasons.json").read_text())
    assert reasons["x.pdf"]["reason"] == "OOM de CUDA"


def test_crash_state_lee_el_formato_enriquecido_antiguo(tmp_path):
    """Debe poder leer lo que escribió la versión intermedia, sin romperse."""
    import json

    from extraction.docling_extractor import CrashStateManager

    (tmp_path / "crash_state.json").write_text(
        json.dumps({"y.pdf": {"count": 2, "reason": "algo", "last": None}})
    )
    mgr = CrashStateManager(tmp_path, max_crashes=3)

    assert mgr._state["y.pdf"] == 2
    assert not mgr.should_skip("y.pdf")  # 2 < 3
    mgr.mark_processing("y.pdf")
    assert mgr.should_skip("y.pdf")  # 3 >= 3


def test_dedup_normaliza_espacios_y_mayusculas():
    chunks = [
        Chunk(text="El mismo contenido exacto aquí presente", page=1),
        Chunk(text="el   MISMO contenido   exacto aquí presente  ", page=5),
        Chunk(text="Un contenido distinto del anterior", page=2),
    ]
    out = dedupe_chunks(chunks)

    assert len(out) == 2
    assert out[0].page == 1  # se conserva la primera aparición
