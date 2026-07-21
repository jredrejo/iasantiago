"""
Tests de la reparación de texto mal decodificado.

Las cadenas de entrada son literales sacados de los índices en vivo el
2026-07-21, no inventadas: si una correspondencia cambia, estos tests dicen
qué documento real se rompe.
"""

import pytest

from chunking.text_repair import repair_chunks, repair_text


class _FakeChunk:
    """Mínimo para repair_chunks: sólo necesita `.text`."""

    def __init__(self, text):
        self.text = text


# --- (A) GLYPH<NNN> ------------------------------------------------------


@pytest.mark.parametrize(
    "corrupt,expected",
    [
        ("MaGLYPH<237>z", "Maíz"),
        ("CGLYPH<243>digo Alimentario", "Código Alimentario"),
        ("ComposiciGLYPH<243>n", "Composición"),
        ("agitaciGLYPH<243>n hasta ebulliciGLYPH<243>n", "agitación hasta ebullición"),
        ("Cloruro sGLYPH<243>dico", "Cloruro sódico"),
        ("subenriquecimiento a 30 GLYPH<176>C", "subenriquecimiento a 30 °C"),
        ("GLYPH<211>xido", "Óxido"),
    ],
)
def test_glyph_decimal(corrupt, expected):
    assert repair_text(corrupt)[0] == expected


def test_c1_range_uses_cp1252_not_chr():
    """
    128-159 son controles C1 en Unicode. El productor del PDF usaba cp1252,
    donde 151 es la raya y 149 el bolo. Con chr() saldrían invisibles.
    1562 casos de GLYPH<151> en Chemistry.
    """
    assert repair_text("Muestras GLYPH<151> Preparación")[0] == "Muestras — Preparación"
    assert repair_text("GLYPH<149> Medio de cultivo")[0] == "• Medio de cultivo"


def test_control_codepoints_are_dropped():
    """DEL (127) no aporta nada; no debe dejarse ni convertirse en basura."""
    assert repair_text("tablaGLYPH<127>final")[0] == "tablafinal"


def test_out_of_range_codepoint_is_left_alone():
    """Si no sabemos decodificarlo, es mejor no tocarlo que emitir basura."""
    original = "raro GLYPH<9999999>"
    assert repair_text(original)[0] == original


# --- (B) Adobe StandardEncoding ------------------------------------------


@pytest.mark.parametrize(
    "corrupt,expected",
    [
        ("mantendrÆ", "mantendrá"),
        ("anÆlisis microbiolGLYPH<243>gico", "análisis microbiológico"),
        ("CrustÆceos y moluscos", "Crustáceos y moluscos"),
        ("gØrmenes", "gérmenes"),
        ("MØtodo del Nœmero MÆs Probable", "Método del Número Más Probable"),
        ("Espaæol", "Español"),
        ("Baæo MarGLYPH<237>a", "Baño María"),
        ("RODRŒGUEZ", "RODRÍGUEZ"),
        ("Œndices de refracciGLYPH<243>n", "Índices de refracción"),
        ("cÆrnicos", "cárnicos"),
        ("Fosfato dipotÆsico", "Fosfato dipotásico"),
    ],
)
def test_standard_encoding(corrupt, expected):
    assert repair_text(corrupt)[0] == expected


# --- Falsos positivos: lo que NO se debe tocar ---------------------------


def test_diameter_notation_is_preserved():
    """
    'Ø 90 mm' es notación de diámetro legítima (Dibujo y catálogos de
    Chemistry). De 3090 coincidencias en rag_dibujo, 3020 eran mojibake y
    exactamente 1 era un diámetro: la guarda de adyacencia a letra las separa.
    """
    for legit in ("Placa Preparada (Ø 90 mm)", "Ø 20", "diámetro Ø 12,5 mm"):
        assert repair_text(legit)[0] == legit


def test_danish_and_french_names_are_preserved():
    """
    'ø' y 'Ł' se dejaron fuera de la tabla a propósito: en el corpus aparecen
    sobre todo en nombres propios legítimos.
    """
    for legit in (
        "teoría de Brønsted-Lowry",
        "ácidos y bases de Brønsted",
        "BibliothŁque Nationale",
    ):
        assert repair_text(legit)[0] == legit


def test_clean_spanish_is_untouched():
    clean = "La valoración coulombimétrica de cloruro en fluidos biológicos año ñu"
    assert repair_text(clean)[0] == clean


def test_empty_and_none_safe():
    assert repair_text("")[0] == ""


# --- (C) irreparable: se cuenta pero no se toca --------------------------


def test_subset_font_is_reported_not_mangled():
    """
    Fuente subconjunto sin ToUnicode: la identidad del carácter no está en el
    PDF. Debe contarse para avisar, nunca 'arreglarse' inventando texto.
    """
    corrupt = "GLYPH<c=1,font=/AAPDCK+LiberationSans-Bold>GLYPH<c=4,font=/AAPDCK+X>"
    fixed, counts = repair_text(corrupt)
    assert fixed == corrupt
    assert counts["unrecoverable"] == 2
    assert counts["glyph_decimal"] == 0


def test_subset_font_not_confused_with_decimal_form():
    """El patrón GLYPH<c=N,font=...> no debe activar la vía de (A)."""
    _, counts = repair_text("GLYPH<c=237,font=/ABCDEF+Foo>")
    assert counts["glyph_decimal"] == 0
    assert counts["unrecoverable"] == 1


# --- repair_chunks -------------------------------------------------------


def test_repair_chunks_mutates_in_place_and_counts():
    chunks = [_FakeChunk("MaGLYPH<237>z"), _FakeChunk("Espaæol"), _FakeChunk("limpio")]
    out = repair_chunks(chunks)
    assert [c.text for c in out] == ["Maíz", "Español", "limpio"]


def test_repair_runs_before_dedup_so_twins_collapse():
    """
    Razón de aplicar la reparación ANTES de deduplicar: dos fragmentos que
    sólo diferían en la corrupción son el mismo texto una vez reparados.
    """
    from chunking.token_chunker import Chunk, dedupe_chunks

    chunks = [Chunk(text="MaGLYPH<237>z", page=1), Chunk(text="Maíz", page=2)]
    assert len(dedupe_chunks(repair_chunks(chunks))) == 1
    # Y al revés no colapsan: por eso el orden importa.
    chunks2 = [Chunk(text="MaGLYPH<237>z", page=1), Chunk(text="Maíz", page=2)]
    assert len(repair_chunks(dedupe_chunks(chunks2))) == 2
