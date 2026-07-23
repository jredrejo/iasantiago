# Archivo: rag-api/tests/test_eval.py
# Descripción: Tests de la métrica de evaluación (eval.py) — tolerancia ±1
#   página y equivalencia de ficheros duplicados byte a byte.
#
# Construidos a partir de casos REALES documentados en PLAN §3.-1 y §3.2, para
# que un fallo nombre el documento concreto que rompe.
#
# Ejecutar con pytest (`pytest tests/`) o directamente:
#   python tests/test_eval.py

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unicodedata  # noqa: E402

from eval import (  # noqa: E402
    _split_page_ref,
    aggregate_eval,
    build_content_alias_map,
    make_file_matcher,
    make_page_matcher,
    mrr,
    normalize_file,
    normalize_page,
    recall_at_k,
)


def _chunk(file_path, page):
    """Un chunk recuperado, tal y como lo devuelve retrieval."""
    return {"file_path": file_path, "page": page}


# ------------------------------------------------------------------
# _split_page_ref
# ------------------------------------------------------------------


def test_split_page_ref():
    assert _split_page_ref("fichero.pdf#12") == ("fichero.pdf", 12)
    assert _split_page_ref("fichero.pdf#12 ") == ("fichero.pdf", 12)
    # Sin '#': no es referencia de página.
    assert _split_page_ref("fichero.pdf") == ("fichero.pdf", None)
    # Centinela de ground truth malformado (normalize_page pone '#?').
    assert _split_page_ref("fichero.pdf#?") == ("fichero.pdf", None)


# ------------------------------------------------------------------
# Normalización Unicode NFC (el corpus vive en un FS NFD: el payload trae la
# 'ó' descompuesta y un golden escrito a mano la trae precompuesta; son el mismo
# fichero pero distintos byte a byte). Sin esto daba fallo fantasma + aviso
# "no aparece en ningún resultado".
# ------------------------------------------------------------------


def test_normalize_file_folds_nfd_and_nfc():
    nfd = unicodedata.normalize("NFD", "/topics/AFD/nutrición.pdf")  # ó descompuesta
    nfc = unicodedata.normalize("NFC", "nutrición.pdf")  # ó precompuesta
    assert nfd != nfc  # distintos byte a byte de partida
    assert normalize_file(nfd) == nfc  # pero normalizan a lo mismo


def test_normalize_page_folds_nfd_and_nfc():
    nfd = unicodedata.normalize("NFD", "/topics/AFD/nutrición.pdf#142")
    assert normalize_page(nfd) == unicodedata.normalize("NFC", "nutrición.pdf#142")


def test_matcher_folds_nfd_ground_truth_against_nfc_retrieval():
    """El caso real de AFD: golden en una forma, retrieval en la otra."""
    match = make_page_matcher(tolerance=1)
    pred = normalize_page(unicodedata.normalize("NFD", "La-guía-completa.pdf#54"))
    truth = normalize_page(unicodedata.normalize("NFC", "La-guía-completa.pdf#54"))
    assert match(pred, truth)


# ------------------------------------------------------------------
# Tolerancia ±1 página (PLAN §3.2: el pasaje de HPLC se movió 346 -> 345/347
# al re-trocear y la métrica exacta lo puntuaba 0 estando bien recuperado).
# ------------------------------------------------------------------


def test_page_tolerance_matches_adjacent_page():
    match = make_page_matcher(tolerance=1)
    truth = "Análisis instrumental.pdf#346"
    assert match("Análisis instrumental.pdf#345", truth)  # -1
    assert match("Análisis instrumental.pdf#346", truth)  # exacta
    assert match("Análisis instrumental.pdf#347", truth)  # +1
    # ±2 queda fuera: la tolerancia es exactamente 1 (verificado en el corpus).
    assert not match("Análisis instrumental.pdf#344", truth)
    assert not match("Análisis instrumental.pdf#348", truth)


def test_page_tolerance_respects_file_boundary():
    """Misma página, otro fichero: nunca casa por mucha tolerancia que haya."""
    match = make_page_matcher(tolerance=1)
    assert not match("otro.pdf#346", "Análisis instrumental.pdf#346")


def test_page_tolerance_zero_is_exact():
    match = make_page_matcher(tolerance=0)
    assert match("f.pdf#10", "f.pdf#10")
    assert not match("f.pdf#11", "f.pdf#10")


def test_malformed_truth_never_matches_by_number():
    """Un ground truth sin página (#?) no debe casar en falso: se sigue
    denunciando como roto en vez de puntuar como acierto."""
    match = make_page_matcher(tolerance=1)
    assert not match("f.pdf#10", "f.pdf#?")


# ------------------------------------------------------------------
# Equivalencia de duplicados byte-idénticos (PLAN §3.-1: el golden nombra
# `Análisis_instrumental.pdf` pero retrieval devuelve la gemela
# `Análisis instrumental.pdf`).
# ------------------------------------------------------------------

# Los 9 pares de Química documentados en §3.-1 (nombre_golden -> canónico).
_CHEM_ALIASES = {
    "Análisis instrumental.pdf": "Análisis instrumental.pdf",
    "Análisis_instrumental.pdf": "Análisis instrumental.pdf",
}


def test_duplicate_files_are_interchangeable_pages():
    match = make_page_matcher(tolerance=1, aliases=_CHEM_ALIASES)
    # Golden nombra una copia; retrieval devuelve la gemela: debe casar.
    assert match("Análisis_instrumental.pdf#346", "Análisis instrumental.pdf#346")
    # Y sigue combinándose con la tolerancia de página.
    assert match("Análisis_instrumental.pdf#347", "Análisis instrumental.pdf#346")


def test_duplicate_files_are_interchangeable_files():
    match = make_file_matcher(aliases=_CHEM_ALIASES)
    assert match("Análisis_instrumental.pdf", "Análisis instrumental.pdf")
    assert not match("otro_libro.pdf", "Análisis instrumental.pdf")


# ------------------------------------------------------------------
# build_content_alias_map (hash real de ficheros en disco)
# ------------------------------------------------------------------


def test_build_content_alias_map_groups_identical_files():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "Análisis instrumental.pdf")
        b = os.path.join(d, "Análisis_instrumental.pdf")
        c = os.path.join(d, "otro.pdf")
        with open(a, "wb") as fh:
            fh.write(b"contenido identico del mismo libro")
        with open(b, "wb") as fh:
            fh.write(b"contenido identico del mismo libro")
        with open(c, "wb") as fh:
            fh.write(b"un libro distinto")

        aliases = build_content_alias_map(
            {
                "Análisis instrumental.pdf": a,
                "Análisis_instrumental.pdf": b,
                "otro.pdf": c,
            }
        )
        # Las dos copias comparten canónico; el determinismo lo fija min().
        canon = min("Análisis instrumental.pdf", "Análisis_instrumental.pdf")
        assert aliases["Análisis instrumental.pdf"] == canon
        assert aliases["Análisis_instrumental.pdf"] == canon
        # El fichero único no genera alias.
        assert "otro.pdf" not in aliases


def test_build_content_alias_map_ignores_unreadable_files():
    """Un PDF que no está montado no debe tumbar la evaluación."""
    aliases = build_content_alias_map({"fantasma.pdf": "/no/existe/fantasma.pdf"})
    assert aliases == {}


# ------------------------------------------------------------------
# recall_at_k / mrr con matcher
# ------------------------------------------------------------------


def test_recall_and_mrr_use_matcher():
    match = make_page_matcher(tolerance=1)
    pred = ["f.pdf#5", "f.pdf#20"]
    truth = ["f.pdf#21"]  # a distancia 1 del segundo predicho
    assert recall_at_k(pred, truth, 1, match) == 0.0  # no en el primero
    assert recall_at_k(pred, truth, 2, match) == 1.0  # sí en los dos primeros
    assert mrr(pred, truth, match) == 0.5  # posición 2

    # Sin matcher explícito: igualdad exacta de siempre.
    assert recall_at_k(["a", "b"], ["b"], 2) == 1.0
    assert mrr(["a", "b"], ["b"]) == 0.5


# ------------------------------------------------------------------
# aggregate_eval de extremo a extremo
# ------------------------------------------------------------------


def test_aggregate_eval_tolerance_saves_shifted_page():
    """El caso §3.2 completo: golden en #346, contenido re-troceado en #345/#347.
    Con casación exacta (tolerance=0) es un fallo; con ±1 es un acierto."""
    queries = [
        {
            "query": "HPLC",
            "topic": "Chemistry",
            "relevant_pages": ["Análisis instrumental.pdf#346"],
            "retrieved": [
                _chunk("/topics/Chemistry/Análisis instrumental.pdf", 345),
                _chunk("/topics/Chemistry/Análisis instrumental.pdf", 347),
            ],
        }
    ]
    exact = aggregate_eval(queries, page_tolerance=0)
    assert exact["pages"]["Recall@1"] == 0.0  # regresión fantasma

    tol = aggregate_eval(queries, page_tolerance=1)
    assert tol["page_tolerance"] == 1
    assert tol["pages"]["Recall@1"] == 1.0  # recuperado, correctamente puntuado


def test_aggregate_eval_duplicate_aware_end_to_end():
    """Golden nombra una copia, retrieval devuelve SOLO la gemela."""
    aliases = {
        "Análisis instrumental.pdf": "Análisis instrumental.pdf",
        "Análisis_instrumental.pdf": "Análisis instrumental.pdf",
    }
    queries = [
        {
            "query": "HPLC",
            "topic": "Chemistry",
            "relevant_pages": ["Análisis_instrumental.pdf#346"],
            "relevant_files": ["Análisis_instrumental.pdf"],
            "retrieved": [
                _chunk("/topics/Chemistry/Análisis instrumental.pdf", 346),
            ],
        }
    ]
    without = aggregate_eval(queries, page_tolerance=1)
    assert without["pages"]["Recall@1"] == 0.0  # gemela no reconocida

    with_dups = aggregate_eval(queries, page_tolerance=1, file_aliases=aliases)
    assert with_dups["duplicate_groups"] == 1
    assert with_dups["pages"]["Recall@1"] == 1.0
    assert with_dups["files"]["Recall@1"] == 1.0


def test_aggregate_eval_excludes_cases_without_ground_truth():
    queries = [
        {
            "query": "sin verdad",
            "topic": "X",
            "relevant_pages": [],
            "retrieved": [_chunk("/topics/X/a.pdf", 1)],
        }
    ]
    agg = aggregate_eval(queries)
    assert agg["pages"]["n"] == 0  # excluido, no puntuado 0.0


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    for t in TESTS:
        t()
    print(f"OK: {len(TESTS)} tests de eval.py pasan")
