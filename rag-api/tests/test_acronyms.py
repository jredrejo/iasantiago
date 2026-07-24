import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from acronyms import expand_acronyms, ACRONYM_EXPANSIONS


def test_rebt_expands():
    out, found = expand_acronyms("dime que es el rebt")
    assert found == ["REBT"]
    assert "Reglamento Electrotécnico para Baja Tensión" in out
    assert out.startswith("dime que es el rebt")  # original preservado


def test_case_insensitive():
    for q in ["REBT", "rebt", "Rebt"]:
        _, found = expand_acronyms(q)
        assert found == ["REBT"], q


def test_no_false_positive_inside_word():
    # 'rebt' embebido en otra palabra no debe expandir
    for q in ["prebtido", "rebtx", "arebt"]:
        out, found = expand_acronyms(q)
        assert found == [], q
        assert out == q


def test_hyphen_code_not_partially_expanded():
    # ITC-BT-44 es un código concreto: no debe disparar la expansión de ITC-BT
    out, found = expand_acronyms("segun la ITC-BT-44")
    assert "ITC-BT" not in found
    assert out == "segun la ITC-BT-44"


def test_itc_bt_alone_expands():
    _, found = expand_acronyms("que dice la ITC-BT")
    assert found == ["ITC-BT"]


def test_multiple_acronyms():
    out, found = expand_acronyms("rebt y cte en una cocina")
    assert "REBT" in found and "CTE" in found
    assert "Reglamento Electrotécnico para Baja Tensión" in out
    assert "Código Técnico de la Edificación" in out


def test_no_acronym_unchanged():
    out, found = expand_acronyms("seccion minima del conductor de proteccion")
    assert found == []
    assert out == "seccion minima del conductor de proteccion"


def test_empty_query():
    assert expand_acronyms("") == ("", [])
    assert expand_acronyms(None) == (None, [])


def test_expansion_added_once():
    # Un acrónimo repetido no debe duplicar la expansión
    out, found = expand_acronyms("rebt rebt")
    assert found == ["REBT"]
    assert out.count("Reglamento Electrotécnico para Baja Tensión") == 1


def test_all_expansions_are_strings():
    for k, v in ACRONYM_EXPANSIONS.items():
        assert k.isupper() or "-" in k
        assert isinstance(v, str) and len(v) > 0
