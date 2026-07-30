"""
Tests de ProcessingState, centrados en la consistencia entre "processed" y
"failed".

Contexto (PLAN.md §6.8): tras la reindexación de Electricidad (run #3, 2026-07-23)
el estado listaba 280 ficheros a la vez en `processed` con `status: success` y en
`failed` con el error 500 de Qdrant de la tirada anterior. `mark_as_processed`
sobrescribía la entrada de `processed` pero nunca borraba la de `failed`, así que
un fallo caducado se quedaba ahí para siempre.
"""

import json

import pytest

from state.processing_state import MAX_RETRIES, ProcessingState


@pytest.fixture
def state(tmp_path):
    """Estado sobre un fichero temporal, sin tocar el volumen real."""
    return ProcessingState(state_file=str(tmp_path / "whoosh" / ".processing_state.json"))


@pytest.fixture
def pdf(tmp_path):
    """Un PDF de mentira: sólo hace falta que exista para el hash MD5."""
    f = tmp_path / "Tema 8. Profibus.pdf"
    f.write_bytes(b"%PDF-1.4 contenido de prueba")
    return str(f)


def test_reproceso_correcto_saca_el_fichero_de_failed(state, pdf):
    """El bug de §6.8: un fichero no puede quedar en processed(success) y failed."""
    state.mark_as_failed(pdf, "Qdrant 500")
    assert pdf in state.state["failed"]

    state.mark_as_processed(pdf, "Electricidad")

    assert state.state["processed"][pdf]["status"] == "success"
    assert pdf not in state.state["failed"], "fallo caducado tras un reproceso correcto"


def test_marcar_procesado_sin_fallo_previo_no_revienta(state, pdf):
    """El camino normal: nunca falló, así que no hay nada que borrar de failed."""
    state.mark_as_processed(pdf, "Electricidad")

    assert state.state["processed"][pdf]["status"] == "success"
    assert state.state["failed"] == {}


def test_la_limpieza_se_persiste_en_disco(state, pdf):
    """Sin persistir, el arranque siguiente recarga el fallo caducado."""
    state.mark_as_failed(pdf, "Qdrant 500")
    state.mark_as_processed(pdf, "Electricidad")

    recargado = json.load(open(state.state_file))

    assert pdf not in recargado["failed"]
    assert recargado["processed"][pdf]["status"] == "success"


def test_stats_no_cuenta_como_fallido_lo_ya_reprocesado(state, pdf):
    """`Fallados previamente: N` en main.py salía inflado por las entradas rancias."""
    state.mark_as_failed(pdf, "Qdrant 500")
    state.mark_as_processed(pdf, "Electricidad")

    stats = state.get_stats()

    assert stats["failed"] == 0
    assert stats["successful"] == 1


def test_reset_failed_pone_a_cero_las_dos_copias_del_contador(state, pdf):
    """
    `reset_failed` escribía el contador de "failed" sobre un dict temporal
    (`.get(path, {})[...] = 0`), así que la copia de "failed" se quedaba alta.
    """
    for _ in range(MAX_RETRIES):
        state.mark_as_failed(pdf, "docling se cuelga")
    assert state.state["failed"][pdf]["retry_count"] == MAX_RETRIES

    state.reset_failed()

    assert state.state["processed"][pdf]["retry_count"] == 0
    assert state.state["failed"][pdf]["retry_count"] == 0


def test_reset_failed_tolera_que_no_haya_entrada_en_failed(state, pdf):
    """Estados heredados pueden tener processed(failed) sin su pareja en failed."""
    state.mark_as_failed(pdf, "error antiguo")
    del state.state["failed"][pdf]

    assert state.reset_failed() == 1
    assert state.state["processed"][pdf]["retry_count"] == 0


def test_un_fichero_reprocesado_deja_de_estar_en_cuarentena(state, pdf):
    """Agotados los intentos se salta el fichero; un reproceso correcto lo libera."""
    for _ in range(MAX_RETRIES):
        state.mark_as_failed(pdf, "docling se cuelga")
    assert state.is_already_processed(pdf) is True  # cuarentena: se salta
    assert pdf in state.get_quarantined()

    state.mark_as_processed(pdf, "Electricidad")

    assert state.get_quarantined() == {}
    assert state.is_already_processed(pdf) is True  # ahora por ser correcto, no por cuarentena
    assert state.state["processed"][pdf]["status"] == "success"
