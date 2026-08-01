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
    return ProcessingState(state_file=str(tmp_path / "state" / ".processing_state.json"))


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


# ============================================================
# §7.3 — negativo rápido por (tamaño, mtime_ns)
# ============================================================
#
# `impact()` marca `is_already_processed` como riesgo ALTO: decide si un fichero
# se reprocesa. Lo que estos tests fijan no es que el atajo sea rápido, sino que
# NUNCA pueda decidir por sí solo que algo no ha cambiado cuando sí lo ha hecho.


def _hash_calls(monkeypatch):
    """Cuenta las veces que se calcula el MD5 real."""
    calls = []
    original = ProcessingState.get_file_hash

    def contador(self, file_path):
        calls.append(file_path)
        return original(self, file_path)

    monkeypatch.setattr(ProcessingState, "get_file_hash", contador)
    return calls


def test_apagado_por_defecto_sigue_hasheando(state, pdf, monkeypatch):
    """Sin FAST_CHANGE_DETECTION el comportamiento es exactamente el de antes."""
    monkeypatch.setattr("state.processing_state.FAST_CHANGE_DETECTION", False)
    state.mark_as_processed(pdf, "Electricidad")
    calls = _hash_calls(monkeypatch)

    assert state.is_already_processed(pdf) is True
    assert calls == [pdf]  # el MD5 se calculó


def test_encendido_evita_el_md5_si_stat_coincide(state, pdf, monkeypatch):
    state.mark_as_processed(pdf, "Electricidad")
    monkeypatch.setattr("state.processing_state.FAST_CHANGE_DETECTION", True)
    calls = _hash_calls(monkeypatch)

    assert state.is_already_processed(pdf) is True
    assert calls == []  # no se hasheó nada


def test_mark_as_processed_guarda_tamano_y_mtime(state, pdf, monkeypatch):
    """Se guardan siempre, esté o no activo el atajo: así encenderlo después no
    obliga a reprocesar el corpus para poblar los metadatos."""
    monkeypatch.setattr("state.processing_state.FAST_CHANGE_DETECTION", False)
    state.mark_as_processed(pdf, "Electricidad")

    info = state.state["processed"][pdf]
    assert info["size"] == len(b"%PDF-1.4 contenido de prueba")
    assert isinstance(info["mtime_ns"], int)
    assert info["hash"]  # el hash sigue siendo la autoridad


def test_entrada_antigua_sin_stat_cae_al_md5(state, pdf, monkeypatch):
    """Estado escrito por una versión anterior: sin `size`/`mtime_ns` el atajo no
    puede opinar y debe pasar el control al hash, no dar por bueno el fichero."""
    state.mark_as_processed(pdf, "Electricidad")
    state.state["processed"][pdf].pop("size")
    state.state["processed"][pdf].pop("mtime_ns")
    monkeypatch.setattr("state.processing_state.FAST_CHANGE_DETECTION", True)
    calls = _hash_calls(monkeypatch)

    assert state.is_already_processed(pdf) is True
    assert calls == [pdf]


def test_contenido_modificado_se_detecta_con_el_atajo_encendido(state, pdf, monkeypatch):
    """El caso que importa: reescribir el fichero cambia tamaño y mtime, el
    atajo no confirma, el MD5 decide y el fichero vuelve a la cola."""
    state.mark_as_processed(pdf, "Electricidad")
    monkeypatch.setattr("state.processing_state.FAST_CHANGE_DETECTION", True)

    from pathlib import Path

    Path(pdf).write_bytes(b"%PDF-1.4 contenido distinto y mas largo")
    # La caché en memoria de hashes es por proceso y guarda el valor anterior.
    from core.cache import _md5_cache

    _md5_cache.clear()

    assert state.is_already_processed(pdf) is False


def test_stat_que_falla_cae_al_md5(state, pdf, monkeypatch):
    """Si `os.stat` revienta, el atajo dice "no lo sé" y no bloquea nada."""
    state.mark_as_processed(pdf, "Electricidad")
    monkeypatch.setattr("state.processing_state.FAST_CHANGE_DETECTION", True)
    monkeypatch.setattr(ProcessingState, "get_file_stat", staticmethod(lambda p: None))
    calls = _hash_calls(monkeypatch)

    assert state.is_already_processed(pdf) is True
    assert calls == [pdf]
