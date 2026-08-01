"""
Tests del detector de muertes duras a nivel de fichero (§6.8).

El hueco que cubre: `crash_state` sólo cuenta las caídas de docling. Una muerte
sin excepción en cualquier otro punto de la ingesta —cadena OCR, VLM,
fragmentación, escritura en Qdrant— no ejecuta ni `mark_as_processed` ni
`mark_as_failed`, así que el estado queda idéntico a antes de empezar. Con
`restart: on-failure` el contenedor vuelve, el escaneo recorre los temas en el
mismo orden, elige el mismo fichero y vuelve a morir: **la ingesta no avanza y
el estado no lo delata**.

Lo que más se prueba aquí es cuándo **no** hay que contar el intento. Contar de
más es la enfermedad cara de este proyecto: ~180 ficheros sanos en cuarentena el
2026-07-22 por la vía del watchdog, y el contador de docling fugándose hasta el
2026-08-01. Un detector de fallos que fabrica fallos es peor que no tenerlo.
"""

import json

import pytest

from state.inflight import InflightTracker, record_hard_death
from state.processing_state import MAX_RETRIES, ProcessingState


@pytest.fixture
def tracker(tmp_path):
    """Detector sobre un fichero temporal, sin tocar el volumen real."""
    return InflightTracker(str(tmp_path / "state" / "inflight.json"))


@pytest.fixture
def state(tmp_path):
    return ProcessingState(state_file=str(tmp_path / "state" / ".processing_state.json"))


# --- La anotación ----------------------------------------------------------


def test_la_anotacion_sobrevive_al_proceso(tracker):
    """Es todo el mecanismo: lo escrito antes de morir es lo que se lee después."""
    tracker.begin("/topics/Electricidad/manual.pdf", context="index_pdf:Electricidad")

    # Otro proceso (el del arranque siguiente) lee el mismo fichero.
    orphan = InflightTracker(str(tracker.path)).consume()

    assert orphan["file"] == "/topics/Electricidad/manual.pdf"
    assert orphan["context"] == "index_pdf:Electricidad"
    assert "started" in orphan


def test_end_borra_la_anotacion_y_es_idempotente(tracker):
    """`end()` corre en un `finally` y en un manejador de señal: no puede reventar."""
    tracker.begin("/topics/FOL/x.pdf")
    tracker.end()
    tracker.end()  # no debe lanzar

    assert tracker.consume() is None


def test_arranque_limpio_no_inventa_nada(tracker):
    assert tracker.consume() is None


def test_consume_destruye_el_rastro(tracker):
    """Describe una muerte concreta, no un estado: se lee una vez."""
    tracker.begin("/topics/Latin/y.pdf")

    assert tracker.consume() is not None
    assert tracker.consume() is None


def test_anotacion_ilegible_se_descarta_sin_contar_nada(tracker):
    """Un fichero a medio escribir no puede paralizar el arranque ni acusar a nadie."""
    tracker.path.parent.mkdir(parents=True, exist_ok=True)
    tracker.path.write_text("{esto no es json")

    assert tracker.consume() is None
    assert not tracker.path.exists()


def test_begin_no_propaga_errores_de_escritura(tmp_path):
    """Es una red, no el trabajo: si no se puede anotar, se indexa igual."""
    imposible = InflightTracker(str(tmp_path / "fichero" / "que" / "no" / "cabe"))
    (tmp_path / "fichero").write_text("soy un fichero, no un directorio")

    imposible.begin("/topics/Dibujo/z.pdf")  # no debe lanzar


# --- La política: cuándo cuenta como intento -------------------------------


def test_muerte_dura_cuenta_como_intento(tracker, state):
    """El caso que arregla esto: sin la anotación, el estado no recordaba nada."""
    pdf = "/topics/Electricidad/escaneado.pdf"
    tracker.begin(pdf, context="index_pdf:Electricidad")

    assert record_hard_death(tracker, state) == pdf
    assert state.get_status(pdf) == "failed"
    assert state.state["processed"][pdf]["retry_count"] == 1
    assert "muerte dura" in state.state["failed"][pdf]["error"]


def test_tres_muertes_duras_ponen_el_fichero_en_cuarentena(tracker, state):
    """
    Es lo que desatasca la ingesta.

    Sin esto, un PDF que tumba el proceso lo tumba en cada reinicio, para siempre.
    La cuarentena existente (`INGESTOR_MAX_RETRIES`) hace el resto en cuanto los
    intentos se contabilizan.
    """
    pdf = "/topics/Electricidad/mata-el-proceso.pdf"

    for _ in range(MAX_RETRIES):
        tracker.begin(pdf)
        record_hard_death(tracker, state)

    assert state.state["processed"][pdf]["retry_count"] == MAX_RETRIES
    # `is_already_processed` devuelve True por cuarentena: el escaneo ya lo salta.
    assert state.is_already_processed(pdf) is True
    assert pdf in state.get_quarantined()


def test_un_fichero_ya_indexado_no_se_marca_como_fallido(tracker, state, tmp_path):
    """
    La ventana entre `mark_as_processed` y `end()`.

    Morir ahí deja anotación sobre un fichero que está perfectamente indexado.
    Contarlo sería fabricar un fallo sobre trabajo bueno — y a los tres, tirar a
    la basura un PDF sano.
    """
    pdf = tmp_path / "bueno.pdf"
    pdf.write_bytes(b"%PDF-1.4 contenido")
    state.mark_as_processed(str(pdf), "Electricidad")

    tracker.begin(str(pdf))

    assert record_hard_death(tracker, state) is None
    assert state.get_status(str(pdf)) == "success"
    assert str(pdf) not in state.state["failed"]


def test_sin_anotacion_no_toca_el_estado(tracker, state):
    """Un arranque limpio no puede mover un solo contador."""
    antes = json.dumps(state.state, sort_keys=True)

    assert record_hard_death(tracker, state) is None
    assert json.dumps(state.state, sort_keys=True) == antes


def test_la_anotacion_se_consume_aunque_no_cuente(tracker, state, tmp_path):
    """Si sobreviviera, el arranque siguiente la volvería a leer y ahí sí contaría."""
    pdf = tmp_path / "bueno.pdf"
    pdf.write_bytes(b"%PDF-1.4 contenido")
    state.mark_as_processed(str(pdf), "Electricidad")
    tracker.begin(str(pdf))

    record_hard_death(tracker, state)

    assert tracker.consume() is None
