"""
Tests del contador de caídas de docling: fugas y atribución.

Dos huecos distintos del §6.8, con la misma víctima (`crash_state`):

1. **Fuga del contador.** `mark_processing` incrementa *antes* de convertir y
   `mark_success` limpia al acabar bien — así se detectan las caídas duras que no
   dejan excepción. Pero `extract` tiene salidas por el respaldo PyPDF que no
   pasan por `mark_success`, así que ficheros que se indexaban perfectamente
   acumulaban fallos fantasma hasta el veto. Espécimen real medido el 2026-08-01:
   `Comunicaciones industriales y WinCC ... Marcombo.pdf`, con `status: success`
   en el estado, 231 puntos `pypdf_fallback` en Qdrant, y un 1 en `crash_state`
   con motivo "conversión interrumpida".

2. **Atribución.** Un kill del watchdog, un segfault nativo y un OOM dejan los
   tres el mismo "conversión interrumpida". El watchdog es el único que sabe que
   fue él, y hasta ahora se moría sin decirlo.
"""

import json

import pytest

from extraction.docling_extractor import CrashStateManager


# --- (A) La fuga del contador ---------------------------------------------


def test_mark_fallback_deshace_el_incremento_de_este_intento(tmp_path):
    """El caso WinCC: docling no se cayó, así que no debe quedar rastro de fallo."""
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("wincc.pdf", reason="conversión interrumpida")
    mgr.mark_fallback("wincc.pdf", "validación previa fallida: cifrado")

    assert "wincc.pdf" not in mgr._state
    assert json.loads((tmp_path / "crash_state.json").read_text()) == {}
    # Y el motivo se va con él: no describe a ningún fallo vivo.
    assert "wincc.pdf" not in json.loads(
        (tmp_path / "crash_reasons.json").read_text()
    )


def test_mark_fallback_conserva_las_caidas_reales_anteriores(tmp_path):
    """
    Resta un intento, no limpia el historial.

    Un fichero que tumbó docling dos veces de verdad y a la tercera se va por el
    respaldo sigue teniendo dos caídas reales que contar.
    """
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("mixto.pdf", reason="conversión interrumpida")
    mgr.record_reason("mixto.pdf", "ConversionError: status FAILURE")
    mgr.mark_processing("mixto.pdf", reason="conversión interrumpida")
    mgr.record_reason("mixto.pdf", "ConversionError: status FAILURE")
    assert mgr._state["mixto.pdf"] == 2

    mgr.mark_processing("mixto.pdf", reason="conversión interrumpida")
    mgr.mark_fallback("mixto.pdf", "validación previa fallida: sin páginas")

    assert mgr._state["mixto.pdf"] == 2, "sólo se resta el intento en curso"
    # Y el motivo ya no dice "interrumpida": si lo dijera, `reingest_false_bans.sh`
    # lo rehabilitaría como falso veto del watchdog, que no es lo que pasó.
    assert not mgr.is_interrupted_only("mixto.pdf")


def test_mark_fallback_sobre_un_fichero_limpio_no_hace_nada(tmp_path):
    """No debe inventar entradas ni bajar de cero."""
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_fallback("nunca_visto.pdf", "lo que sea")

    assert mgr._state == {}


def test_conversion_sin_elementos_mantiene_el_veto_pero_corrige_el_motivo(tmp_path):
    """
    La otra salida por PyPDF (docling convierte y devuelve cero) sí merece
    cuarentena, pero no debe hacerse pasar por una interrupción.
    """
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    for _ in range(3):
        mgr.mark_processing("vacio.pdf", reason="conversión interrumpida")
        mgr.record_reason("vacio.pdf", "conversión sin elementos: respaldo PyPDF")

    assert mgr.should_skip("vacio.pdf"), "3/3: el veto se mantiene"
    assert not mgr.is_interrupted_only("vacio.pdf"), "no es un falso veto"


# --- (B) La atribución del watchdog ---------------------------------------


def _marker(tmp_path, context, age=1834.7):
    p = tmp_path / "watchdog_kill.json"
    p.write_text(json.dumps({"context": context, "age_s": age, "timeout_s": 1200}))
    return p


def test_el_rastro_del_watchdog_reescribe_el_motivo_y_se_borra(tmp_path):
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("manual_abb.pdf", reason="conversión interrumpida")
    marker = _marker(tmp_path, "docling_convert_manual_abb.pdf")

    assert mgr.consume_watchdog_marker(str(marker)) == "manual_abb.pdf"

    reason = mgr._reasons["manual_abb.pdf"]["reason"]
    assert "watchdog" in reason and "1834.7" in reason
    assert not marker.exists(), "describe una muerte concreta, no un estado"
    # Sigue siendo un veto por interrupción: lo que cambia es que ahora se sabe
    # quién lo interrumpió. `reingest_false_bans.sh` debe seguir viéndolo.
    assert mgr.is_interrupted_only("manual_abb.pdf")


def test_el_rastro_no_toca_un_fallo_real_de_docling(tmp_path):
    """
    Si el fichero en vuelo ya tenía una excepción registrada, el proceso murió
    después de que docling fallara: la atribución al watchdog sería mentira.
    """
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("roto.pdf", reason="conversión interrumpida")
    mgr.record_reason("roto.pdf", "ConversionError: status FAILURE")
    marker = _marker(tmp_path, "docling_convert_roto.pdf")

    assert mgr.consume_watchdog_marker(str(marker)) is None
    assert mgr._reasons["roto.pdf"]["reason"].startswith("ConversionError")
    assert not marker.exists(), "el rastro se consume igual, aplique o no"


def test_sin_rastro_ni_con_rastro_ilegible_se_rompe_nada(tmp_path):
    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("x.pdf", reason="conversión interrumpida")

    assert mgr.consume_watchdog_marker(str(tmp_path / "no_existe.json")) is None

    roto = tmp_path / "roto.json"
    roto.write_text("{esto no es json")
    assert mgr.consume_watchdog_marker(str(roto)) is None
    assert mgr._state["x.pdf"] == 1, "el estado no se toca si el rastro no sirve"


# --- (C) El lado que escribe: core.heartbeat -------------------------------


def test_write_kill_marker_deja_el_rastro_completo(tmp_path):
    from core.heartbeat import write_kill_marker

    destino = tmp_path / "sub" / "watchdog_kill.json"
    assert write_kill_marker("docling_convert_a.pdf", 1834.72, 1200, str(destino))

    payload = json.loads(destino.read_text())
    assert payload["context"] == "docling_convert_a.pdf"
    assert payload["age_s"] == 1834.7
    assert payload["timeout_s"] == 1200
    assert payload["at"]
    assert not (tmp_path / "sub" / "watchdog_kill.json.tmp").exists()


def test_write_kill_marker_no_propaga_errores(tmp_path):
    """
    Corre a un `os._exit(1)` de distancia: si no puede escribir, el watchdog
    tiene que matar igual. Nunca una excepción.
    """
    from core.heartbeat import write_kill_marker

    ocupado = tmp_path / "fichero"
    ocupado.write_text("soy un fichero, no un directorio")

    assert write_kill_marker("ctx", 1.0, 1200, str(ocupado / "sub" / "m.json")) is False


def test_ida_y_vuelta_del_watchdog_al_motivo(tmp_path):
    """El rastro que escribe el watchdog es el que sabe leer el estado de fallos."""
    from core.heartbeat import write_kill_marker

    mgr = CrashStateManager(tmp_path, max_crashes=3)
    mgr.mark_processing("grande.pdf", reason="conversión interrumpida")

    destino = tmp_path / "watchdog_kill.json"
    write_kill_marker("docling_convert_grande.pdf", 1300.0, 1200, str(destino))

    assert mgr.consume_watchdog_marker(str(destino)) == "grande.pdf"
    assert "watchdog" in mgr._reasons["grande.pdf"]["reason"]
