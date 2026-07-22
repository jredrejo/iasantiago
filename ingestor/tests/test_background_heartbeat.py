"""
Tests del BackgroundHeartbeat, en particular el max_duration añadido el
2026-07-22.

Contexto: `docling convert()` es una llamada bloqueante que no tick-eaba el
heartbeat; una conversión sana de un manual grande tarda >WATCHDOG_TIMEOUT y el
watchdog la mataba, cargándole un crash (3 -> cuarentena). Se envolvió en
BackgroundHeartbeat para mantener el heartbeat vivo. Pero un cuelgue REAL no
debe quedar vivo para siempre: max_duration deja de tick-ear pasado un tope,
para que el watchdog acabe matándolo.
"""

import time

import core.heartbeat as hb
from core.heartbeat import BackgroundHeartbeat


def _count_ticks(monkeypatch):
    """Cuenta llamadas a update_heartbeat mientras el bloque corre."""
    n = {"c": 0}
    real = hb.update_heartbeat
    monkeypatch.setattr(hb, "update_heartbeat", lambda ctx: n.__setitem__("c", n["c"] + 1))
    monkeypatch.setattr(hb, "call_heartbeat", lambda ctx: None)
    return n, real


def test_ticks_while_block_runs(monkeypatch):
    n, _ = _count_ticks(monkeypatch)
    with BackgroundHeartbeat("op-larga", interval=0.05):
        time.sleep(0.35)
    # ~6-7 ticks en 0.35s con intervalo 0.05; al menos varios
    assert n["c"] >= 3


def test_stops_ticking_after_max_duration(monkeypatch):
    """
    El corazón de la corrección: pasado max_duration deja de tick-ear, aunque
    el bloque siga corriendo, para que el heartbeat quede obsoleto y el watchdog
    mate un cuelgue real.
    """
    n, _ = _count_ticks(monkeypatch)
    with BackgroundHeartbeat("cuelgue-real", interval=0.05, max_duration=0.2):
        time.sleep(0.6)
        ticks_al_parar = n["c"]
        time.sleep(0.4)  # durante esta ventana NO debe tick-ear más
        ticks_despues = n["c"]
    # Tras superar max_duration (0.2s) el conteo se congela.
    assert ticks_despues == ticks_al_parar
    # Y tick-eó algo antes de rendirse.
    assert ticks_al_parar >= 2


def test_no_max_duration_keeps_ticking(monkeypatch):
    """Sin max_duration (None), comportamiento antiguo: tick-ea indefinidamente."""
    n, _ = _count_ticks(monkeypatch)
    with BackgroundHeartbeat("op-sin-tope", interval=0.05):
        time.sleep(0.3)
        a = n["c"]
        time.sleep(0.3)
        b = n["c"]
    assert b > a  # siguió tick-eando


def test_exit_stops_thread_promptly(monkeypatch):
    _count_ticks(monkeypatch)
    bg = BackgroundHeartbeat("op", interval=0.05)
    bg.__enter__()
    bg.__exit__(None, None, None)
    assert bg._thread is not None and not bg._thread.is_alive()
