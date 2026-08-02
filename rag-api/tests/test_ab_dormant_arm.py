# Archivo: rag-api/tests/test_ab_dormant_arm.py
# Descripción: el comparador del §7.1 distingue "todavía no" de "ya nunca".
#
# El cabo del 2026-08-02: `ab_retrieve_vs_chat.py` imprimía "bloqueado hasta que
# haya tráfico real" y salía con 1 mientras la ruta topic:X llevaba dos días sin
# una sola fila. El tráfico se había mudado al Filter cuando entraron los 18
# modelos de workspace —pudiendo seguir usando topic:X, que sigue anunciado en
# GET /v1/models—, así que el consejo era falso: esperar no iba a traer esa
# muestra nunca.
#
# Un criterio de parada que no separa esos dos casos no es un criterio de
# parada, y es lo que estos tests fijan.

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ab_retrieve_vs_chat as ab  # noqa: E402


def _ts(dias_atras: float) -> int:
    """Epoch en milisegundos, que es lo que escribe `telemetry_log`."""
    t = datetime.now(timezone.utc) - timedelta(days=dias_atras)
    return int(t.timestamp() * 1000)


def _fila(source: str, dias_atras: float, **extra) -> dict:
    fila = {
        "ts": _ts(dias_atras),
        "topic": "Electricidad",
        "query": "ley de Ohm",
        "mode": "hybrid",
        "retrieved": [{"file_path": "a.pdf", "page": 1}],
        "generative": False,
    }
    if source:
        fila["source"] = source
    fila.update(extra)
    return fila


def _resumen(filas):
    return {
        arm: ab.summarize([f for f in filas if ab.arm_of(f) == arm])
        for arm in (ab.CHAT, ab.RETRIEVE)
    }


# ------------------------------------------------------- detección de la rama


def test_una_rama_parada_hace_dias_sale_como_inactiva():
    filas = [_fila("chat", 30)] + [_fila("retrieve", d) for d in (5, 3, 0)]
    dormant = ab._dormant_arms(_resumen(filas), stale_days=7)
    assert [d[0] for d in dormant] == [ab.CHAT]


def test_las_dos_ramas_al_dia_no_dan_ninguna_inactiva():
    filas = [_fila("chat", 1), _fila("retrieve", 0)]
    assert ab._dormant_arms(_resumen(filas), stale_days=7) == []


def test_el_retraso_se_mide_contra_la_muestra_no_contra_hoy():
    """Un corpus histórico entero no vuelve inactivas a las dos ramas.

    Si se midiera contra `now`, analizar telemetría vieja marcaría las dos como
    muertas y el criterio dejaría de servir para leer el pasado.
    """
    filas = [_fila("chat", 400), _fila("retrieve", 401)]
    assert ab._dormant_arms(_resumen(filas), stale_days=7) == []


def test_el_umbral_de_dias_es_configurable():
    filas = [_fila("chat", 10), _fila("retrieve", 0)]
    assert ab._dormant_arms(_resumen(filas), stale_days=30) == []
    assert ab._dormant_arms(_resumen(filas), stale_days=3) != []


def test_una_rama_vacia_no_se_marca_inactiva():
    """Sin filas no hay fecha: es "no ha empezado", no "se ha parado"."""
    filas = [_fila("retrieve", 0)]
    assert ab._dormant_arms(_resumen(filas), stale_days=7) == []


# ------------------------------------------------------------ código de salida


def _corre(filas, tmp_path, *args) -> tuple:
    p = tmp_path / "retrieval.jsonl"
    p.write_text("\n".join(json.dumps(f) for f in filas), encoding="utf-8")
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["ab", str(p), *args]
    try:
        with redirect_stdout(buf):
            code = ab.main()
    finally:
        sys.argv = argv
    return code, buf.getvalue()


def test_rama_inactiva_sale_con_3_no_con_1(tmp_path):
    """3 es "esto no se arregla esperando"; 1 es "sigue acumulando"."""
    filas = [_fila("chat", 30)] + [_fila("retrieve", d) for d in (2, 1, 0)]
    code, _ = _corre(filas, tmp_path, "--min-per-arm", "200")
    assert code == 3


def test_muestra_corta_con_las_dos_vivas_sigue_saliendo_con_1(tmp_path):
    filas = [_fila("chat", 1), _fila("retrieve", 0)]
    code, _ = _corre(filas, tmp_path, "--min-per-arm", "200")
    assert code == 1


def test_muestra_suficiente_sale_con_0(tmp_path):
    filas = [_fila("chat", 0) for _ in range(3)] + [
        _fila("retrieve", 0) for _ in range(3)
    ]
    code, _ = _corre(filas, tmp_path, "--min-per-arm", "3")
    assert code == 0


def test_el_informe_no_aconseja_esperar_cuando_una_rama_esta_inactiva(tmp_path):
    """Era exactamente el consejo falso que motivó el arreglo."""
    filas = [_fila("chat", 30)] + [_fila("retrieve", d) for d in (2, 1, 0)]
    _, salida = _corre(filas, tmp_path, "--min-per-arm", "200")
    assert "RAMA INACTIVA" in salida
    assert "hasta que haya tráfico real" not in salida


# --------------------------------------------------------- `generative` ausente


def test_una_rama_sin_generative_se_avisa_aparte_del_contador(tmp_path):
    """Las filas viejas no lo llevan, y sin él la pregunta del §7.1 no se mide.

    El contador de filas no lo puede ver: una rama puede tener 200 filas y cero
    utilizables para la comparación que el §7.1 pone a prueba.
    """
    filas = [_fila("chat", 0, generative=None) for _ in range(3)]
    for f in filas:
        del f["generative"]
    filas += [_fila("retrieve", 0) for _ in range(3)]
    _, salida = _corre(filas, tmp_path, "--min-per-arm", "3")
    assert "SIN `generative`" in salida


def test_con_las_dos_ramas_marcadas_no_se_avisa(tmp_path):
    filas = [_fila("chat", 0), _fila("retrieve", 0)]
    _, salida = _corre(filas, tmp_path, "--min-per-arm", "1")
    assert "SIN `generative`" not in salida
