"""
Tests del supervisor externo (§6.8).

El hueco que cubre es el que ninguna capa de Python puede cubrir: el 2026-07-22
docling entró en `corrupted double-linked list` sobre un PDF de 22 662 páginas y
giró **seis horas** en el manejador de `abort()`, con el GIL retenido. El
watchdog interno estaba vivo y no pudo ejecutar una instrucción. El healthcheck
de compose lo veía, pero docker no reinicia por `unhealthy`: sólo lo pinta.

Aquí se prueban tres propiedades, y las tres son de las que no protege ningún
tipo:

1. Que mate de verdad a un hijo colgado, desde fuera del intérprete.
2. Que **no** mate a nadie más (arranque sin heartbeat, hijo sano, heartbeat
   ilegible): un vigilante con gatillo fácil es la fábrica de vetos falsos que
   este proyecto ya ha pagado dos veces.
3. Que el rastro que deja lo entienda `CrashStateManager`. El acoplamiento entre
   los dos va por el contenido de un fichero JSON escrito por duplicado, y
   `LESSONS.md` es explícito: cuando el acoplamiento va por contenido, hay que
   afirmar la propiedad.
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

import supervise
from extraction.docling_extractor import CrashStateManager


@pytest.fixture
def hb(tmp_path, monkeypatch):
    """Heartbeat y rastro en temporales, y relojes cortos para no dormir el test."""
    heartbeat = tmp_path / "heartbeat"
    marker = tmp_path / "state" / "watchdog_kill.json"
    monkeypatch.setattr(supervise, "HEARTBEAT_FILE", str(heartbeat))
    monkeypatch.setattr(supervise, "WATCHDOG_KILL_MARKER", str(marker))
    monkeypatch.setattr(supervise, "WATCHDOG_CHECK_INTERVAL", 1)
    monkeypatch.setattr(supervise, "KILL_AFTER", 5)
    return heartbeat, marker


def escribe_heartbeat(path, edad_s: float, contexto: str) -> None:
    path.write_text(f"{time.time() - edad_s}\n{contexto}\n")


# --- La capa: el supervisor va DEBAJO del watchdog interno -----------------


def test_el_supervisor_espera_mas_que_el_watchdog_interno():
    """
    El orden importa: el watchdog interno es el único que sabe atribuir la muerte
    (deja el rastro que reescribe el motivo en `crash_state`). El supervisor sólo
    debe entrar cuando aquél no ha podido ni ejecutarse.
    """
    assert supervise.KILL_AFTER > supervise.WATCHDOG_TIMEOUT
    assert supervise.SUPERVISOR_GRACE_SECONDS > 0


# --- Lectura del heartbeat -------------------------------------------------


def test_lee_edad_y_contexto(hb):
    heartbeat, _ = hb
    escribe_heartbeat(heartbeat, 120, "docling_convert_manual.pdf")

    edad, contexto = supervise.read_heartbeat()

    assert 119 <= edad <= 125
    assert contexto == "docling_convert_manual.pdf"


@pytest.mark.parametrize(
    "contenido",
    [None, "", "no-es-un-timestamp\ncontexto\n"],
    ids=["sin-fichero", "vacio", "ilegible"],
)
def test_sin_heartbeat_legible_no_se_vigila(hb, contenido):
    """None significa "no mates": el arranque y una escritura a medias caen aquí."""
    heartbeat, _ = hb
    if contenido is not None:
        heartbeat.write_text(contenido)

    assert supervise.read_heartbeat() is None


def test_borra_el_heartbeat_rancio_al_arrancar(hb):
    """
    `docker restart` conserva la capa de escritura, así que /tmp trae el heartbeat
    del proceso que acaba de morir. Sin este borrado, el arranque siguiente nace
    obsoleto y los dos vigilantes matarían al hijo antes de su primer latido.
    """
    heartbeat, _ = hb
    escribe_heartbeat(heartbeat, 99999, "de la ejecución muerta")

    supervise.clear_stale_heartbeat()

    assert not heartbeat.exists()


# --- El rastro que deja, y quién lo lee ------------------------------------


def test_el_rastro_lo_entiende_el_consumidor_de_docling(hb, tmp_path):
    """
    La prueba del acoplamiento por contenido: el supervisor escribe el rastro por
    duplicado (no importa `core.heartbeat`, a propósito), y quien lo lee es
    `CrashStateManager`. Si los formatos se separan, el motivo del fichero se
    queda en "conversión interrumpida" y nadie se entera de quién mató.
    """
    _, marker = hb
    supervise.write_kill_marker("docling_convert_manual.pdf", age=1700.0)

    crash_dir = tmp_path / "docling_cache"
    crash_dir.mkdir()
    mgr = CrashStateManager(crash_dir, max_crashes=3)
    mgr.mark_processing("manual.pdf", reason="conversión interrumpida")

    assert mgr.consume_watchdog_marker(str(marker)) == "manual.pdf"
    motivo = mgr._reasons["manual.pdf"]["reason"]
    assert "watchdog" in motivo
    # El prefijo es interfaz: `is_interrupted_only` decide por substring, y de él
    # depende que `reset-docling-crashes --interrumpidos` rehabilite el fichero.
    assert mgr.is_interrupted_only("manual.pdf")
    # Y el rastro se consume: describe una muerte, no un estado.
    assert not marker.exists()


def test_el_rastro_dice_quien_mato(hb):
    """
    Distinguir supervisor de watchdog interno no es cosmético: un rastro con
    `by: supervisor` significa que el intérprete estaba tan colgado que no pudo
    ni matarse a sí mismo, que es el diagnóstico del §6.8.
    """
    _, marker = hb
    supervise.write_kill_marker("ocr_pagina_412", age=1600.0)

    datos = json.loads(marker.read_text())
    assert datos["by"] == "supervisor"
    assert datos["context"] == "ocr_pagina_412"
    assert datos["age_s"] == 1600.0


# --- El comportamiento completo: lanzar, vigilar, matar --------------------


@pytest.fixture
def falso_main(tmp_path, monkeypatch):
    """
    Un `main.py` de mentira en el cwd, para ejercitar `main()` de verdad.

    Se prefiere esto a inyectar el comando del hijo: así se prueba también que el
    supervisor lanza lo que dice lanzar.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path / "main.py"


def test_mata_al_hijo_colgado_y_sale_para_forzar_reinicio(hb, falso_main):
    """El caso del §6.8: el hijo deja de latir y no sale por sí mismo."""
    heartbeat, marker = hb
    falso_main.write_text(
        "import sys, time\n"
        f"open({str(heartbeat)!r}, 'w').write(f'{{time.time() - 9999}}\\ncolgado\\n')\n"
        "time.sleep(120)\n"
    )

    inicio = time.time()
    codigo = supervise.main([])

    assert codigo == 1, "tiene que salir ≠ 0 para que restart: on-failure actúe"
    assert time.time() - inicio < 30, "no debe esperar a que el hijo termine"
    assert json.loads(marker.read_text())["context"] == "colgado"


def test_un_hijo_sano_sale_por_su_cuenta_con_su_codigo(hb, falso_main):
    """El supervisor es transparente para todo lo que no esté colgado."""
    heartbeat, marker = hb
    falso_main.write_text(
        "import sys, time\n"
        f"open({str(heartbeat)!r}, 'w').write(f'{{time.time()}}\\nsano\\n')\n"
        "time.sleep(2)\n"
        "sys.exit(7)\n"
    )

    assert supervise.main([]) == 7
    assert not marker.exists(), "un final normal no deja rastro de kill"


def test_no_mata_mientras_no_haya_heartbeat(hb, falso_main):
    """
    Cargar modelos tarda minutos y no late. Un supervisor que matara por ausencia
    de heartbeat haría imposible arrancar.
    """
    _, marker = hb
    falso_main.write_text("import sys, time\ntime.sleep(3)\nsys.exit(0)\n")

    assert supervise.main([]) == 0
    assert not marker.exists()


def test_docker_stop_llega_hasta_el_hijo(tmp_path):
    """
    `docker stop` manda SIGTERM al PID 1, que ahora es el supervisor.

    Si no lo reenviara, el hijo moriría a los 10 s por el SIGKILL de docker sin
    ejecutar su manejador — y el manejador es justo el que borra la anotación en
    vuelo. Cada parada ordenada le costaría un intento a un PDF sano, que es la
    fábrica de cuarentenas falsas de siempre. Se lanza de verdad como proceso
    aparte porque la propiedad que se afirma es entre procesos.
    """
    listo = tmp_path / "listo"
    recibido = tmp_path / "sigterm-recibido"
    (tmp_path / "main.py").write_text(
        "import signal, sys, time\n"
        "def adios(signum, frame):\n"
        f"    open({str(recibido)!r}, 'w').write(str(signum))\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, adios)\n"
        f"open({str(listo)!r}, 'w').write('ok')\n"
        "time.sleep(60)\n"
    )

    entorno = {
        **os.environ,
        "HEARTBEAT_FILE": str(tmp_path / "heartbeat"),
        "STATE_BASE_DIR": str(tmp_path / "state"),
    }
    supervisor = subprocess.Popen(
        [sys.executable, supervise.__file__], cwd=tmp_path, env=entorno
    )
    try:
        limite = time.time() + 15
        while not listo.exists() and time.time() < limite:
            time.sleep(0.1)
        assert listo.exists(), "el hijo no llegó a arrancar"

        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=30) == 0
    finally:
        if supervisor.poll() is None:
            supervisor.kill()

    assert recibido.exists(), "el SIGTERM no llegó al hijo"
    assert recibido.read_text() == str(int(signal.SIGTERM))


def test_pasa_los_argumentos_al_hijo(hb, falso_main, tmp_path):
    """`docker compose run ingestor retry-failed` tiene que seguir funcionando."""
    salida = tmp_path / "argv.txt"
    falso_main.write_text(
        "import sys\n" f"open({str(salida)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
    )

    assert supervise.main(["retry-failed", "--dry-run"]) == 0
    assert salida.read_text() == "retry-failed --dry-run"
