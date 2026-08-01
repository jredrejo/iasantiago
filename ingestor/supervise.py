"""
Supervisor externo del ingestor (§6.8).

El watchdog de `core.heartbeat` es un **hilo** del proceso que vigila, y ahí está
su límite: una falla nativa (C/C++) que se cuelga reteniendo el GIL lo deja sin
poder ejecutarse. Ocurrió de verdad el 2026-07-22 — docling sobre un PDF de
22 662 páginas entró en `corrupted double-linked list` y giró en el manejador de
`abort()` **seis horas**, con el watchdog vivo pero incapaz de correr una sola
instrucción de Python. El healthcheck de compose sí lo detectaba, pero docker no
reinicia por `unhealthy`: sólo lo pinta.

Esto arregla ese hueco por el sitio correcto: **otro proceso**. Corre como PID 1,
lanza `main.py` como hijo y vigila el mismo fichero de heartbeat desde fuera del
intérprete colgado. Si el heartbeat se pasa de rancio, mata al hijo con SIGKILL
—que no necesita ni GIL ni manejadores— y sale con código ≠ 0 para que
`restart: on-failure` levante el contenedor otra vez.

Tres decisiones que conviene no deshacer:

- **Sólo la biblioteca estándar, y ninguna importación del proyecto.** Un
  supervisor que comparte dependencias con lo que vigila comparte también sus
  formas de romperse. Por eso duplica el puñado de líneas que escriben el rastro
  del kill en vez de importar `core.heartbeat.write_kill_marker`; el formato es
  el mismo y hay un test que lo afirma (`test_supervisor.py`), porque el
  acoplamiento va por el contenido del fichero.
- **Espera `WATCHDOG_TIMEOUT + SUPERVISOR_GRACE_SECONDS`, no `WATCHDOG_TIMEOUT`.**
  El watchdog interno tiene que disparar primero siempre que pueda: es el único
  que sabe *qué* estaba haciendo el proceso y deja la atribución que reescribe el
  motivo en `crash_state`. El supervisor es la red de debajo, para cuando aquél
  ni siquiera puede correr.
- **Borra el heartbeat rancio antes de lanzar al hijo.** `docker restart`
  conserva la capa de escritura del contenedor, así que el heartbeat de la
  ejecución que acaba de morir sigue en /tmp. Sin este borrado, el arranque
  siguiente nace ya "obsoleto" y cualquiera de los dos vigilantes mataría al
  hijo antes de que llegue a dar su primer latido.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

HEARTBEAT_FILE = os.getenv("HEARTBEAT_FILE", "/tmp/ingestor_heartbeat")
WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT", "1200"))
WATCHDOG_CHECK_INTERVAL = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60"))
SUPERVISOR_GRACE_SECONDS = int(os.getenv("SUPERVISOR_GRACE_SECONDS", "300"))
STATE_BASE_DIR = os.getenv("STATE_BASE_DIR", "/state")
WATCHDOG_KILL_MARKER = os.getenv(
    "WATCHDOG_KILL_MARKER", os.path.join(STATE_BASE_DIR, "watchdog_kill.json")
)

# Umbral propio: por debajo de esto manda el watchdog interno.
KILL_AFTER = WATCHDOG_TIMEOUT + SUPERVISOR_GRACE_SECONDS


def log(message: str) -> None:
    """Traza por stderr, sin logging: el supervisor no configura nada del proyecto."""
    print(f"[SUPERVISOR] {message}", file=sys.stderr, flush=True)


def read_heartbeat(path=None):
    """
    Lee (edad_en_segundos, contexto) del heartbeat, o None si no hay nada legible.

    None significa "no vigilar": ni el arranque (aún no hay fichero) ni un
    heartbeat a medio escribir pueden costarle la vida al hijo.
    """
    path = path or HEARTBEAT_FILE
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if not lines:
            return None
        age = time.time() - float(lines[0].strip())
        context = lines[1].strip() if len(lines) > 1 else "desconocido"
        return age, context
    except (OSError, ValueError, IndexError):
        return None


def write_kill_marker(context: str, age: float, path=None) -> bool:
    """
    Deja el rastro del kill, en el mismo formato que `core.heartbeat`.

    Lo consume `CrashStateManager.consume_watchdog_marker()` en el arranque
    siguiente para reatribuir el motivo del fichero que estaba en vuelo. El campo
    `by` distingue quién mató, que es justo lo que este supervisor añade: si el
    rastro dice "supervisor", el watchdog interno no pudo ni ejecutarse.
    """
    path = path or WATCHDOG_KILL_MARKER
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(
                {
                    "context": context,
                    "age_s": round(age, 1),
                    "timeout_s": KILL_AFTER,
                    "at": datetime.now().isoformat(),
                    "by": "supervisor",
                },
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def clear_stale_heartbeat(path=None) -> None:
    """Borra el heartbeat de la ejecución anterior (`docker restart` lo conserva)."""
    path = path or HEARTBEAT_FILE
    try:
        os.unlink(path)
        log(f"heartbeat rancio borrado: {path}")
    except FileNotFoundError:
        pass
    except OSError as e:
        log(f"no se pudo borrar el heartbeat rancio: {e}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    clear_stale_heartbeat()

    child = subprocess.Popen([sys.executable, "main.py", *argv])
    log(
        f"hijo {child.pid} lanzado (main.py {' '.join(argv) or '<escaneo>'}); "
        f"mato si el heartbeat pasa de {KILL_AFTER}s "
        f"({WATCHDOG_TIMEOUT}s de watchdog + {SUPERVISOR_GRACE_SECONDS}s de margen)"
    )

    def forward(signum, _frame):
        """Una parada pedida es del hijo: que la gestione él y salga como quiera."""
        log(f"{signal.Signals(signum).name} recibida, reenviando al hijo")
        try:
            child.send_signal(signum)
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward)

    while True:
        try:
            code = child.wait(timeout=WATCHDOG_CHECK_INTERVAL)
            # `Popen.wait` devuelve el número de señal en negativo si el hijo
            # murió por una; traducirlo a la convención del shell evita que un
            # SIGSEGV del hijo (−11) salga del contenedor como un código raro.
            return code if code >= 0 else 128 - code
        except subprocess.TimeoutExpired:
            pass

        beat = read_heartbeat()
        if beat is None:
            continue

        age, context = beat
        if age <= KILL_AFTER:
            continue

        log(
            f"heartbeat obsoleto por {age:.0f}s (límite {KILL_AFTER}s). "
            f"Último contexto: {context}"
        )
        log(
            "el watchdog interno no ha podido actuar: cuelgue nativo con el GIL "
            "retenido. Matando el proceso desde fuera."
        )
        if write_kill_marker(context, age):
            log(f"rastro dejado en {WATCHDOG_KILL_MARKER}")

        child.kill()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # SIGKILL no admite negociación: si el hijo sigue ahí es que está en
            # espera ininterrumpible del kernel (típicamente el driver de GPU) y
            # sólo lo suelta un reinicio del contenedor. Salir es provocarlo.
            log("el hijo no muere ni con SIGKILL; salgo igual para forzar reinicio")
        return 1


if __name__ == "__main__":
    sys.exit(main())
