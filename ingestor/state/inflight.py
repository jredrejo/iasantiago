"""
Detector de muertes duras a nivel de fichero.

`CrashStateManager` (docling) cuenta las caídas de **un** extractor. El resto de
la ingesta no tenía equivalente: si el proceso muere sin excepción —segfault de
una biblioteca nativa en la cadena OCR, OOM al fragmentar, kill del watchdog
mientras se escribe en Qdrant— no se ejecuta ni `mark_as_processed` ni
`mark_as_failed`, así que `.processing_state.json` queda **exactamente igual que
antes de empezar**. Con `restart: on-failure`, el contenedor vuelve, el escaneo
recorre los temas en el mismo orden, llega al mismo fichero y vuelve a morir.
La ingesta no avanza y nada en el estado lo delata.

El mecanismo es el mismo que el de docling, un escalón más arriba: se anota el
fichero **antes** de tocarlo y se borra la anota al terminar (bien o mal). Una
anotación encontrada al arrancar sólo puede significar que el proceso anterior
no llegó a borrarla, es decir, que murió duro. Se convierte entonces en un
intento fallido con `mark_as_failed`, y a partir de ahí manda la cuarentena que
ya existe (`INGESTOR_MAX_RETRIES`).

Dos cosas que el diseño concede a propósito:

- **Un `docker stop` a mitad de fichero cuenta como intento.** No hay forma de
  distinguir desde fuera "lo mataron" de "se murió", que es la misma contrapartida
  que documenta `CrashStateManager` y el motivo de que el umbral no pueda ser 1.
  Se mitiga en el camino limpio (SIGTERM/SIGINT y `atexit` borran la anotación),
  no en el brusco.
- **Un fichero que ya consta indexado con éxito nunca se marca como fallido**,
  aunque deje anotación: la muerte cayó entre el `mark_as_processed` y el
  `end()`, y el fichero está bien. Sin esta comprobación, la ventana fabricaría
  falsos fallos sobre ficheros sanos — la enfermedad que ya costó ~180 ficheros
  en cuarentena el 2026-07-22 por la vía del watchdog.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import INFLIGHT_FILE

logger = logging.getLogger(__name__)


class InflightTracker:
    """
    Anota qué fichero está en proceso, para que una muerte dura deje rastro.

    El escaneo es secuencial (`initial_scan` recorre los temas en un solo hilo),
    así que hay como mucho un fichero en vuelo y el fichero guarda una sola
    entrada. Si algún día la ingesta se paraleliza, esto pasa a ser un mapa —y
    hasta entonces, una entrada sola es más difícil de dejar corrupta.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or INFLIGHT_FILE)

    def begin(self, file_path: str, context: str = "") -> None:
        """Anota el fichero como en vuelo. Escribe con fsync: puede no haber salida limpia."""
        payload = {
            "file": str(file_path),
            "context": context,
            "started": datetime.now().isoformat(),
            "pid": os.getpid(),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            # Que no poder anotar no impida indexar: esto es una red, no el trabajo.
            logger.warning(f"[INFLIGHT] No se pudo anotar {file_path}: {e}")
            try:
                tmp.unlink()
            except OSError:
                pass

    def end(self) -> None:
        """Borra la anotación. Idempotente: se llama también desde `atexit`."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"[INFLIGHT] No se pudo borrar la anotación: {e}")

    def consume(self) -> Optional[Dict[str, Any]]:
        """
        Devuelve la anotación huérfana del proceso anterior, y la borra.

        Una anotación presente al arrancar describe una muerte concreta, no un
        estado: se lee una vez y se destruye, como el rastro del watchdog.

        Returns:
            El diccionario anotado, o None si no había nada (arranque limpio).
        """
        try:
            if not self.path.exists():
                return None
            data = json.loads(self.path.read_text())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[INFLIGHT] Anotación ilegible, se descarta: {e}")
            self.end()
            return None

        self.end()
        if not isinstance(data, dict) or not data.get("file"):
            return None
        return data


def record_hard_death(tracker: "InflightTracker", state: Any) -> Optional[str]:
    """
    Convierte la anotación que dejó una muerte dura en un intento fallido.

    Vive aquí y no en `main.py` para poder probarse sin arrastrar el intérprete
    entero (torch, docling, qdrant): la decisión que toma —contar o no contar el
    intento— es justo la que puede fabricar falsos fallos si se equivoca.

    Args:
        tracker: el detector, del que se consume la anotación huérfana.
        state: `ProcessingState` al que cargarle el intento.

    Returns:
        La ruta a la que se le contabilizó el intento, o None si no había
        anotación o si el fichero consta ya indexado con éxito.
    """
    orphan = tracker.consume()
    if not orphan:
        return None

    path = orphan["file"]
    started = orphan.get("started", "?")
    name = Path(path).name

    # La muerte pudo caer entre `mark_as_processed` y el borrado de la anotación:
    # el fichero está indexado y marcarlo como fallido sería fabricar un fallo.
    if state.get_status(path) == "success":
        logger.info(
            f"[MUERTE-DURA] {name} tenía anotación de {started} pero consta indexado "
            f"con éxito: no se cuenta como intento"
        )
        return None

    logger.error(
        f"[MUERTE-DURA] El proceso anterior murió procesando {name} (en vuelo desde "
        f"{started}) sin dejar excepción. Se contabiliza el intento."
    )
    state.mark_as_failed(
        path, f"muerte dura del proceso durante la ingesta (en vuelo desde {started})"
    )
    return path


_inflight: Optional[InflightTracker] = None


def get_inflight_tracker() -> InflightTracker:
    """Obtiene el singleton global de InflightTracker."""
    global _inflight
    if _inflight is None:
        _inflight = InflightTracker()
    return _inflight
