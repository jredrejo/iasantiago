"""
Sistema de heartbeat y watchdog para el módulo ingestor.

Proporciona monitoreo de salud para operaciones de larga duración y recuperación
automática de procesos bloqueados.
"""

import logging
import os
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Configuración por defecto
DEFAULT_HEARTBEAT_FILE = "/tmp/ingestor_heartbeat"
DEFAULT_WATCHDOG_TIMEOUT = 450  # 7.5 minutos
DEFAULT_CHECK_INTERVAL = 60  # 1 minuto


class HeartbeatManager:
    """
    Gestiona actualizaciones de heartbeat para operaciones de larga duración.

    El archivo de heartbeat contiene:
    - Línea 1: Timestamp Unix del último heartbeat
    - Línea 2: Contexto actual/archivo siendo procesado

    El hilo watchdog monitorea este archivo y fuerza salida si
    el heartbeat se vuelve obsoleto.
    """

    _instance: Optional["HeartbeatManager"] = None

    def __new__(cls, *args, **kwargs) -> "HeartbeatManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        heartbeat_file: str = DEFAULT_HEARTBEAT_FILE,
        watchdog_timeout: int = DEFAULT_WATCHDOG_TIMEOUT,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
    ):
        if self._initialized:
            return

        self._heartbeat_file = heartbeat_file
        self._watchdog_timeout = watchdog_timeout
        self._check_interval = check_interval
        self._watchdog_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[str], None]] = None
        self._initialized = True

    @property
    def heartbeat_file(self) -> str:
        """Obtiene la ruta del archivo de heartbeat."""
        return self._heartbeat_file

    def update(self, context: str = "") -> None:
        """
        Actualiza archivo de heartbeat con timestamp actual y contexto.

        Args:
            context: Descripción de la operación actual (ej. archivo siendo procesado)
        """
        try:
            with open(self._heartbeat_file, "w") as f:
                f.write(f"{time.time()}\n{context}\n")
        except Exception:
            pass  # No dejar que problemas de heartbeat afecten el procesamiento

    def start_watchdog(self) -> None:
        """Inicia el hilo watchdog en segundo plano como daemon."""
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            logger.debug("[WATCHDOG] Ya está ejecutándose")
            return

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="heartbeat-watchdog"
        )
        self._watchdog_thread.start()
        logger.info("[WATCHDOG] Iniciado - monitoreando procesos bloqueados")

    def _watchdog_loop(self) -> None:
        """
        Hilo en segundo plano que monitorea heartbeat y fuerza salida en bloqueo.

        Esto captura casos donde el proceso se cuelga sin levantar una señal,
        como errores de 'corrupted double-linked list' de glibc.
        """
        while True:
            time.sleep(self._check_interval)

            try:
                if os.path.exists(self._heartbeat_file):
                    with open(self._heartbeat_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_heartbeat = float(lines[0].strip())
                            current_file = (
                                lines[1].strip() if len(lines) > 1 else "desconocido"
                            )
                            age = time.time() - last_heartbeat

                            if age > self._watchdog_timeout:
                                logger.error(
                                    f"[WATCHDOG] Heartbeat obsoleto por {age:.0f}s "
                                    f"(límite: {self._watchdog_timeout}s)"
                                )
                                logger.error(
                                    f"[WATCHDOG] Último archivo en proceso: {current_file}"
                                )
                                logger.error(
                                    "[WATCHDOG] Forzando salida para reiniciar contenedor"
                                )
                                sys.stdout.flush()
                                sys.stderr.flush()
                                os._exit(1)
            except Exception as e:
                logger.debug(f"[WATCHDOG] Error verificando heartbeat: {e}")

    def set_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """
        Establece función callback para ser llamada en actualizaciones de heartbeat.

        Args:
            callback: Función que recibe un parámetro de cadena de contexto
        """
        self._callback = callback

    def call_with_callback(self, context: str = "") -> None:
        """
        Actualiza heartbeat y llama al callback registrado si está configurado.

        Args:
            context: Descripción de la operación actual
        """
        self.update(context)
        if self._callback is not None:
            try:
                self._callback(context)
            except Exception:
                pass


# Instancia singleton global
_heartbeat_manager: Optional[HeartbeatManager] = None

# Callback legacy para compatibilidad hacia atrás
_legacy_callback: Optional[Callable[[str], None]] = None


def get_heartbeat_manager() -> HeartbeatManager:
    """Obtiene el singleton global de HeartbeatManager."""
    global _heartbeat_manager
    if _heartbeat_manager is None:
        _heartbeat_manager = HeartbeatManager()
    return _heartbeat_manager


def set_heartbeat_callback(callback: Optional[Callable[[str], None]]) -> None:
    """
    Establece callback para actualizaciones de heartbeat (compatibilidad hacia atrás).

    Permite que código externo (como main.py) registre un callback
    que será llamado durante operaciones largas.

    Args:
        callback: Función a llamar con cadena de contexto, o None para limpiar
    """
    global _legacy_callback
    _legacy_callback = callback


def call_heartbeat(context: str = "") -> None:
    """
    Llama al callback de heartbeat si está configurado (compatibilidad hacia atrás).

    Esta es la función principal usada en toda la base de código para señalar
    actividad durante operaciones de larga duración.

    Args:
        context: Descripción de la operación actual
    """
    if _legacy_callback is not None:
        try:
            _legacy_callback(context)
        except Exception:
            pass


def update_heartbeat(current_file: str = "") -> None:
    """
    Actualiza el archivo de heartbeat directamente (compatibilidad hacia atrás).

    Args:
        current_file: Nombre del archivo actualmente siendo procesado
    """
    get_heartbeat_manager().update(current_file)


def start_watchdog() -> None:
    """Inicia el hilo watchdog (compatibilidad hacia atrás)."""
    get_heartbeat_manager().start_watchdog()
