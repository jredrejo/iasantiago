"""
Gestión del estado de procesamiento.

Rastrea qué archivos han sido procesados, su estado,
y proporciona detección de cambios vía hashing MD5.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "/whoosh/.processing_state.json"


class ProcessingState:
    """
    Gestiona el estado de procesamiento para archivos.

    Rastrea:
    - Qué archivos han sido procesados
    - Sus hashes MD5 para detección de cambios
    - Estado de éxito/fallo
    - Marcas de tiempo de procesamiento
    """

    def __init__(self, state_file: str = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Carga estado desde archivo o crea nuevo."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                logger.info(
                    f"[STATE] Estado cargado con {len(state.get('processed', {}))} archivos procesados"
                )
                return state
            except Exception as e:
                logger.warning(f"[STATE] Error al cargar estado: {e}, creando nuevo")
                return self._create_empty_state()
        else:
            logger.info("[STATE] No se encontró estado previo, creando nuevo")
            return self._create_empty_state()

    def _create_empty_state(self) -> Dict[str, Any]:
        """Crea estructura de estado vacía."""
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "last_scan": None,
            "processed": {},
            "failed": {},
        }

    def _save_state(self) -> None:
        """Guarda estado en archivo."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"[STATE] Error al guardar estado: {e}")

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Calcula hash MD5 del archivo."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"[STATE] Error al calcular hash para {file_path}: {e}")
            return None

    def is_already_processed(self, file_path: str) -> bool:
        """
        Verifica si el archivo ya fue procesado.

        Retorna False si:
        - El archivo nunca fue procesado
        - El archivo falló previamente
        - El hash del archivo cambió (contenido modificado)
        """
        file_path = str(file_path)

        if file_path not in self.state["processed"]:
            return False

        file_info = self.state["processed"][file_path]

        # Reintentar archivos fallidos
        if file_info.get("status") == "failed":
            logger.info(
                f"[STATE] Reintentando archivo que falló previamente: {Path(file_path).name}"
            )
            return False

        # Verificar cambios de contenido
        current_hash = self.get_file_hash(file_path)
        stored_hash = file_info.get("hash")

        if current_hash and stored_hash and current_hash != stored_hash:
            logger.info(
                f"[STATE] Archivo cambió (hash diferente), reprocesando: {Path(file_path).name}"
            )
            return False

        logger.info(f"[STATE] Omitiendo ya procesado: {Path(file_path).name}")
        return True

    def mark_as_processed(self, file_path: str, topic: str) -> None:
        """Marca archivo como procesado exitosamente."""
        file_path = str(file_path)
        self.state["processed"][file_path] = {
            "hash": self.get_file_hash(file_path),
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "status": "success",
        }
        self._save_state()
        logger.info(f"[STATE] Marcado como procesado: {Path(file_path).name}")

    def mark_as_failed(self, file_path: str, error: str) -> None:
        """Marca archivo como fallido."""
        file_path = str(file_path)
        self.state["processed"][file_path] = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(error)[:200],
        }
        self.state["failed"][file_path] = {
            "error": str(error)[:500],
            "timestamp": datetime.now().isoformat(),
        }
        self._save_state()
        logger.warning(f"[STATE] Marcado como fallido: {Path(file_path).name}")

    def update_scan_time(self) -> None:
        """Actualiza marca de tiempo del último escaneo."""
        self.state["last_scan"] = datetime.now().isoformat()
        self._save_state()

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de procesamiento."""
        processed = self.state.get("processed", {})
        successful = sum(1 for v in processed.values() if v.get("status") == "success")
        failed = sum(1 for v in processed.values() if v.get("status") == "failed")

        return {
            "total_processed": len(processed),
            "successful": successful,
            "failed": failed,
            "last_scan": self.state.get("last_scan"),
        }

    def reset(self) -> None:
        """Reinicia todo el estado."""
        logger.warning(
            "[STATE] Reiniciando estado de procesamiento - se reescanearán todos los archivos"
        )
        self.state = self._create_empty_state()
        self._save_state()

    def remove_file(self, file_path: str) -> bool:
        """
        Elimina archivo del estado.

        Args:
            file_path: Ruta del archivo a eliminar

        Returns:
            True si el archivo estaba en estado, False en caso contrario
        """
        file_path = str(file_path)
        removed = False

        if file_path in self.state["processed"]:
            del self.state["processed"][file_path]
            removed = True

        if file_path in self.state["failed"]:
            del self.state["failed"][file_path]
            removed = True

        if removed:
            self._save_state()
            logger.info(f"[STATE] Eliminado del estado: {Path(file_path).name}")

        return removed

    def remove_topic_files(self, topic: str, topic_base_dir: str) -> int:
        """
        Elimina todos los archivos de un tema del estado.

        Args:
            topic: Nombre del tema
            topic_base_dir: Directorio base de temas

        Returns:
            Número de archivos eliminados
        """
        files_to_remove = []
        topic_dir = os.path.join(topic_base_dir, topic)

        # Encontrar archivos procesados para este tema
        for file_path, file_info in self.state["processed"].items():
            if file_info.get("topic") == topic:
                files_to_remove.append(file_path)

        # Eliminar de procesados
        for file_path in files_to_remove:
            self.state["processed"].pop(file_path, None)

        # Eliminar de fallidos
        failed_removed = []
        for file_path in list(self.state["failed"].keys()):
            if topic_dir in file_path:
                self.state["failed"].pop(file_path, None)
                failed_removed.append(file_path)

        total_removed = len(files_to_remove) + len(failed_removed)

        if total_removed > 0:
            self._save_state()
            logger.info(f"[STATE] Eliminados {total_removed} archivos del tema {topic}")

        return total_removed

    def get_processed_files(
        self, topic: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene archivos procesados, opcionalmente filtrados por tema.

        Args:
            topic: Tema opcional para filtrar

        Returns:
            Diccionario de rutas de archivo a info del archivo
        """
        processed = self.state.get("processed", {})

        if topic is None:
            return processed

        return {
            path: info for path, info in processed.items() if info.get("topic") == topic
        }


# Instancia global
_processing_state: Optional[ProcessingState] = None


def get_processing_state() -> ProcessingState:
    """Obtiene o crea el estado de procesamiento global."""
    global _processing_state
    if _processing_state is None:
        _processing_state = ProcessingState()
    return _processing_state
