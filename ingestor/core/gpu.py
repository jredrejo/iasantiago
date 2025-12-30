"""
Gestión unificada de GPU para el módulo ingestor.

Proporciona un GPUManager singleton para estado consistente de GPU en todos los módulos.
Consolida configuración de GPU de chunk.py, main.py y docling_extractor.py.
"""

import gc
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Gestor singleton para estado y operaciones de GPU.

    Proporciona configuración consistente de GPU en todos los módulos del ingestor,
    incluyendo gestión de memoria, detección de dispositivo y manejo de errores.
    """

    _instance: Optional["GPUManager"] = None

    def __new__(cls) -> "GPUManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._available: bool = False
        self._device: str = "cpu"
        self._memory_fraction: float = 0.30
        self._failed: bool = False
        self._initialized = True

        # Realizar configuración inicial
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detecta disponibilidad de GPU y registra información del dispositivo."""
        if not torch.cuda.is_available():
            logger.warning("[GPU] CUDA no disponible, ejecutando en CPU")
            self._available = False
            self._device = "cpu"
            return

        self._available = True
        self._device = "cuda"

        gpu_count = torch.cuda.device_count()
        logger.info(f"[GPU] Encontrados {gpu_count} dispositivo(s) CUDA")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"[GPU] Dispositivo {i}: {props.name}")
            logger.info(f"[GPU]   - Memoria total: {props.total_memory / 1e9:.2f} GB")

    def setup(self, memory_fraction: Optional[float] = None) -> bool:
        """
        Inicializa GPU con restricciones de memoria.

        Args:
            memory_fraction: Fracción de memoria GPU a usar (0.0-1.0).
                           Por defecto usa variable de entorno DOCLING_GPU_MEMORY_FRACTION o 0.30.

        Returns:
            True si la configuración de GPU fue exitosa, False en caso contrario.
        """
        if not self._available:
            return False

        if memory_fraction is None:
            memory_fraction = float(os.getenv("DOCLING_GPU_MEMORY_FRACTION", "0.30"))

        self._memory_fraction = memory_fraction

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=i)

            logger.info(f"[GPU] Fracción de memoria configurada: {memory_fraction:.2%}")
            return True
        except Exception as e:
            logger.error(f"[GPU] Error en configuración: {e}")
            self._failed = True
            return False

    @property
    def is_available(self) -> bool:
        """Verifica si GPU es usable (disponible y no fallida)."""
        return self._available and not self._failed

    @property
    def device(self) -> str:
        """Obtiene la cadena del dispositivo actual ('cuda' o 'cpu')."""
        if self._failed:
            return "cpu"
        return self._device

    @property
    def has_failed(self) -> bool:
        """Verifica si las operaciones de GPU han fallado."""
        return self._failed

    def mark_failed(self) -> None:
        """Marca GPU como fallida, forzando respaldo a CPU."""
        self._failed = True
        logger.warning("[GPU] Marcada como fallida, usando respaldo CPU")

    def reset_failed(self) -> None:
        """Reinicia estado de fallo para reintentar operaciones de GPU."""
        self._failed = False
        logger.info("[GPU] Estado de fallo reiniciado, GPU disponible para reintentar")

    def clear_cache(self) -> None:
        """Limpia caché de GPU y ejecuta recolección de basura."""
        if self._available:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"[GPU] Error al limpiar caché: {e}")
        gc.collect()

    def log_memory_usage(self, context: str = "") -> None:
        """Registra uso actual de memoria GPU para depuración."""
        if not self._available:
            return

        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            prefix = f"[GPU] {context}: " if context else "[GPU] "
            logger.info(
                f"{prefix}Memoria asignada: {allocated:.2f} GB, reservada: {reserved:.2f} GB"
            )
        except Exception as e:
            logger.debug(f"[GPU] No se pudo registrar memoria: {e}")

    def get_memory_allocated(self) -> float:
        """Obtiene memoria GPU actualmente asignada en GB."""
        if not self._available:
            return 0.0
        try:
            return torch.cuda.memory_allocated() / 1e9
        except Exception:
            return 0.0


# Instancia singleton global
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Obtiene el singleton global de GPUManager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
