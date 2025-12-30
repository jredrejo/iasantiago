"""
Utilidades de caché unificadas para el módulo ingestor.

Proporciona caché de hash de archivos, caché de extracción y caché de conteo de páginas PDF.
Consolida patrones de caché de chunk.py, main.py y docling_extractor.py.
"""

import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)


# ============================================================
# CACHÉ DE CONTEO DE PÁGINAS PDF
# ============================================================


@lru_cache(maxsize=256)
def get_pdf_total_pages(pdf_path: str) -> Optional[int]:
    """
    Obtiene el conteo total de páginas de un PDF, cacheado por ruta de archivo.

    Args:
        pdf_path: Ruta al archivo PDF

    Returns:
        Número de páginas, o None si no se puede determinar
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        logger.warning(
            f"[CACHE] No se pudo obtener conteo de páginas para {os.path.basename(str(pdf_path))}: {e}"
        )
        return None


def clear_pdf_page_cache() -> None:
    """Limpia la caché de conteo de páginas PDF."""
    get_pdf_total_pages.cache_clear()


# ============================================================
# CACHÉ DE HASH DE ARCHIVOS
# ============================================================


class FileHashCache:
    """
    Caché para hashes de archivos con algoritmo configurable.

    Proporciona hashing consistente en todo el módulo con caché
    en memoria opcional para solicitudes repetidas de hash.
    """

    def __init__(self, algorithm: str = "md5", max_cache_size: int = 256):
        """
        Inicializa la caché de hash.

        Args:
            algorithm: Algoritmo de hash a usar ('md5' o 'sha256')
            max_cache_size: Número máximo de hashes en caché
        """
        self._algorithm = algorithm
        self._cache: Dict[str, str] = {}
        self._max_size = max_cache_size

    def get_hash(self, file_path: str) -> Optional[str]:
        """
        Obtiene el hash de un archivo, usando caché si está disponible.

        Args:
            file_path: Ruta al archivo

        Returns:
            Cadena de hash, o None si no se puede calcular
        """
        file_path = str(file_path)

        # Verificar caché primero
        if file_path in self._cache:
            return self._cache[file_path]

        # Calcular hash
        try:
            hash_obj = hashlib.new(self._algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            file_hash = hash_obj.hexdigest()

            # Cachear resultado (con límite de tamaño)
            if len(self._cache) >= self._max_size:
                # Eliminar entrada más antigua (FIFO simple)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[file_path] = file_hash

            return file_hash
        except Exception as e:
            logger.error(f"[CACHE] Error al calcular hash para {file_path}: {e}")
            return None

    def clear(self) -> None:
        """Limpia la caché de hash."""
        self._cache.clear()


# Instancias globales de caché de hash
_md5_cache = FileHashCache(algorithm="md5")
_sha256_cache = FileHashCache(algorithm="sha256")


def get_file_hash_md5(file_path: str) -> Optional[str]:
    """Obtiene hash MD5 de un archivo usando caché global."""
    return _md5_cache.get_hash(file_path)


def get_file_hash_sha256(file_path: str) -> Optional[str]:
    """Obtiene hash SHA256 de un archivo usando caché global."""
    return _sha256_cache.get_hash(file_path)


# ============================================================
# CACHÉ DE EXTRACCIÓN
# ============================================================


class ExtractionCache:
    """
    Caché persistente para resultados de extracción de PDF.

    Almacena resultados de extracción por hash de archivo para evitar
    reprocesar archivos sin cambios.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_file_name: str = "extraction_cache.json",
    ):
        """
        Inicializa la caché de extracción.

        Args:
            cache_dir: Directorio para archivos de caché. Por defecto /cache/docling o temp.
            cache_file_name: Nombre del archivo de caché.
        """
        if cache_dir is None:
            if os.path.exists("/cache"):
                cache_dir = Path("/cache/docling")
            else:
                cache_dir = Path(tempfile.gettempdir()) / "docling_cache"

        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / cache_file_name
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    @property
    def cache_dir(self) -> Path:
        """Obtiene la ruta del directorio de caché."""
        return self._cache_dir

    def _load(self) -> None:
        """Carga caché desde disco."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"[CACHE] Cargadas {len(self._cache)} extracciones en caché")
        except Exception as e:
            logger.warning(f"[CACHE] Error al cargar caché: {e}")
            self._cache = {}

    def _save(self) -> None:
        """Guarda caché en disco."""
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"[CACHE] Error al guardar caché: {e}")

    def get(self, file_hash: str) -> Optional[List[Dict[str, Any]]]:
        """
        Obtiene resultados de extracción en caché por hash de archivo.

        Args:
            file_hash: Hash del archivo fuente

        Returns:
            Lista de elementos extraídos, o None si no está en caché
        """
        if file_hash in self._cache:
            logger.info(f"[CACHE] Acierto para hash {file_hash[:8]}...")
            # Retornar copia para prevenir mutación
            import copy

            return copy.deepcopy(self._cache[file_hash])
        return None

    def put(self, file_hash: str, elements: List[Dict[str, Any]]) -> None:
        """
        Cachea resultados de extracción.

        Args:
            file_hash: Hash del archivo fuente
            elements: Lista de elementos extraídos
        """
        import copy

        self._cache[file_hash] = copy.deepcopy(elements)
        self._save()

    def has(self, file_hash: str) -> bool:
        """Verifica si un hash de archivo está en la caché."""
        return file_hash in self._cache

    def remove(self, file_hash: str) -> bool:
        """
        Elimina una entrada de la caché.

        Returns:
            True si la entrada fue eliminada, False si no se encontró
        """
        if file_hash in self._cache:
            del self._cache[file_hash]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Limpia todos los datos en caché."""
        self._cache.clear()
        self._save()

    def __len__(self) -> int:
        """Obtiene el número de entradas en caché."""
        return len(self._cache)
