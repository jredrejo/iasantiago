"""
Operaciones de búsqueda por palabras clave Whoosh BM25.

Proporciona gestión de índices e indexación de documentos para
búsqueda por palabras clave basada en BM25.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from whoosh import index
from whoosh.fields import ID, NUMERIC, TEXT, Schema

from core.config import BM25_BASE_DIR

logger = logging.getLogger(__name__)


# Esquema predeterminado para indexación de documentos.
#
# `doc_key` es el único campo `unique`: sin él, `writer.update_document()` se
# comporta como `add_document()` y cada re-ingesta duplica todos los chunks.
DEFAULT_SCHEMA = Schema(
    doc_key=ID(unique=True, stored=True),
    file_path=ID(stored=True),
    page=NUMERIC(stored=True),
    chunk_id=NUMERIC(stored=True),
    text=TEXT(stored=True),
    chunk_type=TEXT(stored=True),
    source=TEXT(stored=True),
)


def build_doc_key(file_path: str, chunk_id: Any) -> str:
    """Construye la clave única de un chunk dentro del índice."""
    return f"{file_path}:{chunk_id}"


class WhooshService:
    """
    Servicio de búsqueda Whoosh BM25.

    Maneja gestión de índices y operaciones de documentos.
    """

    def __init__(self, base_dir: str = BM25_BASE_DIR):
        self.base_dir = base_dir
        self._indexes: Dict[str, index.Index] = {}

    def get_index_path(self, topic: str) -> str:
        """Obtener ruta de índice para un tema."""
        return os.path.join(self.base_dir, topic)

    def ensure_index(self, topic: str) -> str:
        """
        Asegurar que el índice existe para el tema.

        Args:
            topic: Nombre del tema

        Returns:
            Ruta del índice
        """
        path = self.get_index_path(topic)
        os.makedirs(path, exist_ok=True)

        if not index.exists_in(path):
            logger.info(f"[WHOOSH] Creating index at {path}")
            index.create_in(path, DEFAULT_SCHEMA)
        else:
            logger.info(f"[WHOOSH] Index at {path} already exists")
            self._migrate_schema(path)

        return path

    def _migrate_schema(self, path: str) -> None:
        """
        Añade a un índice existente los campos que le falten del esquema actual.

        Los índices creados por versiones anteriores (o por el esquema más
        estrecho de rag-api) carecen de `doc_key`/`chunk_type`/`source`;
        escribir esos campos fallaría. Los documentos ya indexados quedan sin
        valor en los campos nuevos, algo que la re-indexación completa resuelve.
        """
        try:
            idx = index.open_dir(path)
            missing = [
                name for name in DEFAULT_SCHEMA.names() if name not in idx.schema.names()
            ]
            if not missing:
                return

            logger.info(f"[WHOOSH] Añadiendo campos ausentes en {path}: {missing}")
            writer = idx.writer()
            for name in missing:
                writer.add_field(name, DEFAULT_SCHEMA[name])
            writer.commit()
        except Exception as e:
            logger.warning(f"[WHOOSH] No se pudo migrar el esquema de {path}: {e}")

    def get_index(self, topic: str) -> index.Index:
        """
        Obtener o abrir índice para el tema.

        Args:
            topic: Nombre del tema

        Returns:
            Índice Whoosh
        """
        if topic not in self._indexes:
            path = self.get_index_path(topic)
            if not index.exists_in(path):
                self.ensure_index(topic)
            self._indexes[topic] = index.open_dir(path)

        return self._indexes[topic]

    def index_documents(
        self,
        topic: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Indexar documentos en Whoosh.

        Args:
            topic: Nombre del tema
            documents: Lista de diccionarios de documentos

        Returns:
            Número de documentos indexados
        """
        idx = self.get_index(topic)
        writer = idx.writer(limitmb=512, procs=0, multisegment=True)

        count = 0
        for doc in documents:
            file_path = doc.get("file_path", "")
            chunk_id = doc.get("chunk_id", 0)
            writer.update_document(
                doc_key=build_doc_key(file_path, chunk_id),
                file_path=file_path,
                page=doc.get("page", 1),
                chunk_id=chunk_id,
                text=doc.get("text", ""),
                chunk_type=doc.get("chunk_type", "text"),
                source=doc.get("source", "unknown"),
            )
            count += 1

        writer.commit()
        logger.info(f"[WHOOSH] Indexados {count} documentos en {topic}")
        return count

    def delete_by_file(self, topic: str, file_path: str) -> int:
        """
        Eliminar todos los documentos para un archivo.

        Args:
            topic: Nombre del tema
            file_path: Ruta del archivo a eliminar

        Returns:
            Número de documentos eliminados
        """
        idx = self.get_index(topic)
        writer = idx.writer()

        deleted_count = writer.delete_by_term("file_path", file_path)
        writer.commit()

        logger.info(f"[WHOOSH] Eliminados {deleted_count} documentos de {topic}")
        return deleted_count

    def delete_index(self, topic: str) -> bool:
        """
        Eliminar índice completo.

        Args:
            topic: Nombre del tema

        Returns:
            True si eliminado, False si no existía
        """
        path = self.get_index_path(topic)

        # Close index if open
        if topic in self._indexes:
            del self._indexes[topic]

        if os.path.exists(path):
            # Count documents before deleting
            doc_count = 0
            try:
                idx = index.open_dir(path)
                with idx.searcher() as searcher:
                    doc_count = searcher.doc_count()
            except Exception:
                pass

            shutil.rmtree(path)
            logger.info(f"[WHOOSH] Deleted index '{path}' ({doc_count} documents)")
            return True
        else:
            logger.warning(f"[WHOOSH] Index '{path}' does not exist")
            return False

    def get_document_count(self, topic: str) -> int:
        """Obtener número de documentos en el índice."""
        try:
            idx = self.get_index(topic)
            with idx.searcher() as searcher:
                return searcher.doc_count()
        except Exception:
            return 0

    def close_all(self) -> None:
        """Cerrar todos los índices abiertos."""
        self._indexes.clear()
        logger.info("[WHOOSH] Cerrados todos los índices")


# Global instance
_whoosh_service: Optional[WhooshService] = None


def get_whoosh_service() -> WhooshService:
    """Obtener o crear el servicio Whoosh global."""
    global _whoosh_service
    if _whoosh_service is None:
        _whoosh_service = WhooshService()
    return _whoosh_service


def ensure_whoosh(topic: str) -> str:
    """Función de conveniencia para asegurar que el índice Whoosh existe."""
    return get_whoosh_service().ensure_index(topic)
