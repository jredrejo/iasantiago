"""
Operaciones de base de datos vectorial Qdrant.

Proporciona gestión de colecciones y operaciones de upsert de vectores
para búsqueda semántica.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from core.config import QDRANT_URL

logger = logging.getLogger(__name__)

# Namespace fijo para IDs de puntos deterministas.
# NO cambiar: alterarlo reasigna los IDs de todo el corpus y obliga a re-indexar.
POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def topic_collection(topic: str) -> str:
    """Obtiene nombre de colección Qdrant para un tema."""
    return f"rag_{topic.lower()}"


def build_point_id(file_path: str, chunk_id: Any) -> str:
    """
    Construye un ID de punto determinista para un chunk.

    Usa uuid5 sobre "<file_path>:<chunk_id>" para que el mismo chunk obtenga
    siempre el mismo ID entre ejecuciones y contenedores. `hash()` de Python
    está salteado por proceso (PYTHONHASHSEED), por lo que producía IDs nuevos
    en cada reinicio y los re-ingestas insertaban duplicados en vez de
    actualizar los puntos existentes.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{file_path}:{chunk_id}"))


class QdrantService:
    """
    Servicio de base de datos vectorial Qdrant.

    Maneja gestión de colecciones, upserts de vectores y búsquedas.
    """

    def __init__(self, url: str = QDRANT_URL):
        self.url = url
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Cliente Qdrant con carga perezosa."""
        if self._client is None:
            self._client = QdrantClient(url=self.url)
            logger.info(f"[QDRANT] Conectado a {self.url}")
        return self._client

    def ensure_collection(self, topic: str, dimension: int) -> str:
        """
        Asegura que la colección existe con la dimensión correcta.

        Crea la colección si no existe. La recrea si la dimensión
        no coincide.

        Args:
            topic: Nombre del tema
            dimension: Dimensión del vector

        Returns:
            Nombre de la colección
        """
        coll = topic_collection(topic)

        if not self.client.collection_exists(collection_name=coll):
            logger.info(
                f"[QDRANT] Creando colección '{coll}' con dimensión {dimension}"
            )
            self.client.create_collection(
                collection_name=coll,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            # Verificar que la dimensión coincide
            try:
                collection_info = self.client.get_collection(collection_name=coll)
                existing_dim = collection_info.config.params.vectors.size

                if existing_dim != dimension:
                    logger.warning(
                        f"[QDRANT] Discrepancia de dimensión para '{coll}': "
                        f"existente={existing_dim}, nueva={dimension}"
                    )
                    logger.warning(
                        f"[QDRANT] Recreando colección con dimensión {dimension}"
                    )
                    self.client.delete_collection(collection_name=coll)
                    self.client.create_collection(
                        collection_name=coll,
                        vectors_config=models.VectorParams(
                            size=dimension,
                            distance=models.Distance.COSINE,
                        ),
                    )
                    logger.info(
                        f"[QDRANT] Colección '{coll}' recreada con dimensión {dimension}"
                    )
                else:
                    logger.info(
                        f"[QDRANT] Colección '{coll}' existe con dimensión correcta {dimension}"
                    )
            except Exception as e:
                logger.error(f"[QDRANT] Error verificando colección '{coll}': {e}")
                raise

        self._ensure_file_path_index(coll)

        return coll

    def _ensure_file_path_index(self, coll: str) -> None:
        """
        Asegura el índice de payload sobre `file_path`.

        Necesario para que `delete_by_file` pueda borrar con un filtro
        del lado del servidor en vez de paginar la colección entera.
        Es idempotente: Qdrant ignora la creación si el índice ya existe.
        """
        try:
            self.client.create_payload_index(
                collection_name=coll,
                field_name="file_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"[QDRANT] Índice de payload 'file_path' listo en '{coll}'")
        except Exception as e:
            logger.warning(
                f"[QDRANT] No se pudo crear índice de payload 'file_path' en "
                f"'{coll}': {e}"
            )

    def upsert_vectors(
        self,
        topic: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Inserta o actualiza vectores en la colección.

        Args:
            topic: Nombre del tema
            vectors: Lista de vectores
            payloads: Lista de payloads
            batch_size: Tamaño de lote para upsert

        Returns:
            Número de vectores insertados
        """
        coll = topic_collection(topic)
        total = len(vectors)

        logger.info(
            f"[QDRANT] Insertando {total} vectores en '{coll}' en lotes de {batch_size}"
        )

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)

            # IDs deterministas: mismo chunk -> mismo ID en cada ejecución
            batch_ids = []
            for i, payload in enumerate(
                payloads[batch_start:batch_end], start=batch_start
            ):
                file_path = payload.get("file_path", str(i))
                chunk_id = payload.get("chunk_id", i)
                batch_ids.append(build_point_id(file_path, chunk_id))

            batch_vecs = vectors[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            points = [
                models.PointStruct(
                    id=batch_ids[i],
                    vector=batch_vecs[i],
                    payload=batch_payloads[i],
                )
                for i in range(len(batch_vecs))
            ]

            self.client.upsert(
                collection_name=coll,
                points=points,
                wait=True,
            )

            batch_num = batch_start // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(f"[QDRANT] Lote {batch_num}/{total_batches}")

        logger.info(f"[QDRANT] Todos los {total} vectores subidos")
        return total

    def delete_by_file(self, topic: str, file_path: str) -> int:
        """
        Elimina todos los puntos de un archivo.

        Args:
            topic: Nombre del tema
            file_path: Ruta del archivo a eliminar

        Returns:
            Número de puntos que había para ese archivo antes de borrar
        """
        coll = topic_collection(topic)

        if not self.client.collection_exists(collection_name=coll):
            return 0

        logger.info(f"[QDRANT] Eliminando puntos para archivo: {file_path}")

        file_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchValue(value=file_path),
                )
            ]
        )

        # Contar antes de borrar: el borrado filtrado no devuelve el número
        # de puntos afectados.
        try:
            count = self.client.count(
                collection_name=coll,
                count_filter=file_filter,
                exact=True,
            ).count
        except Exception as e:
            logger.warning(f"[QDRANT] No se pudo contar puntos de {file_path}: {e}")
            count = -1

        if count == 0:
            logger.info(f"[QDRANT] No hay puntos previos para {file_path}")
            return 0

        # Borrado filtrado del lado del servidor: alcanza toda la colección,
        # no sólo los primeros 10 000 puntos.
        self.client.delete(
            collection_name=coll,
            points_selector=models.FilterSelector(filter=file_filter),
            wait=True,
        )

        logger.info(f"[QDRANT] Eliminados {count} puntos de {file_path}")
        return max(count, 0)

    def delete_collection(self, topic: str) -> bool:
        """
        Elimina colección completa.

        Args:
            topic: Nombre del tema

        Returns:
            True si se eliminó, False si no existía
        """
        coll = topic_collection(topic)

        if self.client.collection_exists(collection_name=coll):
            collection_info = self.client.get_collection(collection_name=coll)
            points_count = collection_info.points_count

            self.client.delete_collection(collection_name=coll)
            logger.info(
                f"[QDRANT] Eliminada colección '{coll}' ({points_count} puntos)"
            )
            return True
        else:
            logger.warning(f"[QDRANT] Colección '{coll}' no existe")
            return False

    def get_collection_info(self, topic: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de la colección."""
        coll = topic_collection(topic)

        if not self.client.collection_exists(collection_name=coll):
            return None

        info = self.client.get_collection(collection_name=coll)
        return {
            "name": coll,
            "points_count": info.points_count,
            "dimension": info.config.params.vectors.size,
        }


# Instancia global
_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Obtiene o crea el servicio Qdrant global."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service


def ensure_qdrant(topic: str, dimension: int) -> str:
    """Función de conveniencia para asegurar que la colección Qdrant existe."""
    return get_qdrant_service().ensure_collection(topic, dimension)
