"""
Operaciones de base de datos vectorial Qdrant.

Proporciona gestión de colecciones y operaciones de upsert de vectores
para búsqueda semántica.
"""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from core.config import QDRANT_URL

logger = logging.getLogger(__name__)


def topic_collection(topic: str) -> str:
    """Obtiene nombre de colección Qdrant para un tema."""
    return f"rag_{topic.lower()}"


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

        return coll

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

            # Generar IDs basados en file_path e índice
            batch_ids = []
            for i, payload in enumerate(
                payloads[batch_start:batch_end], start=batch_start
            ):
                file_path = payload.get("file_path", str(i))
                point_id = abs(hash(f"{file_path}:{i}")) % (2**31)
                batch_ids.append(point_id)

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
            Número de puntos eliminados
        """
        coll = topic_collection(topic)

        logger.info(f"[QDRANT] Eliminando puntos para archivo: {file_path}")

        # Encontrar todos los puntos con este file_path
        points_result = self.client.scroll(
            collection_name=coll,
            limit=10000,
            with_payload=True,
        )

        points = points_result[0]
        point_ids = [
            point.id for point in points if point.payload.get("file_path") == file_path
        ]

        if point_ids:
            self.client.delete(
                collection_name=coll,
                points_selector=point_ids,
            )
            logger.info(f"[QDRANT] Eliminados {len(point_ids)} puntos")
            return len(point_ids)
        else:
            logger.warning(f"[QDRANT] No se encontraron puntos para {file_path}")
            return 0

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
