#!/usr/bin/env python3
"""
Optimiza todos los índices Whoosh (BM25) combinando sus segmentos en uno solo,
lo que mejora la velocidad de búsqueda. Ejecutar con el sistema inactivo
(p. ej. un cron nocturno).

Configuración por entorno (mismos valores que usan los servicios):
    BM25_BASE_DIR   Directorio raíz de los índices Whoosh (por defecto /whoosh).
    TOPIC_LABELS    Lista de temas separada por comas. Si no se define, se
                    optimizan todos los subdirectorios de BM25_BASE_DIR que
                    contengan un índice Whoosh válido.

Ejemplos:
    # en el host, contra el volumen montado:
    BM25_BASE_DIR=data/whoosh python scripts/optimiza_whoosh.py
    # dentro del contenedor (si scripts/ está montado):
    docker exec -it rag-api python /app/scripts/optimiza_whoosh.py
"""

import logging
import os

from whoosh import index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BM25_BASE_DIR = os.getenv("BM25_BASE_DIR", "/whoosh")


def _segment_count(idx) -> str:
    """Cuenta segmentos/lectores usando sólo API pública (best-effort).

    El conteo es meramente informativo; nunca debe impedir la optimización, así
    que cualquier fallo se degrada a 'no disponible' en lugar de propagarse.
    """
    try:
        reader = idx.reader()
        try:
            return (
                str(len(reader.leaf_readers()))
                if hasattr(reader, "leaf_readers")
                else "desconocido"
            )
        finally:
            reader.close()
    except Exception:
        return "no disponible"


def optimize_topic_index(base_dir: str, topic: str) -> None:
    """Optimiza el índice Whoosh de un tema."""
    topic_path = os.path.join(base_dir, topic)

    if not index.exists_in(topic_path):
        logger.warning(f"Sin índice Whoosh en: {topic_path}")
        return

    try:
        idx = index.open_dir(topic_path)
        logger.info(f"📊 Índice {topic}:")
        logger.info(f"   - Documentos: {idx.doc_count_all()}")
        logger.info(f"   - Segmentos antes: {_segment_count(idx)}")

        logger.info("   ⏳ Optimizando...")
        writer = idx.writer()
        writer.commit(optimize=True)  # ← combina todos los segmentos en uno

        after = _segment_count(index.open_dir(topic_path))
        logger.info(f"   ✅ Segmentos después: {after}")

    except Exception as e:
        logger.error(f"❌ Error optimizando {topic}: {e}", exc_info=True)


def _topics(base_dir: str) -> list:
    """Temas a optimizar: TOPIC_LABELS si está definido, si no los subdirectorios."""
    env = os.getenv("TOPIC_LABELS")
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]

    if not os.path.isdir(base_dir):
        return []
    return sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )


def main() -> None:
    logger.info(f"🔧 Optimizando índices Whoosh en: {BM25_BASE_DIR}")
    topics = _topics(BM25_BASE_DIR)
    logger.info(f"📚 Temas: {topics}")

    for topic in topics:
        optimize_topic_index(BM25_BASE_DIR, topic)

    logger.info("✅ Optimización completada para todos los temas")


if __name__ == "__main__":
    main()
