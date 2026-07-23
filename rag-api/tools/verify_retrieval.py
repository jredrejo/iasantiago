#!/usr/bin/env python3
"""
verify_retrieval.py

Script para verificar que el retrieval funciona correctamente.
Ejecutar DENTRO del contenedor rag-api:

    docker exec rag-api python /app/tools/verify_retrieval.py
"""

import os
import sys
import logging
from collections import defaultdict

# Permite ejecutar el script desde tools/ resolviendo los imports del paquete rag-api.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Imports
try:
    from qdrant_utils import client, topic_collection, get_collection_stats
    from retrieval import debug_retrieval
    from config.settings import TOPIC_LABELS

    logger.info("✅ Imports exitosos")
except ImportError as e:
    logger.error(f"❌ Error importando: {e}")
    sys.exit(1)


def verify_qdrant_collections():
    """Verifica que las colecciones de Qdrant existan y tengan datos"""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 VERIFICACIÓN 1: Colecciones Qdrant")
    logger.info("=" * 80)

    for topic in TOPIC_LABELS:
        coll = topic_collection(topic)
        try:
            stats = get_collection_stats(topic)
            if stats and stats["points_count"] > 0:
                logger.info(f"✅ {topic}:")
                logger.info(f"   - Colección: {coll}")
                logger.info(f"   - Puntos: {stats['points_count']}")
                logger.info(f"   - Tamaño vector: {stats['vector_size']}")
            else:
                logger.warning(f"⚠️  {topic}: Colección vacía o no existe")
        except Exception as e:
            logger.error(f"❌ {topic}: Error - {e}")


def verify_qdrant_diversity():
    """Verifica que haya vectores de MÚLTIPLES archivos en Qdrant"""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 VERIFICACIÓN 2: Diversidad de archivos en Qdrant")
    logger.info("=" * 80)

    for topic in TOPIC_LABELS:
        coll = topic_collection(topic)
        try:
            # Obtener todos los puntos (primeros 1000)
            points, _ = client.scroll(
                collection_name=coll, limit=1000, with_payload=True
            )

            if not points:
                logger.warning(f"⚠️  {topic}: No hay puntos")
                continue

            # Contar por archivo
            files = defaultdict(int)
            for point in points:
                file_path = point.payload.get("file_path", "unknown")
                files[file_path] += 1

            logger.info(f"✅ {topic}:")
            logger.info(f"   - Archivos únicos: {len(files)}")
            logger.info(f"   - Total puntos: {len(points)}")

            for file_path, count in sorted(files.items(), key=lambda x: -x[1])[:5]:
                filename = file_path.split("/")[-1]
                pct = (count / len(points)) * 100
                logger.info(f"     • {filename}: {count} ({pct:.1f}%)")

            if len(files) == 1:
                logger.error(
                    f"   ❌ CRÍTICO: Solo 1 archivo. Problema de IDs colisionantes"
                )
            elif len(files) < 3:
                logger.warning(f"   ⚠️  Solo {len(files)} archivos (se esperaban >=3)")

        except Exception as e:
            logger.error(f"❌ {topic}: Error - {e}")


def verify_retrieval_queries():
    """Prueba queries reales para ver si traen de múltiples archivos"""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 VERIFICACIÓN 3: Prueba de queries")
    logger.info("=" * 80)

    test_queries = {
        "Programming": [
            "python variables",
            "loops arrays",
            "functions classes",
        ],
        "Electronics": [
            "resistor capacitor",
            "circuit voltage current",
            "transistor semiconductor",
        ],
        "Chemistry": [
            "molecular structure bonds",
            "reactions catalysts",
            "atoms elements periodic",
        ],
    }

    for topic in TOPIC_LABELS:
        if topic not in test_queries:
            continue

        logger.info(f"\n📍 Tema: {topic}")
        queries = test_queries[topic]

        for query in queries:
            try:
                debug_info = debug_retrieval(topic, query)

                logger.info(f"  Query: '{query}'")
                logger.info(f"    - Resultados densos: {debug_info['dense_hits']}")
                logger.info(f"    - Resultados BM25: {debug_info['bm25_hits']}")
                logger.info(f"    - Mezclados: {debug_info['merged_results']}")
                logger.info(f"    - Archivos únicos: {debug_info['unique_files']}")

                if debug_info["unique_files"] > 1:
                    logger.info(f"    ✅ OK - {debug_info['unique_files']} archivos")
                else:
                    logger.warning(f"    ⚠️  Resultado de solo 1 archivo")

            except Exception as e:
                logger.error(f"  ❌ Error: {e}", exc_info=True)


def main():
    logger.info("\n" + "🔍 " * 20)
    logger.info("VERIFICACIÓN COMPLETA DEL SISTEMA RAG")
    logger.info("🔍 " * 20)

    verify_qdrant_collections()
    verify_qdrant_diversity()
    verify_retrieval_queries()

    logger.info("\n" + "=" * 80)
    logger.info("✅ VERIFICACIÓN COMPLETADA")
    logger.info("=" * 80)
    logger.info("\n📋 LISTA DE VERIFICACIÓN:")
    logger.info("  [ ] ¿Todas las colecciones existen y tienen puntos?")
    logger.info("  [ ] ¿Hay múltiples archivos en cada colección?")
    logger.info("  [ ] ¿Las queries traen de múltiples archivos?")
    logger.info("  [ ] ¿Los scores híbridos son razonables?")
    logger.info("\nSi algo falla, revisa los logs de arriba ⬆️")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
