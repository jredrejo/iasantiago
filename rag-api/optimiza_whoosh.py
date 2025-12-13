#!/usr/bin/env python3
"""
Optimiza todos los índices Whoosh para mejorar velocidad de búsqueda
Ejecutar cuando el sistema esté inactivo (cron nocturno)

docker exec -it rag-api python /app/optimiza_whoosh.py 
"""

import os
from whoosh import index
from settings import BM25_BASE_DIR, TOPIC_LABELS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_topic_index(base_dir: str, topic: str):
    """Optimiza un índice de tema específico"""
    topic_path = os.path.join(base_dir, topic)
    
    if not os.path.exists(topic_path):
        logger.warning(f"Path no existe: {topic_path}")
        return
    
    try:
        idx = index.open_dir(topic_path)
        doc_count = idx.doc_count_all()
        logger.info(f"📊 Índice {topic}:")
        logger.info(f"   - Puntos totales: {doc_count}")
        
        # Contar segmentos (forma segura)
        try:
            reader = idx.reader()
            segment_count = len(reader.leaf_readers()) if hasattr(reader, 'leaf_readers') else "desconocido"
            logger.info(f"   - Segmentos/lectores: {segment_count}")
            reader.close()
        except Exception:
            logger.info(f"   - Segmentos: (no disponible)")
        
        logger.info(f"   ⏳ Optimizando...")
        
        # Optimizar: combina todos los segmentos en uno
        writer = idx.writer(optimize=True)
        writer.commit()
        
        logger.info(f"   ✅ Optimización completada")
        
    except Exception as e:
        logger.error(f"❌ Error optimizando {topic}: {e}", exc_info=True)

def main():
    logger.info(f"🔧 Optimizando índices Whoosh en: {BM25_BASE_DIR}")
    logger.info(f"📚 Temas: {TOPIC_LABELS}\n")
    
    for topic in TOPIC_LABELS:
        optimize_topic_index(BM25_BASE_DIR, topic)
        logger.info("")  # línea en blanco
    
    logger.info("✅ Optimización completada para todos los temas")

if __name__ == "__main__":
    main()







