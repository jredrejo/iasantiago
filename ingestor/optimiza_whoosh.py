#!/usr/bin/env python3
"""
Optimiza todos los índices Whoosh para mejorar velocidad de búsqueda
Ejecutar cuando el sistema esté inactivo (cron nocturno)
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
        logger.info(f"📊 Índice {topic}:")
        logger.info(f"   - Puntos totales: {idx.doc_count_all()}")
        logger.info(f"   - Segmentos antes: {len(idx._get_segment_picker().segment_numbers())}")
        
        # Optimizar
        writer = idx.writer()
        writer.commit(optimize=True)  # ← CLAVE: combina todos los segmentos
        
        idx = index.open_dir(topic_path)
        logger.info(f"   ✅ Segmentos después: {len(idx._get_segment_picker().segment_numbers())}")
        logger.info(f"   ✅ Optimización completada")
        
    except Exception as e:
        logger.error(f"❌ Error optimizando {topic}: {e}", exc_info=True)

def main():
    logger.info(f"🔧 Optimizando índices Whoosh en: {BM25_BASE_DIR}")
    logger.info(f"📚 Temas: {TOPIC_LABELS}")
    
    for topic in TOPIC_LABELS:
        optimize_topic_index(BM25_BASE_DIR, topic)
    
    logger.info("✅ Optimización completada")

if __name__ == "__main__":
    main()
