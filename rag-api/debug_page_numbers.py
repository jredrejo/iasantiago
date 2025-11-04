#!/usr/bin/env python3
"""
debug_page_numbers.py

Script para verificar que los números de página se extraen correctamente
del PDF y se almacenan bien en Qdrant.

Uso:
    docker exec rag-api python /app/debug_page_numbers.py Programming
"""

import sys
import logging
from collections import defaultdict
from qdrant_utils import client, topic_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_page_numbers_in_qdrant(topic: str):
    """Verifica los números de página almacenados en Qdrant"""
    
    logger.info("\n" + "="*80)
    logger.info(f"🔍 VERIFICACIÓN DE NÚMEROS DE PÁGINA EN QDRANT")
    logger.info(f"Topic: {topic}")
    logger.info("="*80)
    
    coll = topic_collection(topic)
    
    try:
        points, _ = client.scroll(collection_name=coll, limit=1000, with_payload=True)
        
        if not points:
            logger.error(f"❌ No hay puntos en {coll}")
            return
        
        logger.info(f"\n📊 Analizando {len(points)} puntos...")
        
        # Agrupar por archivo
        file_pages = defaultdict(set)
        page_counts = defaultdict(int)
        
        for point in points:
            payload = point.payload
            file_path = payload.get("file_path", "unknown")
            page = payload.get("page", 0)
            
            filename = file_path.split("/")[-1]
            file_pages[filename].add(page)
            page_counts[page] += 1
        
        # Mostrar por archivo
        logger.info(f"\n📄 Por archivo:")
        for filename in sorted(file_pages.keys()):
            pages = sorted(file_pages[filename])
            min_page = min(pages)
            max_page = max(pages)
            num_chunks = len(pages)
            logger.info(f"   {filename}:")
            logger.info(f"      - Chunks: {num_chunks}")
            logger.info(f"      - Páginas: {min_page}-{max_page}")
            logger.info(f"      - Sample pages: {pages[:10]}...")
        
        # Verificar rangos razonables
        logger.info(f"\n⚠️  VERIFICACIONES:")
        all_pages = list(page_counts.keys())
        min_p = min(all_pages)
        max_p = max(all_pages)
        
        if min_p <= 0:
            logger.error(f"   ❌ ¡Hay páginas <=0! (mín: {min_p})")
        else:
            logger.info(f"   ✅ Páginas comienzan en {min_p}")
        
        if max_p > 2000:
            logger.warning(f"   ⚠️  Páginas muy altas (máx: {max_p}) - ¿numéricas?")
        else:
            logger.info(f"   ✅ Rango de páginas razonable (máx: {max_p})")
        
        # Distribución
        logger.info(f"\n📈 Distribución de chunks por página:")
        sorted_pages = sorted(page_counts.items(), key=lambda x: -x[1])
        for page, count in sorted_pages[:10]:
            logger.info(f"   p.{page}: {count} chunks")
        
        if page_counts[sorted_pages[0][0]] > 100:
            logger.warning(f"   ⚠️  ¡Una página tiene muchos chunks ({sorted_pages[0][1]})!")
            logger.warning(f"       Posible: todos los chunks tienen la misma página")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


def sample_chunks_by_page(topic: str, sample_page: int = None):
    """Muestra chunks de ejemplo para una página específica"""
    
    logger.info("\n" + "="*80)
    logger.info(f"📝 CHUNKS DE EJEMPLO")
    logger.info("="*80)
    
    coll = topic_collection(topic)
    
    try:
        points, _ = client.scroll(collection_name=coll, limit=1000, with_payload=True)
        
        if not points:
            logger.error(f"❌ No hay puntos")
            return
        
        # Obtener páginas únicas
        pages = set(p.payload.get("page", 0) for p in points)
        
        if not sample_page:
            sample_page = sorted(pages)[len(pages)//2]  # Página del medio
        
        logger.info(f"\n🔎 Chunks de la página {sample_page}:")
        
        matching_chunks = [p for p in points if p.payload.get("page") == sample_page]
        
        if not matching_chunks:
            logger.warning(f"   No hay chunks en p.{sample_page}")
            logger.info(f"   Páginas disponibles: {sorted(list(pages))[:20]}")
            return
        
        for i, point in enumerate(matching_chunks[:3], 1):
            payload = point.payload
            filename = payload.get("file_path", "").split("/")[-1]
            text_preview = payload.get("text", "")[:150].replace("\n", " ")
            chunk_id = payload.get("chunk_id", "?")
            
            logger.info(f"\n   [{i}] {filename}")
            logger.info(f"       chunk_id: {chunk_id}")
            logger.info(f"       page: {payload.get('page')}")
            logger.info(f"       text: {text_preview}...")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


def main():
    if len(sys.argv) < 2:
        print("Uso: python debug_page_numbers.py <TOPIC>")
        print(f"Ej:  python debug_page_numbers.py Programming")
        sys.exit(1)
    
    topic = sys.argv[1]
    sample_page = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    check_page_numbers_in_qdrant(topic)
    sample_chunks_by_page(topic, sample_page)
    
    logger.info("\n" + "="*80)
    logger.info("✅ DEBUG COMPLETADO")
    logger.info("="*80)


if __name__ == "__main__":
    main()
