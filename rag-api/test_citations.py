#!/usr/bin/env python3
"""
test_citations.py

Prueba que las citations en el contexto RAG usen la página correcta.
Los chunks deberían mostrar page, NO chunk_id en las citas.

Uso:
    docker exec rag-api python /app/test_citations.py Programming "python variables"
"""

import sys
import logging
from retrieval import hybrid_retrieve, attach_citations_explicit, soft_trim_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_citations(topic: str, query: str):
    """Prueba que attach_citations_explicit use 'page' correctamente"""
    
    logger.info("\n" + "="*80)
    logger.info("🧪 TEST DE CITATIONS")
    logger.info("="*80)
    logger.info(f"Topic: {topic}")
    logger.info(f"Query: {query}\n")
    
    # Retrieve
    logger.info("🔎 Realizando búsqueda...")
    retrieved, meta = hybrid_retrieve(topic, query)
    
    if not retrieved:
        logger.error("❌ No se encontraron resultados")
        return
    
    logger.info(f"✅ Se encontraron {len(retrieved)} chunks\n")
    
    # Mostrar chunks RAW
    logger.info("📋 CHUNKS RAW (antes de attach_citations):")
    for i, chunk in enumerate(retrieved[:3], 1):
        logger.info(f"\n[{i}] {chunk.get('file_path', 'unknown').split('/')[-1]}")
        logger.info(f"    page: {chunk.get('page')}")
        logger.info(f"    chunk_id: {chunk.get('chunk_id')}")
        logger.info(f"    text: {chunk.get('text', '')[:80]}...")
    
    # Soft trim
    logger.info(f"\n\n📏 Aplicando soft_trim_context...")
    from settings import CTX_TOKENS_SOFT_LIMIT
    trimmed = soft_trim_context(retrieved, CTX_TOKENS_SOFT_LIMIT)
    logger.info(f"✅ Después de trim: {len(trimmed)} chunks")
    
    # Attach citations
    logger.info(f"\n\n✍️  Formateando con attach_citations_explicit...")
    context_text, cited_chunks = attach_citations_explicit(trimmed, topic)
    
    # Analizar el contexto formateado
    logger.info("\n" + "="*80)
    logger.info("🔍 ANÁLISIS DE CITATIONS EN EL CONTEXTO")
    logger.info("="*80)
    
    # Buscar patrón [nombre.pdf, p.NUM]
    import re
    citations = re.findall(r"\[([^]]+\.pdf),\s*p\.(\d+)\]", context_text)
    
    logger.info(f"\n📊 Citations encontradas: {len(citations)}")
    
    if not citations:
        logger.error("❌ ¡NO HAY CITATIONS! Algo va mal")
        logger.info(f"\nContext preview:\n{context_text[:500]}")
        return
    
    # Verificar que los números de página coincidan
    logger.info("\n✅ VERIFICANDO CORRESPONDENCIA page ↔ citation:")
    
    all_match = True
    for i, (citation_file, citation_page) in enumerate(citations[:5], 1):
        citation_page_int = int(citation_page)
        
        # Buscar chunk que corresponda
        matching_chunk = None
        for chunk in cited_chunks:
            if citation_file in chunk.get('file_path', ''):
                matching_chunk = chunk
                break
        
        if matching_chunk:
            actual_page = matching_chunk.get('page')
            match = (actual_page == citation_page_int)
            symbol = "✅" if match else "❌"
            logger.info(f"{symbol} [{i}] {citation_file}")
            logger.info(f"      Citation says: p.{citation_page}")
            logger.info(f"      Actual page:   p.{actual_page}")
            
            if not match:
                all_match = False
                logger.error(f"         ⚠️  ¡MISMATCH! {actual_page} ≠ {citation_page_int}")
        else:
            logger.warning(f"⚠️  [{i}] No se encontró chunk para {citation_file}")
    
    logger.info("\n" + "="*80)
    if all_match:
        logger.info("✅ ¡TODAS LAS CITATIONS SON CORRECTAS!")
    else:
        logger.error("❌ ¡HAY MISMATCHES EN LAS CITATIONS!")
        logger.error("   Esto significa que attach_citations_explicit() está usando")
        logger.error("   el valor incorrecto. Revisa que esté usando 'page', no 'chunk_id'")
    logger.info("="*80 + "\n")
    
    # Mostrar ejemplo de citation en contexto
    logger.info("📖 EJEMPLO DE CITATION EN CONTEXTO:\n")
    lines = context_text.split("\n")
    for i, line in enumerate(lines):
        if ".pdf, p." in line:
            logger.info(f"{line}")
            if i > 0:
                logger.info(f"(línea anterior: {lines[i-1][:100]})")
            break


def main():
    if len(sys.argv) < 3:
        print("Uso: python test_citations.py <TOPIC> <QUERY>")
        print('Ej:  python test_citations.py Programming "python variables"')
        sys.exit(1)
    
    topic = sys.argv[1]
    query = sys.argv[2]
    
    test_citations(topic, query)


if __name__ == "__main__":
    main()
