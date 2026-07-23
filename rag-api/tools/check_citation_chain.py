#!/usr/bin/env python3
# rag-api/tools/check_citation_chain.py
"""
Check the complete citation chain: Qdrant → Retrieval → Citations
Verifies that page numbers are correct end-to-end.

Manual diagnostic script (not a pytest test). Run inside the container:
    docker exec rag-api python /app/tools/check_citation_chain.py
"""

import logging
import sys
from pathlib import Path

# Permite ejecutar el script desde tools/ resolviendo los imports del paquete rag-api.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_utils import client, topic_collection
from retrieval import attach_citations, choose_retrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_citation_integrity(topic: str, query: str):
    """
    Test that citations match actual page numbers in Qdrant
    """

    logger.info("\n" + "=" * 80)
    logger.info("🔍 CITATION INTEGRITY TEST")
    logger.info("=" * 80)
    logger.info(f"Topic: {topic}")
    logger.info(f"Query: {query}\n")

    # Step 1: Retrieve chunks
    logger.info("Step 1: Retrieving chunks...")
    retrieved, meta = choose_retrieval(topic, query, is_generative=False)

    if not retrieved:
        logger.error("❌ No chunks retrieved")
        return False

    logger.info(f"✓ Retrieved {len(retrieved)} chunks\n")

    # Step 2: Verify Qdrant data
    logger.info("Step 2: Verifying Qdrant data...")
    coll = topic_collection(topic)

    all_valid = True

    for i, chunk in enumerate(retrieved[:5], 1):
        file_path = chunk.get("file_path")
        chunk_id = chunk.get("chunk_id")
        page_in_chunk = chunk.get("page")

        logger.info(f"\n[{i}] {Path(file_path).name}")
        logger.info(f"    chunk_id: {chunk_id}")
        logger.info(f"    page (in chunk): {page_in_chunk}")

        # Verify against Qdrant
        try:
            # Search for this specific chunk in Qdrant
            points, _ = client.scroll(
                collection_name=coll,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path", match=models.MatchValue(value=file_path)
                        ),
                        models.FieldCondition(
                            key="chunk_id", match=models.MatchValue(value=chunk_id)
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            if points:
                qdrant_page = points[0].payload.get("page")
                logger.info(f"    page (in Qdrant): {qdrant_page}")

                if page_in_chunk == qdrant_page:
                    logger.info(f"    ✅ MATCH")
                else:
                    logger.error(f"    ❌ MISMATCH: {page_in_chunk} ≠ {qdrant_page}")
                    all_valid = False
            else:
                logger.warning(f"    ⚠️  Not found in Qdrant")
                all_valid = False

        except Exception as e:
            logger.error(f"    ❌ Error checking Qdrant: {e}")
            all_valid = False

    # Step 3: Check citations format
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Checking citation format...")

    context_text, cited_chunks = attach_citations(retrieved, topic)

    # Extract citations from context
    import re

    citations = re.findall(r"\[([^]]+\.pdf),\s*p\.(\d+)\]\(([^)]+)\)", context_text)

    logger.info(f"Found {len(citations)} citations in context\n")

    for i, (filename, page_str, url) in enumerate(citations[:5], 1):
        page_num = int(page_str)

        # Find corresponding chunk
        matching_chunk = None
        for chunk in cited_chunks:
            if filename in chunk.get("file_path", ""):
                matching_chunk = chunk
                break

        if matching_chunk:
            actual_page = matching_chunk.get("page")
            match = actual_page == page_num
            symbol = "✅" if match else "❌"

            logger.info(f"{symbol} Citation {i}: [{filename}, p.{page_str}]")
            logger.info(f"   URL: {url}")
            logger.info(f"   Chunk page: {actual_page}")
            logger.info(f"   Citation page: {page_num}")

            if not match:
                logger.error(f"   ⚠️  PAGE MISMATCH!")
                all_valid = False
        else:
            logger.warning(f"⚠️  Citation {i}: No matching chunk found")
            all_valid = False

    # Final verdict
    logger.info("\n" + "=" * 80)
    if all_valid:
        logger.info("✅ ALL CITATIONS ARE CORRECT")
    else:
        logger.error("❌ CITATION ERRORS DETECTED")
        logger.error("\nPossible causes:")
        logger.error("  1. Page numbers wrong in Qdrant (ingestor issue)")
        logger.error("  2. Page numbers lost during retrieval")
        logger.error("  3. Citation formatting using wrong field")
    logger.info("=" * 80 + "\n")

    return all_valid


def test_specific_file(topic: str, filename: str, expected_page: int):
    """
    Test a specific file to verify its page numbers in Qdrant
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"🔍 TESTING SPECIFIC FILE")
    logger.info("=" * 80)
    logger.info(f"File: {filename}")
    logger.info(f"Expected page: {expected_page}\n")

    coll = topic_collection(topic)

    # Get all chunks from this file
    points, _ = client.scroll(
        collection_name=coll,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path", match=models.MatchText(text=filename)
                )
            ]
        ),
        limit=100,
        with_payload=True,
    )

    if not points:
        logger.error(f"❌ No chunks found for {filename}")
        return False

    logger.info(f"Found {len(points)} chunks from {filename}\n")

    # Check page distribution
    pages = {}
    for point in points:
        page = point.payload.get("page")
        pages[page] = pages.get(page, 0) + 1

    logger.info("Page distribution:")
    for page in sorted(pages.keys()):
        count = pages[page]
        marker = "✅" if page == expected_page else "⚠️ "
        logger.info(f"  {marker} Page {page}: {count} chunks")

    # Check for suspicious patterns
    if len(pages) == 1:
        logger.warning("\n⚠️  All chunks have the same page number!")
        logger.warning("    This might indicate page detection failed")

    if any(p < 1 for p in pages.keys()):
        logger.error("\n❌ Found invalid page numbers (<1)")

    if expected_page not in pages:
        logger.error(f"\n❌ Expected page {expected_page} not found!")
        return False

    logger.info("\n✅ Test complete")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Test citations: python test_citation_chain.py <topic> <query>")
        print(
            "  Test file: python test_citation_chain.py <topic> --file <filename> <expected_page>"
        )
        sys.exit(1)

    topic = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == "--file":
        filename = sys.argv[3]
        expected_page = int(sys.argv[4])
        success = test_specific_file(topic, filename, expected_page)
    else:
        query = sys.argv[2]
        success = test_citation_integrity(topic, query)

    sys.exit(0 if success else 1)
