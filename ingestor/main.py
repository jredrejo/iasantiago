import os, glob
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from settings import *
from chunk import pdf_to_chunks
from pathlib import Path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = QdrantClient(url=QDRANT_URL)


def topic_collection(topic: str) -> str:
    return f"rag_{topic.lower()}"


def ensure_qdrant(topic: str, d: int):
    coll = topic_collection(topic)

    if not client.collection_exists(collection_name=coll):
        logger.info(f"[QDRANT] Creating collection '{coll}' with dimension {d}")
        client.create_collection(
            collection_name=coll,
            vectors_config=models.VectorParams(size=d, distance=models.Distance.COSINE),
        )
    else:
        logger.info(f"[QDRANT] Collection '{coll}' already exists")


def ensure_whoosh(topic: str):
    path = os.path.join(BM25_BASE_DIR, topic)
    os.makedirs(path, exist_ok=True)
    if not index.exists_in(path):
        logger.info(f"[WHOOSH] Creating index at {path}")
        schema = Schema(
            file_path=ID(stored=True),
            page=NUMERIC(stored=True),
            chunk_id=NUMERIC(stored=True),
            text=TEXT(stored=True),
        )
        index.create_in(path, schema)
    else:
        logger.info(f"[WHOOSH] Index at {path} already exists")


def index_pdf(topic: str, pdf_path: str):
    """Index a single PDF file to both Qdrant and Whoosh"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting indexing: {pdf_path}")
    logger.info(f"Topic: {topic}")
    logger.info(f"{'=' * 60}")

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")

    # Setup embedding model
    embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
    logger.info(f"Loading embedding model: {embed_name}")

    try:
        model = SentenceTransformer(
            embed_name,
            trust_remote_code=True,
            device=device,
        )
        dims = model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {dims}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load embedding model: {e}", exc_info=True)
        return

    # Ensure collections/indices exist
    ensure_qdrant(topic, dims)
    ensure_whoosh(topic)

    # Extract text chunks from PDF
    try:
        logger.info(f"Extracting chunks from PDF...")
        chunks = pdf_to_chunks(pdf_path)
        logger.info(f"[OK] Extracted {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"[ERROR] Failed to extract text from PDF: {e}", exc_info=True)
        logger.warning(f"[SKIP] Skipping this file and continuing...")
        return

    texts = [c["text"] for c in chunks]

    # Encode chunks
    logger.info(f"Encoding {len(texts)} chunks...")
    try:
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32 if device == "cuda" else 8,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        # Convert tensor to list for Qdrant
        if torch.is_tensor(vecs):
            vecs = vecs.cpu().tolist()
        elif not isinstance(vecs, list):
            vecs = vecs.tolist()

        logger.info(f"[OK] Encoded {len(vecs)} vectors")
    except Exception as e:
        logger.error(f"[ERROR] Failed to encode chunks: {e}", exc_info=True)
        return

    # Prepare payloads
    payloads = []
    for idx, c in enumerate(chunks):
        payloads.append(
            {
                "file_path": pdf_path,
                "page": c["page"],
                "chunk_id": idx,
                "text": c["text"],
            }
        )

    # Upsert to Qdrant in batches
    QDRANT_BATCH_SIZE = 100
    total_chunks = len(vecs)
    logger.info(
        f"Upserting {total_chunks} vectors to Qdrant in batches of {QDRANT_BATCH_SIZE}..."
    )

    try:
        for batch_start in range(0, total_chunks, QDRANT_BATCH_SIZE):
            batch_end = min(batch_start + QDRANT_BATCH_SIZE, total_chunks)
            batch_ids = list(range(batch_start + 1, batch_end + 1))
            batch_vecs = vecs[batch_start:batch_end]
            batch_payloads = payloads[batch_start:batch_end]

            client.upsert(
                collection_name=topic_collection(topic),
                points=models.Batch(
                    ids=batch_ids, vectors=batch_vecs, payloads=batch_payloads
                ),
            )
            batch_num = batch_start // QDRANT_BATCH_SIZE + 1
            total_batches = (total_chunks + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE
            logger.info(f"  [QDRANT] Batch {batch_num}/{total_batches}")

        logger.info(f"[OK] All vectors uploaded to Qdrant")
    except Exception as e:
        logger.error(f"[ERROR] Failed to upsert to Qdrant: {e}", exc_info=True)
        return

    # Index to Whoosh (BM25)
    logger.info(f"Indexing {len(chunks)} chunks to Whoosh (BM25)...")
    try:
        idx = index.open_dir(os.path.join(BM25_BASE_DIR, topic))
        writer = idx.writer(limitmb=512, procs=0, multisegment=True)

        for i, c in enumerate(chunks):
            writer.update_document(
                file_path=pdf_path, page=c["page"], chunk_id=i, text=c["text"]
            )

        writer.commit()
        logger.info(f"[OK] All chunks indexed to Whoosh")
    except Exception as e:
        logger.error(f"[ERROR] Failed to index to Whoosh: {e}", exc_info=True)
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"[SUCCESS] {pdf_path}")
    logger.info(f"  - Chunks: {len(chunks)}")
    logger.info(f"  - Vectors: {len(vecs)}")
    logger.info(f"  - Topic: {topic}")
    logger.info(f"  - Collection: {topic_collection(topic)}")
    logger.info(f"{'=' * 60}\n")


def initial_scan():
    """Scan all topic directories and index PDFs"""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING INITIAL SCAN")
    logger.info("=" * 60)
    logger.info(f"TOPIC_BASE_DIR: {TOPIC_BASE_DIR}")
    logger.info(f"BM25_BASE_DIR: {BM25_BASE_DIR}")
    logger.info(f"QDRANT_URL: {QDRANT_URL}")

    # Check CUDA availability at startup
    logger.info(f"\nPyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info(f"\nTopics to scan: {', '.join(TOPIC_LABELS)}")
    logger.info("=" * 60 + "\n")

    pdf_count = 0
    error_count = 0

    for t in TOPIC_LABELS:
        tdir = os.path.join(TOPIC_BASE_DIR, t)
        logger.info(f"Scanning topic directory: {tdir}")
        os.makedirs(tdir, exist_ok=True)

        pdfs = glob.glob(os.path.join(tdir, "*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {t}")

        for pdf in pdfs:
            abs_pdf = os.path.abspath(pdf)
            pdf_count += 1
            try:
                index_pdf(t, abs_pdf)
            except Exception as e:
                logger.error(
                    f"[ERROR] Unexpected error processing {abs_pdf}: {e}", exc_info=True
                )
                logger.warning(f"[SKIP] Continuing with next file...")
                error_count += 1

    logger.info("\n" + "=" * 60)
    logger.info("INITIAL SCAN COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total PDFs processed: {pdf_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    initial_scan()
