import os, glob
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from settings import *
from chunk import pdf_to_chunks
from pathlib import Path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC

client = QdrantClient(url=QDRANT_URL)


def topic_collection(topic: str) -> str:
    return f"rag_{topic.lower()}"


def ensure_qdrant(topic: str, d: int):
    coll = topic_collection(topic)

    if not client.collection_exists(collection_name=coll):
        client.create_collection(
            collection_name=coll,
            vectors_config=models.VectorParams(size=d, distance=models.Distance.COSINE),
        )


def ensure_whoosh(topic: str):
    path = os.path.join(BM25_BASE_DIR, topic)
    os.makedirs(path, exist_ok=True)
    if not index.exists_in(path):
        schema = Schema(
            file_path=ID(stored=True),
            page=NUMERIC(stored=True),
            chunk_id=NUMERIC(stored=True),
            text=TEXT(stored=True),
        )
        index.create_in(path, schema)


def index_pdf(topic: str, pdf_path: str):
    # Try CUDA with NVIDIA container's PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")

    embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
    model = SentenceTransformer(
        embed_name,
        trust_remote_code=True,
        device=device,
    )
    dims = model.get_sentence_embedding_dimension()
    ensure_qdrant(topic, dims)
    ensure_whoosh(topic)

    chunks = pdf_to_chunks(pdf_path)
    texts = [c["text"] for c in chunks]

    print(f"Encoding {len(texts)} chunks...")
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

    # Qdrant upsert
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
    ids = list(range(1, len(vecs) + 1))
    client.upsert(
        collection_name=topic_collection(topic),
        points=models.Batch(ids=ids, vectors=vecs, payloads=payloads),
    )

    # Whoosh
    idx = index.open_dir(os.path.join(BM25_BASE_DIR, topic))
    writer = idx.writer(limitmb=512, procs=0, multisegment=True)
    for i, c in enumerate(chunks):
        writer.update_document(
            file_path=pdf_path, page=c["page"], chunk_id=i, text=c["text"]
        )
    writer.commit()
    print(f"[OK] {pdf_path} -> {len(chunks)} chunks")


def initial_scan():
    print("Starting initial scan...")
    print(f"TOPIC_BASE_DIR: {TOPIC_BASE_DIR}")

    # Check CUDA availability at startup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    for t in TOPIC_LABELS:
        tdir = os.path.join(TOPIC_BASE_DIR, t)
        print(f"Topic directory: {tdir}")
        os.makedirs(tdir, exist_ok=True)
        for pdf in glob.glob(os.path.join(tdir, "*.pdf")):
            print(f"Processing {pdf}")
            index_pdf(t, os.path.abspath(pdf))


if __name__ == "__main__":
    initial_scan()
    print("Ingestor initial scan finished.")
