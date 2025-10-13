from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
import hashlib
from settings import QDRANT_URL

client = QdrantClient(url=QDRANT_URL)


def topic_collection(topic: str) -> str:
    # un nombre de colección por tema
    return f"rag_{topic.lower()}"


def ensure_collection(topic: str, vector_size: int):
    coll = topic_collection(topic)
    existing = [c.name for c in client.get_collections().collections]
    if coll not in existing:
        client.recreate_collection(
            collection_name=coll,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            replication_factor=1,
            write_consistency_factor=1,
        )


def upsert_points(
    topic: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]
):
    coll = topic_collection(topic)
    ids = [
        int(
            hashlib.md5((p["file_path"] + str(p["chunk_id"])).encode()).hexdigest()[
                :12
            ],
            16,
        )
        for p in payloads
    ]
    client.upsert(
        collection_name=coll,
        points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
    )


def search_dense(topic: str, vector: List[float], topk: int):
    coll = topic_collection(topic)
    res = client.search(
        collection_name=coll, query_vector=vector, limit=topk, with_payload=True
    )
    return res
