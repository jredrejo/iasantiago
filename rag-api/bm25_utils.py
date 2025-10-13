from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
from whoosh.qparser import QueryParser
import os
from typing import List, Dict


def topic_index_dir(base: str, topic: str) -> str:
    return os.path.join(base, topic)


def ensure_whoosh_index(base: str, topic: str):
    path = topic_index_dir(base, topic)
    os.makedirs(path, exist_ok=True)
    if not index.exists_in(path):
        schema = Schema(
            file_path=ID(stored=True),
            page=NUMERIC(stored=True),
            chunk_id=NUMERIC(stored=True),
            text=TEXT(stored=True),
        )
        index.create_in(path, schema)


def add_docs(base: str, topic: str, docs: List[Dict]):
    idx = index.open_dir(topic_index_dir(base, topic))
    writer = idx.writer(limitmb=512, procs=0, multisegment=True)
    for d in docs:
        writer.update_document(
            file_path=d["file_path"],
            page=d["page"],
            chunk_id=d["chunk_id"],
            text=d["text"],
        )
    writer.commit()


def bm25_search(base: str, topic: str, query: str, topk: int) -> List[Dict]:
    idx = index.open_dir(topic_index_dir(base, topic))
    qp = QueryParser("text", schema=idx.schema)
    q = qp.parse(query)
    with idx.searcher() as s:
        res = s.search(q, limit=topk)
        hits = []
        for r in res:
            hits.append(
                dict(
                    file_path=r["file_path"],
                    page=r["page"],
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r.score,
                )
            )
        return hits
