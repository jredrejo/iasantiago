#!/usr/bin/env python3
"""
§7.4 — mide BM25 disperso en Qdrant contra el Whoosh actual, para el go/no-go.

El §7.4 propone ingerir un vector disperso por fragmento y dejar que la Query
API de Qdrant haga la fusión densa+dispersa en el servidor: un único almacén,
escrituras atómicas por documento (arregla la inconsistencia de doble escritura
*por construcción*) y adiós a Whoosh, abandonado desde 2016. La decisión es
humana (§9: "abarca los dos servicios y cambia la ruta de servicio"); lo que
falta para tomarla es saber **si el BM25 disperso recupera al menos tan bien
como el Whoosh que sustituiría**. Eso es lo que mide este script.

Diseño, y por qué así:

- **No reindexa nada.** El texto de cada fragmento ya está en `payload.text` de
  la colección viva, así que los vectores dispersos se construyen leyendo Qdrant.
  No hace falta re-extraer, ni GPU, ni parar la pila web. BM25 es CPU.
- **No toca ninguna colección viva.** Escribe en una colección desechable
  `<colección>_sparse_bench` que borra al terminar (`--keep` para conservarla).
- **Compara sólo la mitad léxica.** Whoosh contra BM25 disperso, sin denso y sin
  reranker. Meter la fusión completa mediría otra cosa: si el híbrido tapa la
  diferencia, el cambio se justifica igual por arquitectura, pero entonces el
  dato que decide no es éste.
- **Puntúa con el evaluador del proyecto** (`rag-api/eval.py`): página, dedup y
  tolerancia ±1. Comparar contra una métrica propia inventada aquí no sería
  comparable con las líneas base del PLAN.

Uso (dentro de un contenedor con la imagen de rag-api, que ya trae whoosh,
qdrant-client y el volumen de índices montado):

    docker run --rm --network iasantiago-rag_default \\
        -v /opt/iasantiago-rag/rag-api:/app:ro \\
        -v /opt/iasantiago-rag/scripts:/scripts:ro \\
        -v /opt/iasantiago-rag/eval:/eval:ro \\
        -v /opt/iasantiago-rag/data/whoosh:/whoosh:ro \\
        -w /app --entrypoint bash iasantiago-rag-rag-api -c \\
        "pip install -q fastembed && python3 /scripts/bench_sparse_vs_whoosh.py --topic FOL"

`fastembed` se instala al vuelo a propósito: si el go/no-go sale "no", no queda
una dependencia nueva en `requirements.txt` que alguien tenga que quitar luego.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "/app")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
BM25_BASE_DIR = os.getenv("BM25_BASE_DIR", "/whoosh")
SPARSE_VECTOR_NAME = "bm25"


def load_golden(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: se esperaba una lista de casos")
    return data


def scroll_all(client, collection: str) -> List[Dict]:
    """Todos los puntos con payload, sin vectores. Es la fuente del texto."""
    points: List[Dict] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in batch:
            payload = p.payload or {}
            if payload.get("text"):
                points.append(
                    {
                        "id": p.id,
                        "text": payload["text"],
                        "file_path": payload.get("file_path", ""),
                        "page": payload.get("page", 0),
                        "chunk_id": payload.get("chunk_id", 0),
                    }
                )
        if offset is None:
            break
    return points


def build_sparse_collection(client, models, source: str, target: str, points: List[Dict]):
    from fastembed import SparseTextEmbedding

    # Stemmer y stopwords en español: sin ellos "instalaciones" y "instalación"
    # son términos distintos y el BM25 pierde justo lo que Whoosh sí resuelve
    # con su StemmingAnalyzer. Comparar sin esto mediría la falta de stemmer.
    embedder = SparseTextEmbedding(model_name="Qdrant/bm25", language="spanish")

    print(f"  · vectorizando {len(points)} fragmentos (BM25, CPU)...", flush=True)
    start = time.time()
    vectors = list(embedder.embed([p["text"] for p in points], batch_size=256))
    print(f"    {time.time() - start:.1f}s", flush=True)

    if client.collection_exists(target):
        client.delete_collection(target)
    client.create_collection(
        collection_name=target,
        vectors_config={},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )

    batch = []
    for point, vec in zip(points, vectors):
        batch.append(
            models.PointStruct(
                id=point["id"],
                vector={
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=vec.indices.tolist(), values=vec.values.tolist()
                    )
                },
                payload={
                    "file_path": point["file_path"],
                    "page": point["page"],
                    "chunk_id": point["chunk_id"],
                },
            )
        )
        if len(batch) >= 512:
            client.upsert(collection_name=target, points=batch, wait=True)
            batch = []
    if batch:
        client.upsert(collection_name=target, points=batch, wait=True)

    print(f"  · colección desechable {target} lista", flush=True)
    return embedder


def search_sparse(client, models, collection: str, embedder, query: str, topk: int) -> List[Dict]:
    vec = next(iter(embedder.query_embed(query)))
    hits = client.query_points(
        collection_name=collection,
        query=models.SparseVector(
            indices=vec.indices.tolist(), values=vec.values.tolist()
        ),
        using=SPARSE_VECTOR_NAME,
        limit=topk,
        with_payload=True,
    ).points
    return [
        {
            "file_path": h.payload.get("file_path", ""),
            "page": h.payload.get("page", 0),
            "chunk_id": h.payload.get("chunk_id", 0),
        }
        for h in hits
    ]


def search_whoosh(topic: str, query: str, topk: int) -> List[Dict]:
    from bm25_utils import bm25_search_safe

    return bm25_search_safe(BM25_BASE_DIR, topic, query, topk)


def score(cases: List[Dict], retrieved_per_case: List[List[Dict]]) -> Dict:
    from eval import aggregate_eval

    queries = [
        {
            "query": c["query"],
            "topic": c.get("topic", ""),
            "relevant_pages": c.get("relevant_pages", []),
            "relevant_files": c.get("relevant_files", []),
            "retrieved": got,
        }
        for c, got in zip(cases, retrieved_per_case)
    ]
    return aggregate_eval(queries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", required=True, help="p.ej. FOL")
    ap.add_argument("--golden", help="por defecto /eval/golden_<topic en minúsculas>.json")
    ap.add_argument("--collection", help="por defecto rag_<topic en minúsculas>")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--keep", action="store_true", help="no borrar la colección desechable")
    ap.add_argument("--json", dest="json_out", help="volcado de resultados")
    args = ap.parse_args()

    topic_lc = args.topic.lower()
    golden_path = Path(args.golden or f"/eval/golden_{topic_lc}.json")
    source = args.collection or f"rag_{topic_lc}"
    target = f"{source}_sparse_bench"

    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=QDRANT_URL, timeout=120)

    cases = load_golden(golden_path)
    print(f"§7.4 — {args.topic}: {len(cases)} consultas de {golden_path}")

    points = scroll_all(client, source)
    print(f"  · {len(points)} fragmentos con texto en {source}")
    if not points:
        print("la colección no tiene texto en el payload: nada que comparar", file=sys.stderr)
        return 2

    embedder = build_sparse_collection(client, models, source, target, points)

    try:
        sparse_got, whoosh_got = [], []
        t_sparse = t_whoosh = 0.0
        for case in cases:
            q = case["query"]

            start = time.time()
            sparse_got.append(search_sparse(client, models, target, embedder, q, args.topk))
            t_sparse += time.time() - start

            start = time.time()
            whoosh_got.append(search_whoosh(args.topic, q, args.topk))
            t_whoosh += time.time() - start

        sparse_scores = score(cases, sparse_got)
        whoosh_scores = score(cases, whoosh_got)
    finally:
        if not args.keep:
            client.delete_collection(target)
            print(f"  · colección desechable {target} borrada")

    n = len(cases)
    print(f"\n  Sólo la mitad léxica: sin denso, sin reranker, top-{args.topk}\n")
    header = f"  {'métrica':<16}{'Whoosh':>12}{'Qdrant sparse':>16}{'delta':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for family, keys in (("pages", ["Recall@1", "Recall@3", "Recall@5", "MRR"]),):
        for key in keys:
            w = whoosh_scores[family].get(key)
            s = sparse_scores[family].get(key)
            if w is None or s is None:
                continue
            print(f"  {key:<16}{w:>12.4f}{s:>16.4f}{s - w:>+10.4f}")
    print(f"  {'ms/consulta':<16}{t_whoosh / n * 1000:>12.1f}{t_sparse / n * 1000:>16.1f}")
    print(f"\n  n = {whoosh_scores['pages'].get('n', 0)} consultas con verdad de referencia de página")

    print("\n  Cómo leerlo:")
    print(
        "  · Un empate ya es un GO por arquitectura: un almacén en vez de dos, "
        "escritura\n    atómica por documento, y fuera una dependencia abandonada "
        "desde 2016."
    )
    print(
        "  · Una pérdida clara es un NO-GO aunque la arquitectura sea mejor: el "
        "híbrido\n    apoya en la mitad léxica las consultas con siglas y "
        "referencias normativas."
    )
    print(
        "  · Esto es un tema. Antes de decidir, repítelo en Electricidad "
        "(el 78 % del\n    corpus) y en Latin (el peor caso, bilingüe)."
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "topic": args.topic,
                    "n": n,
                    "topk": args.topk,
                    "whoosh": whoosh_scores,
                    "sparse": sparse_scores,
                    "ms_per_query": {
                        "whoosh": t_whoosh / n * 1000,
                        "sparse": t_sparse / n * 1000,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  crudos en {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
