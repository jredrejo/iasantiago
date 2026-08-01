#!/usr/bin/env python3
"""
§7.4 — construye las colecciones con vector disperso a partir de las vivas.

Por qué hace falta un script y no basta con un flag: **Qdrant 1.15.5 no deja
añadir un vector disperso a una colección ya creada.** Comprobado el
2026-08-01 contra el servidor real:

    PATCH /collections/<c> {"sparse_vectors":{"bm25":{"modifier":"idf"}}}
    -> {"error": "Wrong input: Not existing vector name error: bm25"}

Así que la migración es: colección nueva con denso+disperso, copiar los puntos
(el vector denso se copia tal cual — **no se re-embebe nada, no hace falta GPU
ni re-extraer**), calcular el disperso desde `payload.text`, y conmutar
`QDRANT_COLLECTION_SUFFIX` cuando la eval lo acepte.

**No borra nada.** Las colecciones originales quedan intactas y la vuelta atrás
es poner `QDRANT_COLLECTION_SUFFIX=""`. Borrar las viejas es una decisión
posterior y humana (`PLAN.md` §9).

El denso de estas colecciones es anónimo, así que la nueva se crea igual
(mismo tamaño y distancia) y el código de consulta densa no cambia.

Uso (dentro de un contenedor con la imagen de rag-api y la red de compose):

    docker run --rm --network iasantiago-rag_default \\
        -v /opt/iasantiago-rag/scripts:/scripts:ro \\
        --entrypoint bash iasantiago-rag-rag-api -c \\
        "pip install -q fastembed && python3 /scripts/migrate_sparse.py --topic FOL"

    ... --all            todos los temas
    ... --verify-only    sólo comprueba lo ya migrado, no escribe
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterator, List

from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "bm25")
SPARSE_LANGUAGE = os.getenv("SPARSE_LANGUAGE", "spanish")
TARGET_SUFFIX = os.getenv("QDRANT_COLLECTION_SUFFIX", "_v2")

TOPICS = [
    "electricidad",
    "chemistry",
    "dibujo",
    "latin",
    "programming",
    "afd",
    "mecanica",
    "fol",
    "sostenibilidad",
]


def scroll_points(
    client: QdrantClient, collection: str, batch: int = 512
) -> Iterator[List[models.Record]]:
    """Pagina la colección con payload **y vector denso**: el denso se copia."""
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if records:
            yield records
        if offset is None:
            break


def dense_of(record: models.Record) -> List[float]:
    """Extrae el vector denso, venga anónimo o bajo la clave vacía."""
    vec = record.vector
    if isinstance(vec, dict):
        return vec.get("", next(iter(vec.values())))
    return vec


def migrate_topic(client: QdrantClient, topic: str, verify_only: bool) -> bool:
    from fastembed import SparseTextEmbedding

    source = f"rag_{topic}"
    target = f"rag_{topic}{TARGET_SUFFIX}"

    if not client.collection_exists(source):
        print(f"  ! '{source}' no existe, saltando")
        return False

    src_count = client.count(source, exact=True).count
    print(f"\n=== {topic}: '{source}' ({src_count} puntos) -> '{target}'")

    if verify_only:
        return verify(client, source, target, src_count)

    info = client.get_collection(source)
    params = info.config.params.vectors
    dim, distance = params.size, params.distance

    if client.collection_exists(target):
        print(f"  ! '{target}' ya existe; se borra y se reconstruye")
        client.delete_collection(target)

    client.create_collection(
        collection_name=target,
        vectors_config=models.VectorParams(size=dim, distance=distance),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )
    # El índice de payload que `delete_by_file` necesita no se hereda.
    client.create_payload_index(
        collection_name=target,
        field_name="file_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    embedder = SparseTextEmbedding(model_name="Qdrant/bm25", language=SPARSE_LANGUAGE)

    done = 0
    empty_text = 0
    start = time.time()

    for records in scroll_points(client, source):
        payloads = [r.payload or {} for r in records]
        texts = [p.get("text", "") or "" for p in payloads]
        empty_text += sum(1 for t in texts if not t.strip())

        sparse = list(embedder.embed(texts, batch_size=256))

        points = [
            models.PointStruct(
                id=records[i].id,
                vector={
                    "": dense_of(records[i]),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse[i].indices.tolist(),
                        values=sparse[i].values.tolist(),
                    ),
                },
                payload=payloads[i],
            )
            for i in range(len(records))
        ]

        client.upsert(collection_name=target, points=points, wait=False)
        done += len(points)

        if done % 10240 < len(points):
            rate = done / max(time.time() - start, 1e-6)
            print(f"    {done}/{src_count} ({rate:.0f} pts/s)", flush=True)

    elapsed = time.time() - start
    print(f"  copiados {done} puntos en {elapsed:.0f}s")
    if empty_text:
        print(f"  ! {empty_text} puntos sin texto: su vector disperso va vacío")

    return verify(client, source, target, src_count)


def verify(client: QdrantClient, source: str, target: str, src_count: int) -> bool:
    """
    Comprueba **por contenido**, no por el resumen de la tirada.

    `PLAN.md`: una tirada puede informar de éxito y no haber hecho nada. Aquí
    eso sería una colección con los puntos copiados y el vector disperso vacío,
    que serviría densa perfectamente y devolvería cero en la rama léxica.
    """
    if not client.collection_exists(target):
        print(f"  FALLO: '{target}' no existe")
        return False

    # Qdrant indexa en segundo plano; el conteo puede ir por detrás del upsert.
    for _ in range(30):
        dst_count = client.count(target, exact=True).count
        if dst_count >= src_count:
            break
        time.sleep(2)

    ok = dst_count == src_count
    print(f"  conteo: {dst_count}/{src_count} {'OK' if ok else 'DESCUADRE'}")

    # Muestreo real de la rama dispersa: un punto cualquiera con texto tiene que
    # tener vector disperso no vacío.
    sample, _ = client.scroll(
        collection_name=target, limit=25, with_payload=True, with_vectors=True
    )
    with_sparse = 0
    for r in sample:
        vec = r.vector if isinstance(r.vector, dict) else {}
        sv = vec.get(SPARSE_VECTOR_NAME)
        if sv is not None and len(getattr(sv, "indices", [])) > 0:
            with_sparse += 1
    print(f"  muestra: {with_sparse}/{len(sample)} puntos con vector disperso")

    if sample and with_sparse == 0:
        print("  FALLO: ningún punto de la muestra tiene vector disperso")
        return False

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", action="append", help="tema (repetible)")
    ap.add_argument("--all", action="store_true", help="todos los temas")
    ap.add_argument("--verify-only", action="store_true", help="no escribe")
    args = ap.parse_args()

    if not args.topic and not args.all:
        ap.error("indica --topic <tema> o --all")

    topics = TOPICS if args.all else [t.lower() for t in args.topic]

    client = QdrantClient(url=QDRANT_URL, timeout=600)
    print(f"Qdrant: {QDRANT_URL} · sufijo destino: '{TARGET_SUFFIX}'")

    results: Dict[str, bool] = {}
    for topic in topics:
        try:
            results[topic] = migrate_topic(client, topic, args.verify_only)
        except Exception as e:
            print(f"  ERROR en {topic}: {e}")
            results[topic] = False

    print("\n=== resumen ===")
    for topic, ok in results.items():
        print(f"  {topic:16} {'OK' if ok else 'FALLO'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
