import os
from typing import Dict, List

# ============================================================
# NORMALIZACIÓN DE REFERENCIAS
# ============================================================
#
# El ground truth se escribe a mano y retrieval devuelve rutas de contenedor
# (/topics/Tema/fichero.pdf). Históricamente los casos usaban rutas absolutas
# del host (/opt/iasantiago-rag/topics/...), así que NINGUNA referencia casaba
# y las métricas daban 0.0 pasara lo que pasara, sin avisar de nada.
# Comparar por nombre base hace la evaluación independiente del formato.


def normalize_file(ref: str) -> str:
    """'/opt/.../tema/fichero.pdf#12' -> 'fichero.pdf'"""
    return os.path.basename(ref.split("#", 1)[0].strip())


def normalize_page(ref: str) -> str:
    """'/opt/.../fichero.pdf#12' -> 'fichero.pdf#12'"""
    path, sep, page = ref.partition("#")
    if not sep:
        # Sin '#' no es una referencia de página. Se devuelve una clave que no
        # puede casar con ninguna página real, y el endpoint lo denuncia como
        # ground truth malformado en vez de puntuar 0.0 en silencio.
        return f"{os.path.basename(path.strip())}#?"
    return f"{os.path.basename(path.strip())}#{page.strip()}"


def dedupe_files(chunks: List[Dict]) -> List[str]:
    """Archivos recuperados, sin repetidos, en orden de recuperación."""
    return list(dict.fromkeys(normalize_file(c["file_path"]) for c in chunks))


def dedupe_pages(chunks: List[Dict]) -> List[str]:
    """
    Páginas recuperadas, sin repetidas, en orden de recuperación.

    La PÁGINA es la unidad primaria de evaluación. A diferencia del chunk, es
    invariante frente a un recorte distinto (una página es una página se trocee
    como se trocee), así que un cambio de chunker no mueve la métrica por sí
    solo. Y a diferencia del archivo, discrimina de verdad: acertar el PDF de un
    manual de 300 páginas no dice casi nada.
    """
    return list(
        dict.fromkeys(f"{normalize_file(c['file_path'])}#{c['page']}" for c in chunks)
    )


# ============================================================
# MÉTRICAS
# ============================================================


def recall_at_k(pred: List[str], relevant: List[str], k: int) -> float:
    """¿Hay algún elemento relevante entre los k primeros?"""
    return 1.0 if any(p in relevant for p in pred[:k]) else 0.0


def mrr(pred: List[str], relevant: List[str]) -> float:
    """Inverso de la posición del primer elemento relevante."""
    for i, p in enumerate(pred, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def _score_family(
    preds: List[List[str]], truths: List[List[str]], k_list: List[int]
) -> Dict:
    """
    Agrega Recall@k y MRR sobre los casos que TIENEN ground truth de esta
    familia. Los casos sin ground truth se excluyen en vez de puntuar 0.0:
    contarlos como fallo hundiría la media y haría creer que el retrieval es
    peor de lo que es.
    """
    pairs = [(p, t) for p, t in zip(preds, truths) if t]
    out = {"n": len(pairs)}
    if not pairs:
        return out

    n = len(pairs)
    for k in k_list:
        out[f"Recall@{k}"] = round(
            sum(recall_at_k(p, t, k) for p, t in pairs) / n, 4
        )
    out["MRR"] = round(sum(mrr(p, t) for p, t in pairs) / n, 4)
    return out


def aggregate_eval(
    queries: List[Dict],
    page_k_list=[1, 3, 5, 10],
    file_k_list=[1, 3, 5],
) -> Dict:
    """
    Agrega métricas de evaluación sobre múltiples queries.

    queries: [{query, topic, relevant_files, relevant_pages, retrieved}]

    Devuelve dos familias:
      - `pages`: métrica PRIMARIA, la que debe decidir si un cambio de chunking
        o de retrieval mejora o empeora.
      - `files`: métrica secundaria y gruesa ("¿encontró el documento?"). Su
        k se queda en 5 a propósito: con MAX_CHUNKS_PER_FILE=10 y FINAL_TOPK=18
        un solo documento fuerte puede ocupar 10 de las 18 plazas, así que rara
        vez aparecen más de 4-6 archivos distintos y Recall@10 sería inalcanzable
        por construcción.
    """
    page_preds = [dedupe_pages(q["retrieved"]) for q in queries]
    file_preds = [dedupe_files(q["retrieved"]) for q in queries]

    page_truths = [
        [normalize_page(r) for r in q.get("relevant_pages", [])] for q in queries
    ]
    file_truths = [
        [normalize_file(r) for r in q.get("relevant_files", [])] for q in queries
    ]

    return {
        "n": len(queries),
        "pages": _score_family(page_preds, page_truths, page_k_list),
        "files": _score_family(file_preds, file_truths, file_k_list),
    }
