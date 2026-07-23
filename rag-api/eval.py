import hashlib
import os
from typing import Callable, Dict, List, Optional, Tuple

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


def _split_page_ref(key: str) -> Tuple[str, Optional[int]]:
    """'fichero.pdf#12' -> ('fichero.pdf', 12).

    Si no hay '#' o la página no es numérica (p.ej. el centinela '#?' que
    `normalize_page` pone en un ground truth malformado) devuelve
    (nombre, None), que nunca casa por número.
    """
    file, sep, page = key.partition("#")
    if not sep:
        return key, None
    try:
        return file, int(page.strip())
    except ValueError:
        return file, None


# ============================================================
# EQUIVALENCIA DE FICHEROS Y PÁGINAS
# ============================================================
#
# Dos ajustes que evitan regresiones fantasma, ambos aprendidos a base de
# perder tiempo (PLAN §3.-1, §3.2, §10):
#
#  - DUPLICADOS: media de Química eran copias byte a byte del mismo libro con
#    dos nombres. Si el golden nombra una copia y el retrieval devuelve la
#    gemela, la casación exacta puntúa 0 estando bien. `file_aliases` mapea cada
#    nombre a un canónico por grupo de hash, así que las copias son
#    intercambiables.
#  - ±1 PÁGINA: re-trocear mueve un pasaje a la página contigua sin empeorar el
#    retrieval (el HPLC que el golden situaba en `...#346` quedó en 345/347).
#    Sin tolerancia, cada cambio de chunking finge una regresión.


def _canonical(name: str, aliases: Optional[Dict[str, str]]) -> str:
    """Nombre canónico del fichero dentro de su grupo de duplicados."""
    return aliases.get(name, name) if aliases else name


def make_file_matcher(
    aliases: Optional[Dict[str, str]] = None,
) -> Callable[[str, str], bool]:
    """Casan dos ficheros si son el mismo o copias byte-idénticas."""

    def match(pred: str, truth: str) -> bool:
        return _canonical(pred, aliases) == _canonical(truth, aliases)

    return match


def make_page_matcher(
    tolerance: int = 1, aliases: Optional[Dict[str, str]] = None
) -> Callable[[str, str], bool]:
    """Casan dos páginas si son del mismo fichero (o copia byte-idéntica) y sus
    números distan <= `tolerance`. Con `tolerance=0` se recupera la casación
    exacta de siempre."""

    def match(pred: str, truth: str) -> bool:
        pf, pp = _split_page_ref(pred)
        tf, tp = _split_page_ref(truth)
        if _canonical(pf, aliases) != _canonical(tf, aliases):
            return False
        if pp is None or tp is None:
            # Página no numérica (ground truth malformado): no inventes casación.
            return pred == truth
        return abs(pp - tp) <= tolerance

    return match


def build_content_alias_map(paths_by_name: Dict[str, str]) -> Dict[str, str]:
    """Mapea nombre-de-fichero -> nombre canónico de su grupo de duplicados
    byte-idénticos (§3.-1).

    `paths_by_name`: {basename: ruta_legible}. Los ficheros ilegibles o
    inexistentes se ignoran (no se agrupan); nunca se lanza excepción: la
    evaluación no debe caerse porque un PDF no esté montado. Sólo entran en un
    grupo los nombres DISTINTOS con el mismo hash — un fichero consigo mismo no
    genera alias.
    """
    by_hash: Dict[str, List[str]] = {}
    for name, path in paths_by_name.items():
        digest = _md5_file(path)
        if digest is None:
            continue
        by_hash.setdefault(digest, []).append(name)

    aliases: Dict[str, str] = {}
    for names in by_hash.values():
        if len(names) > 1:
            canon = min(names)  # determinista, independiente del orden de escaneo
            for n in names:
                aliases[n] = canon
    return aliases


def _md5_file(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """MD5 de un fichero, o None si no se puede leer."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk_size), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# ============================================================
# MÉTRICAS
# ============================================================

# Casador por defecto: igualdad exacta (compatibilidad con cualquier llamada
# que no pase un matcher; los tests directos lo usan).
_EXACT: Callable[[str, str], bool] = lambda a, b: a == b  # noqa: E731


def recall_at_k(
    pred: List[str],
    relevant: List[str],
    k: int,
    match: Callable[[str, str], bool] = _EXACT,
) -> float:
    """¿Hay algún elemento relevante entre los k primeros?"""
    top = pred[:k]
    return 1.0 if any(match(p, t) for t in relevant for p in top) else 0.0


def mrr(
    pred: List[str],
    relevant: List[str],
    match: Callable[[str, str], bool] = _EXACT,
) -> float:
    """Inverso de la posición del primer elemento relevante."""
    for i, p in enumerate(pred, start=1):
        if any(match(p, t) for t in relevant):
            return 1.0 / i
    return 0.0


def _score_family(
    preds: List[List[str]],
    truths: List[List[str]],
    k_list: List[int],
    match: Callable[[str, str], bool] = _EXACT,
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
            sum(recall_at_k(p, t, k, match) for p, t in pairs) / n, 4
        )
    out["MRR"] = round(sum(mrr(p, t, match) for p, t in pairs) / n, 4)
    return out


def aggregate_eval(
    queries: List[Dict],
    page_k_list=[1, 3, 5, 10],
    file_k_list=[1, 3, 5],
    page_tolerance: int = 1,
    file_aliases: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Agrega métricas de evaluación sobre múltiples queries.

    queries: [{query, topic, relevant_files, relevant_pages, retrieved}]

    `page_tolerance` (por defecto 1) y `file_aliases` (mapa de duplicados
    byte-idénticos) evitan las regresiones fantasma descritas en PLAN §3.-1 y
    §3.2. Con `page_tolerance=0` y `file_aliases=None` la métrica es la casación
    exacta de antes.

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

    page_match = make_page_matcher(page_tolerance, file_aliases)
    file_match = make_file_matcher(file_aliases)

    return {
        "n": len(queries),
        "page_tolerance": page_tolerance,
        "duplicate_groups": len(set(file_aliases.values())) if file_aliases else 0,
        "pages": _score_family(page_preds, page_truths, page_k_list, page_match),
        "files": _score_family(file_preds, file_truths, file_k_list, file_match),
    }
