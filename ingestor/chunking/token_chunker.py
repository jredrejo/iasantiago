"""
Fragmentación consciente de estructura y de tokens.

Sustituye la indexación de párrafos en crudo: antes cada párrafo que devolvía el
extractor se embebía tal cual, así que su tamaño era el que casualmente tuviera
en el PDF. Los modelos de embedding truncan a 512 tokens, de modo que la cola de
los párrafos largos (y de páginas enteras, en los extractores de respaldo que
emiten un elemento por página) quedaba fuera del índice denso.

Dos vías, según lo que haya conseguido la extracción:

- Con `DoclingDocument`: `HybridChunker`, que parte primero por estructura
  (secciones, listas, tablas) y luego ajusta al presupuesto real de tokens.
  Además antepone la ruta de encabezados al texto embebido.
- Sin él (extractores de respaldo): agrupación de elementos por página con
  presupuesto de tokens medido con el tokenizador real del modelo.

El presupuesto se calcula con el tokenizador del modelo que embebe *ese* tema
(este proyecto usa e5-instruct, gte-large e instructor-large según el tema), no
con una estimación por caracteres.
"""

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chunking.text_repair import repair_chunks
from extraction.base import Element

logger = logging.getLogger(__name__)

# Margen bajo el límite del modelo: `contextualize()` antepone la ruta de
# encabezados, que también consume tokens.
DEFAULT_MAX_TOKENS = 512

# Un fragmento por debajo de esto es casi siempre un resto de maquetación
# (número de página suelto, línea de cabecera) y sólo añade ruido al top-k.
MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    """Fragmento listo para embeber, con su procedencia."""

    text: str
    page: int
    type: str = "text"
    source: str = "unknown"
    headings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def heading_path(self) -> str:
        """Ruta de encabezados como cadena única para el payload."""
        return " > ".join(self.headings)


def _normalize(text: str) -> str:
    """Normaliza texto para comparar duplicados (espacios y mayúsculas)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _text_hash(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()


class TokenCounter:
    """
    Cuenta tokens con el tokenizador real del modelo de embedding.

    Se cachea por nombre de modelo: cargar un tokenizador por documento sería
    un coste inútil en un corpus de cientos de PDFs.
    """

    _cache: Dict[str, Any] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = self._get_tokenizer(model_name)

    @classmethod
    def _get_tokenizer(cls, model_name: str) -> Optional[Any]:
        if model_name in cls._cache:
            return cls._cache[model_name]
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name)
            cls._cache[model_name] = tok
            logger.info(f"[CHUNK] Tokenizador cargado: {model_name}")
            return tok
        except Exception as e:
            logger.warning(
                f"[CHUNK] No se pudo cargar el tokenizador de {model_name}: {e}. "
                f"Se estimará por caracteres."
            )
            cls._cache[model_name] = None
            return None

    @property
    def tokenizer(self) -> Optional[Any]:
        return self._tokenizer

    def count(self, text: str) -> int:
        """Número de tokens del texto (estimación por caracteres si no hay tokenizador)."""
        if self._tokenizer is None:
            # ~4 caracteres por token es la aproximación habitual; sólo se usa
            # si el tokenizador no está disponible.
            return max(1, len(text) // 4)
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def detect_boilerplate(
    elements: List[Element], min_page_ratio: float = 0.3
) -> set:
    """
    Detecta encabezados y pies de página repetidos.

    Los extractores de respaldo indexan la cabecera y el pie de cada página, lo
    que genera cientos de fragmentos casi idénticos que compiten en el top-k.
    Se considera repetitivo el texto corto que aparece en más del `min_page_ratio`
    de las páginas.

    Args:
        elements: Elementos extraídos del documento
        min_page_ratio: Fracción de páginas en las que debe aparecer una línea

    Returns:
        Conjunto de textos normalizados a descartar
    """
    pages = {e.page for e in elements}
    if len(pages) < 4:
        # En documentos muy cortos no hay evidencia suficiente y se corre el
        # riesgo de borrar contenido legítimo que se repite.
        return set()

    # Sólo se consideran líneas cortas: un párrafo largo repetido es contenido,
    # no maquetación.
    line_pages: Dict[str, set] = {}
    for e in elements:
        for line in e.text.split("\n"):
            norm = _normalize(line)
            if not norm or len(norm) > 100:
                continue
            line_pages.setdefault(norm, set()).add(e.page)

    threshold = max(3, int(len(pages) * min_page_ratio))
    boilerplate = {
        line for line, pgs in line_pages.items() if len(pgs) >= threshold
    }

    if boilerplate:
        logger.info(
            f"[CHUNK] Detectadas {len(boilerplate)} líneas de cabecera/pie "
            f"repetidas en ≥{threshold} de {len(pages)} páginas"
        )
    return boilerplate


def _strip_boilerplate(text: str, boilerplate: set) -> str:
    """Elimina de un texto las líneas marcadas como cabecera/pie."""
    if not boilerplate:
        return text
    kept = [ln for ln in text.split("\n") if _normalize(ln) not in boilerplate]
    return "\n".join(kept).strip()


def dedupe_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Elimina fragmentos con texto idéntico tras normalizar.

    Además de mejorar el top-k, ahorra tiempo de GPU: los duplicados no se
    embeben.
    """
    seen: set = set()
    out: List[Chunk] = []
    for c in chunks:
        h = _text_hash(c.text)
        if h in seen:
            continue
        seen.add(h)
        out.append(c)

    removed = len(chunks) - len(out)
    if removed:
        logger.info(f"[CHUNK] Descartados {removed} fragmentos duplicados")
    return out


def chunk_docling_document(
    doc: Any,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    source: str = "docling",
) -> List[Chunk]:
    """
    Fragmenta un DoclingDocument con HybridChunker.

    Args:
        doc: DoclingDocument
        model_name: Modelo de embedding cuyo tokenizador fija el presupuesto
        max_tokens: Presupuesto de tokens por fragmento
        source: Etiqueta de procedencia para el payload

    Returns:
        Lista de Chunk con ruta de encabezados y página de procedencia
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    counter = TokenCounter(model_name)
    if counter.tokenizer is None:
        raise RuntimeError(
            f"HybridChunker necesita el tokenizador real de {model_name}"
        )

    tokenizer = HuggingFaceTokenizer(tokenizer=counter.tokenizer, max_tokens=max_tokens)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    chunks: List[Chunk] = []
    for dl_chunk in chunker.chunk(dl_doc=doc):
        # `contextualize` antepone la ruta de encabezados. Se embebe ese texto
        # (mejora la recuperación en corpus tipo manual) y no sólo el cuerpo.
        text = chunker.contextualize(chunk=dl_chunk).strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue

        headings = list(getattr(dl_chunk.meta, "headings", None) or [])

        # Página de procedencia: la primera que aporte algún doc_item.
        page = 1
        item_types = set()
        for item in getattr(dl_chunk.meta, "doc_items", []) or []:
            item_types.add(type(item).__name__)
            prov = getattr(item, "prov", None)
            if prov:
                page_no = getattr(prov[0], "page_no", None)
                if page_no:
                    page = int(page_no)
                    break

        chunk_type = "table" if any("Table" in t for t in item_types) else "text"

        chunks.append(
            Chunk(
                text=text,
                page=page,
                type=chunk_type,
                source=source,
                headings=headings,
                metadata={"chunker": "docling_hybrid"},
            )
        )

    return chunks


def chunk_elements(
    elements: List[Element],
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    boilerplate: Optional[set] = None,
) -> List[Chunk]:
    """
    Fragmenta elementos sueltos por presupuesto de tokens.

    Vía para los extractores de respaldo, que no producen estructura. Agrupa
    elementos consecutivos de la misma página hasta agotar el presupuesto, y
    parte por frases los que ya lo superan por sí solos (un elemento por página
    entera es habitual en `TextExtractor`).

    Args:
        elements: Elementos extraídos
        model_name: Modelo cuyo tokenizador fija el presupuesto
        max_tokens: Presupuesto de tokens por fragmento
        boilerplate: Líneas de cabecera/pie a suprimir

    Returns:
        Lista de Chunk
    """
    counter = TokenCounter(model_name)
    boilerplate = boilerplate or set()

    chunks: List[Chunk] = []
    buffer: List[str] = []
    buf_tokens = 0
    buf_page = 1
    buf_source = "unknown"
    buf_type = "text"

    def flush() -> None:
        nonlocal buffer, buf_tokens
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    text=text,
                    page=buf_page,
                    type=buf_type,
                    source=buf_source,
                    metadata={"chunker": "token_budget"},
                )
            )
        buffer = []
        buf_tokens = 0

    for elem in elements:
        text = _strip_boilerplate(elem.text, boilerplate).strip()
        if not text:
            continue

        n_tokens = counter.count(text)

        # Un elemento que por sí solo excede el presupuesto se parte por frases.
        if n_tokens > max_tokens:
            flush()
            for piece in _split_by_tokens(text, counter, max_tokens):
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(
                        Chunk(
                            text=piece,
                            page=elem.page,
                            type=elem.type,
                            source=elem.source,
                            metadata={"chunker": "token_budget", "split": True},
                        )
                    )
            continue

        # Cambio de página o presupuesto agotado -> cerrar el fragmento actual.
        if buffer and (elem.page != buf_page or buf_tokens + n_tokens > max_tokens):
            flush()

        if not buffer:
            buf_page = elem.page
            buf_source = elem.source
            buf_type = elem.type

        buffer.append(text)
        buf_tokens += n_tokens

    flush()
    return chunks


def _hard_split(text: str, counter: TokenCounter, max_tokens: int) -> List[str]:
    """
    Parte un texto sin puntuación aprovechable cortando sobre los propios tokens.

    Estimar el corte por número de palabras no sirve: los tokens por palabra
    varían y el cálculo se pasa del presupuesto (se observaron piezas de 518
    tokens con límite 512), que es justo el truncado silencioso que se quiere
    evitar. Cortando sobre los ids de tokens el límite se respeta por
    construcción.
    """
    tokenizer = counter.tokenizer
    if tokenizer is None:
        # Sin tokenizador sólo cabe estimar; se deja margen para no pasarse.
        words = text.split()
        approx = max(1, (max_tokens * 3) // 4)
        return [" ".join(words[i : i + approx]) for i in range(0, len(words), approx)]

    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces: List[str] = []
    cursor = 0

    while cursor < len(ids):
        window = max_tokens
        # Decodificar N ids y volver a codificar el texto resultante no devuelve
        # siempre N tokens (la normalización de espacios y la fusión de
        # subpalabras cambian el recuento), y quien decide es el embedder, que
        # recodifica el texto. Por eso se verifica y se encoge hasta que quepa.
        while window > 1:
            piece = tokenizer.decode(
                ids[cursor : cursor + window], skip_special_tokens=True
            ).strip()
            if not piece or counter.count(piece) <= max_tokens:
                break
            window = int(window * 0.9) or 1

        if piece:
            pieces.append(piece)
        cursor += window

    return pieces


def _split_by_tokens(
    text: str, counter: TokenCounter, max_tokens: int
) -> List[str]:
    """Parte un texto largo por frases sin superar el presupuesto de tokens."""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    pieces: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        n = counter.count(sent)

        # Una sola "frase" mayor que el presupuesto (tablas, texto sin puntuar):
        # se corta directamente sobre los tokens.
        if n > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            pieces.extend(_hard_split(sent, counter, max_tokens))
            continue

        if current_tokens + n > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0

        current.append(sent)
        current_tokens += n

    if current:
        pieces.append(" ".join(current))

    return [p.strip() for p in pieces if p.strip()]


def build_chunks(
    result: Any,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[Chunk]:
    """
    Punto de entrada: convierte un ExtractionResult en fragmentos embebibles.

    Usa HybridChunker si la extracción trajo estructura y degrada a la vía por
    tokens en caso contrario. Aplica supresión de cabeceras/pies y deduplicación
    en ambos casos.

    Args:
        result: ExtractionResult de ExtractionPipeline.extract_document()
        model_name: Modelo de embedding del tema
        max_tokens: Presupuesto de tokens por fragmento

    Returns:
        Lista de Chunk deduplicados
    """
    if result.has_structure:
        try:
            chunks = chunk_docling_document(
                result.docling_document, model_name, max_tokens
            )
            if chunks:
                logger.info(
                    f"[CHUNK] HybridChunker produjo {len(chunks)} fragmentos "
                    f"desde {len(result.elements)} elementos"
                )
                return dedupe_chunks(repair_chunks(chunks))
            logger.warning(
                "[CHUNK] HybridChunker no produjo fragmentos; se usa la vía por tokens"
            )
        except Exception as e:
            logger.warning(
                f"[CHUNK] HybridChunker falló ({e}); se usa la vía por tokens"
            )

    boilerplate = detect_boilerplate(result.elements)
    chunks = chunk_elements(
        result.elements, model_name, max_tokens, boilerplate=boilerplate
    )
    logger.info(
        f"[CHUNK] Fragmentación por tokens: {len(chunks)} fragmentos "
        f"desde {len(result.elements)} elementos"
    )
    return dedupe_chunks(repair_chunks(chunks))
