"""
Estrategias de fragmentación para diferentes tipos de contenido y requisitos.

Proporciona enfoques de fragmentación semántica, adaptativa y simple.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from core.config import get_sent_tokenizer

logger = logging.getLogger(__name__)


def _fallback_sentence_split(text: str) -> List[str]:
    """Divisor de oraciones de respaldo simple."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def get_tokenizer() -> Callable[[str], List[str]]:
    """Obtiene la función tokenizadora de oraciones."""
    try:
        return get_sent_tokenizer()
    except Exception:
        return _fallback_sentence_split


def semantic_chunk(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
    min_chunk_size: int = 100,
    page_num: int = 1,
    context_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Divide texto en fragmentos usando límites de oraciones.

    Args:
        text: Texto a fragmentar
        chunk_size: Tamaño máximo de fragmento en caracteres
        overlap: Número de caracteres a solapar entre fragmentos
        min_chunk_size: Tamaño mínimo de fragmento a incluir
        page_num: Número de página a asignar a fragmentos
        context_text: Texto de contexto opcional para anteponer al primer fragmento

    Returns:
        Lista de diccionarios de fragmentos
    """
    if not text.strip():
        return []

    tokenize = get_tokenizer()
    sentences = tokenize(text)
    chunks = []

    current = {"sentences": [], "text": "", "char_count": 0}
    first_chunk = True

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        potential_size = current["char_count"] + len(sentence) + 1

        if potential_size <= chunk_size:
            current["sentences"].append(sentence)
            current["text"] = " ".join(current["sentences"])
            current["char_count"] = len(current["text"])
        else:
            if current["char_count"] >= min_chunk_size:
                full_text = (
                    context_text + current["text"]
                    if first_chunk and context_text
                    else current["text"]
                )

                chunks.append(
                    {
                        "page": page_num,
                        "text": full_text,
                        "type": "text",
                        "source": "semantic",
                        "sentence_count": len(current["sentences"]),
                        "has_context": bool(context_text) and first_chunk,
                    }
                )
                first_chunk = False

            # Calcular oraciones de solapamiento
            overlap_sentences = _get_overlap_sentences(current["sentences"], overlap)

            current = {
                "sentences": overlap_sentences + [sentence],
                "text": " ".join(overlap_sentences + [sentence]),
                "char_count": len(" ".join(overlap_sentences + [sentence])),
            }

    # Agregar fragmento final
    if current["char_count"] >= min_chunk_size:
        full_text = (
            context_text + current["text"]
            if first_chunk and context_text
            else current["text"]
        )

        chunks.append(
            {
                "page": page_num,
                "text": full_text,
                "type": "text",
                "source": "semantic",
                "sentence_count": len(current["sentences"]),
                "has_context": bool(context_text) and first_chunk,
            }
        )

    return chunks


def simple_chunk(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
    min_chunk_size: int = 100,
    page_num: int = 1,
    context_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Divide texto en fragmentos de tamaño fijo con solapamiento.

    Args:
        text: Texto a fragmentar
        chunk_size: Tamaño máximo de fragmento en caracteres
        overlap: Número de caracteres a solapar
        min_chunk_size: Tamaño mínimo de fragmento a incluir
        page_num: Número de página a asignar
        context_text: Contexto opcional para primer fragmento

    Returns:
        Lista de diccionarios de fragmentos
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    first_chunk = True

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= min_chunk_size:
            full_text = (
                context_text + chunk_text
                if first_chunk and context_text
                else chunk_text
            )

            chunks.append(
                {
                    "page": page_num,
                    "text": full_text,
                    "type": "text",
                    "source": "simple",
                    "has_context": bool(context_text) and first_chunk,
                }
            )
            first_chunk = False

        # Calcular siguiente inicio con solapamiento
        start = end - overlap if end < len(text) else end

    return chunks


def adaptive_chunk(
    elements: List[Dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 120,
    min_chunk_size: int = 100,
    page_num: int = 1,
    context_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Fragmenta elementos adaptativamente basándose en su tipo.

    - Elementos de texto: fragmentación semántica
    - Tablas: preservadas como están o divididas si son muy grandes
    - Imágenes: preservadas como están

    Args:
        elements: Lista de diccionarios de elementos
        chunk_size: Tamaño máximo de fragmento
        overlap: Solapamiento para fragmentos de texto
        min_chunk_size: Tamaño mínimo de fragmento
        page_num: Número de página a asignar
        context_text: Contexto opcional para primer fragmento

    Returns:
        Lista de diccionarios de fragmentos
    """
    chunks = []

    # Separar por tipo
    text_elements = [e for e in elements if e.get("type") == "text"]
    table_elements = [e for e in elements if e.get("type") == "table"]
    image_elements = [e for e in elements if e.get("type") == "image"]

    # Fragmentar elementos de texto
    if text_elements:
        text_content = "\n\n".join(e.get("text", "") for e in text_elements)
        text_chunks = semantic_chunk(
            text_content,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            page_num=page_num,
            context_text=context_text,
        )
        chunks.extend(text_chunks)

    # Manejar tablas
    for table in table_elements:
        table_text = table.get("text", "")
        if len(table_text) <= chunk_size * 1.5:
            chunks.append(
                {
                    "page": page_num,
                    "text": table_text,
                    "type": "table",
                    "source": "table_standalone",
                    "metadata": table.get("metadata", {}),
                }
            )
        else:
            table_chunks = _split_large_table(table_text, page_num, chunk_size)
            chunks.extend(table_chunks)

    # Manejar imágenes
    for image in image_elements:
        chunks.append(
            {
                "page": page_num,
                "text": image.get("text", ""),
                "type": "image",
                "source": "image_standalone",
                "metadata": image.get("metadata", {}),
            }
        )

    return chunks


def _get_overlap_sentences(sentences: List[str], overlap: int) -> List[str]:
    """Obtiene oraciones para solapamiento."""
    if not sentences:
        return []

    overlap_sentences = []
    char_count = 0

    for sentence in reversed(sentences):
        if char_count + len(sentence) <= overlap:
            overlap_sentences.insert(0, sentence)
            char_count += len(sentence) + 1
        else:
            break

    return overlap_sentences


def _split_large_table(
    table_text: str,
    page_num: int,
    chunk_size: int,
) -> List[Dict[str, Any]]:
    """Divide una tabla grande en fragmentos."""
    rows = table_text.split("\n")

    if not rows:
        return []

    header = rows[0] if rows else ""
    data_rows = rows[1:] if len(rows) > 1 else []

    chunks = []
    current_rows = [header]
    current_size = len(header)

    for row in data_rows:
        row_size = len(row)

        if current_size + row_size <= chunk_size * 1.5:
            current_rows.append(row)
            current_size += row_size
        else:
            if len(current_rows) > 1:
                chunks.append(
                    {
                        "page": page_num,
                        "text": "\n".join(current_rows),
                        "type": "table",
                        "source": "table_split",
                        "is_continuation": len(chunks) > 0,
                    }
                )

            current_rows = [header, row]
            current_size = len(header) + row_size

    if len(current_rows) > 1:
        chunks.append(
            {
                "page": page_num,
                "text": "\n".join(current_rows),
                "type": "table",
                "source": "table_split",
                "is_continuation": len(chunks) > 0,
            }
        )

    return chunks
