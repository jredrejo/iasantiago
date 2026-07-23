# Archivo: rag-api/retrieval_lib/citations.py
# Descripción: Construcción de contexto RAG con citaciones

import logging
import os
from typing import Dict, List, Tuple
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Separador visual para el contexto
SEPARATOR = "=" * 70


def build_context_with_citations(
    chunks: List[Dict],
    topic: str = "",
) -> Tuple[str, List[Dict]]:
    """
    Construye el contexto RAG con citaciones clicables.

    Genera un bloque de texto con:
    1. Contenido de cada chunk con su fuente
    2. Instrucciones para el LLM sobre cómo citar

    Args:
        chunks: Lista de chunks recuperados
        topic: Tema para construir las URLs

    Returns:
        Tupla (contexto_con_citaciones, chunks_originales)
    """
    if not chunks:
        return "No se encontró información relevante.", []

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        filename = os.path.basename(chunk["file_path"])
        page = chunk["page"]
        text = chunk["text"]
        encoded_filename = quote(filename, safe=".")

        # Construir URL del documento
        if topic:
            doc_url = f"/docs/{topic}/{encoded_filename}#page={page}"
        else:
            doc_url = f"/docs/{encoded_filename}#page={page}"

        # Formato del chunk con fuente
        chunk_with_citation = f"""{text}

FUENTE:
[{filename}, p.{page}]({doc_url})"""

        context_parts.append(chunk_with_citation)
        logger.info(f"[{i}] {filename}, p.{page} -> {doc_url}")

    # Construir cuerpo del contexto
    context_body = f"\n\n{SEPARATOR}\n"
    context_body += "CONTEXTO RAG - INFORMACIÓN DE DOCUMENTOS\n"
    context_body += f"{SEPARATOR}\n\n"
    context_body += "\n\n".join(context_parts)
    context_body += f"\n\n{SEPARATOR}"

    # Añadir instrucciones para el LLM
    instructions = _build_citation_instructions()

    return context_body + instructions, chunks


def _build_citation_instructions() -> str:
    """Construye las instrucciones de citación para el LLM"""
    return """

Cita las fuentes copiando el formato exacto que aparece despues de FUENTE:
Ejemplo: [archivo.pdf, p.N](/docs/TOPIC/archivo.pdf#page=N)
"""
