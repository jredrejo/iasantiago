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
    return f"""

{SEPARATOR}
INSTRUCCIONES PARA RESPUESTAS CON FUENTES
{SEPARATOR}

Usa la información de los documentos anteriores para responder:

1. Cada sección del texto incluye su fuente al final en este formato:
   [archivo.pdf, p.N](/docs/TOPIC/archivo.pdf#page=N)

2. Cuando uses información de un documento, cita la fuente EXACTAMENTE como aparece:
   Fuente: [archivo.pdf, p.N](/docs/TOPIC/archivo.pdf#page=N)

3. IMPORTANTE: Las citas deben ser enlaces clicables en formato markdown.
   NO escribas las URLs como texto plano.

4. NUNCA menciones "fragmento", "sección", "parte" o referencias internas.
   Solo habla del contenido y cita el archivo y página.

5. Si usas información del documento Manual.pdf página 42, termina con:
   Fuente: [Manual.pdf, p.42](/docs/Chemistry/Manual.pdf#page=42)

EJEMPLO DE RESPUESTA:
"La ley de Ohm establece que V = I x R, donde V es el voltaje, I es la corriente y R es la resistencia.
Esta relación fundamental permite calcular cualquier parámetro eléctrico si se conocen los otros dos.
Fuente: [Electrónica_Básica.pdf, p.15](/docs/Electronics/Electrónica_Básica.pdf#page=15)"

FORMATO CORRECTO DE CITAS:
OK: Fuente: [archivo.pdf, p.25](/docs/TOPIC/archivo.pdf#page=25)  <- Enlace clicable
MAL: Fuente: [archivo.pdf, p.25]                             <- No es enlace
MAL: Fuente: archivo.pdf, p.25                             <- No es enlace

REGLAS IMPORTANTES:
- NO inventes información que no esté en los documentos
- NO uses referencias internas como "según el documento 3" o "en el fragmento"
- Sé natural y conversacional, solo citando las fuentes
- SOLO permite markdown para: bloques de código (```), fórmulas LaTeX ($formula$) y citas [enlaces](/docs/...)
- NO uses negritas (**), títulos (##) o listas (-)

{SEPARATOR}
"""


def format_simple_context(chunks: List[Dict]) -> str:
    """
    Formato simple de contexto sin instrucciones elaboradas.

    Útil para casos donde se quiere un contexto más compacto.

    Args:
        chunks: Lista de chunks recuperados

    Returns:
        Contexto formateado de forma simple
    """
    if not chunks:
        return "No se encontró información relevante en la base de datos."

    parts = []
    for chunk in chunks:
        filename = os.path.basename(chunk["file_path"])
        page = chunk["page"]
        text = chunk["text"]
        parts.append(f"[{filename}, p.{page}]\n{text}")

    return "\n\n---\n\n".join(parts)


def validate_context_usage(
    retrieved_chunks: List[Dict],
    model_response: str,
) -> Dict:
    """
    Valida si el modelo usó el contexto correctamente.

    Args:
        retrieved_chunks: Chunks proporcionados al modelo
        model_response: Respuesta generada por el modelo

    Returns:
        Diccionario con métricas de uso del contexto
    """
    import re

    context_files = set(c["file_path"] for c in retrieved_chunks)
    citations = re.findall(r"\[([^]]+\.pdf),\s*p\.(\d+)\]", model_response)
    cited_files = set(filename for filename, _ in citations)

    coverage = len(cited_files & context_files) / max(len(context_files), 1)
    has_not_found = "No encontré información" in model_response

    result = {
        "archivos_contexto": list(context_files),
        "archivos_citados": list(cited_files),
        "cobertura": round(coverage, 2),
        "dijo_no_encontro": has_not_found,
        "num_citaciones": len(citations),
    }

    logger.info(f"Validación de contexto: {result}")
    return result
