# Archivo: rag-api/chat/intent.py
# Descripción: Detección de intención del usuario (generativa vs respuesta)

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Verbo creativo: imperativo (con clítico opcional: hazME, prepáraNOS),
# subjuntivo dirigido al asistente (que me hagas) o infinitivo (puedes crear).
# NO incluye formas neutras como "hace"/"hacer a secas" para no disparar con
# "¿cómo se hace la prueba?" o "no sé hacer el ejercicio 2".
_CREATIVE_VERB = (
    r"(?:crea|elabora|genera|dise[ñn]a|prep[aá]ra|redacta|inventa|"
    r"prop[oó]n|plantea|formula|escribe|haz|hagas)(?:me|nos|te)?"
    r"|crear|elaborar|generar|dise[ñn]ar|preparar|redactar|inventar|"
    r"proponer|plantear|formular|escribir|hacerme|hacernos"
)

# Objeto generable: lo que tiene sentido pedir que se cree
_GENERATIVE_OBJECT = (
    r"(?:examen|ex[aá]menes|test|prueba|evaluaci[oó]n|cuestionario|"
    r"pregunta|cuesti[oó]n|ejercicio|actividad|problema|"
    r"esquema|resumen|mapa)(?:e?s)?"
    r"|lista(?:dos?|s)?"
)

# Patrones para detectar intención generativa.
# Requieren verbo creativo + objeto generable (§2.4): mencionar "ejercicio"
# o "pregunta" a secas ya NO dispara el modo generativo.
GENERATIVE_PATTERNS: List[str] = [
    # Verbo creativo seguido a corta distancia de un objeto generable:
    # "crea un examen...", "hazme 5 ejercicios...", "¿puedes generar preguntas...?"
    rf"\b(?:{_CREATIVE_VERB})\b.{{0,80}}?\b(?:{_GENERATIVE_OBJECT})\b",
    # Cantidad explícita de ítems a producir: "10 preguntas", "5 ejercicios"
    r"\b\d+\s*(?:preguntas?|ejercicios?|cuestiones?|actividades?|problemas?)\b",
    # Reorganización de contenido en un formato: "resume el tema 4 en un esquema"
    r"\b(?:resume|res[uú]me\w*|sintetiza|organiza)\b.{0,60}?\b(?:esquema|mapa|listas?|listados?|tabla)\b",
]


def detect_generative_intent(user_message: str) -> bool:
    """
    Detecta si el usuario quiere GENERAR contenido (examen, ejercicios, etc.)
    vs. simplemente RESPONDER una pregunta con el contexto.

    La intención generativa requiere:
    - Más tokens de respuesta
    - Más contexto RAG
    - Prompt de sistema específico para creación

    Args:
        user_message: Mensaje del usuario a analizar

    Returns:
        True si se detecta intención generativa, False para respuesta normal
    """
    message_lower = user_message.lower()

    for pattern in GENERATIVE_PATTERNS:
        if re.search(pattern, message_lower):
            logger.info(f"Intención GENERATIVA detectada: patrón '{pattern}'")
            return True

    logger.info("Intención de RESPUESTA detectada (default)")
    return False


def load_system_prompt(
    is_generative: bool,
    prompts_dir: str = "/app/templates/system_prompts",
) -> str:
    """
    Carga el prompt de sistema correcto según la intención.

    Args:
        is_generative: True para prompt generativo, False para default
        prompts_dir: Directorio donde están los prompts

    Returns:
        Contenido del prompt de sistema
    """
    if is_generative:
        path = f"{prompts_dir}/generative.txt"
        logger.info("Usando prompt GENERATIVO (crear contenido)")
    else:
        path = f"{prompts_dir}/default.txt"
        logger.info("Usando prompt DEFAULT (responder con contexto)")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt no encontrado: {path}")
        # Fallback al prompt default
        default_path = f"{prompts_dir}/default.txt"
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt default tampoco encontrado: {default_path}")
            return _get_fallback_prompt()


def _get_fallback_prompt() -> str:
    """Prompt de respaldo si no se encuentran los archivos"""
    return """Eres un asistente docente experto. Responde usando el contexto proporcionado.
Cita las fuentes con formato: [archivo.pdf, p.X](/docs/TOPIC/archivo.pdf#page=X)
Si no encuentras información relevante en el contexto, indica que no encontraste información."""


def get_last_user_message(messages: list) -> str:
    """
    Obtiene el último mensaje del usuario de una lista de mensajes.

    Args:
        messages: Lista de objetos Message con atributos role y content

    Returns:
        Contenido del último mensaje del usuario, o cadena vacía si no hay
    """
    for msg in reversed(messages):
        if hasattr(msg, "role") and msg.role == "user":
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""
