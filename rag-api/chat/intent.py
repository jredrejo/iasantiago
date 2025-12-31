# Archivo: rag-api/chat/intent.py
# Descripción: Detección de intención del usuario (generativa vs respuesta)

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Patrones para detectar intención generativa
GENERATIVE_PATTERNS: List[str] = [
    # Creación de exámenes
    r"\b(crea|elabora|genera|diseña|prepara|haz|hacer)\b.*\b(examen|test|prueba|evaluaci[oó]n)\b",
    r"\b(preguntas?)\b.*\b(sobre|de|acerca)\b",
    r"\b\d+\s*(preguntas?|ejercicios?|cuestiones?)\b",  # "10 preguntas"
    # Creación de ejercicios
    r"\b(ejercicios?|actividades?|pr[aá]cticas?)\b",
    # Creación de contenido educativo
    r"\b(resume|sintetiza|organiza)\b.*\b(en|como)\b.*\b(esquema|mapa|lista)\b",
    r"\blistado\b.*\b(de|con)\b",
    # Comandos explícitos al inicio
    r"^(crea|elabora|genera|diseña|prepara|haz)\b",
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
