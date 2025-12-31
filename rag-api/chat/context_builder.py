# Archivo: rag-api/chat/context_builder.py
# Descripción: Construcción de contexto y mensajes para el LLM

import logging
from typing import Dict, List, Optional

from core.cache import ModelCache

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Construye mensajes y contexto para enviar al LLM.

    Responsabilidades:
    - Enriquecer system prompt con contexto RAG
    - Truncar system prompt si es necesario
    - Construir lista de mensajes final
    """

    def __init__(
        self,
        max_system_percent: float = 0.25,
        model_max_len: int = 32768,
    ):
        """
        Inicializa el builder.

        Args:
            max_system_percent: Máximo % del modelo para system prompt
            model_max_len: Longitud máxima del modelo
        """
        self.max_system_percent = max_system_percent
        self.model_max_len = model_max_len

    def count_tokens(self, text: str) -> int:
        """Cuenta tokens en un texto"""
        return ModelCache.count_tokens(text)

    def build_enhanced_system_prompt(
        self,
        base_prompt: str,
        context_text: str,
        no_context_message: str = "No se encontró información relevante en la base de datos.",
    ) -> str:
        """
        Construye el system prompt enriquecido con contexto RAG.

        Args:
            base_prompt: Prompt de sistema base
            context_text: Contexto RAG a incluir
            no_context_message: Mensaje que indica sin contexto

        Returns:
            System prompt enriquecido (y posiblemente truncado)
        """
        # Si hay contexto válido, enriquecerlo
        if context_text and context_text != no_context_message:
            enhanced = f"""{base_prompt}

[Contexto RAG - Información relevante de la base de datos]
{context_text}

Usa este contexto para responder las preguntas del usuario. Siempre cita las fuentes usando los enlaces proporcionados."""
        else:
            enhanced = base_prompt

        # Verificar si necesita truncado
        max_tokens = int(self.model_max_len * self.max_system_percent)
        current_tokens = self.count_tokens(enhanced)

        if current_tokens > max_tokens:
            logger.warning(
                f"System prompt demasiado largo ({current_tokens} tokens). "
                f"Truncando a {max_tokens} tokens."
            )
            enhanced = self._truncate_system_prompt(context_text, max_tokens)
            new_tokens = self.count_tokens(enhanced)
            logger.info(
                f"System prompt truncado: {current_tokens} -> {new_tokens} tokens"
            )

        return enhanced

    def _truncate_system_prompt(
        self,
        context_text: Optional[str],
        max_tokens: int,
    ) -> str:
        """Crea una versión truncada del system prompt"""
        if context_text:
            # Versión minimalista con contexto
            return f"""Eres un asistente docente experto. Responde usando el contexto proporcionado.

Contexto RAG:
{context_text}

Responde usando solo información del contexto. Cita las fuentes con formato: [archivo.pdf, p.X](/docs/TOPIC/archivo.pdf#page=X)"""
        else:
            # Versión minimalista sin contexto
            return """Eres un asistente docente experto. Responde usando el contexto proporcionado y cita las fuentes con formato: [archivo.pdf, p.X](/docs/TOPIC/archivo.pdf#page=X)"""

    def build_messages(
        self,
        enhanced_system_prompt: str,
        user_messages: list,
    ) -> List[Dict[str, str]]:
        """
        Construye la lista final de mensajes para el LLM.

        Args:
            enhanced_system_prompt: System prompt ya enriquecido
            user_messages: Lista de mensajes del usuario (objetos o dicts)

        Returns:
            Lista de mensajes en formato dict
        """
        messages = []

        # 1. System prompt
        messages.append(
            {
                "role": "system",
                "content": enhanced_system_prompt,
            }
        )

        # 2. Historial de conversación (sin mensajes system del usuario)
        for msg in user_messages:
            if hasattr(msg, "role"):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")

            if role != "system":
                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        logger.info(
            f"Mensajes construidos: {len(messages)} "
            f"(system + {len(user_messages)} historial)"
        )

        return messages

    def log_context_status(
        self,
        context_text: str,
        num_chunks: int,
        no_context_message: str = "No se encontró información relevante en la base de datos.",
    ):
        """Log del estado del contexto RAG"""
        if not context_text or context_text == no_context_message:
            logger.warning("SIN contexto RAG disponible - modelo en riesgo de alucinar")
        else:
            logger.info(f"Contexto proporcionado: {num_chunks} chunks")
            logger.debug(f"Preview contexto: {context_text[:200]}...")
