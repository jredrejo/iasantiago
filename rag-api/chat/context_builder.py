# Archivo: rag-api/chat/context_builder.py
# Descripción: Construcción de contexto y mensajes para el LLM
#
# Arquitectura: Context-in-User-Message
# - System prompt: instrucciones estáticas (cacheable por vLLM prefix caching)
# - User message: contexto RAG dinámico + query del usuario

import logging
from typing import Dict, List

from core.cache import ModelCache

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Construye mensajes y contexto para enviar al LLM.

    Arquitectura Context-in-User-Message:
    - System prompt contiene solo instrucciones estáticas (cacheable)
    - Contexto RAG se inyecta en el último mensaje del usuario
    - Mejora latencia via vLLM prefix caching
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,
        model_max_len: int = 32768,
    ):
        """
        Inicializa el builder.

        Args:
            max_context_tokens: Máximo tokens para contexto RAG
            model_max_len: Longitud máxima del modelo
        """
        self.max_context_tokens = max_context_tokens
        self.model_max_len = model_max_len

    def count_tokens(self, text: str) -> int:
        """Cuenta tokens en un texto"""
        return ModelCache.count_tokens(text)

    def get_system_prompt(self, base_prompt: str) -> str:
        """
        Retorna el system prompt estático (sin contexto RAG).

        Args:
            base_prompt: Prompt de sistema base desde template

        Returns:
            System prompt estático (cacheable por vLLM)
        """
        return base_prompt

    def build_user_message_with_context(
        self,
        user_query: str,
        context_text: str,
        max_context_tokens: int = 0,
        no_context_message: str = "No se encontró información relevante en la base de datos.",
    ) -> str:
        """
        Construye el mensaje del usuario con contexto RAG inyectado.

        Args:
            user_query: Query original del usuario
            context_text: Contexto RAG a incluir
            max_context_tokens: Límite de tokens para contexto (0 = usar default)
            no_context_message: Mensaje que indica sin contexto

        Returns:
            Mensaje del usuario enriquecido con contexto
        """
        effective_limit = (
            max_context_tokens if max_context_tokens > 0 else self.max_context_tokens
        )
        has_context = context_text and context_text != no_context_message

        if has_context:
            # Truncar contexto si excede límite
            context_tokens = self.count_tokens(context_text)
            if context_tokens > effective_limit:
                logger.warning(
                    f"Contexto demasiado largo ({context_tokens} tokens). "
                    f"Truncando a ~{effective_limit} tokens."
                )
                context_text = self._truncate_context(context_text, effective_limit)

            return f"""CONTEXTO RAG (información de documentos para responder):

{context_text}

---
PREGUNTA:
{user_query}"""
        else:
            return user_query

    def _truncate_context(self, context_text: str, max_tokens: int) -> str:
        """
        Trunca el contexto para ajustarse al límite de tokens.

        Preserva chunks completos (no corta a mitad de chunk).
        """
        chunks = context_text.split("\n---\n")
        truncated_chunks = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            truncated_chunks.append(chunk)
            current_tokens += chunk_tokens

        if not truncated_chunks and chunks:
            # Al menos incluir el primer chunk truncado
            truncated_chunks = [chunks[0][: max_tokens * 4]]

        result = "\n---\n".join(truncated_chunks)
        logger.info(
            f"Contexto truncado: {len(chunks)} -> {len(truncated_chunks)} chunks"
        )
        return result

    def build_messages(
        self,
        system_prompt: str,
        user_messages: list,
        context_text: str = "",
        max_context_tokens: int = 0,
        no_context_message: str = "No se encontró información relevante en la base de datos.",
    ) -> List[Dict[str, str]]:
        """
        Construye la lista final de mensajes para el LLM.

        Inyecta contexto RAG en el último mensaje del usuario.

        Args:
            system_prompt: System prompt estático (sin contexto)
            user_messages: Lista de mensajes del usuario (objetos o dicts)
            context_text: Contexto RAG a inyectar
            max_context_tokens: Límite de tokens para contexto (0 = usar default)
            no_context_message: Mensaje que indica sin contexto

        Returns:
            Lista de mensajes en formato dict
        """
        # Usar límite pasado o el default de la instancia
        effective_limit = (
            max_context_tokens if max_context_tokens > 0 else self.max_context_tokens
        )
        messages = []

        # 1. System prompt (estático, cacheable)
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

        # 2. Historial de conversación
        user_msg_list = []
        for msg in user_messages:
            if hasattr(msg, "role"):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")

            if role != "system":
                user_msg_list.append({"role": role, "content": content})

        # 3. Inyectar contexto RAG en el último mensaje del usuario
        if user_msg_list:
            last_idx = None
            for i in range(len(user_msg_list) - 1, -1, -1):
                if user_msg_list[i]["role"] == "user":
                    last_idx = i
                    break

            if last_idx is not None:
                original_query = user_msg_list[last_idx]["content"]
                user_msg_list[last_idx]["content"] = (
                    self.build_user_message_with_context(
                        user_query=original_query,
                        context_text=context_text,
                        max_context_tokens=effective_limit,
                        no_context_message=no_context_message,
                    )
                )

        messages.extend(user_msg_list)

        logger.info(
            f"Mensajes construidos: {len(messages)} "
            f"(1 system + {len(user_msg_list)} conversación)"
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
