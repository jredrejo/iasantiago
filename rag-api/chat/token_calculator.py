# Archivo: rag-api/chat/token_calculator.py
# Descripción: Cálculo dinámico de tokens para respuestas

import logging
from dataclasses import dataclass
from typing import Dict, List

from core.cache import ModelCache

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Presupuesto de tokens calculado para una request"""

    system_tokens: int
    context_tokens: int
    history_tokens: int
    total_input: int
    available_for_response: int
    max_tokens: int
    is_truncated: bool


class TokenCalculator:
    """
    Calcula el presupuesto de tokens para respuestas del LLM.

    Determina dinámicamente cuántos tokens usar basándose en:
    - Modo (generativo vs respuesta)
    - Tokens de input (system + context + historial)
    - Límites del modelo
    """

    def __init__(
        self,
        model_max_len: int,
        max_tokens_limit: int,
        generative_percent: int = 60,
        response_percent: int = 25,
        min_response_tokens: int = 512,
        safety_margin: int = 100,
    ):
        """
        Inicializa el calculador.

        Args:
            model_max_len: Longitud máxima del modelo (VLLM_MAX_MODEL_LEN)
            max_tokens_limit: Límite de max_tokens (VLLM_MAX_TOKENS)
            generative_percent: % del modelo para modo generativo
            response_percent: % del modelo para modo respuesta
            min_response_tokens: Mínimo absoluto de tokens para respuesta
            safety_margin: Margen de seguridad en tokens
        """
        self.model_max_len = model_max_len
        self.max_tokens_limit = max_tokens_limit
        self.generative_percent = generative_percent
        self.response_percent = response_percent
        self.min_response_tokens = min_response_tokens
        self.safety_margin = safety_margin

    def count_tokens(self, text: str) -> int:
        """Cuenta tokens en un texto"""
        return ModelCache.count_tokens(text)

    def calculate_budget(
        self,
        system_prompt: str,
        context_text: str,
        messages: List[Dict],
        is_generative: bool,
    ) -> TokenBudget:
        """
        Calcula el presupuesto de tokens para una request.

        Args:
            system_prompt: Prompt de sistema (ya enriquecido con contexto)
            context_text: Texto del contexto RAG
            messages: Lista de mensajes (sin system)
            is_generative: True si es modo generativo

        Returns:
            TokenBudget con todos los cálculos
        """
        # Contar tokens de cada componente
        system_tokens = self.count_tokens(system_prompt)
        context_tokens = self.count_tokens(context_text)
        history_tokens = sum(
            self.count_tokens(m.get("content", ""))
            for m in messages
            if m.get("role") != "system"
        )

        total_input = system_tokens + history_tokens
        available = self.model_max_len - total_input - self.safety_margin

        # Calcular max_tokens deseados según modo
        if is_generative:
            desired = self._calculate_generative_tokens()
            logger.info(
                f"MODO GENERATIVO: Objetivo {desired} tokens "
                f"({self.generative_percent}% de {self.model_max_len})"
            )
        else:
            desired = self._calculate_response_tokens()
            logger.info(
                f"MODO RESPUESTA: Objetivo {desired} tokens "
                f"({self.response_percent}% de {self.model_max_len})"
            )

        # Usar el mínimo entre lo deseado y lo disponible
        max_tokens = max(
            self.min_response_tokens,
            min(desired, available)
        )

        is_truncated = max_tokens < desired

        # Log detallado
        self._log_token_breakdown(
            system_tokens,
            context_tokens,
            history_tokens,
            total_input,
            available,
            desired,
            max_tokens,
            is_generative,
        )

        return TokenBudget(
            system_tokens=system_tokens,
            context_tokens=context_tokens,
            history_tokens=history_tokens,
            total_input=total_input,
            available_for_response=available,
            max_tokens=max_tokens,
            is_truncated=is_truncated,
        )

    def _calculate_generative_tokens(self) -> int:
        """Calcula tokens para modo generativo"""
        desired = min(
            self.max_tokens_limit,
            int(self.model_max_len * (self.generative_percent / 100.0)),
        )

        # Mínimo garantizado para generación (45% del modelo)
        min_for_generation = int(self.model_max_len * 0.45)
        return max(desired, min_for_generation)

    def _calculate_response_tokens(self) -> int:
        """Calcula tokens para modo respuesta"""
        return min(
            self.max_tokens_limit,
            int(self.model_max_len * (self.response_percent / 100.0)),
        )

    def _log_token_breakdown(
        self,
        system_tokens: int,
        context_tokens: int,
        history_tokens: int,
        total_input: int,
        available: int,
        desired: int,
        max_tokens: int,
        is_generative: bool,
    ):
        """Log detallado del desglose de tokens"""
        logger.info("Desglose de tokens:")
        logger.info(f"   - Longitud máxima modelo: {self.model_max_len} tokens")
        logger.info(f"   - Límite max_tokens vLLM: {self.max_tokens_limit}")
        logger.info(f"   - System prompt: ~{system_tokens} tokens")
        logger.info(f"   - Contexto RAG: ~{context_tokens} tokens")
        logger.info(f"   - Historial conversación: ~{history_tokens} tokens")
        logger.info(f"   - TOTAL INPUT: ~{total_input} tokens")
        logger.info(f"   - Disponible para respuesta: {available} tokens")
        logger.info(f"   - max_tokens deseados: {desired} tokens")
        logger.info(f"   - FINAL max_tokens: {max_tokens} tokens")

        # Warnings
        if is_generative and max_tokens < 10000:
            logger.error(
                f"MODO GENERATIVO: Solo {max_tokens} tokens disponibles "
                f"(se necesitan ~15k para 40 preguntas). "
                f"Considerar reducir CTX_TOKENS_GENERATIVE"
            )

        input_percent = (total_input / self.model_max_len) * 100
        if input_percent > 70:
            logger.warning(
                f"Input muy largo ({total_input} tokens, {input_percent:.1f}% del límite)"
            )

    def validate_input_size(self, total_input: int) -> None:
        """
        Valida que el input no sea demasiado grande.

        Args:
            total_input: Tokens totales de input

        Raises:
            ValueError: Si el input excede el límite
        """
        available = self.model_max_len - total_input - self.safety_margin

        if available < self.min_response_tokens:
            raise ValueError(
                f"El contexto de entrada es demasiado largo ({total_input} tokens). "
                f"El modelo solo soporta {self.model_max_len} tokens totales. "
                f"Solo quedan {available} tokens para respuesta "
                f"(mínimo requerido: {self.min_response_tokens})."
            )
