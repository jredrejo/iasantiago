# Archivo: rag-api/core/vllm_client.py
# Descripción: Cliente unificado para comunicación con vLLM

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import HTTPException

from core.retry import RetryConfig, with_retry
from core.cache import VLLMState

logger = logging.getLogger(__name__)

# Excepciones que ameritan reintento
RETRIABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


class VLLMClient:
    """
    Cliente asíncrono para comunicación con vLLM.

    Centraliza toda la lógica de:
    - Health checks
    - Espera de modelo listo
    - Llamadas con reintentos
    - Streaming SSE

    Uso:
        client = VLLMClient(base_url="http://vllm:8000/v1")
        await client.wait_for_model_ready("Qwen/Qwen2.5-7B-Instruct")
        response = await client.chat_completion(payload, stream=False)
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: float = 300.0,
        connect_timeout: float = 20.0,
    ):
        """
        Inicializa el cliente vLLM.

        Args:
            base_url: URL base de la API (ej: "http://vllm:8000/v1")
            api_key: API key para autenticación
            timeout: Timeout para requests (default: 300s)
            connect_timeout: Timeout para conexión (default: 20s)
        """
        self.base_url = base_url or os.getenv("UPSTREAM_OPENAI_URL", "http://vllm:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self.streaming_timeout = httpx.Timeout(600.0, connect=connect_timeout)

    @property
    def health_url(self) -> str:
        """URL para health check (sin /v1)"""
        return self.base_url.replace("/v1", "") + "/health"

    @property
    def models_url(self) -> str:
        """URL para listar modelos"""
        return f"{self.base_url}/models"

    @property
    def completions_url(self) -> str:
        """URL para chat completions"""
        return f"{self.base_url}/chat/completions"

    @property
    def headers(self) -> Dict[str, str]:
        """Headers para requests"""
        return {"Authorization": f"Bearer {self.api_key}"}

    async def check_health(self, max_retries: int = 3) -> bool:
        """
        Verifica si vLLM está disponible.

        Args:
            max_retries: Número máximo de reintentos

        Returns:
            True si vLLM responde correctamente
        """
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=2.0,
            exceptions=RETRIABLE_EXCEPTIONS + (Exception,),
        )

        async def _check():
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(self.health_url)
                if resp.status_code == 200:
                    return True
                raise Exception(f"Health check devolvió status {resp.status_code}")

        try:
            return await with_retry(
                _check,
                config=config,
                operation_name="health check vLLM",
            )
        except Exception as e:
            logger.error(f"vLLM no responde después de {max_retries} intentos: {e}")
            return False

    async def wait_for_model_ready(
        self,
        model_name: str,
        max_wait_seconds: int = 300,
        check_interval: float = 2.0,
    ) -> bool:
        """
        Espera a que un modelo esté listo en vLLM.

        Args:
            model_name: Nombre del modelo (ej: "Qwen/Qwen2.5-7B-Instruct")
            max_wait_seconds: Tiempo máximo de espera
            check_interval: Intervalo entre verificaciones

        Returns:
            True si el modelo está listo, False si timeout
        """
        timeout = httpx.Timeout(10.0, connect=5.0)
        elapsed = 0
        attempt = 0

        logger.info(
            f"Esperando a que el modelo '{model_name}' esté listo "
            f"(máximo {max_wait_seconds}s)..."
        )

        while elapsed < max_wait_seconds:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    # Verificar health
                    try:
                        health_resp = await client.get(self.health_url)
                        if health_resp.status_code != 200:
                            logger.debug(
                                f"[{attempt}] Health check falló (status {health_resp.status_code})"
                            )
                            await asyncio.sleep(check_interval)
                            elapsed += check_interval
                            continue
                    except Exception as e:
                        logger.debug(f"[{attempt}] Error en health check: {e}")
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval
                        continue

                    # Verificar lista de modelos
                    try:
                        models_resp = await client.get(self.models_url)
                        if models_resp.status_code != 200:
                            logger.debug(
                                f"[{attempt}] No se pudo obtener lista de modelos "
                                f"(status {models_resp.status_code})"
                            )
                            await asyncio.sleep(check_interval)
                            elapsed += check_interval
                            continue

                        models_data = models_resp.json()
                        available_models = [m["id"] for m in models_data.get("data", [])]

                        if model_name in available_models:
                            logger.info(
                                f"Modelo '{model_name}' LISTO (tardó {elapsed:.0f}s)"
                            )
                            return True
                        else:
                            logger.debug(
                                f"[{attempt}] Modelo '{model_name}' no disponible aún. "
                                f"Disponibles: {available_models}. "
                                f"Esperando... ({elapsed:.0f}s/{max_wait_seconds}s)"
                            )
                            await asyncio.sleep(check_interval)
                            elapsed += check_interval

                    except Exception as e:
                        logger.debug(f"[{attempt}] Error parseando respuesta de modelos: {e}")
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval

            except Exception as e:
                logger.debug(f"[{attempt}] Error inesperado verificando modelo: {e}")
                await asyncio.sleep(check_interval)
                elapsed += check_interval

        logger.error(
            f"Modelo '{model_name}' no estuvo listo después de {max_wait_seconds}s"
        )
        return False

    async def ensure_model_ready(self, model_name: str) -> None:
        """
        Asegura que el modelo esté listo, esperando si hay cambio de modelo.

        Args:
            model_name: Modelo requerido

        Raises:
            HTTPException: Si el modelo no está listo después del timeout
        """
        # Verificar si hay cambio de modelo
        model_changed = await VLLMState.check_model_change(model_name)

        if model_changed:
            logger.info("Esperando a que el nuevo modelo esté listo...")
            model_ready = await self.wait_for_model_ready(model_name)

            if not model_ready:
                current = await VLLMState.get_current_model()
                raise HTTPException(
                    status_code=503,
                    detail=f"El modelo '{model_name}' no está listo. "
                    f"Actualmente cargando (cambiando desde '{current}'). "
                    f"Por favor intente de nuevo en un momento.",
                )

            await VLLMState.set_current_model(model_name)
            logger.info(f"Cambiado a modelo: {model_name}")

    async def chat_completion(
        self,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> httpx.Response:
        """
        Ejecuta una llamada de chat completion sin streaming.

        Args:
            payload: Payload de la request
            max_retries: Número máximo de reintentos

        Returns:
            Respuesta HTTP

        Raises:
            HTTPException: Si la llamada falla después de reintentos
        """
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=2.0,
            exceptions=RETRIABLE_EXCEPTIONS,
        )

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.completions_url,
                    headers=self.headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp

        try:
            return await with_retry(
                _call,
                config=config,
                operation_name="chat completion vLLM",
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP de vLLM: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Error de vLLM: {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Servicio vLLM no disponible: {str(e)}",
            )

    async def stream_chat_completion(
        self,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> AsyncIterator[bytes]:
        """
        Genera un stream de chat completion (SSE).

        Args:
            payload: Payload de la request (debe incluir stream=True)
            max_retries: Número máximo de reintentos

        Yields:
            Chunks de bytes del stream SSE
        """
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.streaming_timeout) as client:
                    async with client.stream(
                        "POST",
                        self.completions_url,
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        # Manejar error 404 (modelo no encontrado)
                        if response.status_code == 404:
                            logger.error(
                                f"404: Modelo '{payload.get('model')}' no encontrado"
                            )
                            error_data = {
                                "error": {
                                    "message": f"Modelo '{payload.get('model')}' no disponible. "
                                    f"Puede estar cargándose después de un cambio. "
                                    f"Por favor intente de nuevo.",
                                    "type": "model_not_ready",
                                    "code": 404,
                                }
                            }
                            yield f"data: {json.dumps(error_data)}\n\n".encode()
                            return

                        response.raise_for_status()
                        logger.info(f"Stream establecido con vLLM (intento {attempt + 1})")

                        async for chunk in response.aiter_bytes():
                            yield chunk

                        logger.info("Stream completado exitosamente")
                        return

            except RETRIABLE_EXCEPTIONS as e:
                if attempt == max_retries - 1:
                    logger.error(f"Stream falló después de {max_retries} intentos: {e}")
                    error_data = {
                        "error": {
                            "message": f"Conexión con vLLM falló después de {max_retries} reintentos: {str(e)}",
                            "type": "connection_error",
                            "code": 503,
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n".encode()
                    return

                wait_time = 2**attempt
                logger.warning(
                    f"Stream interrumpido (intento {attempt + 1}/{max_retries}), "
                    f"reintentando en {wait_time}s... Error: {type(e).__name__}"
                )
                await asyncio.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                try:
                    error_text = await e.response.aread()
                    error_text = error_text.decode("utf-8", errors="ignore")
                except Exception:
                    error_text = "<no legible>"

                logger.error(f"Error HTTP de vLLM: {e.response.status_code} - {error_text}")
                error_data = {
                    "error": {
                        "message": f"Error vLLM: HTTP {e.response.status_code} - {error_text}",
                        "type": "upstream_error",
                        "code": e.response.status_code,
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                return

            except Exception as e:
                logger.error(f"Error inesperado en streaming: {str(e)}", exc_info=True)
                error_data = {
                    "error": {
                        "message": f"Error de streaming: {str(e)}",
                        "type": "internal_error",
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                return


# Instancia global del cliente (se puede configurar al inicio)
_vllm_client: Optional[VLLMClient] = None


def get_vllm_client() -> VLLMClient:
    """Obtiene la instancia global del cliente vLLM"""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient()
    return _vllm_client


def configure_vllm_client(**kwargs) -> VLLMClient:
    """Configura y retorna una nueva instancia del cliente vLLM"""
    global _vllm_client
    _vllm_client = VLLMClient(**kwargs)
    return _vllm_client
