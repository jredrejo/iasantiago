# Archivo: rag-api/core/vllm_client.py
# Descripción: Cliente unificado para comunicación con vLLM

import asyncio
import contextlib
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import HTTPException

from core.retry import RetryConfig, with_retry

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
    - Llamadas con reintentos
    - Streaming SSE

    Uso:
        client = VLLMClient(base_url="http://vllm:8000/v1")
        response = await client.chat_completion(payload, stream=False)
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: float = 300.0,
        connect_timeout: float = 20.0,
        httpx_client=None,
    ):
        """
        Inicializa el cliente vLLM.

        Args:
            base_url: URL base de la API (ej: "http://vllm:8000/v1")
            api_key: API key para autenticación
            timeout: Timeout para requests (default: 300s)
            connect_timeout: Timeout para conexión (default: 20s)
            httpx_client: Cliente httpx compartido (opcional, para testing)
        """
        self.base_url = base_url or os.getenv(
            "UPSTREAM_OPENAI_URL", "http://vllm:8000/v1"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self.streaming_timeout = httpx.Timeout(600.0, connect=connect_timeout)
        self.httpx_client = httpx_client  # Cliente compartido (opcional)

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

    @contextlib.asynccontextmanager
    async def _client(self, timeout: Optional[httpx.Timeout] = None):
        """
        Proporciona el cliente httpx apropiado: el compartido si existe,
        o uno temporal que se cierra al salir del contexto.

        Nota: el timeout del cliente compartido se ignora; cada request
        debe pasar su propio timeout explícito.

        Args:
            timeout: Timeout para el cliente temporal (opcional)
        """
        if self.httpx_client is not None:
            yield self.httpx_client
        else:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                yield client

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
            timeout = httpx.Timeout(10.0)
            async with self._client(timeout) as client:
                resp = await client.get(self.health_url, timeout=timeout)
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
            async with self._client() as client:
                resp = await client.post(
                    self.completions_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
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
            # El detalle crudo de vLLM se registra pero NO se devuelve al cliente
            # (puede filtrar rutas de modelo, trazas internas, etc.).
            logger.error(
                f"Error HTTP de vLLM: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail="Error al procesar la solicitud en el servicio de lenguaje.",
            )
        except Exception as e:
            logger.error(f"Servicio vLLM no disponible: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Servicio de lenguaje no disponible temporalmente.",
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
                async with self._client(self.streaming_timeout) as client:
                    async with client.stream(
                        "POST",
                        self.completions_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.streaming_timeout,
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
                        logger.info(
                            f"Stream establecido con vLLM (intento {attempt + 1})"
                        )

                        async for chunk in response.aiter_bytes():
                            yield chunk

                        logger.info("Stream completado exitosamente")
                        return

            except RETRIABLE_EXCEPTIONS as e:
                if attempt == max_retries - 1:
                    logger.error(f"Stream falló después de {max_retries} intentos: {e}")
                    error_data = {
                        "error": {
                            "message": "Servicio de lenguaje no disponible temporalmente.",
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

                logger.error(
                    f"Error HTTP de vLLM: {e.response.status_code} - {error_text}"
                )
                error_data = {
                    "error": {
                        "message": "Error al procesar la solicitud en el servicio de lenguaje.",
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
                        "message": "Error interno al generar la respuesta.",
                        "type": "internal_error",
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                return


# Instancia global del cliente (se puede configurar al inicio)
_vllm_client: Optional[VLLMClient] = None


def get_vllm_client(httpx_client=None) -> VLLMClient:
    """
    Obtiene la instancia global del cliente vLLM.

    Args:
        httpx_client: Cliente httpx compartido (opcional, desde lifespan)
    """
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient(httpx_client=httpx_client)
    return _vllm_client


def configure_vllm_client(**kwargs) -> VLLMClient:
    """Configura y retorna una nueva instancia del cliente vLLM"""
    global _vllm_client
    _vllm_client = VLLMClient(**kwargs)
    return _vllm_client
