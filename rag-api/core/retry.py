# Archivo: rag-api/core/retry.py
# Descripción: Lógica de reintentos con backoff exponencial centralizada

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuración para reintentos"""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    exceptions: Tuple[Type[Exception], ...] = (Exception,)


async def with_retry(
    operation: Callable,
    config: Optional[RetryConfig] = None,
    operation_name: str = "operación",
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Any:
    """
    Ejecuta una operación asíncrona con reintentos exponenciales.

    Args:
        operation: Función asíncrona a ejecutar (sin argumentos, usar lambda si es necesario)
        config: Configuración de reintentos (usa valores por defecto si es None)
        operation_name: Nombre descriptivo para los logs
        on_retry: Callback opcional que se ejecuta antes de cada reintento

    Returns:
        Resultado de la operación si tiene éxito

    Raises:
        Exception: Si todos los reintentos fallan, lanza la última excepción

    Ejemplo:
        result = await with_retry(
            lambda: client.get(url),
            config=RetryConfig(max_retries=5),
            operation_name="obtener datos"
        )
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for intento in range(config.max_retries):
        try:
            return await operation()
        except config.exceptions as e:
            last_exception = e

            if intento == config.max_retries - 1:
                logger.error(
                    f"Operación '{operation_name}' fallida después de "
                    f"{config.max_retries} intentos: {e}"
                )
                raise

            # Calcular tiempo de espera con backoff exponencial
            tiempo_espera = min(config.base_delay * (2**intento), config.max_delay)

            logger.warning(
                f"Intento {intento + 1}/{config.max_retries} de '{operation_name}' "
                f"fallido, reintentando en {tiempo_espera:.1f}s... "
                f"Error: {type(e).__name__}: {e}"
            )

            if on_retry:
                on_retry(intento, e)

            await asyncio.sleep(tiempo_espera)

    # Este punto no debería alcanzarse, pero por seguridad
    if last_exception:
        raise last_exception


def sync_with_retry(
    operation: Callable,
    config: Optional[RetryConfig] = None,
    operation_name: str = "operación",
) -> Any:
    """
    Versión síncrona de with_retry para operaciones bloqueantes.

    Args:
        operation: Función síncrona a ejecutar
        config: Configuración de reintentos
        operation_name: Nombre descriptivo para los logs

    Returns:
        Resultado de la operación si tiene éxito
    """
    import time

    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for intento in range(config.max_retries):
        try:
            return operation()
        except config.exceptions as e:
            last_exception = e

            if intento == config.max_retries - 1:
                logger.error(
                    f"Operación '{operation_name}' fallida después de "
                    f"{config.max_retries} intentos: {e}"
                )
                raise

            tiempo_espera = min(config.base_delay * (2**intento), config.max_delay)

            logger.warning(
                f"Intento {intento + 1}/{config.max_retries} de '{operation_name}' "
                f"fallido, reintentando en {tiempo_espera:.1f}s... "
                f"Error: {type(e).__name__}: {e}"
            )

            time.sleep(tiempo_espera)

    if last_exception:
        raise last_exception
