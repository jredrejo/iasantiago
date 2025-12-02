# ingestor/docling_client.py (NEW FILE)
"""
Client for Docling microservice
Provides fallback to existing extraction methods
"""

import httpx
import logging
import asyncio
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import docker
except ImportError:
    docker = None

from docling_monitor import DoclingMonitor

logger = logging.getLogger(__name__)


class DoclingClient:
    """
    Client for Docling PDF extraction service
    With automatic fallback to existing methods
    Can restart the docling-service container if it crashes
    """

    def __init__(
        self,
        docling_url: str = "http://docling-service:8003",
        timeout: float = None,  # Will be calculated per PDF
        enable_fallback: bool = True,
        max_retries: int = 2,  # Reduced retries for long-running jobs
        retry_delay: float = 50.0,  # Longer delay between retries
        min_timeout: float = 300.0,  # Minimum timeout of 5 minutes for docling processing
        max_timeout: float = 900.0,  # Maximum timeout (15 minutes)
        timeout_per_mb: float = 5.0,  # Seconds per MB of PDF
        container_name: str = "docling-service",  # Docker container name
        enable_segfault_detection: bool = True,  # Enable segfault monitoring
    ):
        self.docling_url = docling_url
        self.base_timeout = timeout  # For legacy use (if explicitly set)
        self.min_timeout = min_timeout  # 5 minutes minimum
        self.max_timeout = max_timeout  # 15 minutes maximum
        self.timeout_per_mb = timeout_per_mb  # 5 seconds per MB
        self.enable_fallback = enable_fallback
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.container_name = container_name
        self._service_available = None
        self._docker_client = None
        self._docker_available = docker is not None and os.path.exists(
            "/var/run/docker.sock"
        )

        # Initialize segfault monitoring
        self.monitor = (
            DoclingMonitor(container_name=container_name)
            if enable_segfault_detection
            else None
        )

        if self._docker_available:
            try:
                self._docker_client = docker.DockerClient(
                    base_url="unix:///var/run/docker.sock"
                )
                # Test connection
                self._docker_client.ping()
                logger.info("[DOCLING] Docker API connection established")
            except Exception as e:
                logger.warning(f"[DOCLING] Failed to connect to Docker API: {e}")
                self._docker_available = False
                self._docker_client = None

    def _calculate_timeout(self, file_size_mb: float) -> tuple[httpx.Timeout, float]:
        """
        Calculate timeout proportional to PDF size.

        Returns: (httpx.Timeout object, timeout_seconds as float)

        Formula: timeout = min(max_timeout, max(min_timeout, size_mb * timeout_per_mb + buffer))

        Examples:
          - 0.1 MB (100 KB)  → 300s (5 minutes minimum)
          - 1 MB             → 305s (1 * 5 + 30s base, rounded to 5 min)
          - 10 MB            → 350s (10 * 5 + 30s base, min 5 min)
          - 100 MB           → 530s (100 * 5 + 30s base, capped at 15 min)
          - 200 MB           → 900s (capped at 15 minutes maximum)
        """
        # Calculate based on file size
        buffer = 30.0  # 30 second buffer for overhead
        calculated_timeout = (file_size_mb * self.timeout_per_mb) + buffer

        # Apply min/max bounds (min 5 minutes, max 15 minutes)
        timeout_seconds = max(
            self.min_timeout, min(self.max_timeout, calculated_timeout)
        )

        logger.info(
            f"[DOCLING] Timeout calculation: {file_size_mb:.1f}MB → "
            f"{timeout_seconds:.0f}s "
            f"(formula: {file_size_mb:.1f} * {self.timeout_per_mb} + {buffer} = {calculated_timeout:.0f}s)"
        )

        # Create timeout object with proportional values
        timeout_obj = httpx.Timeout(
            timeout=timeout_seconds,
            connect=60.0,  # Connection timeout (fixed)
            read=timeout_seconds,  # Read timeout (proportional)
            write=min(
                120.0, timeout_seconds / 2
            ),  # Write timeout (proportional, max 120s)
        )

        return timeout_obj, timeout_seconds

    def _restart_container(self) -> bool:
        """
        Restart the docling-service container after a crash.
        Uses monitor to check logs before restarting.
        Returns True if restart was successful.
        """
        if not self._docker_available or not self._docker_client:
            logger.warning(
                "[DOCLING] Docker API not available - cannot restart container"
            )
            return False

        # Check logs before restarting (using monitor)
        if self.monitor and not self.monitor.should_restart_container():
            logger.info("[DOCLING] No segfault detected in logs - skipping restart")
            return False

        try:
            logger.info(
                f"[DOCLING] Attempting to restart container: {self.container_name}"
            )
            container = self._docker_client.containers.get(self.container_name)
            container.restart()
            logger.info(f"[DOCLING] ✓ Container restart command sent")
            return True

        except docker.errors.NotFound:
            logger.error(f"[DOCLING] ✗ Container '{self.container_name}' not found")
            return False
        except Exception as e:
            logger.error(f"[DOCLING] Error restarting container: {e}")
            return False

    async def check_health(self) -> bool:
        """Check if Docling service is available with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                # Use increased timeout for health check (service might be busy)
                timeout = httpx.Timeout(30.0, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    logger.debug(f"[DOCLING] Health check attempt {attempt + 1}")
                    resp = await client.get(f"{self.docling_url}/health")

                    if resp.status_code == 200:
                        logger.debug(f"[DOCLING] Health check passed")
                        return True
                    else:
                        logger.warning(
                            f"[DOCLING] Health check returned {resp.status_code}"
                        )

            except httpx.ConnectError as e:
                logger.warning(
                    f"[DOCLING] Health check connection failed (attempt {attempt + 1}): {e}"
                )
            except httpx.TimeoutException as e:
                logger.warning(
                    f"[DOCLING] Health check timeout (attempt {attempt + 1}): {e}"
                )
            except Exception as e:
                logger.warning(
                    f"[DOCLING] Health check failed (attempt {attempt + 1}): {e}"
                )

            if attempt < self.max_retries:
                logger.info(
                    f"[DOCLING] Retrying health check in {self.retry_delay}s..."
                )
                await asyncio.sleep(self.retry_delay)

        return False

    async def extract_pdf(
        self, pdf_path: Path, fallback_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract elements from PDF using Docling service with retry logic

        Args:
            pdf_path: Path to PDF file
            fallback_func: Function to call if Docling fails (existing extraction)

        Returns:
            List of extracted elements
        """

        # Check if document has history of consecutive segfaults
        if self.monitor and self.monitor.should_use_fallback(str(pdf_path.name)):
            logger.warning(
                f"[DOCLING] Document {pdf_path.name} has 2+ consecutive segfaults - using fallback only"
            )
            if self.enable_fallback and fallback_func:
                logger.info(
                    "[DOCLING] Using fallback extraction due to segfault history"
                )
                return fallback_func(pdf_path)
            raise RuntimeError(
                f"Document skipped due to segfault history and no fallback available"
            )

        # Check service health if not cached
        if self._service_available is None:
            self._service_available = await self.check_health()

            if not self._service_available:
                logger.warning("[DOCLING] Service not available")
                if self.enable_fallback and fallback_func:
                    logger.info("[DOCLING] Using fallback extraction")
                    return fallback_func(pdf_path)
                raise RuntimeError("Docling service unavailable and no fallback")

        last_error = None
        file_size_mb = pdf_path.stat().st_size / 1_000_000


        # Calculate dynamic timeout based on file size
        timeout_obj, timeout_seconds = self._calculate_timeout(file_size_mb)

        # Estimate processing time based on file size
        estimated_seconds = (
            file_size_mb * 0.5 * 60
        )  # 0.5 minutes per MB = 30 seconds per MB
        estimated_minutes = estimated_seconds / 60

        # For VERY large files, the monitor timeout must be much longer than HTTP timeout
        # because docling_service processes at variable speed depending on content complexity
        # Use the maximum of:
        # 1. HTTP timeout (for network reliability)
        # 2. Estimated processing time (for actual document processing)
        # Plus 10 minute buffer for overhead
        buffer_seconds = 600
        monitor_timeout_seconds = max(
            timeout_seconds, estimated_seconds + buffer_seconds
        )

        if estimated_minutes > 5:
            logger.info(
                f"[DOCLING] Large file detected - estimated processing time: ~{estimated_minutes:.1f} minutes"
            )
            logger.info(
                f"[DOCLING] HTTP timeout: {timeout_seconds:.0f}s, Monitor timeout: {monitor_timeout_seconds:.0f}s ({monitor_timeout_seconds/60:.1f} minutes)"
            )

        # Retry logic for extraction
        for attempt in range(self.max_retries + 1):
            start_time = time.time()

            try:
                logger.info(
                    f"[DOCLING] Extracting: {pdf_path.name} ({file_size_mb:.1f}MB, attempt {attempt + 1}/{self.max_retries + 1}, timeout={timeout_seconds:.0f}s)"
                )
                if estimated_minutes > 5:
                    logger.info(
                        f"[DOCLING] Processing large file - please wait (estimated {estimated_minutes:.1f} minutes)..."
                    )

                # Send extraction request (don't wait for response yet)
                async with httpx.AsyncClient(timeout=timeout_obj) as client:
                    with open(pdf_path, "rb") as f:
                        files = {"file": (pdf_path.name, f, "application/pdf")}

                        logger.debug(
                            f"[DOCLING] Sending request to {self.docling_url}/extract"
                        )

                        # Create a task for the HTTP request
                        request_task = asyncio.create_task(
                            client.post(f"{self.docling_url}/extract", files=files)
                        )

                        # Concurrently monitor for segfaults while waiting for response
                        monitor_task = None
                        if self.monitor:
                            monitor_task = asyncio.create_task(
                                self.monitor.wait_for_completion_or_segfault(
                                    pdf_path.name, timeout=monitor_timeout_seconds
                                )
                            )

                        # Wait for either request to complete or segfault to be detected
                        if monitor_task:
                            done, pending = await asyncio.wait(
                                [request_task, monitor_task],
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            # If monitor detected a segfault first, cancel the request
                            if monitor_task in done:
                                monitor_result, monitor_error = await monitor_task
                                if not monitor_result and monitor_error:
                                    # Cancel the pending request task
                                    request_task.cancel()
                                    if request_task in pending:
                                        try:
                                            await request_task
                                        except asyncio.CancelledError:
                                            pass

                                    # If it's a segfault, mark for fallback
                                    if monitor_error == "segfault":
                                        logger.error(
                                            f"[DOCLING] Segfault detected during processing"
                                        )
                                        # Try to restart container if on first attempt
                                        if attempt == 0:
                                            if self._restart_container():
                                                logger.info(
                                                    "[DOCLING] Container restart command sent"
                                                )
                                            else:
                                                logger.warning(
                                                    "[DOCLING] Could not restart container (Docker API unavailable)"
                                                )
                                                logger.info(
                                                    "[DOCLING] Waiting for container to auto-restart..."
                                                )

                                            # Wait for restart regardless of explicit restart success
                                            await asyncio.sleep(15)
                                            if await self.check_health():
                                                logger.info(
                                                    "[DOCLING] Service recovered - retrying extraction"
                                                )
                                                continue
                                            else:
                                                logger.warning(
                                                    "[DOCLING] Service still unhealthy after restart attempt"
                                                )

                                        # Mark unavailable and allow retry logic to handle
                                        self._service_available = False
                                    elif monitor_error == "timeout":
                                        logger.error(
                                            f"[DOCLING] Timeout waiting for completion"
                                        )
                                        self._service_available = False
                                        break

                            # Wait for request to complete
                            await asyncio.gather(*done)

                        # Get response from request task
                        resp = await request_task
                        logger.debug(f"[DOCLING] Response status: {resp.status_code}")

                        if resp.status_code != 200:
                            error_msg = f"Docling service returned {resp.status_code}: {resp.text[:500]}"
                            logger.error(f"[DOCLING] {error_msg}")
                            raise RuntimeError(error_msg)

                        data = resp.json()

                        if not data.get("success"):
                            error_msg = f"Docling extraction failed: {data.get('error', 'Unknown error')}"
                            logger.error(f"[DOCLING] {error_msg}")
                            raise RuntimeError(error_msg)

                        elements = data["elements"]
                        stats = data["stats"]

                        processing_time = time.time() - start_time
                        processing_minutes = processing_time / 60

                        logger.info(
                            f"[DOCLING] ✓ Extracted {stats['total_elements']} elements "
                            f"from {stats['pages']} pages in {file_size_mb:.1f}MB file"
                        )
                        logger.info(
                            f"[DOCLING] Processing time: {processing_minutes:.1f} minutes"
                        )
                        logger.info(f"[DOCLING] By type: {stats['by_type']}")

                        # Mark service as available on success
                        self._service_available = True

                        # Record successful completion in monitor
                        if self.monitor:
                            self.monitor.record_processing_end(pdf_path.name)

                        return elements

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    f"[DOCLING] Connection failed (attempt {attempt + 1}): {e}"
                )
                self._service_available = False  # Mark as unavailable

            except httpx.TimeoutException as e:
                last_error = e
                logger.error(
                    f"[DOCLING] Timeout after {timeout_seconds:.0f}s (attempt {attempt + 1}) - likely segfault: {e}"
                )
                self._service_available = False

                # Timeout often indicates segfault - try to restart container on first attempt
                if attempt == 0:
                    if self._restart_container():
                        logger.info("[DOCLING] Container restart command sent")
                    else:
                        logger.warning(
                            "[DOCLING] Could not restart container (Docker API unavailable)"
                        )
                        logger.info(
                            "[DOCLING] Waiting for container to auto-restart..."
                        )

                    # Wait for restart regardless of explicit restart success
                    await asyncio.sleep(15)
                    if await self.check_health():
                        logger.info("[DOCLING] Service recovered - retrying extraction")
                        continue
                    else:
                        logger.warning(
                            "[DOCLING] Service still unhealthy after restart attempt"
                        )

            except httpx.RemoteProtocolError as e:
                last_error = e
                logger.error(
                    f"[DOCLING] Server disconnected - likely service crashed/segfault (attempt {attempt + 1}): {e}"
                )
                self._service_available = False  # Likely service crashed (segfault)

                # Try to restart the container on first attempt
                if attempt == 0:
                    if self._restart_container():
                        logger.info("[DOCLING] Container restart command sent")
                    else:
                        logger.warning(
                            "[DOCLING] Could not restart container (Docker API unavailable)"
                        )
                        logger.info(
                            "[DOCLING] Waiting for container to auto-restart..."
                        )

                    # Wait for restart regardless of explicit restart success
                    await asyncio.sleep(15)
                    if await self.check_health():
                        logger.info("[DOCLING] Service recovered - retrying extraction")
                        continue
                    else:
                        logger.warning(
                            "[DOCLING] Service still unhealthy after restart attempt"
                        )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[DOCLING] Extraction failed (attempt {attempt + 1}): {e}"
                )

            # Retry logic
            if attempt < self.max_retries:
                logger.info(f"[DOCLING] Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)

                # Re-check health before retry
                if await self.check_health():
                    logger.info(f"[DOCLING] Service health restored")
                else:
                    logger.warning(f"[DOCLING] Service still unhealthy")
                    self._service_available = False
                    break

        # All retries failed
        logger.error(f"[DOCLING] All {self.max_retries + 1} attempts failed")

        # Fallback to existing method
        if self.enable_fallback and fallback_func:
            logger.warning("[DOCLING] Falling back to existing extraction")
            return fallback_func(pdf_path)

        raise RuntimeError(
            f"Docling service failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def extract_pdf_sync(
        self, pdf_path: Path, fallback_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for extract_pdf
        For use in existing sync code
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.extract_pdf(pdf_path, fallback_func))
