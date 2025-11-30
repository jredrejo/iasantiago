# ingestor/docling_client.py (NEW FILE)
"""
Client for Docling microservice
Provides fallback to existing extraction methods
"""

import httpx
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DoclingClient:
    """
    Client for Docling PDF extraction service
    With automatic fallback to existing methods
    """

    def __init__(
        self,
        docling_url: str = "http://docling-service:8003",
        timeout: float = 960.0,  # 16 minutes (15 min + 1 min buffer)
        enable_fallback: bool = True,
        max_retries: int = 1,  # Reduced retries for long-running jobs
        retry_delay: float = 10.0,  # Longer delay between retries
    ):
        self.docling_url = docling_url
        # Extended timeouts for large PDF processing (up to 15 minutes)
        self.timeout = httpx.Timeout(
            timeout=timeout,
            connect=60.0,  # Increased connection timeout
            read=timeout,  # Use full timeout for read operations
            write=120.0,  # Increased write timeout for large file uploads
        )
        self.enable_fallback = enable_fallback
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._service_available = None

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

        # Estimate processing time based on file size
        estimated_minutes = max(
            1, int(file_size_mb * 0.5)
        )  # Rough estimate: 0.5 min per MB
        if estimated_minutes > 5:
            logger.info(
                f"[DOCLING] Large file detected - estimated processing time: ~{estimated_minutes} minutes"
            )

        # Retry logic for extraction
        for attempt in range(self.max_retries + 1):
            start_time = time.time()

            try:
                logger.info(
                    f"[DOCLING] Extracting: {pdf_path.name} ({file_size_mb:.1f}MB, attempt {attempt + 1}/{self.max_retries + 1})"
                )
                if estimated_minutes > 5:
                    logger.info(
                        f"[DOCLING] Processing large file - please wait (estimated {estimated_minutes} minutes)..."
                    )

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    with open(pdf_path, "rb") as f:
                        files = {"file": (pdf_path.name, f, "application/pdf")}

                        logger.debug(
                            f"[DOCLING] Sending request to {self.docling_url}/extract"
                        )
                        resp = await client.post(
                            f"{self.docling_url}/extract", files=files
                        )

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
                        return elements

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    f"[DOCLING] Connection failed (attempt {attempt + 1}): {e}"
                )
                self._service_available = False  # Mark as unavailable

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"[DOCLING] Timeout after {self.timeout.timeout}s (attempt {attempt + 1}): {e}"
                )

            except httpx.RemoteProtocolError as e:
                last_error = e
                logger.warning(
                    f"[DOCLING] Server disconnected (attempt {attempt + 1}): {e}"
                )
                self._service_available = False  # Likely service crashed

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
