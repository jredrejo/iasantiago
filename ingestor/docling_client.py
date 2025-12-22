# ingestor/docling_client.py
"""
Client for Docling PDF extraction (simplified - direct function calls)
No longer uses HTTP - calls docling_extractor directly
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from docling_extractor import extract_elements_from_pdf, DoclingCrashLimitExceeded

logger = logging.getLogger(__name__)


# Re-export exception for main.py to catch
__all__ = ["DoclingClient", "DoclingCrashLimitExceeded"]


class DoclingClient:
    """
    Client for Docling PDF extraction.
    Now uses direct function calls instead of HTTP to docling-service.
    """

    def __init__(
        self,
        docling_url: str = None,  # Ignored - kept for backwards compatibility
        timeout: float = None,
        enable_fallback: bool = True,
        **kwargs,  # Ignore other legacy parameters
    ):
        self.enable_fallback = enable_fallback
        self._service_available = True  # Always available (local)

        if docling_url:
            logger.info(
                "[DOCLING] Note: docling_url is ignored - using local extraction"
            )

    async def check_health(self) -> bool:
        """Always healthy - local extraction"""
        return True

    async def extract_pdf(
        self, pdf_path: Path, fallback_func: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract elements from PDF using local Docling extraction.

        Args:
            pdf_path: Path to PDF file
            fallback_func: Function to call if Docling fails (legacy - now handled internally)

        Returns:
            List of extracted elements
        """
        try:
            logger.info(f"[DOCLING] Extracting: {pdf_path.name}")
            elements = extract_elements_from_pdf(pdf_path)
            logger.info(
                f"[DOCLING] Extracted {len(elements)} elements from {pdf_path.name}"
            )
            return elements

        except Exception as e:
            logger.error(f"[DOCLING] Extraction failed for {pdf_path.name}: {e}")

            # Fallback is now handled inside extract_elements_from_pdf
            # But keep legacy fallback support
            if self.enable_fallback and fallback_func:
                logger.warning("[DOCLING] Using external fallback function")
                return fallback_func(pdf_path)

            raise

    def extract_pdf_sync(
        self, pdf_path: Path, fallback_func: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous extraction (no async needed for local calls).
        """
        try:
            logger.info(f"[DOCLING] Extracting: {pdf_path.name}")
            elements = extract_elements_from_pdf(pdf_path)
            logger.info(
                f"[DOCLING] Extracted {len(elements)} elements from {pdf_path.name}"
            )
            return elements

        except Exception as e:
            logger.error(f"[DOCLING] Extraction failed for {pdf_path.name}: {e}")

            if self.enable_fallback and fallback_func:
                logger.warning("[DOCLING] Using external fallback function")
                return fallback_func(pdf_path)

            raise
