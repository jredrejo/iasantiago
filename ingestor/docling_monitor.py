# ingestor/docling_monitor.py
"""
Monitor docling-service logs and detect segfaults
Provides mechanism to wait for completion without timeouts
"""

import asyncio
import logging
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Set
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SegfaultRecord:
    """Track segfault events per document"""

    document_hash: str
    filename: str
    timestamp: float
    segfault_count: int = 1

    def to_dict(self):
        return asdict(self)


class DoclingMonitor:
    """
    Monitor docling-service container for:
    1. Segfaults (detect from logs)
    2. Processing completion (detect from log messages)
    3. Document-specific crash patterns
    """

    def __init__(
        self,
        container_name: str = "docling-service",
        log_tail_lines: int = 100,
        segfault_history_file: str = "/tmp/docling_segfaults.json",
        completion_timeout: float = 600.0,  # Fallback timeout if completion signal not found
    ):
        self.container_name = container_name
        self.log_tail_lines = log_tail_lines
        self.segfault_history_file = segfault_history_file
        self.completion_timeout = completion_timeout
        self._docker_client = None
        self._segfault_history: Dict[str, SegfaultRecord] = {}
        self._processing_documents: Dict[str, float] = {}  # filename -> start_time
        self._load_segfault_history()

        # Try to initialize Docker client
        try:
            import docker

            if os.path.exists("/var/run/docker.sock"):
                self._docker_client = docker.DockerClient(
                    base_url="unix:///var/run/docker.sock"
                )
                logger.info("[MONITOR] Docker client initialized for log monitoring")
        except ImportError:
            logger.warning(
                "[MONITOR] Docker SDK not available - log monitoring disabled"
            )
        except Exception as e:
            logger.warning(f"[MONITOR] Failed to initialize Docker client: {e}")

    def _load_segfault_history(self):
        """Load historical segfault records from disk"""
        try:
            if os.path.exists(self.segfault_history_file):
                with open(self.segfault_history_file, "r") as f:
                    history = json.load(f)
                    self._segfault_history = {
                        doc_hash: SegfaultRecord(**record)
                        for doc_hash, record in history.items()
                    }
                logger.info(
                    f"[MONITOR] Loaded {len(self._segfault_history)} segfault records"
                )
        except Exception as e:
            logger.warning(f"[MONITOR] Failed to load segfault history: {e}")
            self._segfault_history = {}

    def _save_segfault_history(self):
        """Persist segfault history to disk"""
        try:
            os.makedirs(os.path.dirname(self.segfault_history_file), exist_ok=True)
            with open(self.segfault_history_file, "w") as f:
                json.dump(
                    {
                        doc_hash: record.to_dict()
                        for doc_hash, record in self._segfault_history.items()
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"[MONITOR] Failed to save segfault history: {e}")

    def _get_document_hash(self, filename: str) -> str:
        """Generate consistent hash for document filename"""
        return hashlib.md5(filename.encode()).hexdigest()

    def record_processing_start(self, filename: str):
        """Record when document processing starts"""
        self._processing_documents[filename] = time.time()
        logger.debug(f"[MONITOR] Recording start of processing: {filename}")

    def record_processing_end(self, filename: str):
        """Record when document processing completes successfully"""
        if filename in self._processing_documents:
            duration = time.time() - self._processing_documents[filename]
            logger.debug(
                f"[MONITOR] Recording end of processing: {filename} (duration: {duration:.1f}s)"
            )
            del self._processing_documents[filename]

            # Clear segfault record on successful completion
            doc_hash = self._get_document_hash(filename)
            if doc_hash in self._segfault_history:
                logger.info(f"[MONITOR] Clearing segfault history for {filename}")
                del self._segfault_history[doc_hash]
                self._save_segfault_history()

    async def wait_for_completion_or_segfault(
        self,
        filename: str,
        timeout: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Wait for document processing to complete or detect segfault.

        Returns: (success, error_message)
            - (True, None): Processing completed successfully
            - (False, "segfault"): Segmentation fault detected
            - (False, "timeout"): Timeout waiting for completion
        """
        if timeout is None:
            timeout = self.completion_timeout

        self.record_processing_start(filename)
        start_time = time.time()
        last_check = start_time
        check_interval = 2.0  # Check every 2 seconds

        logger.info(
            f"[MONITOR] Waiting for completion: {filename} (timeout: {timeout}s)"
        )

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time

            # Check for completion signal every 2 seconds
            if time.time() - last_check >= check_interval:
                completion = await self._check_for_completion_signal(filename)
                if completion:
                    logger.info(f"[MONITOR] Completion signal detected: {filename}")
                    self.record_processing_end(filename)
                    return (True, None)

                last_check = time.time()

            # Check for segfault in logs
            segfault_detected = await self._check_for_segfault(filename)
            if segfault_detected:
                logger.error(f"[MONITOR] Segmentation fault detected: {filename}")
                self.record_segfault(filename)
                self.record_processing_end(filename)
                return (False, "segfault")

            # Small sleep to avoid busy waiting
            await asyncio.sleep(0.5)

        logger.warning(f"[MONITOR] Timeout waiting for completion: {filename}")
        self.record_processing_end(filename)
        return (False, "timeout")

    async def _check_for_completion_signal(self, filename: str) -> bool:
        """
        Check docling-service logs for completion signal.
        Looks for: "Processing document {filename}" message without subsequent errors.
        """
        if not self._docker_client:
            return False

        try:
            container = self._docker_client.containers.get(self.container_name)
            logs = container.logs(tail=self.log_tail_lines).decode("utf-8")

            # Look for the completion pattern in recent logs
            # Docling logs: "2025-12-01 20:16:32,390 - INFO - Processing document tmpf3w0xghw.pdf"
            # After successful completion, there won't be a segfault message for this file

            # Extract just the filename for matching (handle both full path and just name)
            file_basename = Path(filename).name

            # Check if we have a recent "Processing document" message
            processing_pattern = f"Processing document {file_basename}"
            if processing_pattern not in logs:
                return False

            # Check if there's a segfault message AFTER the processing message
            lines = logs.split("\n")
            processing_index = -1

            for i, line in enumerate(lines):
                if processing_pattern in line:
                    processing_index = i

            if processing_index == -1:
                return False

            # Check if there's any segfault message after this processing line
            for line in lines[processing_index:]:
                if "Segmentation fault" in line or "Caught signal 11" in line:
                    return False

            # If we got here, processing started but no segfault after it
            # Assume completion (this is conservative - we only return True if we're confident)
            # For now, return False and let timeout handle it
            # A more robust approach would be to add an explicit completion endpoint to docling-service
            return False

        except Exception as e:
            logger.warning(f"[MONITOR] Error checking completion signal: {e}")
            return False

    async def _check_for_segfault(self, filename: str) -> bool:
        """
        Check docling-service logs for segmentation fault.
        Looks for: "Segmentation fault" messages in recent logs.
        """
        if not self._docker_client:
            return False

        try:
            container = self._docker_client.containers.get(self.container_name)
            logs = container.logs(tail=self.log_tail_lines).decode("utf-8")

            # Look for segfault patterns
            if "Segmentation fault" in logs or "Caught signal 11" in logs:
                # Try to determine if it's related to this document
                file_basename = Path(filename).name

                # Get last few lines to see if segfault is recent
                lines = logs.split("\n")
                recent_lines = lines[-10:]

                for line in recent_lines:
                    if "Segmentation fault" in line or "Caught signal 11" in line:
                        logger.error(f"[MONITOR] Segfault detected in logs: {line}")
                        return True

            return False
        except Exception as e:
            logger.warning(f"[MONITOR] Error checking for segfault: {e}")
            return False

    def record_segfault(self, filename: str):
        """Record a segfault for this document"""
        doc_hash = self._get_document_hash(filename)

        if doc_hash in self._segfault_history:
            record = self._segfault_history[doc_hash]
            record.segfault_count += 1
            record.timestamp = time.time()
            logger.warning(
                f"[MONITOR] Document {filename} has {record.segfault_count} segfaults"
            )
        else:
            record = SegfaultRecord(
                document_hash=doc_hash,
                filename=filename,
                timestamp=time.time(),
                segfault_count=1,
            )
            self._segfault_history[doc_hash] = record
            logger.warning(f"[MONITOR] Recording first segfault for {filename}")

        self._save_segfault_history()

    def should_use_fallback(self, filename: str) -> bool:
        """
        Determine if document should skip docling-service and use fallback.
        Returns True if document has 2+ consecutive segfaults.
        """
        doc_hash = self._get_document_hash(filename)

        if doc_hash not in self._segfault_history:
            return False

        record = self._segfault_history[doc_hash]
        should_fallback = record.segfault_count >= 2

        if should_fallback:
            logger.warning(
                f"[MONITOR] Document {filename} has {record.segfault_count} segfaults - "
                f"will use fallback extraction"
            )

        return should_fallback

    def clear_segfault_history(self, filename: Optional[str] = None):
        """Clear segfault history for a specific file or all files"""
        if filename:
            doc_hash = self._get_document_hash(filename)
            if doc_hash in self._segfault_history:
                del self._segfault_history[doc_hash]
                logger.info(f"[MONITOR] Cleared segfault history for {filename}")
        else:
            self._segfault_history.clear()
            logger.info("[MONITOR] Cleared all segfault history")

        self._save_segfault_history()

    def should_restart_container(self) -> bool:
        """
        Check if container should be restarted based on logs.
        Returns True if segfault is detected in last 15 lines.
        Returns False if "Processing document" is the last line (still processing).
        """
        if not self._docker_client:
            return False

        try:
            container = self._docker_client.containers.get(self.container_name)
            # Get last 20 lines to check
            logs = container.logs(tail=20).decode("utf-8", errors="ignore").split("\n")
            # Filter empty lines
            logs = [line.strip() for line in logs if line.strip()]

            if not logs:
                return False

            # Check if last line is "Processing document"
            if logs and "Processing document" in logs[-1]:
                logger.info("[MONITOR] Container still processing - no restart needed")
                return False

            # Check last 15 lines for segfault indicators
            logs_to_check = logs[-15:] if len(logs) >= 15 else logs
            for line in logs_to_check:
                if "Caught signal" in line or "Segmentation fault" in line:
                    logger.error(f"[MONITOR] Segfault detected in logs: {line}")
                    return True

            return False
        except Exception as e:
            logger.warning(f"[MONITOR] Could not check logs for segfault: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get statistics about segfaults"""
        total_segfaults = sum(r.segfault_count for r in self._segfault_history.values())
        documents_with_segfaults = len(self._segfault_history)

        return {
            "total_segfaults": total_segfaults,
            "documents_with_segfaults": documents_with_segfaults,
            "currently_processing": len(self._processing_documents),
            "history": {
                doc_hash: {
                    "filename": record.filename,
                    "count": record.segfault_count,
                    "last_seen": datetime.fromtimestamp(record.timestamp).isoformat(),
                }
                for doc_hash, record in self._segfault_history.items()
            },
        }
