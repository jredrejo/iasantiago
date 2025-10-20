"""
Watcher de nuevos PDFs.
Monitoriza un directorio y si llegan nuevos PDFs, los indexa usando la función index_pdf.
"""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time, os
import torch
from main import index_pdf
from settings import TOPIC_LABELS, TOPIC_BASE_DIR
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing = set()  # Track files being processed

    def on_created(self, event):
        if event.is_directory:
            return

        if not event.src_path.lower().endswith(".pdf"):
            return

        abs_path = os.path.abspath(event.src_path)

        # Avoid duplicate processing
        if abs_path in self.processing:
            logger.warning(f"[SKIP] {abs_path} is already being processed")
            return

        # Find the topic this file belongs to
        topic_found = False
        for t in TOPIC_LABELS:
            tdir = os.path.join(TOPIC_BASE_DIR, t)
            if event.src_path.startswith(tdir):
                topic_found = True

                # Wait for file to be fully written (check file size stability)
                logger.info(f"[DETECTED] {abs_path}")
                if not self._wait_for_file_ready(abs_path):
                    logger.error(f"[TIMEOUT] File not ready: {abs_path}")
                    return

                # Mark as processing
                self.processing.add(abs_path)

                try:
                    logger.info(f"[PROCESSING] Indexing {abs_path} to topic '{t}'")
                    index_pdf(t, abs_path)
                    logger.info(f"[SUCCESS] {abs_path} indexed successfully")
                except Exception as e:
                    logger.error(
                        f"[ERROR] Failed to process {abs_path}: {e}", exc_info=True
                    )
                finally:
                    # Remove from processing set
                    self.processing.discard(abs_path)

                break

        if not topic_found:
            logger.warning(f"[SKIP] {abs_path} - no matching topic directory")

    def _wait_for_file_ready(self, filepath, timeout=30, check_interval=0.5):
        """
        Espera a que el archivo se haya escrito completamente.
        Comprueba que el tamaño del archivo se estabilice.
        """
        start_time = time.time()
        last_size = -1
        stable_checks = 0
        required_stable_checks = 3

        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(filepath)

                if current_size == last_size and current_size > 0:
                    stable_checks += 1
                    if stable_checks >= required_stable_checks:
                        logger.debug(f"[READY] {filepath} ({current_size} bytes)")
                        return True
                else:
                    stable_checks = 0

                last_size = current_size
                time.sleep(check_interval)
            except OSError:
                # File might be getting written, continue waiting
                stable_checks = 0
                time.sleep(check_interval)

        return False


def run():
    logger.info("=" * 60)
    logger.info("Starting PDF Watcher")
    logger.info("=" * 60)
    logger.info(f"Topics to monitor: {', '.join(TOPIC_LABELS)}")
    logger.info(f"Base directory: {TOPIC_BASE_DIR}")

    # Check CUDA availability at startup
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    logger.info("=" * 60)

    obs = Observer()
    for t in TOPIC_LABELS:
        path = os.path.join(TOPIC_BASE_DIR, t)
        os.makedirs(path, exist_ok=True)
        obs.schedule(PDFHandler(), path, recursive=False)
        logger.info(f"Watching: {path}")

    obs.start()
    logger.info("\n✓ Watcher is running. Waiting for new PDFs...\n")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("\nStopping watcher...")
        obs.stop()

    obs.join()
    logger.info("Watcher stopped.")


if __name__ == "__main__":
    run()
