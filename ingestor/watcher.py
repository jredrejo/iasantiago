"""
Este archivo es de uso opcional.
Es un módulo watcher de nuevos PDFs.
Monitoriza un directorio y si llegan nuevos PDFs, los indexa usando la función index_pdf.

Puedes elegir arrancar `ingestor/main.py` (escaneo inicial) y/o `ingestor/watcher.py`
con `command` en Docker.
Para simplicidad hemos dejado solo `main.py` en `docker-compose`.
"""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time, os
import torch
from main import index_pdf
from settings import TOPIC_LABELS, TOPIC_BASE_DIR


class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            # Determine which topic this file belongs to
            for t in TOPIC_LABELS:
                tdir = os.path.join(TOPIC_BASE_DIR, t)
                if event.src_path.startswith(tdir):
                    print(f"\n[NEW FILE DETECTED] {event.src_path}")
                    # Give the file system a moment to finish writing
                    time.sleep(1)
                    try:
                        index_pdf(t, os.path.abspath(event.src_path))
                    except Exception as e:
                        print(f"[ERROR] Failed to process new file: {e}")
                        import traceback

                        traceback.print_exc()
                    break


def run():
    print("Starting PDF watcher...")
    print(f"Monitoring directories for topics: {', '.join(TOPIC_LABELS)}")
    print(f"Base directory: {TOPIC_BASE_DIR}")

    # Check CUDA availability at startup
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    obs = Observer()
    for t in TOPIC_LABELS:
        path = os.path.join(TOPIC_BASE_DIR, t)
        os.makedirs(path, exist_ok=True)
        obs.schedule(PDFHandler(), path, recursive=False)
        print(f"Watching: {path}")

    obs.start()
    print("\nWatcher is running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        obs.stop()
    obs.join()
    print("Watcher stopped.")


if __name__ == "__main__":
    run()
