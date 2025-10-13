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
from .main import index_pdf
from .settings import TOPIC_LABELS, TOPIC_BASE_DIR


class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            for t in TOPIC_LABELS:
                tdir = os.path.join(TOPIC_BASE_DIR, t)
                if event.src_path.startswith(tdir):
                    try:
                        index_pdf(t, event.src_path)
                    except Exception as e:
                        print("Error", e)


def run():
    obs = Observer()
    for t in TOPIC_LABELS:
        path = os.path.join(TOPIC_BASE_DIR, t)
        os.makedirs(path, exist_ok=True)
        obs.schedule(PDFHandler(), path, recursive=False)
    obs.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


if __name__ == "__main__":
    run()
