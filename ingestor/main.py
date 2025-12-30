"""
Ingestor de PDF - Punto de entrada principal.

Orquesta la extracción de PDF, chunking, embedding e indexación.
"""

import faulthandler
import glob
import logging
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

# Habilitar faulthandler antes de las importaciones CUDA
faulthandler.enable(file=sys.stderr, all_threads=True)


def _force_exit_on_signal(signum: int, frame) -> None:
    """Forzar salida en señales fatales para activar el reinicio del contenedor."""
    signal_name = signal.Signals(signum).name
    print(
        f"\n[FATAL] Capturado {signal_name} - forzando salida para activar reinicio del contenedor",
        file=sys.stderr,
        flush=True,
    )
    os._exit(128 + signum)


# Instalar manejadores de señales antes de cualquier importación de biblioteca C
for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT):
    try:
        signal.signal(sig, _force_exit_on_signal)
    except (OSError, ValueError):
        pass

# Ahora importar módulos del proyecto
from core.cache import get_pdf_total_pages
from core.config import (
    EMBED_DEFAULT,
    EMBED_PER_TOPIC,
    TOPIC_BASE_DIR,
    TOPIC_LABELS,
)
from core.heartbeat import get_heartbeat_manager, update_heartbeat
from extraction.pipeline import ExtractionPipeline
from indexing.embeddings import get_embedding_service, validate_and_fix_vectors
from indexing.qdrant import ensure_qdrant, get_qdrant_service
from indexing.whoosh_bm25 import ensure_whoosh, get_whoosh_service
from pages.page_validator import validate_page_number
from state.processing_state import get_processing_state

# Re-registrar manejadores de señales después de la inicialización de CUDA/PyTorch
for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT):
    try:
        signal.signal(sig, _force_exit_on_signal)
    except (OSError, ValueError):
        pass

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Inicializar servicios
state = get_processing_state()
embedding_service = get_embedding_service()
qdrant_service = get_qdrant_service()
whoosh_service = get_whoosh_service()
extraction_pipeline = ExtractionPipeline()


def index_pdf(topic: str, pdf_path: str) -> bool:
    """
    Indexar un archivo PDF a Qdrant y Whoosh.

    Args:
        topic: Categoría del tema
        pdf_path: Ruta al archivo PDF

    Returns:
        True si es exitoso, False en caso contrario
    """
    if state.is_already_processed(pdf_path):
        return True

    filename = Path(pdf_path).name
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Indexando: {filename}")
    logger.info(f"Tema: {topic}")
    logger.info(f"{'=' * 60}")

    try:
        # Extract elements from PDF
        logger.info("Extracting content...")
        elements = extraction_pipeline.extract(Path(pdf_path))
        logger.info(f"Extracted {len(elements)} elements")

        # Get embedding model
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
        model = embedding_service.get_model(embed_name)
        dims = embedding_service.get_dimension(model)

        # Ensure indexes exist
        ensure_qdrant(topic, dims)
        ensure_whoosh(topic)

        # Prepare texts for encoding
        texts = [elem.text for elem in elements]

        # Apply E5 prefix if needed
        if "e5" in embed_name.lower():
            texts = [f"Represent this document for retrieval: {t}" for t in texts]
            logger.info("[E5] Usando prefijo de recuperación de documentos")

        # Encode texts
        logger.info(f"Encoding {len(texts)} chunks...")
        vecs = embedding_service.encode(
            model,
            texts,
            batch_size=32,
            heartbeat_callback=update_heartbeat,
        )

        # Validate vectors
        vecs = validate_and_fix_vectors(vecs, dims)
        logger.info(f"Generated {len(vecs)} vectors")

        # Get total pages for validation
        total_pages = get_pdf_total_pages(pdf_path)

        # Build payloads with validated page numbers
        payloads = []
        for idx, elem in enumerate(elements):
            page = validate_page_number(elem.page, total_pages)
            payloads.append(
                {
                    "file_path": pdf_path,
                    "page": page,
                    "chunk_id": idx,
                    "text": elem.text,
                    "chunk_type": elem.type,
                    "source": elem.source,
                }
            )

        # Upload to Qdrant
        logger.info("Subiendo a Qdrant...")
        qdrant_service.upsert_vectors(topic, vecs, payloads)

        # Index in Whoosh
        logger.info("Indexando en Whoosh...")
        whoosh_service.index_documents(topic, payloads)

        logger.info(f"[ÉXITO] {filename}")
        state.mark_as_processed(pdf_path, topic)
        return True

    except Exception as e:
        logger.error(f"[ERROR] Falló al indexar {filename}: {e}", exc_info=True)
        state.mark_as_failed(pdf_path, str(e))
        return False


def initial_scan() -> None:
    """Escanear todos los directorios de temas e indexar PDFs."""
    heartbeat = get_heartbeat_manager()
    heartbeat.start_watchdog()

    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO ESCANEO INICIAL")
    logger.info(f"Temas: {', '.join(TOPIC_LABELS)}")

    stats = state.get_stats()
    logger.info(f"Procesados previamente: {stats['successful']} archivos")
    logger.info(f"Fallados previamente: {stats['failed']} archivos")

    pdf_count = 0
    skipped_count = 0
    error_count = 0

    for topic in TOPIC_LABELS:
        topic_dir = os.path.join(TOPIC_BASE_DIR, topic)
        os.makedirs(topic_dir, exist_ok=True)

        pdfs = glob.glob(os.path.join(topic_dir, "*.pdf"))
        logger.info(f"\nEncontrados {len(pdfs)} PDFs en {topic}")

        for pdf in pdfs:
            abs_pdf = os.path.abspath(pdf)

            if state.is_already_processed(abs_pdf):
                skipped_count += 1
                continue

            pdf_count += 1
            update_heartbeat(os.path.basename(abs_pdf))

            try:
                if not index_pdf(topic, abs_pdf):
                    error_count += 1
            except Exception as e:
                logger.error(f"[ERROR] {abs_pdf}: {e}", exc_info=True)
                state.mark_as_failed(abs_pdf, str(e))
                error_count += 1

    state.update_scan_time()

    logger.info("\n" + "=" * 60)
    logger.info("ESCANEO COMPLETADO")
    logger.info(
        f"Procesados: {pdf_count} | Omitidos: {skipped_count} | Errores: {error_count}"
    )
    logger.info("=" * 60)


def delete_pdf(topic: str, pdf_path: str) -> bool:
    """Eliminar un PDF de los índices."""
    logger.info(f"Eliminando {Path(pdf_path).name} de {topic}")

    try:
        qdrant_service.delete_by_file(topic, pdf_path)
        whoosh_service.delete_by_file(topic, pdf_path)
        state.remove_file(pdf_path)
        logger.info("[ÉXITO] PDF eliminado")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Falló la eliminación: {e}")
        return False


def delete_topic(topic: str) -> bool:
    """Eliminar todos los archivos de un tema."""
    if topic not in TOPIC_LABELS:
        logger.error(f"Invalid topic: {topic}")
        return False

    logger.info(f"Eliminando todos los archivos del tema: {topic}")

    try:
        qdrant_service.delete_collection(topic)
        whoosh_service.delete_index(topic)
        state.remove_topic_files(topic, TOPIC_BASE_DIR)

        # Recreate empty indexes
        ensure_whoosh(topic)
        embed_name = EMBED_PER_TOPIC.get(topic, EMBED_DEFAULT)
        model = embedding_service.get_model(embed_name, device="cpu")
        dims = embedding_service.get_dimension(model)
        ensure_qdrant(topic, dims)

        logger.info("[ÉXITO] Tema limpiado y listo para re-indexación")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Falló la eliminación del tema: {e}")
        return False


def main() -> None:
    """Punto de entrada CLI."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "delete" and len(sys.argv) >= 4:
            topic, pdf_path = sys.argv[2], sys.argv[3]
            sys.exit(0 if delete_pdf(topic, pdf_path) else 1)

        elif command == "delete-topic" and len(sys.argv) >= 3:
            topic = sys.argv[2]
            sys.exit(0 if delete_topic(topic) else 1)

        else:
            print("Uso:")
            print("  python main.py                      - Escanear e indexar todo")
            print("  python main.py delete <topic> <pdf> - Eliminar un PDF")
            print("  python main.py delete-topic <topic> - Eliminar todo de un tema")
            sys.exit(1)
    else:
        initial_scan()


if __name__ == "__main__":
    main()
