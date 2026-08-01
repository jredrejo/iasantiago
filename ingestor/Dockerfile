FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ============================================================
# CRITICAL: Install ALL system dependencies BEFORE Python packages
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR and PDF processing
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-cpp-dev \
    libmagic1 \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    # PDF utilities
    ghostscript \
    # Build tools
    build-essential \
    pkg-config \
    # Development headers
    libssl-dev \
    libffi-dev \
    # OpenCV dependencies (required by unstructured-inference)
    libgl1 \
    libglib2.0-0 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download docling models during image build to avoid runtime downloads
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN mkdir -p /root/.cache/huggingface && \
    python3 -c "from docling.document_converter import DocumentConverter; DocumentConverter()" || true && \
    echo "[DOCKERFILE] Docling model caching completed"

# El supervisor (§6.8) corre como PID 1 y lanza main.py como hijo, para poder
# matarlo desde FUERA del intérprete cuando una falla nativa se cuelga reteniendo
# el GIL y el watchdog interno no puede ejecutarse. Reenvía señales y propaga el
# código de salida del hijo, así que se comporta igual para todo lo demás.
# Los scripts que invocan subcomandos puntuales pasan `--entrypoint python`, que
# se lo salta: un `retry-failed` no necesita vigilancia.
ENTRYPOINT ["python", "supervise.py"]
