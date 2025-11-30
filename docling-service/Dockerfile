FROM nvcr.io/nvidia/pytorch:25.04-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download docling models during image build to avoid runtime downloads
# This significantly speeds up startup and first extraction
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN mkdir -p /root/.cache/huggingface && \
    python3 -c "from docling.document_converter import DocumentConverter; DocumentConverter()" || true && \
    echo "[DOCKERFILE] Model caching completed"

EXPOSE 8003

# Single worker with extended timeouts for large PDF processing
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1", "--timeout-keep-alive", "900", "--timeout-graceful-shutdown", "30", "--limit-concurrency", "2"]

