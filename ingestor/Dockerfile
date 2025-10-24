# Use NVIDIA's official PyTorch container - has better RTX 5090 support
FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    # Libmagic (para detección de tipo MIME)
    libmagic1 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# The container already has PyTorch with better CUDA support
# Just install your additional requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV PYTESSERACT_PATH=/usr/bin/tesseract

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["python", "main.py"]
