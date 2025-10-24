FROM nvcr.io/nvidia/pytorch:25.04-py3

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
    # PDF utilities
    ghostscript \
    # Build tools
    build-essential \
    pkg-config \
    # Development headers
    libssl-dev \
    libffi-dev \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "main.py"]
