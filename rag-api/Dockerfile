# Use NVIDIA's official PyTorch container - has better RTX 5090 support
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install pip requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download embedding models during build
RUN python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import os

# SHAs auditados (deben coincidir con PINNED_MODEL_REVISIONS en config/settings.py):
# trust_remote_code=True ejecuta código remoto, así que se fija la revisión para
# que la imagen no descargue una versión distinta a la auditada.
models = {
    "intfloat/multilingual-e5-large-instruct": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
    "jinaai/jina-reranker-v2-base-multilingual": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
}

for model, revision in models.items():
    try:
        print(f"Downloading {model}@{revision}...")
        if "reranker" in model:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
            AutoModelForSequenceClassification.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
        else:
            SentenceTransformer(model, revision=revision, trust_remote_code=True)
        print(f"✓ {model} downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading {model}: {e}")

print("Model pre-download complete!")
EOF

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
