# Use NVIDIA's official PyTorch container - has better RTX 5090 support
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# The container already has PyTorch with better CUDA support
# Just install your additional requirements
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]

