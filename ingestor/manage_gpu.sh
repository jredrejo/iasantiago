#!/bin/bash

# manage_gpu.sh - Orquesta dos servicios vLLM sin conflicto
# vLLM (Qwen) para rag-api: siempre corriendo
# vLLM-LLaVA para ingestor: bajo demanda

set -e

VLLM_CONTAINER="vllm"
VLLM_LLAVA_CONTAINER="vllm-llava"
INGESTOR_CONTAINER="ingestor"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    cat << EOF
Usage: $(basename "$0") [COMMAND]

Commands:
  status              Show GPU memory and container status
  check               Check if containers are running
  ingest              RECOMMENDED: Ingest PDFs WITH LLaVA
  pause-llava         Stop vLLM-LLaVA (frees GPU)
  resume-llava        Start vLLM-LLaVA
  help                Show this help message

Examples:
  $(basename "$0") status
  $(basename "$0") ingest              # Orchestrates: pause vLLM -> start vLLM-LLaVA -> ingest -> stop vLLM-LLaVA -> resume vLLM
  $(basename "$0") pause-llava
  $(basename "$0") resume-llava
EOF
}

show_status() {
    echo -e "${BLUE}GPU Memory Status${NC}"
    echo "======================================"

    # Check containers
    vllm_running=$(docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER}$" && echo "[OK] Running" || echo "[STOP] Stopped")
    vllm_llava_running=$(docker ps --format '{{.Names}}' | grep -q "^${VLLM_LLAVA_CONTAINER}$" && echo "[OK] Running" || echo "[STOP] Stopped")
    ingestor_running=$(docker ps --format '{{.Names}}' | grep -q "^${INGESTOR_CONTAINER}$" && echo "[OK] Running" || echo "[STOP] Stopped")

    echo "vLLM (Qwen): $vllm_running"
    echo "vLLM-LLaVA: $vllm_llava_running"
    echo "Ingestor: $ingestor_running"
    echo ""

    # GPU usage
    echo -e "${BLUE}GPU Memory Usage${NC}"
    echo "======================================"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"

    echo ""
    echo -e "${BLUE}GPU Processes${NC}"
    echo "======================================"
    nvidia-smi pmon -c 1 2>/dev/null | head -20 || echo "No GPU processes"
}

check_containers() {
    echo "Checking container status..."

    vllm_running=$(docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER}$" && echo "true" || echo "false")
    vllm_llava_running=$(docker ps --format '{{.Names}}' | grep -q "^${VLLM_LLAVA_CONTAINER}$" && echo "true" || echo "false")
    ingestor_running=$(docker ps --format '{{.Names}}' | grep -q "^${INGESTOR_CONTAINER}$" && echo "true" || echo "false")

    echo "vLLM (Qwen): $vllm_running"
    echo "vLLM-LLaVA: $vllm_llava_running"
    echo "Ingestor: $ingestor_running"
}

pause_vllm() {
    echo -e "${YELLOW}Pausing vLLM (Qwen) to free GPU memory...${NC}"

    if ! docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER}$"; then
        echo "vLLM is not running"
        return 0
    fi

    docker stop "$VLLM_CONTAINER" 2>/dev/null || true
    echo -e "${GREEN}[OK] vLLM stopped${NC}"
    sleep 3
    return 0
}

resume_vllm() {
    echo -e "${YELLOW}Resuming vLLM (Qwen)...${NC}"

    docker start "$VLLM_CONTAINER"

    # Wait for vLLM to be ready
    for i in {1..60}; do
        if docker exec "$VLLM_CONTAINER" curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo -e "${GREEN}[OK] vLLM is ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "${RED}[ERROR] vLLM failed to start${NC}"
    return 1
}

start_vllm_llava() {
    pause_vllm
    echo -e "${YELLOW}Starting vLLM-LLaVA...${NC}"

    if docker ps --format '{{.Names}}' | grep -q "^${VLLM_LLAVA_CONTAINER}$"; then
        echo "vLLM-LLaVA already running"
        return 0
    fi

    docker start "$VLLM_LLAVA_CONTAINER"

    # Wait 2 minutos for vLLM-LLaVA to be ready
    for i in {1..120}; do
        if docker exec "$VLLM_LLAVA_CONTAINER" curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo -e "${GREEN}[OK] vLLM-LLaVA is ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "${RED}[ERROR] vLLM-LLaVA failed to start${NC}"
    return 1
}

stop_vllm_llava() {
    echo -e "${YELLOW}Stopping vLLM-LLaVA...${NC}"

    if ! docker ps --format '{{.Names}}' | grep -q "^${VLLM_LLAVA_CONTAINER}$"; then
        echo "vLLM-LLaVA is not running"
        return 0
    fi

    docker stop "$VLLM_LLAVA_CONTAINER" 2>/dev/null || true
    echo -e "${GREEN}[OK] vLLM-LLaVA stopped${NC}"
    sleep 3
    resume_vllm
    return 0
}

ingest_with_llava() {
    echo -e "${BLUE}Starting ingest orchestration${NC}"
    echo "======================================"
    echo "Strategy:"
    echo "  1. Start vLLM-LLaVA (for ingestor)"
    echo "  2. Run ingestor"
    echo "  3. Stop vLLM-LLaVA"
    echo ""

    # 1. Start vLLM-LLaVA
    echo "[1/3] Starting vLLM-LLaVA..."
    start_vllm_llava
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to start vLLM-LLaVA${NC}"
        return 1
    fi

    # 2. Show GPU memory
    echo ""
    echo "[2/3] GPU memory with vLLM-LLaVA running:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU info unavailable"

    # 3. Run ingestor
    echo ""
    echo -e "${BLUE}[3/3] Running ingestor scan WITH LLaVA...${NC}"
    echo "======================================"
    docker start "$INGESTOR_CONTAINER" 2>/dev/null || true
    docker exec "$INGESTOR_CONTAINER" python main.py
    INGEST_STATUS=$?

    # Stop vLLM-LLaVA regardless of ingest status
    echo ""
    echo "Stopping vLLM-LLaVA..."
    stop_vllm_llava
    docker stop "$INGESTOR_CONTAINER" 2>/dev/null || true
    echo ""
    if [ $INGEST_STATUS -eq 0 ]; then
        echo -e "${GREEN}[OK] Ingest completed successfully${NC}"
        echo ""
        echo "GPU is now back to normal:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU info unavailable"
    else
        echo -e "${RED}[ERROR] Ingest failed${NC}"
        return 1
    fi
    # Restart RAG web
    docker compose up -d oauth2-proxy
}

# ============================================================
# MAIN
# ============================================================

COMMAND="${1:-help}"

case "$COMMAND" in
    status)
        show_status
        ;;
    check)
        check_containers
        ;;
    pause-llava)
        stop_vllm_llava
        ;;
    resume-llava)
        start_vllm_llava
        ;;
    ingest)
        ingest_with_llava
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
