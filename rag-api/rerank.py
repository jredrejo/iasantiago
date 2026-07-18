from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from typing import List
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        # RERANK_DEVICE=cpu evita competir por VRAM con vLLM (gpu_mem_util 0.95)
        self.device = os.getenv(
            "RERANK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"[RERANKER] Cargando modelo: {model_name}")
        logger.info(f"[RERANKER] Dispositivo: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info(f"[RERANKER] Tokenizer cargado")

            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if self.device == "cpu" and hasattr(config, "use_flash_attn"):
                # El código custom de jina usa flash-attn, que exige CUDA
                # (assert qkv.is_cuda); en CPU hay que usar la atención estándar
                config.use_flash_attn = False
                logger.info("[RERANKER] flash-attn desactivado (CPU)")

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,  # ← Ignora diferencias de tamaño
            ).to(self.device)
            logger.info(f"[RERANKER] Modelo cargado exitosamente")

            self.model.eval()
        except Exception as e:
            logger.error(f"[RERANKER] Error al cargar modelo: {e}", exc_info=True)
            raise

    @torch.inference_mode()
    def rerank(self, query: str, passages: List[str], topk: int) -> List[int]:
        """Reordena passages según relevancia con el query"""
        if not passages:
            return []

        logger.debug(f"[RERANKER] Reranking {len(passages)} passages")

        pairs = [(query, p) for p in passages]
        batch = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        scores = self.model(**batch).logits.squeeze(-1)
        order = torch.argsort(scores, descending=True).tolist()

        result = order[:topk]
        logger.debug(f"[RERANKER] Top-{topk} reranked indices: {result}")

        return result
