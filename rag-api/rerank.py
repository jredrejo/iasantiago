from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[RERANKER] Loading model: {model_name}")
        logger.info(f"[RERANKER] Device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info(f"[RERANKER] Tokenizer loaded")

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,  # ← Ignora mismatches de tamaño
            ).to(self.device)
            logger.info(f"[RERANKER] Model loaded successfully")

            self.model.eval()
        except Exception as e:
            logger.error(f"[RERANKER] Error loading model: {e}", exc_info=True)
            raise

    @torch.inference_mode()
    def rerank(self, query: str, passages: List[str], topk: int) -> List[int]:
        """Rerank passages based on relevance to query"""
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
