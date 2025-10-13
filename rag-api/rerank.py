from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, passages: List[str], topk: int) -> List[int]:
        pairs = [(query, p) for p in passages]
        batch = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        scores = self.model(**batch).logits.squeeze(-1)
        order = torch.argsort(scores, descending=True).tolist()
        return order[:topk]
