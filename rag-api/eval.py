from typing import List, Dict
import math


def recall_at_k(pred_files: List[str], rel_files: List[str], k: int) -> float:
    pred = pred_files[:k]
    return 1.0 if any(p in rel_files for p in pred) else 0.0


def mrr(pred_files: List[str], rel_files: List[str]) -> float:
    for i, p in enumerate(pred_files, start=1):
        if p in rel_files:
            return 1.0 / i
    return 0.0


def aggregate_eval(queries: List[Dict], k_list=[1, 3, 5, 10]) -> Dict:
    # queries: [{query, topic, relevant_files:[...], retrieved:[{file_path,...}, ...]}]
    out = {"n": len(queries)}
    for k in k_list:
        out[f"Recall@{k}"] = round(
            sum(
                recall_at_k(
                    [r["file_path"] for r in q["retrieved"]], q["relevant_files"], k
                )
                for q in queries
            )
            / max(1, len(queries)),
            4,
        )
    out["MRR"] = round(
        sum(
            mrr([r["file_path"] for r in q["retrieved"]], q["relevant_files"])
            for q in queries
        )
        / max(1, len(queries)),
        4,
    )
    return out
