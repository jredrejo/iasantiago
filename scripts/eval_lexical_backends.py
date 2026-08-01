#!/usr/bin/env python3
"""
§7.4 — corre los golden sets contra rag-api y vuelca la tabla por tema.

Mide **el camino completo de servicio**: densa + léxica, fusión RRF, dedup,
límite por fichero y reranker. Es el dato que `FINDINGS.md` §7.4 exige antes de
sacar Whoosh — "RRF sobre un ranking léxico distinto no es el mismo sistema".

No conmuta nada por su cuenta: mide el rag-api que esté levantado. Se guarda un
JSON por tirada y se comparan dos con `--compare`, que es como se comparó Whoosh
contra el disperso antes de retirar Whoosh el 2026-08-01.

    python3 scripts/eval_lexical_backends.py --out /tmp/whoosh.json
    # (cambiar LEXICAL_BACKEND y recrear rag-api)
    python3 scripts/eval_lexical_backends.py --out /tmp/qdrant.json
    python3 scripts/eval_lexical_backends.py --compare /tmp/whoosh.json /tmp/qdrant.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

RAG_API = os.getenv("RAG_API", "http://127.0.0.1:8001")
EVAL_DIR = Path(os.getenv("EVAL_DIR", "/opt/iasantiago-rag/eval"))
API_KEY = os.getenv("OPENAI_API_KEY", "")

GOLDEN = [
    ("FOL", "golden_fol.json"),
    ("Mecanica", "golden_mecanica.json"),
    ("Programming", "golden_programming.json"),
    ("Sostenibilidad", "golden_sostenibilidad.json"),
    ("AFD", "golden_afd.json"),
    ("Dibujo", "golden_dibujo.json"),
    ("Chemistry", "golden_chemistry.json"),
    ("Chemistry-docling", "golden_chemistry_docling.json"),
    ("Electricidad", "golden_electricidad.json"),
    ("Latin", "golden_latin.json"),
]

# El agregado del proyecto anida las métricas bajo "pages" (nivel página, que es
# el que decide) y "files". Se miden las de página, que son las de las líneas
# base de `PLAN.md`.
METRICS = ("Recall@1", "Recall@3", "MRR")


def post_eval(cases: List[Dict], rerank: bool, timeout: int = 1800) -> Dict:
    url = f"{RAG_API}/v1/eval/offline?rerank={'true' if rerank else 'false'}"
    body = json.dumps(cases).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick(agg: Dict, name: str) -> Optional[float]:
    """Saca una métrica de nivel página del agregado de `rag-api/eval.py`."""
    pages = agg.get("pages")
    if isinstance(pages, dict) and name in pages:
        return pages[name]
    return agg.get(name)


def run(out_path: Optional[str], rerank: bool) -> int:
    results: Dict[str, Dict] = {}

    for label, filename in GOLDEN:
        path = EVAL_DIR / filename
        if not path.exists():
            print(f"  ! {filename} no existe, saltando")
            continue

        cases = json.loads(path.read_text(encoding="utf-8"))
        start = time.time()
        try:
            resp = post_eval(cases, rerank)
        except Exception as e:
            print(f"  ERROR {label}: {e}")
            results[label] = {"error": str(e), "n": len(cases)}
            continue

        agg = resp.get("aggregate", {})
        elapsed = time.time() - start
        row = {
            "n": len(cases),
            "seconds": round(elapsed, 1),
            **{m: pick(agg, m) for m in METRICS},
            "warnings": resp.get("warnings", []),
            "aggregate": agg,
        }
        results[label] = row

        vals = "  ".join(
            f"{m}={row[m]:.3f}" if isinstance(row[m], (int, float)) else f"{m}=?"
            for m in METRICS
        )
        print(f"  {label:20} n={row['n']:>3}  {vals}  ({elapsed:.0f}s)")
        if row["warnings"]:
            print(f"      ! {len(row['warnings'])} avisos")

    if out_path:
        Path(out_path).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nEscrito {out_path}")

    return 0


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))

    print(f"A = {a_path}")
    print(f"B = {b_path}\n")
    header = f"{'tema':22} {'n':>3} {'A R@1':>7} {'B R@1':>7} {'A R@3':>7} {'B R@3':>7} {'A MRR':>7} {'B MRR':>7} {'ΔMRR':>8}"
    print(header)
    print("-" * len(header))

    tot_n = 0
    tot_a = 0.0
    tot_b = 0.0

    for label in a:
        if label not in b or "error" in a[label] or "error" in b[label]:
            print(f"{label:22} (sin datos en ambos)")
            continue
        ra, rb = a[label], b[label]
        n = ra["n"]

        # Se releen del agregado guardado, no de las claves aplanadas: así un
        # JSON escrito por una versión anterior del script sigue comparándose.
        va = {m: pick(ra.get("aggregate", {}), m) for m in METRICS}
        vb = {m: pick(rb.get("aggregate", {}), m) for m in METRICS}

        def fmt(v):
            return f"{v:7.3f}" if isinstance(v, (int, float)) else "      ?"

        d = ""
        if isinstance(va["MRR"], (int, float)) and isinstance(vb["MRR"], (int, float)):
            delta = vb["MRR"] - va["MRR"]
            d = f"{delta:+8.3f}"
            # Chemistry-docling solapa con Chemistry; no se pondera dos veces.
            if label != "Chemistry-docling":
                tot_n += n
                tot_a += va["MRR"] * n
                tot_b += vb["MRR"] * n

        print(
            f"{label:22} {n:>3} {fmt(va['Recall@1'])} {fmt(vb['Recall@1'])} "
            f"{fmt(va['Recall@3'])} {fmt(vb['Recall@3'])} "
            f"{fmt(va['MRR'])} {fmt(vb['MRR'])} {d}"
        )

    if tot_n:
        ma, mb = tot_a / tot_n, tot_b / tot_n
        print("-" * len(header))
        print(
            f"{'ponderado':22} {tot_n:>3} {'':>7} {'':>7} {'':>7} {'':>7} "
            f"{ma:7.3f} {mb:7.3f} {mb - ma:+8.3f}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="escribe los resultados a este JSON")
    ap.add_argument(
        "--no-rerank",
        action="store_true",
        help="mide antes del reranker (separa fallo de recuperación de fallo de orden)",
    )
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compara dos JSON")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    return run(args.out, rerank=not args.no_rerank)


if __name__ == "__main__":
    sys.exit(main())
