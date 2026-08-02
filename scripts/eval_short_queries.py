#!/usr/bin/env python3
"""Mide `eval/golden_short.json` por `POST /v1/eval/offline`, tema a tema.

Existe porque `eval_lexical_backends.py` lleva la lista de golden sets a fuego y
uno por tema; el corto es un solo fichero con los diez dentro (campo
`source_set`). Reutiliza el **mismo endpoint y el mismo agregado**, así que las
cifras son comparables en metodología con las líneas base — pero **no** en
significado: leer el aviso de `eval/build_golden_short.py`.

Uso:
    python3 scripts/eval_short_queries.py --out ruta.json [--no-rerank]
    python3 scripts/eval_short_queries.py --compare A.json B.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

RAG_API = os.getenv("RAG_API", "http://127.0.0.1:8001")
EVAL_DIR = Path(os.getenv("EVAL_DIR", "/opt/iasantiago-rag/eval"))
API_KEY = os.getenv("OPENAI_API_KEY", "")
METRICS = ("Recall@1", "Recall@3", "MRR")

# Etiquetas iguales a las de `eval_lexical_backends.py`, para poder leer las dos
# tablas seguidas sin traducir nombres.
LABELS = {
    "golden_fol.json": "FOL",
    "golden_mecanica.json": "Mecanica",
    "golden_programming.json": "Programming",
    "golden_sostenibilidad.json": "Sostenibilidad",
    "golden_afd.json": "AFD",
    "golden_dibujo.json": "Dibujo",
    "golden_chemistry.json": "Chemistry",
    "golden_chemistry_docling.json": "Chemistry-docling",
    "golden_electricidad.json": "Electricidad",
    "golden_latin.json": "Latin",
}


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
    pages = agg.get("pages")
    if isinstance(pages, dict) and name in pages:
        return pages[name]
    return agg.get(name)


def run(out_path: Optional[str], rerank: bool) -> int:
    all_cases = json.loads((EVAL_DIR / "golden_short.json").read_text(encoding="utf-8"))

    groups: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for fname, label in LABELS.items():
        sel = [c for c in all_cases if c.get("source_set") == fname]
        if sel:
            groups[label] = sel

    results: Dict[str, Dict] = {}
    for label, cases in groups.items():
        # `long_query`/`source_set` no los entiende el modelo EvalCase; fuera.
        payload = [
            {k: v for k, v in c.items() if k in ("query", "topic", "relevant_pages")}
            for c in cases
        ]
        start = time.time()
        try:
            resp = post_eval(payload, rerank)
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
            "aggregate": agg,
        }
        results[label] = row
        vals = "  ".join(
            f"{m}={row[m]:.3f}" if isinstance(row[m], (int, float)) else f"{m}=?"
            for m in METRICS
        )
        print(f"  {label:20} n={row['n']:>3}  {vals}  ({elapsed:.0f}s)")

    scored = [(l, r) for l, r in results.items() if isinstance(r.get("MRR"), float)]
    # Chemistry-docling solapa con Chemistry: no se pondera, igual que en PLAN.md.
    pond = [(l, r) for l, r in scored if l != "Chemistry-docling"]
    if pond:
        n = sum(r["n"] for _, r in pond)
        w = sum(r["MRR"] * r["n"] for _, r in pond) / n
        print(f"\n  ponderado (sin Chemistry-docling)  n={n}  MRR={w:.3f}")
        results["_ponderado"] = {"n": n, "MRR": w}

    if out_path:
        Path(out_path).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nEscrito {out_path}")
    return 0


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    print(f"\n  A = {a_path}\n  B = {b_path}\n")
    print(f"  {'tema':20} {'n':>3} {'A R@1':>7} {'B R@1':>7} {'A R@3':>7} {'B R@3':>7} {'A MRR':>7} {'B MRR':>7} {'ΔMRR':>8}")
    print("  " + "-" * 78)

    def fmt(v):
        return f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'?':>7}"

    for label in [l for l in a if not l.startswith("_")]:
        va, vb = a.get(label, {}), b.get(label, {})
        if "MRR" not in va or "MRR" not in vb:
            continue
        d = vb["MRR"] - va["MRR"]
        print(
            f"  {label:20} {va['n']:>3} {fmt(va['Recall@1'])} {fmt(vb['Recall@1'])} "
            f"{fmt(va['Recall@3'])} {fmt(vb['Recall@3'])} "
            f"{fmt(va['MRR'])} {fmt(vb['MRR'])} {d:>+8.3f}"
        )
    pa, pb = a.get("_ponderado"), b.get("_ponderado")
    if pa and pb:
        print("  " + "-" * 78)
        print(f"  {'PONDERADO':20} {pa['n']:>3} {'':>7} {'':>7} {'':>7} {'':>7} "
              f"{fmt(pa['MRR'])} {fmt(pb['MRR'])} {pb['MRR'] - pa['MRR']:>+8.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    return run(args.out, rerank=not args.no_rerank)


if __name__ == "__main__":
    sys.exit(main())
