#!/usr/bin/env python3
"""Compara las dos rutas de servicio del §7.1 sobre `retrieval.jsonl`.

`POST /retrieve` (el Filter de Open WebUI) y `POST /v1/chat/completions` (los
modelos `topic:X`) corren **la misma cadena de recuperación**
(`choose_retrieval → rerank → soft_trim → attach_citations`). Por construcción
no pueden diferir en calidad de recuperación para una misma consulta, así que
este comparador no intenta medirla: mide lo que las rutas sí deciden de forma
distinta, que es lo que el §7.1 puso en juego.

  1. **Resolución de tema.** El Filter deriva el tema del nombre del modelo de
     workspace (con `strip_suffixes`/`topic_map`); la ruta topic:X lo saca del
     nombre falso del modelo. Un fallo se ve como contexto vacío.
  2. **Modo generativo.** La ruta topic:X lo *adivina* con una regex de
     intención; el Filter lo *lee* de la variante de modelo que eligió el
     usuario. Ésta es la afirmación central del §7.1 y aquí es donde se decide.
  3. **Contexto vacío.** El Filter degrada a "sin RAG" cuando rag-api no
     responde o el tema no existe; la ruta topic:X devuelve un error. Un exceso
     de filas con 0 chunks en la ruta /retrieve es degradación silenciosa.
  4. **Cobertura y volumen** por tema, para saber si la muestra da para decidir.

Uso:
    python3 rag-api/tools/ab_retrieve_vs_chat.py [--since YYYY-MM-DD]
                                                 [--min-per-arm N]
                                                 [--json]
                                                 [ruta/al/retrieval.jsonl ...]

Sin argumentos lee `data/telemetry/retrieval.jsonl` y sus archivos rotados
(`retrieval.jsonl.YYYY-MM`).

Salida distinta de cero si alguna rama no llega a `--min-per-arm` filas: el A/B
no está listo para decidir. Es el criterio de parada, no un error.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

# Las filas anteriores al marcado explícito no llevan `source`; son de la ruta
# topic:X, que es la única que existía sin marcar.
CHAT = "chat"
RETRIEVE = "retrieve"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TELEMETRY = os.path.join(REPO_ROOT, "data", "telemetry", "retrieval.jsonl")


def default_paths() -> List[str]:
    return [DEFAULT_TELEMETRY] + sorted(glob.glob(DEFAULT_TELEMETRY + ".*"))


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        f"aviso: {path}:{lineno} no es JSON válido, se omite",
                        file=sys.stderr,
                    )
    return rows


def row_ts(row: Dict[str, Any]) -> Optional[datetime]:
    """`ts` es epoch en milisegundos (`telemetry_log`)."""
    ts = row.get("ts")
    if not isinstance(ts, (int, float)):
        return None
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)


def arm_of(row: Dict[str, Any]) -> str:
    return RETRIEVE if row.get("source") == RETRIEVE else CHAT


def pct(part: int, whole: int) -> str:
    return "—" if not whole else f"{100.0 * part / whole:.1f} %"


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    stamps = [t for t in (row_ts(r) for r in rows) if t is not None]
    n_retrieved = [len(r.get("retrieved") or []) for r in rows]
    empty = sum(1 for k in n_retrieved if k == 0)

    # `generative` sólo existe desde el marcado explícito; None = fila antigua,
    # no "no generativa". Se cuenta aparte para no inventar una tasa.
    gen_known = [r for r in rows if isinstance(r.get("generative"), bool)]
    gen_true = sum(1 for r in gen_known if r["generative"])

    # Unicidad de fuentes: cuántos ficheros distintos aporta un contexto. Un
    # valor bajo con muchos chunks es concentración en un documento.
    spreads = []
    for r in rows:
        got = r.get("retrieved") or []
        if got:
            spreads.append(len({c.get("file_path") for c in got}))

    return {
        "n": n,
        "first": min(stamps).isoformat(timespec="seconds") if stamps else None,
        "last": max(stamps).isoformat(timespec="seconds") if stamps else None,
        "topics": Counter(r.get("topic") for r in rows),
        "modes": Counter(r.get("mode") for r in rows),
        "empty_context": empty,
        "empty_context_pct": pct(empty, n),
        "generative_known": len(gen_known),
        "generative_true": gen_true,
        "generative_pct": pct(gen_true, len(gen_known)),
        "mean_chunks": (sum(n_retrieved) / n) if n else 0.0,
        "mean_files_per_context": (sum(spreads) / len(spreads)) if spreads else 0.0,
        # `translated_query` lleva el eco de la consulta original siempre que el
        # idioma no sea inglés, esté o no activa la traducción. Con
        # TRANSLATE_QUERIES=false es idéntica a `query`: contar el campo a secas
        # da 106 de 145 "traducciones" que no ocurrieron. Sólo cuenta si difiere.
        "translated": sum(
            1
            for r in rows
            if r.get("translated_query") and r["translated_query"] != r.get("query")
        ),
        "non_english": sum(
            1 for r in rows if (r.get("original_language") or "en") != "en"
        ),
    }


def print_report(by_arm: Dict[str, Dict[str, Any]], min_per_arm: int) -> None:
    arms = [CHAT, RETRIEVE]
    label = {CHAT: "topic:X (chat)", RETRIEVE: "/retrieve (Filter)"}

    def row(name: str, fn) -> None:
        cells = "".join(f"{str(fn(by_arm[a])):>22}" for a in arms)
        print(f"  {name:<28}{cells}")

    print("\n§7.1 A/B — ruta de servicio\n")
    print(f"  {'':<28}{''.join(f'{label[a]:>22}' for a in arms)}")
    print("  " + "-" * 72)
    row("filas", lambda s: s["n"])
    row("desde", lambda s: (s["first"] or "—")[:10])
    row("hasta", lambda s: (s["last"] or "—")[:10])
    row("temas distintos", lambda s: len(s["topics"]))
    row("contexto vacío", lambda s: f"{s['empty_context']} ({s['empty_context_pct']})")
    row("chunks/consulta (media)", lambda s: f"{s['mean_chunks']:.1f}")
    row("ficheros/contexto (media)", lambda s: f"{s['mean_files_per_context']:.1f}")
    row("modo bm25-only", lambda s: s["modes"].get("bm25", 0))
    row("consultas no inglesas", lambda s: s["non_english"])
    row("traducidas de verdad", lambda s: s["translated"])
    row(
        "generativo (marcadas)",
        lambda s: (
            f"{s['generative_true']}/{s['generative_known']} ({s['generative_pct']})"
            if s["generative_known"]
            else "sin marcar"
        ),
    )

    print("\n  Cobertura por tema:")
    topics = sorted({t for a in arms for t in by_arm[a]["topics"]}, key=str)
    for topic in topics:
        cells = "".join(f"{by_arm[a]['topics'].get(topic, 0):>22}" for a in arms)
        print(f"  {str(topic):<28}{cells}")

    print("\n  Lectura:")
    print(
        "  · `generativo` compara la regex de intención (chat) con la elección "
        "explícita\n    del usuario (Filter). Es la afirmación que el §7.1 pone a "
        "prueba; sólo\n    tiene sentido con las dos ramas marcadas y sobre los "
        "mismos temas."
    )
    print(
        "  · `contexto vacío` en /retrieve es degradación silenciosa (tema "
        "desconocido o\n    rag-api caído): el modelo responde sin RAG en vez de "
        "dar error."
    )
    print(
        "  · La calidad de recuperación NO se compara aquí: ambas rutas corren la "
        "misma\n    cadena. Para eso está `POST /v1/eval/offline` sobre los golden "
        "sets."
    )

    short = [a for a in arms if by_arm[a]["n"] < min_per_arm]
    print()
    if short:
        falta = ", ".join(f"{label[a]} ({by_arm[a]['n']}/{min_per_arm})" for a in short)
        print(f"  NO DECIDIBLE — muestra insuficiente: {falta}")
        print("  El rip-out del §7.1 sigue bloqueado hasta que haya tráfico real.")
    else:
        print(f"  Muestra suficiente en ambas ramas (≥ {min_per_arm}).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="ficheros JSONL (por defecto, los del repo)")
    ap.add_argument("--since", help="descarta filas anteriores a YYYY-MM-DD (UTC)")
    ap.add_argument(
        "--min-per-arm",
        type=int,
        default=200,
        help="filas mínimas por rama para considerar el A/B decidible (defecto 200)",
    )
    ap.add_argument("--json", action="store_true", help="volcado JSON en vez de tabla")
    args = ap.parse_args()

    rows = load_rows(args.paths or default_paths())
    if not rows:
        print("sin telemetría que leer", file=sys.stderr)
        return 2

    if args.since:
        cutoff = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        rows = [r for r in rows if (row_ts(r) or cutoff) >= cutoff]

    by_arm = {arm: summarize([r for r in rows if arm_of(r) == arm]) for arm in (CHAT, RETRIEVE)}

    if args.json:
        serializable = {
            arm: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in s.items()}
            for arm, s in by_arm.items()
        }
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
    else:
        print_report(by_arm, args.min_per_arm)

    return 0 if all(s["n"] >= args.min_per_arm for s in by_arm.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
