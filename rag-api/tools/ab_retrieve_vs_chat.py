#!/usr/bin/env python3
"""Compara las dos rutas de servicio del §7.1 sobre `retrieval.jsonl`.

**LEE HISTORIA, NO EL SISTEMA VIVO (desde el 2026-08-02).** El rip-out del §7.1
retiró la ruta `topic:X`: `/v1/chat/completions` ya no existe y no volverá a
producirse una sola fila `source=chat`. Este comparador se conserva porque
`retrieval.jsonl` no se reescribe y sigue conteniendo las 82 filas de aquella
rama, pero **su A/B ya no se puede completar** y su código de salida 3 ("una
rama está inactiva; esperar no lo arregla") es ahora permanente y correcto. Si
lo que buscas es el estado de la decisión, está en PLAN.md punto 6.

`POST /retrieve` (el Filter de Open WebUI) y `POST /v1/chat/completions` (los
modelos `topic:X`) corrían **la misma cadena de recuperación**
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
                                                 [--stale-days N]
                                                 [--json]
                                                 [ruta/al/retrieval.jsonl ...]

Sin argumentos lee `data/telemetry/retrieval.jsonl` y sus archivos rotados
(`retrieval.jsonl.YYYY-MM`).

Códigos de salida — son el criterio de parada, no errores:

    0  muestra suficiente en las dos ramas: el A/B se puede decidir.
    1  falta muestra, pero las dos ramas siguen vivas: sigue acumulando.
    2  no hay telemetría que leer.
    3  falta muestra y **una rama está inactiva**: esperar no lo arregla.

La diferencia entre 1 y 3 es la que motivó el arreglo del 2026-08-02. Este
comparador decía "bloqueado hasta que haya tráfico real" mientras la ruta
topic:X llevaba dos días sin una sola fila, así que aconsejaba esperar a una
muestra que no iba a llegar nunca. Un criterio de parada que no distingue
"todavía no" de "ya nunca" no es un criterio de parada.

**Corrección del 2026-08-02 (rip-out), medida contra `webui.db`:** aquel arreglo
se escribió creyendo que el tráfico había abandonado `topic:X` *pudiendo*
usarlo, porque rag-api seguía anunciándolo en `GET /v1/models`. Falso: Open WebUI
tenía los nueve modelos `topic:X` con `is_active = 0` desde el **2026-07-29
15:15:30**, una hora después de la última consulta real de esa rama (14:16:44).
No estaban disponibles para nadie. La rama no se apagó por desuso: se apagó
administrativamente, y el contador de filas tampoco podía ver *eso*.
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


def _gap_days(by_arm: Dict[str, Dict[str, Any]], arm: str) -> str:
    """Días entre la última fila de `arm` y la más reciente de toda la muestra."""
    newest = max((s["last"] for s in by_arm.values() if s["last"]), default=None)
    last = by_arm[arm]["last"]
    if not newest or not last:
        return "—"
    return str((datetime.fromisoformat(newest) - datetime.fromisoformat(last)).days)


def _dormant_arms(
    by_arm: Dict[str, Dict[str, Any]], stale_days: int
) -> List[tuple]:
    """Ramas cuya última fila va muy por detrás de la muestra: `(rama, fecha, días)`.

    Una rama corta no dice **por qué** es corta, y la respuesta cambia el consejo
    por completo: "faltan datos, espera" y "esta rama ya no la usa nadie" se ven
    igual en el contador de filas. Medido el 2026-08-02: la ruta topic:X llevaba
    parada desde el 07-31 mientras `/retrieve` seguía recibiendo, y el mensaje de
    parada seguía diciendo "espera a que haya tráfico real".

    Lo usan el informe y el código de salida, para que no puedan discrepar.
    """
    newest = max((s["last"] for s in by_arm.values() if s["last"]), default=None)
    if not newest:
        return []
    out = []
    for arm, s in by_arm.items():
        if not s["last"]:
            continue
        gap = (datetime.fromisoformat(newest) - datetime.fromisoformat(s["last"])).days
        if gap >= stale_days:
            out.append((arm, s["last"][:10], gap))
    return out


def print_report(
    by_arm: Dict[str, Dict[str, Any]], min_per_arm: int, stale_days: int = 7
) -> None:
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
        "  · `contexto vacío` en /retrieve **puede** ser degradación silenciosa (tema\n"
        "    desconocido o rag-api caído): el modelo responde sin RAG en vez de dar\n"
        "    error. Pero no lo des por hecho — cuando se miraron una a una el\n"
        "    2026-08-02, ninguna de las 7 lo era: 4 eran un bug de consulta corta ya\n"
        "    arreglado, 2 el atajo BM25 del §7.5 y 1 una pregunta fuera de su tema.\n"
        "    Este contador dice dónde mirar, no qué encontraste."
    )
    print(
        "  · La calidad de recuperación NO se compara aquí: ambas rutas corren la "
        "misma\n    cadena. Para eso está `POST /v1/eval/offline` sobre los golden "
        "sets."
    )

    short = [a for a in arms if by_arm[a]["n"] < min_per_arm]
    print()

    dormant = _dormant_arms(by_arm, stale_days)
    if dormant:
        for a, last, gap in dormant:
            print(
                f"  RAMA INACTIVA — {label[a]}: última fila {last}, "
                f"{gap} días por detrás de la muestra."
            )
        print(
            "  Esperar NO la desbloquea: el tráfico se mudó a la otra rama. Lo que\n"
            "  queda es una decisión, no una medida — retirar la ruta inactiva y ver\n"
            "  si alguien la echa de menos, o decidir por preferencia revelada."
        )
    elif short:
        # Se enseña el retraso de cada rama aunque no llegue a `stale_days`: es
        # el dato que distingue "va despacio" de "se está muriendo", y el
        # contador de filas por sí solo no lo puede mostrar. Deliberadamente NO
        # se baja el umbral para que el veredicto automático coincida con lo que
        # uno ya sospecha — se enseña el dato y se deja juzgar.
        falta = ", ".join(
            f"{label[a]} ({by_arm[a]['n']}/{min_per_arm}, "
            f"última hace {_gap_days(by_arm, a)} d)"
            for a in short
        )
        print(f"  NO DECIDIBLE — muestra insuficiente: {falta}")
        print(
            "  Sigue acumulando **si las dos ramas siguen recibiendo**. Mira la fila\n"
            "  `hasta`: una rama que lleva días sin una sola consulta no va a llegar a\n"
            f"  {min_per_arm} por esperar, y a los {stale_days} días esto lo dirá solo."
        )
    else:
        print(f"  Muestra suficiente en ambas ramas (≥ {min_per_arm}).")

    # El contador de filas no basta para saber si la pregunta del §7.1 se puede
    # responder: se responde con `generative`, y una rama entera puede no
    # llevarlo. Pasó — la instrumentación de la ruta de chat entró el 07-31,
    # después de su última fila real, así que 0 de 82 lo llevan.
    sin_marcar = [a for a in arms if by_arm[a]["n"] and not by_arm[a]["generative_known"]]
    if sin_marcar:
        print()
        for a in sin_marcar:
            print(
                f"  SIN `generative` — {label[a]}: 0 de {by_arm[a]['n']} filas lo llevan."
            )
        print(
            "  La afirmación central del §7.1 (la regex adivina, la variante de modelo\n"
            "  se elige) no se puede medir con esta muestra, tenga las filas que tenga."
        )


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
    ap.add_argument(
        "--stale-days",
        type=int,
        default=7,
        help="días de retraso de una rama respecto a la muestra para darla por "
        "inactiva (defecto 7)",
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
        print_report(by_arm, args.min_per_arm, args.stale_days)

    if all(s["n"] >= args.min_per_arm for s in by_arm.values()):
        return 0
    # 3 y 1 son los dos motivos de no poder decidir, y piden cosas distintas: 3
    # es "esto no se arregla esperando" y 1 es "sigue acumulando". Separarlos es
    # el punto del arreglo — con un solo código, quien automatice esto seguirá
    # esperando a una rama que no va a crecer.
    return 3 if _dormant_arms(by_arm, args.stale_days) else 1


if __name__ == "__main__":
    sys.exit(main())
