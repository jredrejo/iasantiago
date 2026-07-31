#!/usr/bin/env python3
"""
Compara el VLM (granite-docling) con la cadena EasyOCR sobre escaneados reales.

Es la medición que el §9 exige **antes** de borrar `unstructured` y la cadena
EasyOCR/Tesseract (§7.2): "sólo después de confirmar que la vía VLM cubre los
PDFs escaneados en una muestra del corpus real". Este script produce esa
confirmación —o la desmiente— y no borra nada.

Se corre dentro del contenedor ingestor, que es donde viven docling y easyocr:

    docker compose --profile ingest run --rm --entrypoint python3 \\
        ingestor /app/../scripts/compare_vlm_vs_ocr.py --sample 6

o, más cómodo, montando el script:

    docker run --rm --gpus all \\
        -v /opt/iasantiago-rag/ingestor:/app:ro \\
        -v /opt/iasantiago-rag/scripts:/scripts:ro \\
        -v /opt/iasantiago-rag/topics:/topics:ro \\
        -v /opt/iasantiago-rag/huggingface_cache:/models_cache \\
        -w /app --entrypoint python3 iasantiago-rag-ingestor:latest \\
        /scripts/compare_vlm_vs_ocr.py --sample 6

**Necesita ventana de GPU**, y esto está medido, no supuesto: el 2026-07-31 se
intentó en CPU y a los **19 minutos seguía en el primer fichero de 5 páginas**.
La vía funciona en CPU (una página de sólo imagen convierte en 8,4 s y devuelve
`DoclingDocument`), pero una página con texto de verdad son minutos, así que ni
el tiempo ni la paciencia dan para una muestra. Con vLLM ocupando la tarjeta
entera (32 032 / 32 607 MiB) esto no se puede correr: hay que pararlo antes
(la GPU es del ingestor durante la ingesta).

Para dimensionar la ventana: `--discover-only` contó **79 escaneados / 13 564
páginas** en los nueve temas.

Qué mide, y por qué esas métricas:

- **caracteres y elementos**: volumen bruto. Por sí solo no dice nada —
  `LESSONS.md`: un fichero puede devolver 15 588 fragmentos de basura de fuente
  subseteada y sólo 2 legibles.
- **ratio de palabras funcionales** (`de la que en el y ...`): proxy barato de
  "esto es español legible". Es lo que separa texto de glifos sin decodificar.
  Un OCR sano ronda 0.10–0.25; por debajo de ~0.03 el texto no es lenguaje.
- **segundos por página**: el coste real del cambio. El VLM es generativo, así
  que se espera perder aquí; la pregunta es cuánto.
- **estructura**: el VLM devuelve `DoclingDocument`, así que la fragmentación
  usa HybridChunker en vez de aplanar. EasyOCR no puede.

Salida: tabla por fichero + veredicto agregado, y `--json` para archivarlo junto
al `FINDINGS.md` del día.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, "/app")

# Palabras funcionales del español: si el texto es lenguaje, aparecen. Si es
# ruido de OCR o glifos sin mapa de caracteres, no.
STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un",
    "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más",
    "o", "pero", "sus", "le", "ha", "me", "si", "sin", "sobre", "este", "ya",
    "entre", "cuando", "todo", "esta", "ser", "son", "dos", "también", "fue",
}
WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


@dataclass
class Run:
    ok: bool
    seconds: float = 0.0
    chars: int = 0
    elements: int = 0
    pages_with_text: int = 0
    stopword_ratio: float = 0.0
    structured: bool = False
    error: str = ""
    sample: str = ""


@dataclass
class FileResult:
    path: str
    pages: int
    size_mb: float
    vlm: Run = field(default_factory=lambda: Run(ok=False))
    ocr: Run = field(default_factory=lambda: Run(ok=False))


def stopword_ratio(text: str) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    return sum(1 for w in words if w in STOPWORDS) / len(words)


def page_count(path: Path) -> Optional[int]:
    try:
        import pypdf

        with open(path, "rb") as fh:
            return len(pypdf.PdfReader(fh).pages)
    except Exception:
        return None


def find_scanned(base: Path, topics: List[str], max_pages: int, limit: int) -> List[Path]:
    """PDFs sin capa de texto: los que hoy resuelve la cadena EasyOCR.

    Es la única población en la que la comparación significa algo. Un PDF con
    texto no llega nunca al OCR, así que meterlo en la muestra sólo mediría
    ruido.
    """
    from extraction.base import check_pdf_has_text

    found: List[Path] = []
    roots = [base / t for t in topics] if topics else [base]
    for root in roots:
        if not root.is_dir():
            print(f"aviso: {root} no existe, se omite", file=sys.stderr)
            continue
        for pdf in sorted(root.rglob("*.pdf")):
            if check_pdf_has_text(pdf):
                continue
            pages = page_count(pdf)
            if pages is None or pages > max_pages:
                continue
            found.append(pdf)
            if limit and len(found) >= limit:
                return found
    return found


def run_extractor(extractor, pdf: Path) -> Run:
    start = time.time()
    try:
        elements = extractor.extract(pdf)
    except Exception as e:
        return Run(ok=False, seconds=time.time() - start, error=f"{type(e).__name__}: {e}"[:200])

    seconds = time.time() - start
    text = "\n".join(e.text for e in elements)
    return Run(
        ok=bool(elements),
        seconds=seconds,
        chars=len(text),
        elements=len(elements),
        pages_with_text=len({e.page for e in elements}),
        stopword_ratio=stopword_ratio(text),
        structured=getattr(extractor, "last_document", None) is not None,
        sample=text[:300].replace("\n", " "),
    )


def compare(pdf: Path, vlm, ocr) -> FileResult:
    pages = page_count(pdf) or 0
    result = FileResult(
        path=str(pdf), pages=pages, size_mb=round(pdf.stat().st_size / 1e6, 2)
    )
    print(f"\n=== {pdf.name} ({pages} pág, {result.size_mb} MB)", flush=True)

    print("  · VLM...", end="", flush=True)
    result.vlm = run_extractor(vlm, pdf)
    print(f" {result.vlm.seconds:.1f}s, {result.vlm.chars} car", flush=True)

    print("  · OCR...", end="", flush=True)
    result.ocr = run_extractor(ocr, pdf)
    print(f" {result.ocr.seconds:.1f}s, {result.ocr.chars} car", flush=True)

    return result


def print_table(results: List[FileResult]) -> None:
    print("\n\n§7.2 — VLM (granite-docling) frente a la cadena EasyOCR\n")
    head = f"{'fichero':<34}{'pág':>5}{'  car/pág VLM':>14}{'  car/pág OCR':>14}{'  s/pág VLM':>12}{'  s/pág OCR':>12}{'  stop VLM':>11}{'  stop OCR':>11}"
    print(head)
    print("-" * len(head))
    for r in results:
        p = max(r.pages, 1)
        name = Path(r.path).name
        name = name if len(name) <= 33 else name[:30] + "..."
        print(
            f"{name:<34}{r.pages:>5}"
            f"{(r.vlm.chars / p if r.vlm.ok else 0):>14.0f}"
            f"{(r.ocr.chars / p if r.ocr.ok else 0):>14.0f}"
            f"{(r.vlm.seconds / p):>12.1f}"
            f"{(r.ocr.seconds / p):>12.1f}"
            f"{r.vlm.stopword_ratio:>11.3f}"
            f"{r.ocr.stopword_ratio:>11.3f}"
        )

    for r in results:
        if not r.vlm.ok or not r.ocr.ok:
            who = "VLM" if not r.vlm.ok else "OCR"
            err = r.vlm.error if not r.vlm.ok else r.ocr.error
            print(f"\n  FALLO {who} en {Path(r.path).name}: {err or 'sin elementos'}")

    print("\n  Veredicto (lo que el §9 tiene que decidir):")
    n = len(results)
    vlm_ok = sum(1 for r in results if r.vlm.ok)
    ocr_ok = sum(1 for r in results if r.ocr.ok)
    print(f"  · ficheros cubiertos: VLM {vlm_ok}/{n} · OCR {ocr_ok}/{n}")

    # La cobertura es la condición de bloqueo: si el VLM no cubre lo que hoy
    # cubre EasyOCR, borrar EasyOCR pierde documentos, cueste lo que cueste el
    # resto de métricas.
    perdidos = [
        Path(r.path).name for r in results if r.ocr.ok and not r.vlm.ok
    ]
    if perdidos:
        print(f"  · NO-GO para borrar EasyOCR: el VLM no cubre {len(perdidos)} que sí cubre OCR")
        for name in perdidos:
            print(f"      - {name}")
    elif vlm_ok == n and n:
        print("  · el VLM cubre toda la muestra; queda comparar legibilidad y coste")

    legibles = [r for r in results if r.vlm.ok and r.ocr.ok]
    if legibles:
        mejor = sum(1 for r in legibles if r.vlm.stopword_ratio >= r.ocr.stopword_ratio)
        print(
            f"  · legibilidad (palabras funcionales): VLM ≥ OCR en {mejor}/{len(legibles)}"
        )
        v = sum(r.vlm.seconds for r in legibles)
        o = sum(r.ocr.seconds for r in legibles)
        factor = (v / o) if o else float("inf")
        print(f"  · coste: el VLM tarda {factor:.1f}× lo que EasyOCR en esta muestra")
        estruct = sum(1 for r in legibles if r.vlm.structured)
        print(
            f"  · estructura (DoclingDocument → HybridChunker): VLM {estruct}/{len(legibles)}, "
            f"OCR {sum(1 for r in legibles if r.ocr.structured)}/{len(legibles)}"
        )

    print(
        "\n  Esto NO decide nada por sí solo: cubrir y ser legible es el mínimo, "
        "pero el\n  cambio de extractor sólo entra con un delta de eval sobre el "
        "golden set del\n  tema afectado (PLAN.md, cierre de fase)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topics", default="", help="lista separada por comas; vacío = todos")
    ap.add_argument("--sample", type=int, default=6, help="ficheros a comparar (defecto 6)")
    ap.add_argument("--max-pages", type=int, default=40, help="descarta escaneados más largos")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--base", default=os.getenv("TOPIC_BASE_DIR", "/topics"))
    ap.add_argument("--json", dest="json_out", help="escribe los resultados crudos aquí")
    ap.add_argument("--files", nargs="*", help="rutas concretas en vez de muestrear")
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="lista los escaneados y sale, sin cargar ningún modelo (sirve para "
        "dimensionar la ventana de GPU antes de pedirla)",
    )
    args = ap.parse_args()

    if args.discover_only:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        found = find_scanned(Path(args.base), topics, args.max_pages, limit=0)
        total = sum(page_count(p) or 0 for p in found)
        for pdf in found:
            print(f"{page_count(pdf) or '?':>5}  {pdf}")
        print(f"\n{len(found)} escaneados de ≤ {args.max_pages} páginas, {total} páginas en total")
        return 0

    if args.files:
        candidates = [Path(f) for f in args.files]
    else:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        print("Buscando escaneados (PDFs sin capa de texto)...", flush=True)
        candidates = find_scanned(Path(args.base), topics, args.max_pages, limit=0)
        print(f"  {len(candidates)} candidatos de ≤ {args.max_pages} páginas")
        if not candidates:
            print("no hay escaneados que comparar", file=sys.stderr)
            return 2
        random.Random(args.seed).shuffle(candidates)
        candidates = candidates[: args.sample]

    from extraction.ocr_extractor import OCRExtractor
    from extraction.vlm_extractor import VlmExtractor

    # `enabled=True` explícito: el extractor está apagado por entorno a propósito
    # y aquí se enciende sólo para medirlo, sin tocar la configuración real.
    vlm = VlmExtractor(enabled=True, only_scanned=False, max_pages=args.max_pages)
    ocr = OCRExtractor()

    results = [compare(pdf, vlm, ocr) for pdf in candidates]
    print_table(results)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  crudos en {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
