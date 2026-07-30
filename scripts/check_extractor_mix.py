#!/usr/bin/env python3
"""
Comprueba QUÉ extractor produjo los chunks de cada tema, leyendo el índice vivo.

Por qué existe: la run #3 de Electricidad informó "312/312 ficheros, status
success" y aun así 44 ficheros se habían indexado sin docling, porque su veto en
`crash_state.json` seguía puesto (PLAN.md §6.8-bis). El resumen de la tirada no
puede ver eso; el payload sí. `source` guarda el extractor de cada chunk, así que
un fichero con 0 % de docling delata un veto —falso o no— aunque la tirada lo
haya dado por bueno.

Uso:
  python3 scripts/check_extractor_mix.py                    # todos los temas
  python3 scripts/check_extractor_mix.py Electricidad       # uno
  python3 scripts/check_extractor_mix.py Electricidad --files  # desglose
"""

import collections
import json
import os
import sys
import urllib.request

QDRANT = "http://127.0.0.1:6333"
# 0 = recorrer la colección entera. Antes había un tope de 60.000 puntos y los
# porcentajes salían sobre la muestra pero se imprimían como si fueran del tema:
# con 202.189 puntos en Electricidad eso es el 30 %, y un fichero podía aparecer
# como "sin docling" sólo porque sus chunks no habían entrado en la muestra. El
# recorrido completo son ~41 peticiones de payload: no merece la pena arriesgar
# un dato engañoso. SAMPLE=n lo limita si hace falta ir rápido.
SAMPLE = int(os.environ.get("SAMPLE", "0"))


def _post(path, body):
    req = urllib.request.Request(
        QDRANT + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))["result"]


def collections_list():
    req = urllib.request.Request(QDRANT + "/collections")
    result = json.load(urllib.request.urlopen(req))["result"]
    return sorted(c["name"] for c in result["collections"])


def points_count(collection):
    req = urllib.request.Request(f"{QDRANT}/collections/{collection}")
    return json.load(urllib.request.urlopen(req))["result"]["points_count"]


def scan(collection):
    """Devuelve {fichero: Counter(extractor)} muestreando la colección."""
    per_file = collections.defaultdict(collections.Counter)
    offset, seen = None, 0
    while not SAMPLE or seen < SAMPLE:
        body = {
            "limit": 5000,
            "with_payload": ["file_path", "source"],
            "with_vector": False,
        }
        if offset:
            body["offset"] = offset
        page = _post(f"/collections/{collection}/points/scroll", body)
        for point in page["points"]:
            payload = point["payload"]
            name = str(payload.get("file_path", "?")).split("/")[-1]
            per_file[name][payload.get("source", "?")] += 1
        seen += len(page["points"])
        offset = page.get("next_page_offset")
        if not offset:
            break
    return per_file, seen


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_files = "--files" in sys.argv

    names = collections_list()
    if args:
        wanted = {f"rag_{t.lower()}" for t in args}
        names = [n for n in names if n in wanted]
        if not names:
            print(f"Sin colecciones para {args}. Hay: {collections_list()}")
            return 1

    for name in names:
        per_file, seen = scan(name)
        totals = collections.Counter()
        for mix in per_file.values():
            totals.update(mix)
        total = sum(totals.values()) or 1

        real = points_count(name)
        aviso = f", MUESTRA de {real}" if seen < real else ""
        print(f"\n=== {name} ({seen} puntos{aviso}, {len(per_file)} ficheros) ===")
        if aviso:
            print("  OJO: recorrido parcial, un '0 % docling' puede ser del muestreo.")
        for source, n in totals.most_common():
            print(f"  {source:22s} {n:7d}  {100 * n / total:5.1f} %")

        # Lo que importa: ficheros sin una sola línea de docling.
        sin_docling = sorted(
            ((f, sum(m.values())) for f, m in per_file.items() if not m.get("docling")),
            key=lambda x: -x[1],
        )
        if sin_docling:
            puntos = sum(n for _, n in sin_docling)
            print(
                f"\n  {len(sin_docling)} ficheros SIN chunks de docling "
                f"({puntos} puntos, {100 * puntos / total:.1f} % del tema):"
            )
            for f, n in sin_docling[: (None if show_files else 10)]:
                print(f"    {n:6d} pts  {f[:70]}")
            if not show_files and len(sin_docling) > 10:
                print(f"    ... y {len(sin_docling) - 10} más (--files para verlos)")
        else:
            print("\n  Todos los ficheros tienen chunks de docling.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
