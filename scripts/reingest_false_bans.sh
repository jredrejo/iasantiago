#!/usr/bin/env bash
#
# Re-ingesta los ficheros que se indexaron SIN docling por un veto falso.
#
# Contexto (PLAN.md §6.8-bis): la tirada rota de Electricidad (run #2) mató
# conversiones sanas por un fallo del heartbeat —arreglado en c481071— y cada
# muerte contó como caída de docling. A los 3 strikes el fichero queda vetado en
# `crash_state.json`. `main.py retry-failed` reactivó `.processing_state.json`,
# pero NO toca `crash_state.json`: son ficheros distintos en volúmenes distintos.
# Resultado: run #3 reindexó esos ficheros con docling vetado y cayeron a la
# cadena de extracción alternativa. Medido sobre el índice vivo, esos 44 ficheros
# de Electricidad tienen 0 % de chunks con `source=docling` frente al 78,9 % del
# resto del tema.
#
# El veto, además, sólo era la mitad del problema. Medido sobre el índice vivo con
# recuentos exactos, Electricidad tiene 99 ficheros sin un solo chunk de docling
# (56.452 puntos, el 28 % del tema), y sólo 43 de ellos llegaron a estar vetados:
# los otros ~56 fallaron una o dos veces —por debajo del umbral de 3 strikes—,
# cayeron a la cadena alternativa y la tirada los dio por buenos.
#
# Este script deshace eso: rehabilita SÓLO los vetos por conversión interrumpida
# (los PDFs rotos de verdad siguen vetados), busca en el ÍNDICE los ficheros sin
# docling, los saca de `processed` y de cuarentena para que el escaneo los vuelva
# a ver, y lanza una ingesta incremental. El resto del corpus no se toca:
# `index_pdf` borra los chunks previos de cada fichero antes de reescribirlo, así
# que es idempotente y no deja huérfanos.
#
# Uso:
#   DRY_RUN=1 ./scripts/reingest_false_bans.sh              # plan, no toca nada
#   ./scripts/reingest_false_bans.sh                        # sólo Electricidad
#   ./scripts/reingest_false_bans.sh Electricidad Chemistry # ambos temas
#   MAX_CHUNKS=0 ./scripts/reingest_false_bans.sh           # incluye los manuales gigantes
#
# Variables: DRY_RUN=1 · SKIP_BACKUP=1 · MAX_CHUNKS=1000 · SKIP_TEXTLAYER_CHECK=1
#
# SKIP_TEXTLAYER_CHECK=1 desactiva la criba del paso 2-bis (los escaneados
# entran también). Sólo tiene sentido si algún día se activa el OCR de docling.
#
# PARA LA WEB ABAJO mientras corre (el ingestor necesita la GPU entera). Lánzalo
# fuera de horario de clase y con nohup: sobrevive al cierre de la sesión SSH.

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

TOPICS=("${@:-Electricidad}")
DRY_RUN="${DRY_RUN:-0}"
SKIP_BACKUP="${SKIP_BACKUP:-0}"
# Tope de chunks por fichero, para poder acotar una tirada corta. Por defecto no
# hay tope: el tamaño no distingue lo que interesa. Con MAX_CHUNKS=1000 se caían
# los manuales gigantes de Omron (1.000–4.100 chunks) pero también apuntes del
# curso del mismo tamaño —"Control Avanzado de Procesos" (1.149), "Comunicaciones
# y Redes de Computadoras" (2.097), "OperacionesBásicasLaboratorioQuímica"
# (1.100)—, y ésos son justo los que hay que arreglar.
MAX_CHUNKS="${MAX_CHUNKS:-0}"

STATE="$ROOT/data/whoosh/.processing_state.json"
LOG="$ROOT/data/reingest-falsebans-$(date +%F_%H%M%S).log"
WEB_SERVICES=(vllm rag-api openwebui oauth2-proxy)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

restore_web() {
  log "Restaurando la web..."
  docker compose up -d "${WEB_SERVICES[@]}" >/dev/null 2>&1
  log "Web restaurada."
}

# La web vuelve aunque el script muera a mitad: es lo que dejó el sitio caído
# 9 h en la run #1.
cleanup() { [[ "$DRY_RUN" == "1" ]] || restore_web; }

log "=== Re-ingesta de vetos falsos de docling ==="
log "Temas: ${TOPICS[*]}"
[[ "$DRY_RUN" == "1" ]] && log "*** DRY RUN — no se modifica nada ***"

# ---------------------------------------------------------------------------
# 1. Qué se rehabilitaría. `--interrumpidos` deja vetados los PDFs que docling
#    no sabe abrir (bgpython_a4_c_2.pdf, el catálogo de Panreac).
# ---------------------------------------------------------------------------
log ""
log "--- Vetos por conversión interrumpida (candidatos) ---"
docker compose --profile ingest run --rm --no-deps -T --entrypoint python ingestor \
  main.py reset-docling-crashes --interrumpidos --dry-run 2>/dev/null | tee -a "$LOG"

# ---------------------------------------------------------------------------
# 2. Qué ficheros se van a reprocesar. El criterio es el CONTENIDO del índice:
#    ficheros de `processed` que no tienen NI UN chunk con `source=docling`.
#
#    No se usa `crash_state.json` para esto. El veto es estado transitorio —el
#    paso 4 lo borra, así que tras una tirada ya no queda de dónde deducir la
#    lista— y además sólo cubre la mitad del problema: en Electricidad hay 99
#    ficheros sin docling, de los que sólo 43 llegaron a estar vetados. Los
#    otros ~56 fallaron una o dos veces (por debajo del umbral de 3), cayeron a
#    la cadena alternativa y se dieron por buenos. El índice sí los delata.
#
#    Se excluyen los vetados POR UN MOTIVO REAL —el nombre de una excepción—:
#    ésos son los PDFs que docling no sabe abrir y volverían a caer a la cadena
#    alternativa igual. Los vetados por interrupción NO se excluyen: el paso 4 los
#    rehabilita justo después, así que excluirlos aquí los dejaría sin reprocesar
#    para siempre.
#
#    El recuento es exacto (`"exact": true` en /points/count), no muestreado:
#    check_extractor_mix.py muestrea 60 k de 202 k puntos y sirve para ver el
#    panorama, pero para decidir qué se reprocesa hace falta certeza.
# ---------------------------------------------------------------------------
BANNED=$(docker run --rm -v iasantiago-rag_docling_cache:/c:ro \
  python:3.11-slim python -c "
import json
crash = json.load(open('/c/crash_state.json'))
reasons = json.load(open('/c/crash_reasons.json'))
print('\n'.join(
    k for k, v in crash.items()
    if v >= 3 and 'interrump' not in reasons.get(k, {}).get('reason', '').lower()
))
" 2>/dev/null)

STDERR_TMP=$(mktemp)
TARGETS=$(MAX_CHUNKS="$MAX_CHUNKS" BANNED="$BANNED" python3 - "${TOPICS[@]}" 2>"$STDERR_TMP" <<'PY'
import json, os, sys, urllib.request

QDRANT = "http://127.0.0.1:6333"
topics = sys.argv[1:]
max_chunks = int(os.environ.get("MAX_CHUNKS") or 0)
banned = {os.path.basename(b) for b in os.environ.get("BANNED", "").split("\n") if b.strip()}
state = json.load(open("/opt/iasantiago-rag/data/whoosh/.processing_state.json"))


def count(collection, must):
    req = urllib.request.Request(
        f"{QDRANT}/collections/{collection}/points/count",
        data=json.dumps({"filter": {"must": must}, "exact": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))["result"]["count"]


targets, skipped_big, skipped_banned = [], [], []
for path, info in state.get("processed", {}).items():
    if info.get("topic") not in topics:
        continue
    coll = f"rag_{info['topic'].lower()}"
    f = [{"key": "file_path", "match": {"value": path}}]
    total = count(coll, f)
    if total == 0:
        continue
    if count(coll, f + [{"key": "source", "match": {"value": "docling"}}]):
        continue
    if os.path.basename(path) in banned:
        skipped_banned.append(path)
    elif max_chunks and total > max_chunks:
        skipped_big.append((path, total))
    else:
        targets.append((path, total))

for path, _ in targets:
    print(path)
# El resumen va por stderr para no contaminar la lista.
w = sys.stderr.write
w(f"  ({len(targets)} candidatos, {sum(n for _, n in targets)} chunks; "
  f"el paso 2-bis aún quita los escaneados)\n")
if skipped_banned:
    w(f"  ({len(skipped_banned)} excluidos por seguir vetados: docling no los abre)\n")
if skipped_big:
    pts = sum(n for _, n in skipped_big)
    w(f"  ({len(skipped_big)} excluidos por pasar de {max_chunks} chunks, {pts} puntos:\n")
    for path, n in sorted(skipped_big, key=lambda x: -x[1]):
        w(f"     {n:6d}  {os.path.basename(path)[:70]}\n")
    w(f"   sube MAX_CHUNKS para incluirlos; son manuales de miles de páginas)\n")
PY
)

# ---------------------------------------------------------------------------
# 2-bis. Descartar los PDFs que son imágenes escaneadas.
#
#    "Sin chunks de docling" no significa "mal indexado". 30 de los ficheros que
#    el paso 2 marca son escaneados puros: 0 caracteres de capa de texto. Docling
#    corre con do_ocr=False, así que devuelve un documento vacío —correctamente—
#    y la cadena los manda a easyocr_gpu, que es justo lo que hay que hacer con
#    ellos. Reprocesarlos son horas de OCR de GPU para dejarlos donde ya estaban.
#
#    El criterio es la CAPA DE TEXTO medida sobre el PDF, no el extractor que
#    figura en el índice. Probé el atajo "excluir lo que cubrió easyocr_gpu" y
#    se llevaba por delante 4 ficheros que sí tienen texto (entre ellos
#    4072940-7_nx-ecc20, cifrado, que docling ya recupera): easyocr aparece
#    también cuando lo que falló fue otra cosa.
#
#    Se mira dentro del contenedor porque el host no tiene pypdf, y sólo las 10
#    primeras páginas: con una portada escaneada y el cuerpo en texto basta para
#    decidir, y abrir 300 páginas por fichero costaría más que la propia criba.
# ---------------------------------------------------------------------------
if [[ -n "$TARGETS" && "${SKIP_TEXTLAYER_CHECK:-0}" != "1" ]]; then
  mapfile -t CANDIDATES <<< "$TARGETS"
  TEXT_TMP=$(mktemp)
  docker compose --profile ingest run --rm --no-deps -T --entrypoint python ingestor -c '
import sys
import pypdf

TEXT_MIN = 200
for path in [p for p in sys.argv[1:] if p.strip()]:
    try:
        reader = pypdf.PdfReader(path)
        if reader.is_encrypted:
            reader.decrypt("")
        chars = sum(len((pg.extract_text() or "").strip()) for pg in reader.pages[:10])
    except Exception as e:
        # Si no se puede ni mirar, que lo intente la ingesta: este paso sólo
        # está para ahorrar trabajo, no para vetar.
        print("CONTEXTO", path, sep="\t")
        print(f"  aviso: no se pudo leer la capa de texto ({e})", file=sys.stderr)
        continue
    print("CONTEXTO" if chars >= TEXT_MIN else "ESCANEADO", path, sep="\t")
' "${CANDIDATES[@]}" 2>/dev/null > "$TEXT_TMP"

  if [[ -s "$TEXT_TMP" ]]; then
    SCANNED=$(sed -n 's/^ESCANEADO\t//p' "$TEXT_TMP")
    TARGETS=$(sed -n 's/^CONTEXTO\t//p' "$TEXT_TMP")
    if [[ -n "$SCANNED" ]]; then
      log ""
      log "--- Excluidos por ser escaneados (sin capa de texto): $(echo "$SCANNED" | grep -c .) ---"
      echo "$SCANNED" | sed 's|.*/|  |' | tee -a "$LOG"
      log "    docling no puede con ellos sin OCR; easyocr_gpu ya los cubre."
    fi
  else
    log "AVISO: la criba de capa de texto no devolvió nada; sigo con la lista del paso 2."
  fi
  rm -f "$TEXT_TMP"
fi

COUNT=$(echo "$TARGETS" | grep -c . || true)

# El total de chunks se recuenta sobre la lista DEFINITIVA. El resumen del paso 2
# es anterior a la criba y hablaba de 102 ficheros mientras la cabecera decía 71:
# dos cifras distintas para lo mismo en la misma pantalla es cómo se cuela una
# tirada que no hace lo que dice (§10).
if [[ -n "$TARGETS" ]]; then
  mapfile -t FINAL_LIST <<< "$TARGETS"
  CHUNKS=$(python3 - "${FINAL_LIST[@]}" <<'PY'
import json, sys, urllib.request

state = json.load(open("/opt/iasantiago-rag/data/whoosh/.processing_state.json"))
total = 0
for path in [p for p in sys.argv[1:] if p.strip()]:
    topic = state.get("processed", {}).get(path, {}).get("topic")
    if not topic:
        continue
    req = urllib.request.Request(
        f"http://127.0.0.1:6333/collections/rag_{topic.lower()}/points/count",
        data=json.dumps(
            {"filter": {"must": [{"key": "file_path", "match": {"value": path}}]},
             "exact": True}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    total += json.load(urllib.request.urlopen(req))["result"]["count"]
print(total)
PY
)
fi

log ""
log "--- Ficheros sin un solo chunk de docling en ${TOPICS[*]}: $COUNT ($CHUNKS chunks) ---"
echo "$TARGETS" | sed 's|.*/|  |' | tee -a "$LOG"
[[ -s "$STDERR_TMP" ]] && tee -a "$LOG" < "$STDERR_TMP"
rm -f "$STDERR_TMP"
log ""
log "Además se sacará de cuarentena lo que agotó los reintentos (paso 5-bis)."

if [[ "$COUNT" -eq 0 ]]; then
  log "Nada que hacer."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log ""
  log "DRY RUN terminado. Para ejecutarlo de verdad, sin DRY_RUN=1."
  exit 0
fi

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 3. Copia de seguridad. No se borra ninguna colección (cada fichero se
#    reescribe solo), pero es barato: ~1,6 GB.
# ---------------------------------------------------------------------------
if [[ "$SKIP_BACKUP" != "1" ]]; then
  BACKUP_NAME="qdrant-$(date +%F_%H%M%S).tar.gz"
  mkdir -p "$ROOT/backups"
  log ""
  log "Copia de seguridad -> backups/$BACKUP_NAME"
  docker compose stop qdrant >/dev/null 2>&1   # parado: evita un tar inconsistente
  # tar dentro del contenedor y como root: si no, no puede leer los
  # payload_index de qdrant y archiva de menos en silencio (ver 4c56e04).
  docker run --rm -v "$ROOT/data:/data:ro" -v "$ROOT/backups:/backups" \
    alpine tar -czf "/backups/$BACKUP_NAME" -C /data storage
  rc=$?
  docker compose start qdrant >/dev/null 2>&1
  if [[ $rc -ne 0 ]]; then
    log "ERROR: la copia de seguridad falló (rc=$rc). Abortando."
    exit 1
  fi
  log "Copia hecha."
  sleep 5
fi

# ---------------------------------------------------------------------------
# 4. Rehabilitar docling en los vetos falsos.
# ---------------------------------------------------------------------------
log ""
log "Rehabilitando docling en los vetos por interrupción..."
docker compose --profile ingest run --rm --no-deps -T --entrypoint python ingestor \
  main.py reset-docling-crashes --interrumpidos 2>/dev/null | tee -a "$LOG"

# ---------------------------------------------------------------------------
# 5. Sacar los ficheros de `processed` para que el escaneo los vuelva a ver.
#    El JSON es de root y vive en un volumen: se edita desde un contenedor.
#
#    Las rutas van por ARGV, no por una tubería: `cmd <<'PY'` redirige stdin al
#    heredoc, así que un `echo "$TARGETS" | docker run ... python - <<'PY'` se
#    come la lista en silencio —python lee el script y `sys.stdin` ya está en
#    EOF—. El paso decía "Eliminados 0" y la tirada seguía como si nada: es lo
#    que dejó la ejecución del 2026-07-29 sin reprocesar ni un fichero.
# ---------------------------------------------------------------------------
mapfile -t TARGET_LIST <<< "$TARGETS"

log ""
log "Sacando $COUNT ficheros de 'processed'..."
OUT=$(docker run --rm -i -v "$ROOT/data/whoosh:/w" python:3.11-slim python - "${TARGET_LIST[@]}" <<'PY'
import json, os, sys
paths = [p for p in sys.argv[1:] if p.strip()]
p = '/w/.processing_state.json'
state = json.load(open(p))
removed = 0
for path in paths:
    if state.get('processed', {}).pop(path, None) is not None:
        removed += 1
    state.get('failed', {}).pop(path, None)
tmp = p + '.tmp'
json.dump(state, open(tmp, 'w'), indent=2)
os.replace(tmp, p)
print(f"Eliminados {removed} de 'processed' (quedan {len(state['processed'])})")
print(f"REMOVED={removed}")
PY
)
echo "$OUT" | grep -v '^REMOVED=' | tee -a "$LOG"
REMOVED=$(echo "$OUT" | sed -n 's/^REMOVED=\([0-9]*\)$/\1/p')

# Sin este paso la ingesta salta los 43 ficheros y la tirada no hace NADA, pero
# termina con "Errores: 0" y aspecto de éxito. Mejor abortar aquí.
if [[ "${REMOVED:-0}" != "$COUNT" ]]; then
  log "ERROR: se esperaba sacar $COUNT ficheros y se sacaron ${REMOVED:-0}."
  log "       Abortando antes de la ingesta: no habría nada que reprocesar."
  exit 1
fi

# ---------------------------------------------------------------------------
# 5-bis. Sacar de cuarentena lo que agotó los reintentos. `is_already_processed`
#    no basta: un fichero en cuarentena se salta con un WARNING, no se reintenta.
#    En Electricidad es el caso de 'Sistemas Programables Avanzados Marcombo.pdf'
#    ("No se produjo ningún chunk"), cuyo fallo viene probablemente de haberse
#    extraído sin docling.
# ---------------------------------------------------------------------------
log ""
log "Sacando de cuarentena los ficheros con reintentos agotados..."
docker compose --profile ingest run --rm --no-deps -T --entrypoint python ingestor \
  main.py retry-failed 2>/dev/null | tee -a "$LOG"

# ---------------------------------------------------------------------------
# 6. Ingesta incremental. Escanea todo y salta lo ya hecho, así que sólo
#    procesa los que acabamos de sacar de `processed`.
# ---------------------------------------------------------------------------
log ""
log "Parando la web para dejar la GPU libre..."
docker compose stop "${WEB_SERVICES[@]}" >/dev/null 2>&1

SINCE=$(date -u +%Y-%m-%dT%H:%M:%S)
log "Lanzando la ingesta incremental (esto tarda horas)..."
docker compose --profile ingest up -d ingestor >/dev/null 2>&1

while [[ "$(docker compose ps ingestor --format '{{.State}}' 2>/dev/null)" == "running" ]]; do
  sleep 60
done

log "Ingesta terminada."

# ---------------------------------------------------------------------------
# 7. Comprobar el resultado por CONTENIDO, no por el resumen de la tirada: run
#    #3 dijo "312/312 success" y aun así 44 ficheros salieron sin docling.
# ---------------------------------------------------------------------------
errs=$(docker compose logs ingestor --since "$SINCE" 2>&1 | grep -c ' - ERROR - ' || true)
log "Errores en el log: $errs"
[[ "$errs" -gt 0 ]] && log "  revisar: docker compose logs ingestor --since $SINCE | grep ERROR"

log ""
log "Vetos que siguen puestos (deberían ser sólo los PDFs rotos de verdad):"
docker compose --profile ingest run --rm --no-deps -T --entrypoint python ingestor \
  main.py reset-docling-crashes --dry-run 2>/dev/null | tee -a "$LOG"

restore_web
trap - EXIT INT TERM

log ""
log "=== Hecho. Log: $LOG ==="
log "Comprueba que los ficheros llevan ya source=docling con:"
log "  scripts/check_extractor_mix.py"
