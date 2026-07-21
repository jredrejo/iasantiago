#!/usr/bin/env bash
#
# Re-indexa temas de uno en uno, en vez de `make reset`.
#
# Por qué no `make reset`: es todo-o-nada. Si muere a mitad de Electricidad
# (316 ficheros, 4,4 GB) te quedas con el corpus a medias, sin instantáneas
# —que no sobreviven a `docker compose down`, viven en la capa del contenedor,
# no en el bind mount— y sin forma de volver atrás. Aquí un fallo cuesta un
# tema, no el corpus: los ya terminados quedan hechos y el script para.
#
# Cada tema: limpiar su estado -> borrar su colección -> ingestar -> verificar.
# El ingestor escanea todo pero omite lo ya procesado, así que sólo rehace el
# tema que se acaba de limpiar.
#
# Uso:
#   scripts/reindex_topics.sh                  # secuencia por defecto
#   scripts/reindex_topics.sh Dibujo FOL       # sólo esos temas
#   DRY_RUN=1 scripts/reindex_topics.sh        # enseña el plan y sale
#
# Electricidad NO está en la lista por defecto a propósito: 316 ficheros y el
# camino de OCR/escaneado, entre 10 y 15 horas estimadas. Pásalo explícitamente
# cuando tengas la ventana:  scripts/reindex_topics.sh Electricidad

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# Orden: primero el de mayor ganancia conocida (Dibujo, 3282 fragmentos
# reparables), luego los pequeños. Chemistry y Programming ya están hechos.
DEFAULT_TOPICS=(Dibujo Sostenibilidad Mecanica FOL AFD Latin)
TOPICS=("${@:-}")
[[ -z "${TOPICS[0]:-}" ]] && TOPICS=("${DEFAULT_TOPICS[@]}")

STATE="$ROOT/data/whoosh/.processing_state.json"
LOG="$ROOT/data/reindex-$(date +%F_%H%M%S).log"
QDRANT="http://localhost:6333"
WEB_SERVICES=(vllm rag-api openwebui oauth2-proxy)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORTA: $*"; restore_web; exit 1; }

collection_for() { echo "rag_$(echo "$1" | tr '[:upper:]' '[:lower:]')"; }

points_in() {
  curl -s "$QDRANT/collections/$1" \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',{}).get('points_count','0'))" 2>/dev/null \
    || echo 0
}

# El fichero de estado es de root: se edita desde un contenedor de usar y tirar.
state_py() {
  docker run --rm -v "$ROOT/data/whoosh:/w" python:3.11-alpine python -c "$1" 2>/dev/null
}

restore_web() {
  log "Restaurando servicios web..."
  docker compose up -d "${WEB_SERVICES[@]}" >/dev/null 2>&1
  for _ in $(seq 1 60); do
    curl -sf http://localhost:8001/healthz >/dev/null 2>&1 && { log "rag-api OK"; return; }
    sleep 5
  done
  log "AVISO: rag-api no respondió en 5 min; revisar a mano"
}

# ---------------------------------------------------------------- comprobaciones

log "Temas: ${TOPICS[*]}"
log "Log: $LOG"

for t in "${TOPICS[@]}"; do
  [[ -d "$ROOT/topics/$t" ]] || die "no existe topics/$t"
done

# Reconciliación estado vs disco. `make reset` borra el estado entero y se lleva
# por delante estas discrepancias sin decir nada; aquí se informan.
log "--- Reconciliando estado contra disco ---"
state_py "
import json,os
d=json.load(open('/w/.processing_state.json'))
stale=[k for k in d['processed'] if not os.path.exists('/w/../..'+k)]
print(f'  procesados en estado: {len(d[\"processed\"])}')
print(f'  fallidos en estado  : {len(d.get(\"failed\",{}))}')
" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import json,os
d=json.load(open('data/whoosh/.processing_state.json'))
stale=[k for k in d['processed'] if not os.path.exists('.'+k)]
disk={'/topics/'+os.path.relpath(os.path.join(r,f),'topics')
      for r,_,fs in os.walk('topics') for f in fs if f.lower().endswith('.pdf')}
missing=sorted(disk-set(d['processed']))
print(f"  en estado pero borrados del disco: {len(stale)}")
for k in stale[:8]: print(f"     - {k}")
print(f"  en disco pero nunca indexados    : {len(missing)}")
for k in missing[:8]: print(f"     + {k}")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  log "DRY_RUN=1: plan mostrado, no se toca nada."
  for t in "${TOPICS[@]}"; do
    c=$(collection_for "$t")
    log "  $t -> limpiar estado, borrar $c ($(points_in "$c") puntos), reingestar"
  done
  exit 0
fi

# ---------------------------------------------------------------- copia de seguridad

log "--- Copia de seguridad de Qdrant (única marcha atrás real) ---"
BACKUP="$ROOT/backups/qdrant-$(date +%F_%H%M%S).tar.zst"
mkdir -p "$ROOT/backups"
docker compose stop qdrant >/dev/null 2>&1   # parado: evita un tar inconsistente
tar --use-compress-program='zstd -T0 -3' -cf "$BACKUP" -C "$ROOT/data" storage 2>/dev/null \
  || tar -czf "${BACKUP%.zst}.gz" -C "$ROOT/data" storage
docker compose start qdrant >/dev/null 2>&1
for _ in $(seq 1 40); do curl -sf "$QDRANT/readyz" >/dev/null 2>&1 && break; sleep 3; done
log "Copia: $(ls -la "$ROOT/backups" | tail -1 | awk '{print $9, $5" bytes"}')"

# ---------------------------------------------------------------- GPU

log "--- Liberando GPU (los servicios web bajan) ---"
docker compose stop "${WEB_SERVICES[@]}" >/dev/null 2>&1
sleep 8
log "VRAM en uso: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# ---------------------------------------------------------------- bucle

OK=(); FAILED=""
for t in "${TOPICS[@]}"; do
  c=$(collection_for "$t")
  before=$(points_in "$c")
  files=$(find "topics/$t" -iname '*.pdf' | wc -l)
  log "=== $t: $files ficheros, $before puntos antes ==="

  state_py "
import json,os
p='/w/.processing_state.json'
d=json.load(open(p))
b=len(d['processed'])
d['processed']={k:v for k,v in d['processed'].items() if v.get('topic')!='$t'}
d['failed']={k:v for k,v in d.get('failed',{}).items() if v.get('topic')!='$t'}
json.dump(d,open(p+'.tmp','w')); os.replace(p+'.tmp',p)
print(f'  estado: {b} -> {len(d[\"processed\"])}')
" | tee -a "$LOG"

  # Borrar la colección es obligatorio: si las dimensiones coinciden,
  # ensure_collection la da por buena y se mezclarían vectores viejos y nuevos.
  curl -s -X DELETE "$QDRANT/collections/$c" >/dev/null
  rm -rf "$ROOT/data/whoosh/$t" 2>/dev/null || \
    docker run --rm -v "$ROOT/data/whoosh:/w" alpine rm -rf "/w/$t" >/dev/null 2>&1

  log "  ingestando..."
  docker compose --profile ingest up -d ingestor >/dev/null 2>&1
  start=$SECONDS
  while [[ "$(docker compose ps ingestor --format '{{.State}}' 2>/dev/null)" == "running" ]]; do
    sleep 30
  done
  mins=$(( (SECONDS-start)/60 ))

  after=$(points_in "$c")
  repaired=$(docker compose logs ingestor 2>&1 | grep -c 'REPAIR].*reparados' || true)
  log "  hecho en ${mins}m: $before -> $after puntos ($repaired lotes reparados)"

  if [[ "$after" == "0" || "$after" == "None" ]]; then
    FAILED="$t"
    log "  FALLO: $t quedó con 0 puntos"
    break            # parar: un fallo sistémico no debe vaciar más temas
  fi
  OK+=("$t")
done

# ---------------------------------------------------------------- fin

restore_web

log "--- Resumen ---"
log "Completados: ${OK[*]:-ninguno}"
[[ -n "$FAILED" ]] && log "FALLÓ en: $FAILED (los siguientes NO se han tocado)"
for t in "${OK[@]:-}"; do
  [[ -n "$t" ]] && log "  $t: $(points_in "$(collection_for "$t")") puntos"
done
log "Copia de seguridad: $BACKUP"
log "Pendiente a propósito: Electricidad (316 ficheros, ~10-15 h)"
[[ -n "$FAILED" ]] && exit 1
exit 0
