# IASantiago RAG 

## 1) ¿Qué es y qué puede hacer? (visión general)

**IASantiago RAG** es un stack de Recuperación Aumentada con Generación que monta un “ChatGPT interno” sobre PDFs organizados por **temas**. Arquitectura:

```
Open WebUI  ──>  RAG API (FastAPI, OpenAI-compatible)
                  ├─ Hybrid Retrieval: Qdrant (denso) + BM25 (Whoosh)
                  ├─ Re-ranker (Jina/BGE)
                  ├─ Streaming + citas [archivo.pdf, p.X] + límites dinámicos
                  ├─ /v1/models (selector de tema)  ──> colecciones por tema
                  └─ /v1/eval/offline (Recall@k, MRR)
Ingestor (PDF→chunks→embeddings→Qdrant + BM25)
Qdrant (vectores)    vLLM (LLM servido en GPU)
Nginx (TLS + OIDC Google Workspace)
```

### Funcionalidades clave

* **Selector de tema** dentro de Open WebUI (usando `/v1/models`), p. ej. `topic:Chemistry / Electronics / Programming`.
* **RAG híbrido**: búsqueda **densa** (embeddings `intfloat/multilingual-e5-large-instruct` por defecto; configurables por tema) + **BM25** (Whoosh).
* **Heurísticas de calidad**:

  * **Fallback BM25** si la consulta es muy corta (< 4 tokens).
  * **Re-ranking** cruzado (Jina Reranker multilingüe por defecto).
  * **Límite por archivo** (máx. N fragmentos por documento) para evitar monopolios.
  * **Límite de contexto dinámico** por tokens.
* **Citas embebidas y clicables** (texto marcado “[…]”; si publicas `/docs/` en Nginx, se vuelven enlaces).
* **Streaming** de tokens desde vLLM.
* **Telemetría**: `retrieval.jsonl` con consultas y resultados; **logrotate** incluido.
* **Evaluación**: endpoint `/v1/eval/offline` y **cron nocturno** (Recall@k / MRR a `eval_summary.csv`).
* **Seguridad**: autenticación **Google Workspace (OIDC)** y restricción por **dominio**; Nginx escucha en **443** (`iasantiago.santiagoapostol.net`) sobre **intranet** (`172.23.120.11`).

---

## 2) Guía rápida para usuarios

### Acceso y login

1. Abre `https://iasantiago.santiagoapostol.net` (puede pedirte aceptar el certificado si es self-signed).
2. Inicia sesión con tu cuenta de **Google Workspace** del dominio permitido (p. ej. `@santiagoapostol.net`).

### Elegir el **tema**

* En la barra superior de Open WebUI, abre el desplegable de **Modelos** y elige uno de la forma `topic:<Tema>` (ej.: `topic:Chemistry`).
* Todo lo que preguntes en esa conversación **solo** buscará PDFs del tema seleccionado.

### Subir PDFs (profesorado)

* Copia tus PDFs a la carpeta del tema correspondiente en el servidor:

  * `/opt/iasantiago-rag/topics/Chemistry/`
  * `/opt/iasantiago-rag/topics/Electronics/`
  * `/opt/iasantiago-rag/topics/Programming/`
* El **ingestor** indexa en segundo plano; tras unos segundos/minutos los documentos quedarán disponibles.

### Hacer preguntas y entender las citas

* Formula tu pregunta con palabras clave (si es muy corta, el sistema usará BM25).
* La respuesta incluye citas como: `[…/archivo.pdf, p.12]`.
* Si el admin publicó `/docs/` en Nginx, esas citas serán **clicables** y abrirán el PDF en esa página/archivo.

### Modos de respuesta (explica / guía / examen)

* Verás referencias a “modo” en las respuestas. Puedes pedirlo en el primer mensaje (“modo: **explica**/**guía**/**examen**”) o el administrador puede publicar variantes del modelo (por ejemplo `topic:Chemistry [examen]`).

### Consejos de consulta

* Para **definiciones** o términos breves, añade **más contexto** (“definición + autor/tema/curso”) para mejorar el ranking.
* Para **preguntas amplias**, incluye **palabras clave** concretas (títulos de secciones, leyes, fechas, etc.).
* Si la respuesta parece “genérica”, pide **citas más específicas** o reformula con más detalles.

---

## 3) Para desarrolladores y admins (montar y operar el sistema)

### Requisitos

* **Ubuntu Server 24.04**, GPU NVIDIA (ej.: RTX 5090), drivers + CUDA instalados.
* Docker + Docker Compose plugin, Nginx, Python 3.
* Acceso a Internet saliente (para JWKS de Google y descarga de modelos en el primer arranque).

### Estructura de proyecto

```
/opt/iasantiago-rag/
├─ docker-compose.yml              ── vLLM, Qdrant, RAG API, Ingestor, Open WebUI
├─ Makefile                        ── make up / seed / reset / bench
├─ .env.example  → .env            ── configuración
├─ topics/                         ── PDFs por tema (montado read-only en RAG API)
├─ data/storage                    ── datos Qdrant
├─ data/whoosh                     ── índices BM25 por tema
├─ rag-api/ …                      ── FastAPI (OpenAI-compatible)
├─ ingestor/ …                     ── indexación PDF→Qdrant+BM25
├─ openwebui/.env.openwebui
├─ nginx/nginx.conf (+certs, dhparam.pem)
├─ systemd/*.service|*.timer|logrotate-telemetry
└─ scripts/deploy_all.sh
```

### Variables clave en `.env`

* **Temas**: `TOPIC_LABELS=Chemistry,Electronics,Programming`
* **Carpetas**: `TOPIC_BASE_DIR=/opt/iasantiago-rag/topics`
* **Embeddings** (por tema):

  * `EMBED_MODEL_DEFAULT=intfloat/multilingual-e5-large-instruct`
  * `EMBED_MODEL_PROGRAMMING=thenlper/gte-large` (ejemplo)
* **Re-ranker**: `RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual`
* **Límites**: `CTX_TOKENS_SOFT_LIMIT`, `MAX_CHUNKS_PER_FILE`, `HYBRID_*_K`, `FINAL_TOPK`
* **Auth**: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `ALLOWED_EMAIL_DOMAIN=@santiagoapostol.net`
* **Rutas**: `QDRANT_URL`, `BM25_BASE_DIR`, `TELEMETRY_PATH`

### Despliegue rápido

```bash
sudo mkdir -p /opt/iasantiago-rag && cd /opt/iasantiago-rag
# Copia dentro todos los archivos del proyecto (los de este README).
cp .env.example .env   # y rellena OIDC si procede
make up                # build + up -d
```

* **Nginx (bare-metal)**: usa `nginx/nginx.conf`, certificado en `nginx/certs/`.
* **Systemd**:

  ```bash
  sudo ln -sf /opt/iasantiago-rag/systemd/iasantiago-rag.service /etc/systemd/system/
  sudo ln -sf /opt/iasantiago-rag/systemd/iasantiago-rag-eval.service /etc/systemd/system/
  sudo ln -sf /opt/iasantiago-rag/systemd/iasantiago-rag-eval.timer /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now iasantiago-rag.service
  sudo systemctl enable --now iasantiago-rag-eval.timer
  ```
* **Logrotate**:

  ```bash
  sudo ln -sf /opt/iasantiago-rag/systemd/logrotate-telemetry /etc/logrotate.d/iasantiago-rag
  ```

### Seeds y reindexado

```bash
make seed   # crea carpetas/ejemplos y reinicia ingestor
make reset  # borra Qdrant + Whoosh y reindexa desde cero
```

### Open WebUI — selector por tema

* La RAG API expone `/v1/models` con entradas `topic:<Tema>`.
* Open WebUI mostrará esas opciones en su desplegable de modelos.
* Todo chat que use ese “modelo” queda **limitado** a la colección Qdrant + índice BM25 del **tema**.

### Endpoints de la RAG API (OpenAI-compat)

* `/v1/models` → lista de “modelos” (temas).
* `/v1/chat/completions` → streaming permitido (`stream=true`).

  * Añade **citas** tipo `[archivo.pdf, p.X]` en el texto.
  * Respeta **límite dinámico** de tokens y **límite por archivo**.
* `/v1/eval/offline` → `POST` con `[{query, topic, relevant_files:[…]}]` → `{aggregate, details}`.

### Evaluación y telemetría

* **Cron (systemd timer)** nocturno ejecuta evaluación “toy” y guarda `eval_summary.csv`.
* **Telemetría de retrieval** a `rag-api/retrieval.jsonl` (rotado diariamente).

### Seguridad (intranet + OIDC Google)

* **Open WebUI** usa OIDC para login (redirect a Google y vuelta).
* **Nginx** protege `/` con `auth_request` hacia `/auth` de la RAG API, que valida el **ID Token** y **dominio permitido**.
* Asegúrate de permitir **salida** a `https://accounts.google.com` para JWKS.

### Rendimiento y GPU

* vLLM corre con `--gpu-memory-utilization 0.85` (ajusta según VRAM).
* `rag-api` puede usar GPU para embeddings y re-ranker. Si necesitas aislar GPUs:

  * Exporta `CUDA_VISIBLE_DEVICES` por servicio en `docker-compose.yml`.

### Publicar PDFs como **clicables**

* Opción rápida (opcional): en Nginx, sirve `/opt/iasantiago-rag/topics` como `/docs/`:

  ```nginx
  location /docs/ {
    autoindex on;
    alias /opt/iasantiago-rag/topics/;
  }
  ```
* En la RAG API, convierte `file_path` → URL `/docs/<Tema>/<archivo.pdf>` para que Open WebUI los muestre como enlaces (el ejemplo actual deja el texto marcado listo para enlazar).

### Operación diaria

* **Añadir PDFs**: ponlos en `/opt/iasantiago-rag/topics/<Tema>/`.
* **Ver estado**:

  * Qdrant: `http://<host>:6333/dashboard` (si lo expones a la LAN).
  * RAG API: `http://<host>:8001/healthz`.
  * vLLM: `http://<host>:8000/v1/models`.
* **Logs**:

  ```bash
  docker logs -f rag-api
  docker logs -f ingestor
  docker logs -f vllm
  ```

### Troubleshooting rápido

* **No sale el tema en Open WebUI** → comprueba `/v1/models` en `rag-api` y variables `TOPIC_LABELS`.
* **No hay resultados** → revisa que el PDF esté en la carpeta correcta, que el **ingestor** lo haya indexado y que Qdrant/Woosh tengan datos en `data/`.
* **Auth falla** → revisa `GOOGLE_CLIENT_ID/SECRET`, redirect URI en GCP (`/_auth/callback`), y conectividad a `accounts.google.com`.
* **VRAM insuficiente** → prueba un modelo de LLM más pequeño o baja `--max-model-len`/`--gpu-memory-utilization`.


## 4) Operación & Mantenimiento (recetas rápidas)

### Estado y logs

```bash
cd /opt/iasantiago-rag
docker compose ps
docker compose logs -f rag-api
docker compose logs -f ingestor
docker compose logs -f vllm
docker compose logs -f qdrant
```

### Ciclo de vida de la pila

```bash
make up            # build + up -d
make down          # docker compose down
make reset         # borra Qdrant+Whoosh y reindexa
make seed          # crea carpetas y PDFs de ejemplo e indexa
```

### Servicios del sistema (arranque automático)

```bash
sudo systemctl status iasantiago-rag.service
sudo systemctl restart iasantiago-rag.service

# Evaluación nocturna
sudo systemctl list-timers | grep iasantiago
sudo systemctl start iasantiago-rag-eval.service   # ejecutar ahora
sudo systemctl restart iasantiago-rag-eval.timer
```

### Nginx + TLS

```bash
sudo nginx -t
sudo systemctl reload nginx
# Cert & clave en: /opt/iasantiago-rag/nginx/certs/
```

### Logrotate de telemetría

```bash
sudo logrotate -d /etc/logrotate.d/iasantiago-rag     # dry-run (ver plan)
sudo logrotate -f /etc/logrotate.d/iasantiago-rag     # forzar rotación
# Archivo: /opt/iasantiago-rag/rag-api/retrieval.jsonl
```

---

## 5) Backups, Snapshots y Migraciones

> Recomendado programar una tarea (cron/systemd timer) diaria fuera del horario lectivo.

### 5.1 Qdrant — snapshots

**Crear snapshot** (detiene escrituras unos instantes; seguro para lecturas):

```bash
# Dentro de Qdrant (o vía API). Con Docker:
docker exec -it qdrant bash -lc 'curl -X POST http://localhost:6333/snapshots -s'
# Listar snapshots:
docker exec -it qdrant bash -lc 'curl http://localhost:6333/snapshots -s | jq'
# Copiar a host (si usa el volumen bind /opt/iasantiago-rag/data/storage ya persiste)
# opcional: rsync a almacenamiento externo
sudo rsync -av --delete /opt/iasantiago-rag/data/storage/ /backups/qdrant/storage-$(date +%F)/
```

**Restaurar snapshot** (ventana de mantenimiento):

```bash
docker compose down
# Restituir carpeta de almacenamiento
sudo rsync -av /backups/qdrant/storage-YYYY-MM-DD/ /opt/iasantiago-rag/data/storage/
docker compose up -d
```

> Alternativa: **export/import por colección** usando la API de Qdrant si prefieres granularidad (collections: `rag_chemistry`, `rag_electronics`, `rag_programming`).

### 5.2 Whoosh (BM25) — backup simple

Los índices Whoosh son ficheros en disco:

```bash
sudo rsync -av --delete /opt/iasantiago-rag/data/whoosh/ /backups/whoosh/$(date +%F)/
```

**Re-construir** (si hay corrupción o tras migración):

```bash
make reset                 # borra Qdrant+Whoosh
docker compose restart ingestor
```

### 5.3 PDFs fuente

Respaldar los documentos originales (la **verdad** del sistema):

```bash
sudo rsync -av --delete /opt/iasantiago-rag/topics/ /backups/topics/$(date +%F)/
```

### 5.4 Exportar/importar por tema (migración a otro servidor)

1. Copiar **topics/**, **data/storage/** (Qdrant) y **data/whoosh/** a la nueva máquina.
2. Replicar `.env` y el repo.
3. `make up` en el destino.
4. Verificar `/healthz` y `/v1/models`.

---

## 6) Variantes y Extensiones

### 6.1 Cambiar el **re-ranker** (Jina ↔ BGE)

En `.env`:

```
RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
# o
RERANK_MODEL=BAAI/bge-reranker-large
```

Recrea el contenedor de `rag-api`:

```bash
docker compose up -d --build rag-api
```

### 6.2 Embeddings **por tema** (p. ej. uno para código)

En `.env`:

```
EMBED_MODEL_DEFAULT=intfloat/multilingual-e5-large-instruct
EMBED_MODEL_PROGRAMMING=thenlper/gte-large
EMBED_MODEL_ELECTRONICS=intfloat/multilingual-e5-large-instruct
EMBED_MODEL_CHEMISTRY=intfloat/multilingual-e5-large-instruct
```

Reindexa solo si cambias embeddings (para reflejar vectores nuevos):

```bash
make reset   # borra Qdrant+Whoosh y reindexa todo
```

### 6.3 Límite por archivo y contexto

Ajusta en `.env`:

```
MAX_CHUNKS_PER_FILE=3
CTX_TOKENS_SOFT_LIMIT=6000
HYBRID_DENSE_K=40
HYBRID_BM25_K=40
FINAL_TOPK=12
BM25_FALLBACK_TOKEN_THRESHOLD=4
```

Reinicia `rag-api`:

```bash
docker compose restart rag-api
```

### 6.4 Activar “**watcher**” de PDFs en caliente

Si quieres indexación automática al **crear** ficheros:

1. Cambia `ingestor` para que ejecute `watcher.py` (en `docker-compose.yml`):

```yaml
  ingestor:
    build: ./ingestor
    command: ["python", "-u", "watcher.py"]
```

2. Recomiendo mantener un **scan inicial** al arranque (llamando a `main.py`) o ejecutarlo manualmente:

```bash
docker compose run --rm ingestor python -u main.py
```

### 6.5 “Modelos por modo” (explica/guía/examen) visibles en Open WebUI

Opción rápida: **duplicar** modelos en `/v1/models` con sufijos y parsearlos:

* Edita `rag-api/app.py` en `list_models` para publicar:

  * `topic:Chemistry [explica]`, `topic:Chemistry [guía]`, `topic:Chemistry [examen]`, etc.
* Ajusta `extract_topic_from_model_name` para extraer **tema** y **modo** (y propágalo como `iasantiago_mode`).

### 6.6 Hacer **citas clicables** hacia PDFs

1. Publica `/docs/` en Nginx:

```nginx
location /docs/ {
  autoindex on;
  alias /opt/iasantiago-rag/topics/;
}
```

2. En `rag-api/retrieval.py`, al construir citas, convierte `file_path` → URL:

```python
base_url = os.getenv("DOCS_BASE_URL", "https://iasantiago.santiagoapostol.net/docs")
name = os.path.basename(c["file_path"])
topic = os.path.basename(os.path.dirname(c["file_path"]))
url = f"{base_url}/{topic}/{name}"
context.append(f'{c["text"]}\n— según <{url}> [{name}, p.{c["page"]}]')
```

> Open WebUI renderiza enlaces de texto plano. También puedes añadir un **campo `sources`** en la respuesta si construyes una capa UI (no estándar OpenAI).

### 6.7 Cambiar el **LLM** en vLLM

En `.env`:

```
VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
# ejemplos alternativos (según licencia/VRAM):
# VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# VLLM_MODEL=mistralai/Mistral-Nemo-Instruct-2407
```

Recrear `vllm`:

```bash
docker compose up -d --build vllm
```

#### Quantización y memoria

* vLLM soporta cuantización (AWQ/GPTQ en algunos modelos). Si el modelo elegido la admite:

  * Añade flags de vLLM (según doc del modelo) o usa un **checkpoint ya cuantizado**.
* Ajusta:

```
--gpu-memory-utilization 0.85
--max-model-len 8192
--tensor-parallel-size 1
```

para encajar en tu VRAM.

### 6.8 Multi-GPU (aislar GPU para cada servicio)

En `docker-compose.yml`, por servicio:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
          device_ids: ["0"]   # o ["1"] etc.
```

Y en `rag-api`/`ingestor` puedes exportar:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
```

---

## 7) Benchmarks y Métricas

### 7.1 Latencia de vLLM (simple)

```bash
make bench
# Usa scripts/bench_vllm.py (p50/avg/max).
```

### 7.2 Evaluación offline manual (Recall@k / MRR)

```bash
cat <<'JSON' > /tmp/cases.json
[
  {"query": "definición de ley de Ohm", "topic": "Electronics", "relevant_files": ["/opt/iasantiago-rag/topics/Electronics/sample1.pdf"]},
  {"query": "ácidos y bases", "topic": "Chemistry", "relevant_files": ["/opt/iasantiago-rag/topics/Chemistry/sample1.pdf"]}
]
JSON

curl -s http://127.0.0.1:8001/v1/eval/offline \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/cases.json | jq
```

### 7.3 Telemetría de recuperación

* Archivo JSONL: `/opt/iasantiago-rag/rag-api/retrieval.jsonl`
* Formato por línea:

```json
{"ts": 1733920000000, "query": "ohm", "topic": "Electronics", "mode": "hybrid", "...": "..."}
```

* Analízalo con tu herramienta favorita (pandas, jq, etc.).

---

## 8) Seguridad y Red

* El servidor vive en **intranet** (`172.23.120.11`) y expone **443**.
* **OIDC Google**:

  * `GOOGLE_CLIENT_ID/SECRET` en `.env`.
  * Redirect URI: `https://iasantiago.santiagoapostol.net/_auth/callback`.
  * Dominio permitido: `ALLOWED_EMAIL_DOMAIN=@santiagoapostol.net`.
* **Salida a Internet** necesaria para:

  * Validar JWKS (`accounts.google.com`).
  * Descargar modelos en primer arranque (Hugging Face).

> Si no hay salida a Internet, **pre-cacha** modelos en un mirror interno y desactiva OIDC (o usa un IdP local).

---

## 9) Resolución de Problemas (checklist)

* **Open WebUI no muestra temas**

  * `curl http://127.0.0.1:8001/v1/models | jq`
  * Revisa `TOPIC_LABELS` y reinicia `rag-api`.
* **No recupera nada**

  * ¿PDF en la carpeta correcta? (`/opt/iasantiago-rag/topics/<Tema>/`).
  * `docker compose logs ingestor` para ver si se indexó.
  * ¿Colecciones existen? (`rag_<tema>` en Qdrant).
* **Errores de memoria GPU**

  * Cambia LLM por uno más pequeño o reduce `--max-model-len`, `--gpu-memory-utilization`.
  * Separa GPUs con `CUDA_VISIBLE_DEVICES`.
* **Auth 401/403**

  * Comprueba `GOOGLE_CLIENT_ID/SECRET`, redirect URI en GCP, reloj del servidor (NTP), conectividad a `accounts.google.com`.

---

## 10) Checklist de Actualización del Sistema

1. **Parar stack** (opcional en actualizaciones mayores):

```bash
docker compose down
```

2. **Pull/build** imágenes y servicios:

```bash
docker compose pull
docker compose build
docker compose up -d
```

3. **Migraciones**:

   * Si cambias **embeddings**, ejecuta `make reset` para reindexar.
   * Si solo cambias **reranker** o **knobs**, basta con reiniciar `rag-api`.

