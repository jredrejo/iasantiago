# IASantiago RAG

Sistema de Recuperación Aumentada con Generación (RAG) para el Colegio Santiago Apóstol que permite consultar documentos PDF organizados por temas mediante una interfaz de chat.

> **Plan de mejoras pendientes**: ver [IMPROVEMENTS.md](IMPROVEMENTS.md) (seguridad, bugs conocidos, limpieza y evolución hacia funcionalidades nativas de Open WebUI).

## Arquitectura

```
Internet / LAN
      │
┌─────▼─────────┐
│  nginx (443)  │ ← TLS Let's Encrypt
└─────┬─────────┘
      │
┌─────▼─────────┐
│ oauth2-proxy  │ ← Autenticación Google Workspace (OIDC + PKCE)
│    (4180)     │
└─────┬─────────┘
      │
┌─────▼──────────────────────────┐
│      Open WebUI (8080)         │ ← Interfaz Y orquestación del chat:
│                                │   historial, prompt de sistema y
│  ┌──────────────────────────┐  │   muestreo por modelo de workspace.
│  │ Filter (inlet)           │  │   Cabeceras confiadas:
│  │ resuelve tema → contexto │  │   X-Forwarded-Email
│  └────────────┬─────────────┘  │
└───────┬───────┼────────────────┘
        │       │
        │       │ POST /retrieve
        │       │
        │  ┌────▼──────────────────────────┐
        │  │       RAG API (8001)          │
        │  │   Servicio de retrieval PURO  │
        │  │  • Búsqueda híbrida: Qdrant   │
        │  │    denso + disperso (BM25)    │
        │  │  • Re-ranking (Jina, CPU)     │
        │  │  • Contexto + citas en JSON   │
        │  └────────────┬──────────────────┘
        │               │
        │               └──► Qdrant (6333/6334, vectores)
        │
        │ generación (streaming, OpenAI-compat)
        │
        └──────────► vLLM (8000)
                     Qwen3.6-27B-NVFP4, 64k ctx, MTP, tool calling

                     También accesible por Tailnet (Tailscale) sin pasar
                     por el RAG: es como conecta opencode.

┌─────────────────┐
│    Ingestor     │ ← Indexación de PDFs (bajo demanda, EXCLUYENTE
│                 │   con vLLM: no caben juntos en la GPU)
│                 │   • Extracción con Docling + Unstructured.io
│                 │   • Embeddings a Qdrant (denso + disperso)
└─────────────────┘
```

> **rag-api no genera ni habla con vLLM.** Dejó de ser un proxy de chat compatible
> con OpenAI el 2026-08-02: no expone `/v1/models` ni `/v1/chat/completions`. Quien
> orquesta el chat es Open WebUI, y rag-api sólo responde `POST /retrieve`,
> `POST /v1/eval/offline` y `GET /healthz`.

## Funcionalidades Principales

### Para Usuarios

- **Selector de temas**: un "modelo" por tema en el desplegable (AFD, Chemistry, Dibujo, Electricidad, FOL, Latin, Mecanica, Programming, Sostenibilidad — configurable en `TOPIC_LABELS`)
- **Búsqueda híbrida**: Combina embeddings densos + BM25 léxico
- **Citas clicables**: Enlaces directos a PDFs con número de página
- **Streaming**: Respuestas en tiempo real
- **Autenticación Google**: Login con cuentas @santiagoapostol.net

### Características Técnicas

- **LLM**: `unsloth/Qwen3.6-27B-NVFP4` en vLLM — 64k de contexto, speculative decoding (MTP), tool calling habilitado (necesario para opencode)
- **Retrieval inteligente**: búsqueda híbrida (denso + BM25 disperso, fusión RRF) para **todas** las consultas. El atajo que desviaba las consultas de menos de 4 tokens a BM25 solo se desactivó el 2026-08-02 (`BM25_FALLBACK_TOKEN_THRESHOLD=0`): afectaba al 41 % del tráfico real y ningún banco lo medía
- **Re-ranking**: Jina Reranker multilingüe mejora relevancia
- **Límites por archivo**: Máximo N fragmentos por documento (evita monopolios)
- **Límite de contexto dinámico**: Control por tokens (6000 default)
- **Telemetría**: Logs de consultas en JSONL (ver estado y pendientes en IMPROVEMENTS.md §1.6)
- **Estado persistente**: Tracking de archivos procesados (evita reindexación)

### Acceso desde opencode (uso como LLM de programación)

vLLM es accesible directamente por Tailscale para usarlo desde [opencode](https://opencode.ai) sin pasar por el RAG:

- URL: `http://iasantiago.tailbc4440.ts.net:8000/v1` (ver `opencode.json` de ejemplo en la raíz)
- Modelo: `unsloth/Qwen3.6-27B-NVFP4`, contexto 65536, salida 4096
- Requiere que vLLM conserve los flags `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (opencode usa function calling en cada petición; sin ellos vLLM rechaza la petición o no arranca)

## Requisitos del Sistema

- **Sistema Operativo**: Ubuntu Server 24.04
- **GPU**: NVIDIA con soporte CUDA (ej: RTX 5090)
- **Software**: Docker + Docker Compose, Nginx, Python 3
- **Conectividad**: Acceso a Internet para descarga de modelos
- **Almacenamiento**: ~50GB para modelos + datos

## Instalación Rápida

### 1. Preparar el entorno

```bash
# Crear directorio del proyecto
sudo mkdir -p /opt/iasantiago-rag
cd /opt/iasantiago-rag

# Descargar código
wget https://github.com/jredrejo/iasantiago/archive/refs/heads/main.zip
unzip main.zip
mv iasantiago-main/* .
rm -rf iasantiago-main main.zip

# Configurar variables de entorno
cp .env.example .env
nano .env  # Editar configuración
```

### 2. Configurar `.env`

Variables esenciales:

```bash
# Temas (uno por carpeta en topics/)
TOPIC_LABELS=AFD,Chemistry,Dibujo,Electricidad,FOL,Latin,Mecanica,Programming,Sostenibilidad
TOPIC_BASE_DIR=/opt/iasantiago-rag/topics

# Google OAuth
OAUTH2_CLIENT_ID=tu-client-id.apps.googleusercontent.com
OAUTH2_CLIENT_SECRET=tu-secret
OAUTH2_REDIRECT_URL=https://iasantiago.santiagoapostol.net/oauth2/callback
OAUTH2_EMAIL_DOMAINS=santiagoapostol.net
OAUTH2_COOKIE_SECRET=$(openssl rand -base64 32)

# Modelos (configurables por tema)
EMBED_MODEL_DEFAULT=intfloat/multilingual-e5-large-instruct
RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
VLLM_MODEL=unsloth/Qwen3.6-27B-NVFP4

# Límites
CTX_TOKENS_SOFT_LIMIT=6000
MAX_CHUNKS_PER_FILE=3
FINAL_TOPK=12
```

### 3. Configurar Google OAuth

1. Ir a [Google Cloud Console](https://console.cloud.google.com)
2. Crear proyecto nuevo o usar existente
3. Activar **Google+ API**
4. Crear credenciales OAuth 2.0:
   - Tipo: Web application
   - URIs autorizados: `https://iasantiago.santiagoapostol.net`
   - URIs de redirección: `https://iasantiago.santiagoapostol.net/oauth2/callback`
5. Copiar Client ID y Secret al `.env`

### 4. Iniciar servicios

```bash
# Construir e iniciar contenedores
make up

# Ver estado
docker compose ps

# Ver logs
docker compose logs -f rag-api
```

### 5. Configurar Nginx (opcional)

Si deseas exponer el servicio públicamente:

```bash
# Copiar configuración
cd nginx/nginx.conf
sudo ./configuracion_nginx.sh
# Probar y recargar
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Configurar servicios systemd (arranque automático)

```bash
# Enlazar servicios
sudo ln -sf /opt/iasantiago-rag/systemd/iasantiago-rag.service /etc/systemd/system/
sudo ln -sf /opt/iasantiago-rag/systemd/iasantiago-rag-eval.service /etc/systemd/system/
sudo ln -sf /opt/iasantiago-rag/systemd/iasantiago-rag-eval.timer /etc/systemd/system/

# Activar
sudo systemctl daemon-reload
sudo systemctl enable --now iasantiago-rag.service
sudo systemctl enable --now iasantiago-rag-eval.timer

# Configurar logrotate
sudo ln -sf /opt/iasantiago-rag/systemd/logrotate-telemetry /etc/logrotate.d/iasantiago-rag
```

## Uso del Sistema

### Añadir un tema nuevo

Un tema son **siete piezas en cuatro sitios distintos**, y sólo las tres primeras
son obligatorias para que responda. Las otras cuatro son las que hacen que
responda *bien* y que se sepa si lo hace. El orden importa: el nombre elegido en
el paso 1 se repite en casi todos los demás.

> **Elige la etiqueta sin acentos ni espacios.** La etiqueta viaja a Qdrant como
> nombre de colección —`rag_<etiqueta en minúsculas><QDRANT_COLLECTION_SUFFIX>`, hoy
> `_v2`—, así que `Química` crearía `rag_química_v2`. El nombre bonito se pone
> después, en Open WebUI, y se conecta con la etiqueta mediante el `topic_map` del
> paso 5. Los temas existentes que llevan acento (`Química`, `Latín`, `Mecánica`,
> `Programación`) usan exactamente ese mecanismo.

**1. Carpeta con los PDFs.** El nombre de la carpeta tiene que ser **idéntico** a
la etiqueta: el ingestor recorre `TOPIC_LABELS` y busca `topics/<etiqueta>`.

```bash
sudo mkdir -p /opt/iasantiago-rag/topics/Musica
sudo cp *.pdf /opt/iasantiago-rag/topics/Musica/
```

Los PDFs se buscan **recursivamente** y sin distinguir mayúsculas en la extensión,
así que puedes organizarlos en subcarpetas.

**2. Declarar la etiqueta en `.env`.** Añádela a `TOPIC_LABELS`, separada por comas
y sin espacios alrededor:

```bash
TOPIC_LABELS=AFD,Chemistry,Dibujo,Electricidad,FOL,Latin,Mecanica,Musica,Programming,Sostenibilidad
```

Un tema que esté en la carpeta pero no en `TOPIC_LABELS` **no se indexa**, y uno que
esté en `TOPIC_LABELS` sin carpeta se salta con un aviso. `/retrieve` rechaza con
**400** cualquier tema que no esté en esta lista, así que una errata aquí se ve en
seguida en vez de parecer que el corpus no tiene la respuesta.

Los cambios de `.env` exigen **recrear** el contenedor, no reiniciarlo — un
`restart` no vuelve a leer `env_file`:

```bash
docker compose up -d rag-api      # `up -d`, NO `restart`
```

**3. Ingestar.** El ingestor y vLLM **no caben juntos en la GPU**, así que `make ingest`
para el stack web y `make web` lo devuelve:

```bash
make ingest    # detiene vllm/rag-api/openwebui y lanza el ingestor
make web       # ⚠️ SIEMPRE al terminar, o el sitio entero da 502
```

Crea la colección `rag_musica_v2` en Qdrant con el vector denso y el disperso (BM25)
en el mismo punto, en un solo `upsert`. **Comprueba por contenido, no por el resumen
de la tirada** —un proceso puede informar de éxito sin haber escrito nada:

```bash
# ¿Existe la colección y cuántos puntos tiene?
curl -s http://localhost:6333/collections | jq '.result.collections[].name'
curl -s http://localhost:6333/collections/rag_musica_v2 | jq '.result.points_count'

# ¿Cuántos ficheros dio por procesados el ingestor, y cuántos falló?
jq '.processed | length' data/ingestor-state/.processing_state.json
jq '.failed' data/ingestor-state/.processing_state.json
```

`scripts/ragctl topic-stats` da el mismo conteo con más contexto, pero **necesita
`qdrant_client`**, que no está en el Python del host: hay que lanzarlo desde un
entorno con los requisitos de `ingestor` o `rag-api` instalados.

A partir de aquí el tema ya responde por `POST /retrieve`. Lo que falta es la UI y
la calidad.

**4. Prompt de sistema del tema.** Los prompts viven versionados en
`rag-api/templates/system_prompts/per_topic/` (respuesta) y `per_topic_generador/`
(generación de exámenes); Open WebUI guarda su propia copia, así que el repo es la
fuente de verdad y la UI el destino. El de examen **no se escribe a mano**: se genera
añadiendo una cabecera de materia en `scripts/gen_generador.py` y ejecutándolo, que
copia el reglamento de `generative.txt` intacto.

```bash
$EDITOR scripts/gen_generador.py     # añadir la entrada "Musica" a HEADERS
python3 scripts/gen_generador.py
```

**5. Modelos de workspace en Open WebUI.** Dos por tema —el normal y su
`- Generador`— con **modelo base `santiago`** (vLLM directo, no rag-api: desde el
rip-out del §7.1 rag-api no sirve chat) y el prompt del paso 4 pegado en el campo
*System Prompt*.

- Si el modelo se llama **exactamente igual que la etiqueta** (`Musica`), no hay
  nada más que hacer: el Filter resuelve el tema del nombre.
- Si le pones nombre bonito (`Música`, `Historia de la Música`), **añádelo al
  `topic_map`** del Filter (Admin → Functions → *IASantiago RAG Retrieval* →
  Valves), que traduce nombre visible → etiqueta:

```json
{"Química": "Chemistry", "Latín": "Latin", "Música": "Musica"}
```

El sufijo `- Generador` se quita solo y **vuelve a consultar el mapa**, así que basta
con una entrada por tema, no una por variante.

**6. Golden set, para poder medir el tema.** Sin él, el tema no aparece en ninguna
línea base y cualquier cambio futuro de recuperación lo mueve a ciegas. Se escribe
como `eval/golden_musica.json`, con la página como ground truth (invariante frente
a cambios de chunking):

```json
[{"query": "cadencia perfecta", "topic": "Musica",
  "relevant_pages": ["armonia.pdf#42"]}]
```

Añádelo a la lista `GOLDEN` de `scripts/eval_lexical_backends.py` y mídelo. Ojo con
dos trampas ya pagadas: un `relevant_pages` que apunte a un fichero que no está en
el tema mide **0.000** para siempre (el endpoint ya avisa de ello en `warnings`), y
las consultas escritas por quien conoce el corpus salen mucho más largas que las de
los alumnos —mediana 4 tokens—, así que conviene mirar también
`eval/golden_short.json`.

```bash
export OPENAI_API_KEY=$(grep -m1 '^OPENAI_API_KEY=' .env | cut -d= -f2)
python3 scripts/eval_lexical_backends.py
```

**7. Comprobación de punta a punta.** Antes de darlo por hecho, contra el servicio
vivo y no contra el log:

```bash
KEY=$(grep -m1 '^OPENAI_API_KEY=' .env | cut -d= -f2)
curl -s http://127.0.0.1:8001/healthz | jq .topics        # la etiqueta aparece
curl -s -X POST http://127.0.0.1:8001/retrieve \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"query":"cadencia perfecta","topic":"Musica"}' | jq '.meta, (.citations|length)'
```

`num_chunks: 0` con `meta.error: "missing_collection"` significa que falta ingestar;
un **400** significa que la etiqueta no está en `TOPIC_LABELS`. Son fallos distintos
a propósito: el primero es de operación y el segundo de quien llama.

Por último, en el navegador: elige el modelo nuevo y comprueba que la respuesta trae
**citas clicables** a `/docs/Musica/...`.

#### Quitar un tema

```bash
docker compose --profile ingest run --rm --no-deps ingestor delete-topic Musica
```

Sin el prefijo `python main.py`: el `ENTRYPOINT` del contenedor es el supervisor y
reenvía los argumentos. **`delete-topic` no borra el PDF del disco**, así que hay que
sacar también la carpeta de `topics/` o el siguiente escaneo lo reingesta. Y quitar
la etiqueta de `TOPIC_LABELS`, los modelos de workspace y su entrada del `topic_map`.

### Añadir Documentos

1. Copiar PDFs a la carpeta del tema:

```bash
sudo cp documento.pdf /opt/iasantiago-rag/topics/Chemistry/
```

2. Ejecutar ingestor:


El ingestor:
- Solo procesa archivos nuevos o modificados (tracking con hash MD5)
- Guarda estado en `/state/.processing_state.json` (volumen `data/ingestor-state`)
- Cachea análisis de imágenes/tablas en SQLite

### Hacer Consultas

1. Sólo accesible desde la LAN del centro: acceder a `https://ia.santiagoapostol.net`
2. Login con Google Workspace
3. Seleccionar tema en el desplegable de modelos (ej: `Química`)
4. Escribir pregunta
5. Las respuestas incluyen citas clicables: `[documento.pdf, p.5](/docs/Chemistry/documento.pdf#page=5)`



## Operación y Mantenimiento

### Comandos Útiles

```bash
# Ver estado
make status                    # Docker compose ps
docker compose logs -f rag-api # Logs en tiempo real

# Gestión de contenedores
make up                        # Iniciar todo
make down                      # Detener todo
make rag-restart               # Reiniciar solo rag-api

# Reindexación
make reset                     # Borra todo y reindexa
make seed                      # Crea ejemplos

# Modos de operación (vLLM e ingestor son EXCLUYENTES: no caben juntos en la GPU)
make ingest                    # Modo ingestión: detiene vllm/rag-api/openwebui y lanza ingestor
make web                       # Modo web: detiene ingestor y lanza oauth2-proxy
# ⚠️ Tras `make ingest`, volver SIEMPRE con `make web`: si oauth2-proxy queda
# parado, el sitio entero devuelve 502 (nginx enruta todo a través de él).

# Ver estado del ingestor
docker compose logs ingestor
cat /opt/iasantiago-rag/data/ingestor-state/.processing_state.json | jq
```

### Backups

```bash
# Backup completo
sudo rsync -av /opt/iasantiago-rag/data/ /backups/iasantiago-$(date +%F)/
sudo rsync -av /opt/iasantiago-rag/topics/ /backups/topics-$(date +%F)/

# Solo Qdrant
sudo rsync -av /opt/iasantiago-rag/data/storage/ /backups/qdrant-$(date +%F)/

# Solo el estado del ingestor (lo único que no se reconstruye sin horas de GPU)
make backup-state
```

### Monitoreo

**Salud de servicios:**

```bash
curl http://localhost:8001/healthz  # RAG API
curl http://localhost:8000/v1/models # vLLM
curl http://localhost:6333/        # Qdrant
```

**Telemetría:**

```bash
# Ver últimas consultas
tail -f /opt/iasantiago-rag/rag-api/retrieval.jsonl | jq

# Estadísticas
cat retrieval.jsonl | jq '.topic' | sort | uniq -c
```

**Estado de indexación:**

```bash
# Ver archivos procesados
cat /opt/iasantiago-rag/data/ingestor-state/.processing_state.json | jq '.processed | length'

# Ver archivos fallidos
cat /opt/iasantiago-rag/data/ingestor-state/.processing_state.json | jq '.failed'
```

## Configuración Avanzada

### Cambiar Modelo de Embeddings por Tema

En `.env`:

```bash
EMBED_MODEL_DEFAULT=intfloat/multilingual-e5-large-instruct
EMBED_MODEL_PROGRAMMING=thenlper/gte-large
EMBED_MODEL_ELECTRONICS=intfloat/multilingual-e5-large-instruct
EMBED_MODEL_CHEMISTRY=intfloat/multilingual-e5-large-instruct
```

**Importante**: Tras cambiar embeddings, reiniciar y reindexar:

```bash
make reset
```

### Cambiar Re-ranker

```bash
RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
# o
RERANK_MODEL=BAAI/bge-reranker-large
```

Reiniciar RAG API:

```bash
docker compose restart rag-api
```

### Ajustar Límites de Recuperación

```bash
# Contexto máximo (tokens)
CTX_TOKENS_SOFT_LIMIT=6000

# Máximo de fragmentos por archivo
MAX_CHUNKS_PER_FILE=3

# Recuperación híbrida
HYBRID_DENSE_K=40      # Top-K búsqueda densa
HYBRID_BM25_K=40       # Top-K búsqueda BM25
FINAL_TOPK=12          # Fragmentos finales tras fusión

# Umbral para fallback BM25
BM25_FALLBACK_TOKEN_THRESHOLD=4
```

### Cambiar Modelo LLM

Modelo actual (RTX 5090, 32 GB):

```bash
VLLM_MODEL=unsloth/Qwen3.6-27B-NVFP4    # 21.1 GiB de pesos en VRAM
VLLM_MAX_MODEL_LEN=65536                # pool KV medido: ~132k tokens
VLLM_GPU_MEMORY_UTILIZATION=0.95
VLLM_MAX_NUM_SEQS=16                    # ~11-13 peticiones de clase simultáneas
```

Hay perfiles `.env.*` de modelos probados anteriormente en la raíz del repo.
Al cambiar de modelo, tener en cuenta:

- Poner `HF_HUB_OFFLINE=0` en el servicio vllm de `docker-compose.yml` para
  permitir la descarga (volver a `1` después: arranque más rápido y sin red).
- Revisar `--tool-call-parser` en el command de vLLM (para Qwen3.x: `qwen3_coder`).
  **No quitar** `--enable-auto-tool-choice --tool-call-parser`: los necesita opencode.
- Actualizar el id del modelo en `opencode.json` (vLLM exige el id exacto).
- Las líneas del log de arranque `GPU KV cache size: N tokens` y
  `Maximum concurrency` dicen la concurrencia real que soporta la nueva config.

Reiniciar:

```bash
docker compose up -d vllm
```

### Multi-GPU

Editar `docker-compose.yml`:

```yaml
services:
  vllm:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ["0"]  # GPU 0
  
  rag-api:
    environment:
      - CUDA_VISIBLE_DEVICES=1  # GPU 1
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ["1"]
```

## Evaluación Offline

### Ejecutar Evaluación Manual

```bash
cat > /tmp/eval_cases.json <<'EOF'
[
  {
    "query": "definición de ley de Ohm",
    "topic": "Electronics",
    "relevant_files": ["/opt/iasantiago-rag/topics/Electronics/sample1.pdf"]
  },
  {
    "query": "ácidos y bases",
    "topic": "Chemistry",
    "relevant_files": ["/opt/iasantiago-rag/topics/Chemistry/sample1.pdf"]
  }
]
EOF

curl -X POST http://localhost:8001/v1/eval/offline \
  -H 'Content-Type: application/json' \
  -d @/tmp/eval_cases.json | jq
```

Métricas:
- **Recall@k**: Proporción de documentos relevantes recuperados
- **MRR (Mean Reciprocal Rank)**: Posición del primer documento relevante

### Evaluación Automática (Cron)

El timer systemd ejecuta evaluación nocturna:

```bash
# Ver próxima ejecución
sudo systemctl list-timers | grep iasantiago

# Ejecutar ahora
sudo systemctl start iasantiago-rag-eval.service

# Ver resultados
cat /opt/iasantiago-rag/eval_summary.csv
```

## Estructura del Proyecto

```
/opt/iasantiago-rag/
├── docker-compose.yml          # Orquestación de servicios
├── Makefile                    # Comandos útiles
├── .env                        # Configuración
│
├── topics/                     # PDFs por tema (entrada)
│   ├── Chemistry/
│   ├── Electronics/
│   └── Programming/
│
├── data/                       # Datos persistentes
│   ├── storage/               # Base de datos Qdrant
│   └── ingestor-state/        # .processing_state.json (estado de ingesta)
│       └── .processing_state.json
│
├── rag-api/                    # FastAPI
│   ├── app.py                 # Endpoints OpenAI-compatible
│   ├── config/                # settings.py: configuración desde .env
│   ├── chat/                  # intent, context_builder, token_calculator
│   ├── core/                  # vllm_client, cache de modelos, retry
│   ├── retrieval_lib/         # fusión RRF, citas, helpers de búsqueda
│   ├── retrieval.py           # Orquestación del retrieval híbrido
│   ├── rerank.py              # Re-ranking (CrossEncoder)
│   ├── templates/             # System prompts (default/generative)
│   └── requirements.txt
│
├── ingestor/                   # Indexador de PDFs
│   ├── main.py                # Punto de entrada (scan / delete / delete-topic)
│   ├── extraction/            # Pipeline: docling, unstructured, OCR, texto
│   ├── chunking/              # Estrategias de chunking
│   ├── indexing/              # Embeddings, Qdrant (denso + disperso BM25)
│   ├── pages/                 # Validación de números de página
│   ├── state/                 # Estado de procesamiento (.processing_state.json)
│   ├── core/                  # Config, heartbeat/watchdog, GPU
│   └── requirements.txt
│
├── openwebui/                  # Frontend
│   ├── data/                  # BD SQLite de Open WebUI
│   └── custom/                # Logos personalizados
│
├── oauth2-proxy/               # Autenticación
│   ├── oauth2-proxy.cfg
│   └── templates/
│
├── nginx/                      # Reverse proxy (opcional)
│   ├── nginx.conf
│   └── certs/
│
└── systemd/                    # Servicios del sistema
    ├── iasantiago-rag.service
    ├── iasantiago-rag-eval.service
    ├── iasantiago-rag-eval.timer
    └── logrotate-telemetry
```

## Resolución de Problemas

### 502 Bad Gateway en el navegador

nginx enruta todo el sitio a través de oauth2-proxy; si ese contenedor está
parado (p. ej. tras un `make ingest` sin su `make web` posterior), todo devuelve 502.

```bash
docker ps -a | grep oauth2          # ¿Exited?
docker compose up -d oauth2-proxy   # Arrancarlo (o `make web`)
curl http://127.0.0.1:4180/ping     # Debe responder 200
```

### Open WebUI no muestra temas

Los temas **no salen de rag-api**: son modelos de workspace de Open WebUI. rag-api
ya no expone `/v1/models` ni `/v1/chat/completions` (rip-out del §7.1), así que si
falta un tema en el desplegable, el problema está en Open WebUI, no aquí.

```bash
# ¿Está el modelo de workspace creado y activo? (Admin -> Workspace -> Models)
# Un is_active=0 lo esconde sin que nada dé error.

# rag-api sí publica la lista de etiquetas que acepta:
curl -s http://localhost:8001/healthz | jq .topics

# Verificar variables (los cambios de .env necesitan `up -d`, no `restart`)
docker compose exec rag-api env | grep TOPIC
docker compose up -d rag-api openwebui
```

### El tema sale pero responde "no encuentro fragmentos relevantes"

Casi siempre es configuración, no calidad de recuperación. Los dos casos se
distinguen desde fuera:

```bash
KEY=$(grep -m1 '^OPENAI_API_KEY=' .env | cut -d= -f2)
curl -s -X POST http://127.0.0.1:8001/retrieve \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"query":"prueba","topic":"Musica"}' | jq .meta
```

- **HTTP 400** → la etiqueta no está en `TOPIC_LABELS` (errata, o falta el `topic_map`
  del Filter si el modelo tiene nombre con acento). El cuerpo trae las válidas.
- **`meta.error: "missing_collection"`** → la etiqueta es correcta pero el tema no se
  ha ingestado.
- **200 con `num_chunks > 0`** → recuperación real; entonces sí es calidad.

### No recupera resultados

```bash
# ¿Archivos indexados?
ls -lh /opt/iasantiago-rag/topics/Chemistry/

# ¿Ingestor ejecutado?
docker compose logs ingestor | tail -50

# ¿Estado del ingestor?
cat /opt/iasantiago-rag/data/ingestor-state/.processing_state.json | jq

# ¿Colecciones en Qdrant?
curl http://localhost:6333/collections | jq

# Forzar reindexación
make reset
```

### Error de memoria GPU

```bash
# Ver uso actual
nvidia-smi

# Reducir memoria: bajar en este orden
nano .env
# VLLM_GPU_MEMORY_UTILIZATION=0.92   # primero (cada 0.01 ≈ 0.3 GiB)
# VLLM_MAX_NUM_BATCHED_TOKENS=2048   # segundo (reduce pico de activaciones)
# VLLM_MAX_MODEL_LEN=32768           # último recurso (limita contexto de opencode)

# Reiniciar
docker compose up -d --build vllm
```

### Error de autenticación 401/403

```bash
# Verificar OAuth
docker compose logs oauth2-proxy | tail -50

# Verificar redirect URI en Google Cloud Console
# Debe ser: https://iasantiago.santiagoapostol.net/oauth2/callback

# Verificar reloj del servidor (NTP)
timedatectl status

# Verificar conectividad a Google
curl -I https://accounts.google.com
```

### Ingestor falla con archivos grandes

```bash
# Ver errores
docker compose logs ingestor | grep ERROR

# Ver archivos fallidos
cat /opt/iasantiago-rag/data/ingestor-state/.processing_state.json | jq '.failed'

# Aumentar memoria del contenedor
nano docker-compose.yml
# shm_size: 32gb  # En servicio ingestor

# Reiniciar
docker compose up -d --build ingestor
docker compose run --rm ingestor
```

## Seguridad

### Acceso a la Red

- **Intranet**: Servidor expuesto solo en red interna (172.23.120.11)
- **Puerto**: HTTPS 443
- **Autenticación**: Google Workspace OAuth2
- **Restricción**: Solo emails del dominio configurado

### Salida a Internet Necesaria

- Validación de tokens (accounts.google.com)
- Descarga de modelos en primer arranque (huggingface.co)

### Sin Internet

Si no hay salida a Internet:

1. Pre-cachear modelos en un servidor con acceso
2. Copiar `/root/.cache/huggingface` al servidor destino
3. Desactivar OAuth o usar IdP local (Keycloak, etc.)

## Actualizaciones

```bash
cd /opt/iasantiago-rag

# Backup
sudo rsync -av data/ /backups/data-$(date +%F)/

# Pull código nuevo
git pull  # o descargar ZIP

# Actualizar imágenes
docker compose pull
docker compose build

# Reiniciar
docker compose up -d

# Si cambian embeddings
make reset
```

## Licencia

Proyecto interno del IES Santiago Apóstol.
GNU GENERAL PUBLIC LICENSE V3

## Soporte

Para problemas o preguntas:
- Issues: https://github.com/jredrejo/iasantiago/issues
