# IASantiago RAG

Sistema de Recuperación Aumentada con Generación (RAG) para el Colegio Santiago Apóstol que permite consultar documentos PDF organizados por temas mediante una interfaz de chat.

## Arquitectura

```
┌─────────────────┐
│   Open WebUI    │ ← Interfaz de usuario (puerto 8080)
└────────┬────────┘
         │
┌────────▼────────┐
│  oauth2-proxy   │ ← Autenticación Google Workspace (puerto 4180)
└────────┬────────┘
         │
┌────────▼────────┐
│    RAG API      │ ← FastAPI - OpenAI compatible (puerto 8001)
│                 │   • Hybrid Retrieval (Qdrant + BM25/Whoosh)
│                 │   • Re-ranking (Jina/BGE)
│                 │   • Streaming + citas con enlaces
│                 │   • Límites dinámicos de contexto
└─┬───┬───┬───┬───┘
  │   │   │   │
  │   │   │   └─────► vLLM (puerto 8000)
  │   │   │           Generación de texto (LLM en GPU)
  │   │   │
  │   │   └─────────► vLLM-LLaVA (puerto 8002)
  │   │               Análisis de imágenes/tablas
  │   │
  │   └─────────────► Whoosh (BM25)
  │                   Búsqueda léxica (índices locales)
  │
  └─────────────────► Qdrant (puertos 6333/6334)
                      Base de datos vectorial

┌─────────────────┐
│    Ingestor     │ ← Indexación de PDFs (ejecución única)
│                 │   • Extracción con Unstructured.io
│                 │   • Análisis LLaVA (imágenes/tablas)
│                 │   • Cache SQLite (70x speedup)
│                 │   • Embeddings a Qdrant + BM25
└─────────────────┘
```

## Funcionalidades Principales

### Para Usuarios

- **Selector de temas**: Elige entre Chemistry, Electronics, Programming
- **Búsqueda híbrida**: Combina embeddings densos + BM25 léxico
- **Citas clicables**: Enlaces directos a PDFs con número de página
- **Streaming**: Respuestas en tiempo real
- **Autenticación Google**: Login con cuentas @santiagoapostol.net

### Características Técnicas

- **Retrieval inteligente**: Fallback automático a BM25 para consultas cortas (<4 tokens)
- **Re-ranking**: Jina Reranker multilingüe mejora relevancia
- **Límites por archivo**: Máximo N fragmentos por documento (evita monopolios)
- **Límite de contexto dinámico**: Control por tokens (6000 default)
- **Cache LLaVA**: SQLite para análisis de imágenes/tablas (70x speedup)
- **Telemetría**: Logs en `retrieval.jsonl` con rotación automática
- **Estado persistente**: Tracking de archivos procesados (evita reindexación)

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
# Temas
TOPIC_LABELS=Chemistry,Electronics,Programming
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
VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

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

### Añadir Documentos

1. Copiar PDFs a la carpeta del tema:

```bash
sudo cp documento.pdf /opt/iasantiago-rag/topics/Chemistry/
```

2. Ejecutar ingestor:

```bash
cd ingestor
./manage_gpu.sh ingest
```

El ingestor:
- Solo procesa archivos nuevos o modificados (tracking con hash MD5)
- Guarda estado en `/whoosh/.processing_state.json`
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
make restart                   # Reiniciar

# Reindexación
make reset                     # Borra todo y reindexa
make seed                      # Crea ejemplos

# Ver estado del ingestor
docker compose logs ingestor
cat /opt/iasantiago-rag/data/whoosh/.processing_state.json | jq
```

### Backups

```bash
# Backup completo
sudo rsync -av /opt/iasantiago-rag/data/ /backups/iasantiago-$(date +%F)/
sudo rsync -av /opt/iasantiago-rag/topics/ /backups/topics-$(date +%F)/

# Solo Qdrant
sudo rsync -av /opt/iasantiago-rag/data/storage/ /backups/qdrant-$(date +%F)/

# Solo Whoosh
sudo rsync -av /opt/iasantiago-rag/data/whoosh/ /backups/whoosh-$(date +%F)/
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
cat /opt/iasantiago-rag/data/whoosh/.processing_state.json | jq '.processed | length'

# Ver archivos fallidos
cat /opt/iasantiago-rag/data/whoosh/.processing_state.json | jq '.failed'
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

```bash
VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
# o
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# o
VLLM_MODEL=mistralai/Mistral-Nemo-Instruct-2407
```

Ajustar memoria GPU:

```bash
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_TENSOR_PARALLEL_SIZE=1
```

Reconstruir:

```bash
docker compose up -d --build vllm
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
│   └── whoosh/                # Índices BM25 + estado
│       └── .processing_state.json
│
├── rag-api/                    # FastAPI
│   ├── app.py                 # Endpoints OpenAI-compatible
│   ├── retrieval.py           # Lógica de búsqueda
│   ├── rerank.py              # Re-ranking
│   ├── settings.py            # Configuración
│   └── requirements.txt
│
├── ingestor/                   # Indexador de PDFs
│   ├── main.py                # Loop principal
│   ├── chunk.py               # Extracción con Unstructured + LLaVA
│   ├── settings.py
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

### Open WebUI no muestra temas

```bash
# Verificar endpoint
curl http://localhost:8001/v1/models | jq

# Verificar variables
docker compose exec rag-api env | grep TOPIC

# Reiniciar
docker compose restart rag-api openwebui
```

### No recupera resultados

```bash
# ¿Archivos indexados?
ls -lh /opt/iasantiago-rag/topics/Chemistry/

# ¿Ingestor ejecutado?
docker compose logs ingestor | tail -50

# ¿Estado del ingestor?
cat /opt/iasantiago-rag/data/whoosh/.processing_state.json | jq

# ¿Colecciones en Qdrant?
curl http://localhost:6333/collections | jq

# Forzar reindexación
make reset
```

### Error de memoria GPU

```bash
# Ver uso actual
nvidia-smi

# Reducir modelo o memoria
nano .env
# VLLM_MODEL=meta-llama/Llama-3.2-3B-Instruct  # Modelo más pequeño
# VLLM_GPU_MEMORY_UTILIZATION=0.7
# VLLM_MAX_MODEL_LEN=4096

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
cat /opt/iasantiago-rag/data/whoosh/.processing_state.json | jq '.failed'

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
