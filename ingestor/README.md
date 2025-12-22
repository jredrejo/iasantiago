# 📋 RAG Ingestor - Documentación

## 🎯 Objetivo

**Sistema de indexación de PDFs que procesa automáticamente:**

- ✅ **Texto** (extracción con Docling + Unstructured.io fallback)
- ✅ **Tablas** (detección y extracción con pdfplumber)
- ✅ **Sistema de estado** para evitar reprocesamiento
- ✅ **GPU/CPU auto-fallback** para máxima robustez
- ✅ **Watchdog** para detectar procesos colgados

---

## 📦 Archivos del Sistema

### Código Core

```
ingestor/
├── chunk.py                 # Extracción de texto con validación de páginas
├── main.py                  # Pipeline indexación + ModelCache + ProcessingState
├── settings.py              # Configuración centralizada
├── docling_client.py        # Cliente Docling para extracción
├── docling_extractor.py     # Extracción con Docling GPU
├── setup_nltk.py           # Descarga datos NLTK
├── requirements.txt         # Dependencias Python
└── Dockerfile               # Build ingestor
```

### Scripts de Gestión

```
├── manage_gpu.sh           # Estado GPU y contenedores
└── manage_state.sh         # Gestión de estado de procesamiento
```

---

## 🗃️ Arquitectura del Sistema

```
PDF INPUT
    ↓
┌──────────────────────────────────────────────────────────┐
│   ProcessingState (main.py)                             │
│   Verifica si archivo ya fue procesado                  │
│   ├─ Hash MD5 del archivo                               │
│   ├─ Estado: success/failed                             │
│   └─ Skip si ya procesado con mismo hash                │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│   Docling Extractor (docling_client.py)                 │
│   Motor principal: Docling con GPU                      │
│   Fallback: Unstructured.io                             │
└──────────────────────────────────────────────────────────┘
    ↓
    ├─ TEXTO
    │  ├─ Docling extrae texto con preservación de páginas
    │  ├─ Fallback: pypdf + pdfplumber
    │  ├─ Split: 900 chars + 120 overlap
    │  └─ Chunk → type: "text"
    │
    └─ TABLAS
       ├─ pdfplumber detecta tablas
       ├─ Extracción estructurada
       └─ Chunk → type: "table"
    ↓
┌──────────────────────────────────────────────────────────┐
│  ModelCache (main.py)                                    │
│  Gestiona modelos de embedding con GPU/CPU fallback     │
└──────────────────────────────────────────────────────────┘
    ↓
    ├─ Intenta cargar modelo en GPU (float16)
    │  └─ Si falla → Automático fallback a CPU (float32)
    │
    ├─ Cache en memoria (evita recargas)
    │
    └─ encode_with_gpu() → batch processing
           ├─ Batch size: 32
           └─ Normalización L2
    ↓
┌──────────────────────────────────────────────────────────┐
│  Embedding: intfloat/multilingual-e5-large-instruct     │
│  Dimensión: 1024                                         │
│  Device: Auto-detecta GPU/CPU                            │
└──────────────────────────────────────────────────────────┘
    ↓
    ├─ Qdrant (Dense vector search)
    │  ├─ Collection por topic
    │  ├─ Batch upsert (100 vectores)
    │  └─ Metadata completo por chunk
    │
    └─ Whoosh (BM25 + metadata)
       ├─ Índice por topic
       ├─ Schema: file_path, page, chunk_id, text, type, source
       └─ Update por documento
```

---

## 💾 Estado de Procesamiento

```
┌───────────────────────────────────────────────────────────┐
│   /whoosh/.processing_state.json                          │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  {                                                         │
│    "version": 1,                                           │
│    "created_at": "2025-01-15T10:00:00",                   │
│    "last_scan": "2025-01-20T15:30:00",                    │
│    "processed": {                                          │
│      "/topics/Chemistry/libro.pdf": {                     │
│        "hash": "abc123def456...",                         │
│        "timestamp": "2025-01-20T15:25:00",                │
│        "topic": "Chemistry",                              │
│        "status": "success"                                │
│      }                                                     │
│    },                                                      │
│    "failed": {                                             │
│      "/topics/Physics/corrupted.pdf": {                   │
│        "error": "Failed to extract...",                   │
│        "timestamp": "2025-01-20T15:26:00"                 │
│      }                                                     │
│    }                                                       │
│  }                                                         │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuración

### Variables de Entorno

```bash
# Directorios
TOPIC_BASE_DIR=/topics
BM25_BASE_DIR=/whoosh

# Qdrant
QDRANT_URL=http://qdrant:6333

# Docling
ENABLE_DOCLING=true
DOCLING_GPU_MEMORY_FRACTION=0.30

# Embeddings
EMBED_MODEL_DEFAULT=intfloat/multilingual-e5-large-instruct
```

---

## 🚀 Comandos de Gestión

### manage_gpu.sh

```bash
./manage_gpu.sh status    # Ver estado GPU y contenedores
./manage_gpu.sh check     # Verificar contenedores
./manage_gpu.sh help      # Ayuda
```

### manage_state.sh

```bash
./manage_state.sh status        # Ver estado del indexador
./manage_state.sh stats         # Estadísticas de procesamiento
./manage_state.sh reset         # Resetear estado (re-indexar todo)
./manage_state.sh failed        # Ver archivos fallidos
./manage_state.sh retry-failed  # Reintentar fallidos
```

---

## 🔄 Flujo de Procesamiento

1. **Inicio**: Ingestor escanea `/topics/{topic}/` buscando PDFs
2. **Verificación**: ProcessingState verifica hash MD5 de cada archivo
3. **Extracción**:
   - Docling (primario, GPU)
   - Unstructured.io (fallback)
   - pypdf + pdfplumber (último recurso)
4. **Chunking**: Texto dividido en fragmentos de 900 chars con 120 overlap
5. **Embedding**: Modelo multilingual-e5-large genera vectores de 1024 dims
6. **Indexación**:
   - Qdrant: vectores densos
   - Whoosh: índice BM25 léxico
7. **Estado**: Se actualiza `.processing_state.json`

---

## 🛡️ Tolerancia a Fallos

- **Signal handlers**: Capturan SIGSEGV, SIGBUS, SIGABRT
- **Watchdog thread**: Detecta procesos colgados (heartbeat >300s)
- **Docker restart**: `restart: on-failure` reinicia el contenedor
- **Fallback chain**: Docling → Unstructured → pypdf+pdfplumber

---

## 📊 Métricas de Rendimiento

| Operación | Tiempo típico |
|-----------|---------------|
| Extracción PDF (Docling) | 2-5s/página |
| Extracción PDF (fallback) | 1-3s/página |
| Embedding (GPU) | ~100ms/chunk |
| Indexación Qdrant | ~50ms/batch |
| Indexación Whoosh | ~30ms/doc |
