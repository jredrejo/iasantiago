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

### Estructura Modular

```
ingestor/
├── core/                        # Infraestructura base
│   ├── config.py                   - Configuración centralizada (topics, modelos, URLs)
│   ├── cache.py                    - Cache de hashes MD5 y extracción
│   ├── gpu.py                      - Gestión de estado GPU
│   └── heartbeat.py                - Heartbeat y watchdog para health checks
│
├── extraction/                  # Pipeline de extracción PDF
│   ├── base.py                     - Element dataclass, ExtractorProtocol
│   ├── pipeline.py                 - Orquestación con fallback chain
│   ├── docling_extractor.py        - Extracción con Docling GPU
│   ├── text_extractor.py           - pypdf + pdfplumber
│   ├── ocr_extractor.py            - EasyOCR + Tesseract
│   └── unstructured_extractor.py   - Estrategias Unstructured.io
│
├── pages/                       # Utilidades de número de página
│   ├── page_validator.py           - Validación unificada de páginas
│   ├── page_extractor.py           - Extracción multi-estrategia
│   └── page_boundary.py            - Detección de límites de página
│
├── chunking/                    # Fragmentación de documentos
│   ├── strategies.py               - Chunking semántico, simple, adaptativo
│   └── chunker.py                  - ContextAwareChunker
│
├── indexing/                    # Búsqueda vectorial y léxica
│   ├── embeddings.py               - EmbeddingService con fallback GPU/CPU
│   ├── qdrant.py                   - Operaciones Qdrant
│   └── whoosh_bm25.py              - Operaciones Whoosh BM25
│
├── state/                       # Gestión de estado
│   └── processing_state.py         - Tracking MD5, estado success/failed
│
├── main.py                      # CLI entry point (~285 líneas)
├── setup_nltk.py                # Descarga datos NLTK
├── download_easyocr_models.py   # Descarga modelos EasyOCR
├── requirements.txt             # Dependencias Python
└── Dockerfile                   # Build ingestor
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
│   state/processing_state.py                              │
│   ProcessingState - Verifica si archivo ya procesado     │
│   ├─ Hash MD5 del archivo                                │
│   ├─ Estado: success/failed                              │
│   └─ Skip si ya procesado con mismo hash                 │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│   extraction/pipeline.py                                 │
│   ExtractionPipeline - Fallback chain automático         │
│   ├─ DoclingExtractor (GPU, mejor calidad)               │
│   ├─ TextExtractor (pypdf + pdfplumber, rápido)          │
│   ├─ UnstructuredExtractor (hi_res, layout-aware)        │
│   └─ OCRExtractor (EasyOCR + Tesseract, scanned docs)    │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│   extraction/base.py                                     │
│   Element dataclass - Representación unificada           │
│   ├─ text: str                                           │
│   ├─ type: "text" | "table" | "image"                    │
│   ├─ page: int (validado)                                │
│   └─ source: str (extractor usado)                       │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│   pages/page_validator.py                                │
│   Validación unificada de números de página              │
│   ├─ Clamp a rango [1, total_pages]                      │
│   ├─ Conversión de tipos (str/float → int)               │
│   └─ Detección de gaps y secuencias inválidas            │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│   indexing/embeddings.py                                 │
│   EmbeddingService - GPU/CPU con fallback automático     │
│   ├─ Modelo en GPU (float16) ~650MB                      │
│   ├─ Fallback automático a CPU si GPU falla              │
│   ├─ Mega-batch processing para docs grandes             │
│   └─ Cache de modelos en memoria                         │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│   Embedding: intfloat/multilingual-e5-large-instruct     │
│   Dimensión: 1024                                        │
│   Device: Auto-detecta GPU/CPU                           │
└──────────────────────────────────────────────────────────┘
    ↓
    ├─ indexing/qdrant.py
    │  QdrantService (Dense vector search)
    │  ├─ Collection por topic
    │  ├─ Batch upsert (100 vectores)
    │  └─ Metadata completo por chunk
    │
    └─ indexing/whoosh_bm25.py
       WhooshService (BM25 + metadata)
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

- **Signal handlers** (main.py): Capturan SIGSEGV, SIGBUS, SIGABRT
- **Watchdog thread** (core/heartbeat.py): Detecta procesos colgados (heartbeat >450s)
- **Docker restart**: `restart: on-failure` reinicia el contenedor
- **Fallback chain** (extraction/pipeline.py): Docling → Text → Unstructured → OCR
- **GPU fallback** (indexing/embeddings.py): GPU (float16) → CPU (float32) automático

---

## 📊 Métricas de Rendimiento

| Operación | Tiempo típico |
|-----------|---------------|
| Extracción PDF (Docling) | 2-5s/página |
| Extracción PDF (fallback) | 1-3s/página |
| Embedding (GPU) | ~100ms/chunk |
| Indexación Qdrant | ~50ms/batch |
| Indexación Whoosh | ~30ms/doc |
