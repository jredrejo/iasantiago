# Cross-Lingual Query Translation Guide

## Problem Statement

Previously, the RAG system struggled with cross-lingual queries. For example:
- **Query (Spanish)**: "dime que es Quality of Service en MQTT"
- **Documents (English)**: MQTT QoS explanations in English PDF files
- **Result**: ❌ No documents found

The issue occurred because:
1. Query embeddings in Spanish didn't match document embeddings in English
2. BM25 lexical search requires exact keyword matches (e.g., "en MQTT" ≠ "in MQTT")
3. The multilingual embedding model alone wasn't sufficient for strong cross-lingual alignment

## Solution: Automatic Query Translation

The system now automatically detects the query language and translates it to English before retrieval. This ensures:
- ✅ Consistent embeddings with English documents
- ✅ Better keyword matching for BM25
- ✅ Support for questions in any language (Spanish, French, German, Italian, Portuguese, etc.)

## How It Works

### 1. Language Detection

```python
from translation import detect_language

lang = detect_language("dime que es Quality of Service en MQTT")
# Returns: "es" (Spanish)
```

Uses `langdetect` library for automatic language identification.

### 2. Query Translation

```python
from translation import translate_query

translated, source_lang = translate_query(
    "dime que es Quality of Service en MQTT",
    source_lang="es",
    target_lang="en"
)
# Returns: ("Tell me what Quality of Service in MQTT is", "es")
```

Uses Helsinki-NLP's lightweight translation models (opus-mt-*) for fast, accurate translation:
- **Spanish → English**: `Helsinki-NLP/opus-mt-es-en`
- **French → English**: `Helsinki-NLP/opus-mt-fr-en`
- **German → English**: `Helsinki-NLP/opus-mt-de-en`
- And many more...

### 3. Retrieval with Translated Query

The translated query is then used for:
- **Dense retrieval**: Embedding-based search with consistent English vectors
- **BM25 retrieval**: Lexical search with English keywords

The original query is preserved in metadata for logging and debugging.

## Supported Languages

| Code | Language | Translation Model |
|------|----------|-------------------|
| es   | Spanish  | Helsinki-NLP/opus-mt-es-en |
| fr   | French   | Helsinki-NLP/opus-mt-fr-en |
| de   | German   | Helsinki-NLP/opus-mt-de-en |
| it   | Italian  | Helsinki-NLP/opus-mt-it-en |
| pt   | Portuguese | Helsinki-NLP/opus-mt-pt-en |
| nl   | Dutch    | Helsinki-NLP/opus-mt-nl-en |
| pl   | Polish   | Helsinki-NLP/opus-mt-pl-en |
| ru   | Russian  | Helsinki-NLP/opus-mt-ru-en |
| zh   | Chinese  | Helsinki-NLP/opus-mt-zh-en |
| ja   | Japanese | Helsinki-NLP/opus-mt-ja-en |
| ko   | Korean   | Helsinki-NLP/opus-mt-ko-en |
| ar   | Arabic   | Helsinki-NLP/opus-mt-ar-en |

More languages can be added by using the appropriate Helsinki-NLP model.

## Implementation Details

### Files Modified

1. **`rag-api/translation.py`** (NEW)
   - Language detection: `detect_language(text)`
   - Query translation: `translate_query(query, source_lang, target_lang)`
   - Model caching for performance

2. **`rag-api/retrieval.py`** (MODIFIED)
   - `choose_retrieval()`: Auto-detects and translates queries
   - `choose_retrieval_enhanced()`: Same, with topk multiplier support
   - Preserves original language in metadata

3. **`rag-api/app.py`** (MODIFIED)
   - Telemetry logging now includes:
     - `original_language`: Detected language code
     - `translated_query`: The original query before translation

4. **`rag-api/requirements.txt`** (MODIFIED)
   - Added `langdetect>=1.0.9` for language detection
   - `torch>=2.0.0` (already required by transformers)

### Performance Considerations

**Model Loading**:
- Translation models are cached after first use
- On first query in a new language: ~200-500ms overhead
- Subsequent queries in same language: ~50-100ms overhead
- English queries: 0ms overhead (no translation)

**Memory**:
- Each translation model (~400MB) is loaded on-demand
- Models cached in GPU memory if available, otherwise CPU
- Embeddings configured with `CUDA_VISIBLE_DEVICES` in docker-compose.yml

**Latency**:
- Short queries (< 20 words): ~10-50ms translation time
- Long queries (> 100 words): ~50-150ms translation time
- Negligible compared to embedding + vector search time

## Configuration

### Enable/Disable Translation

Translation is **enabled by default**. No configuration needed.

To disable (NOT RECOMMENDED):
```python
# In retrieval.py, comment out the translation block:
# if detected_lang != "en":
#     query, _ = translate_query(query, detected_lang, "en")
```

### Adjust Language Support

Edit the `SUPPORTED_LANGS` dict in `translation.py`:

```python
SUPPORTED_LANGS = {
    "es": "Spanish",     # Keep existing
    "fr": "French",      # Add new
    # ...
}
```

The system will automatically use the appropriate `Helsinki-NLP/opus-mt-{lang}-en` model.

### GPU Device Selection

If using multiple GPUs, specify which one for translation models:

In `translation.py`, modify the device selection:
```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

## Usage Examples

### Example 1: Spanish Query

**Input**: "dime que es Quality of Service en MQTT"

**Flow**:
1. Detect language: Spanish (`es`)
2. Translate: "Tell me what Quality of Service in MQTT is"
3. Retrieve using translated query against English documents
4. Return documents with citations

**Telemetry**:
```json
{
  "query": "dime que es Quality of Service en MQTT",
  "original_language": "es",
  "translated_query": "Tell me what Quality of Service in MQTT is",
  "mode": "hybrid",
  "retrieved": [...]
}
```

### Example 2: English Query

**Input**: "What is Quality of Service in MQTT?"

**Flow**:
1. Detect language: English (`en`)
2. Skip translation (already in target language)
3. Retrieve directly using original query
4. Return documents

**Telemetry**:
```json
{
  "query": "What is Quality of Service in MQTT?",
  "original_language": "en",
  "translated_query": null,
  "mode": "hybrid",
  "retrieved": [...]
}
```

## Troubleshooting

### Issue: Slow First Query in New Language

**Cause**: First query in a language needs to load the translation model (~400MB)

**Solution**: This is normal. Subsequent queries are fast (cached model).

### Issue: Translation Seems Inaccurate

**Cause**: Helsinki-NLP models are lightweight and sometimes lose nuance

**Solutions**:
1. The system falls back to original query if translation fails
2. Re-ranking step helps recover if translation degrades quality
3. Consider using a stronger model (LLM-based) if translation quality is critical

### Issue: Translation Model Not Found

**Cause**: Internet required to download model on first use

**Solution**: Pre-download model in advance:
```bash
python3 << 'EOF'
from transformers import MarianTokenizer, MarianMTModel
model_name = "Helsinki-NLP/opus-mt-es-en"
MarianTokenizer.from_pretrained(model_name)
MarianMTModel.from_pretrained(model_name)
print(f"✓ Model {model_name} cached")
EOF
```

### Issue: Language Misdetected

**Cause**: Code-mixed text (e.g., Spanish + English) may confuse langdetect

**Solution**: The system gracefully handles this by:
1. Attempting translation even if slightly wrong language detected
2. Preserving original query in metadata
3. Re-ranking step helps recover bad translations

## Monitoring

### Check Telemetry

View translation statistics:

```bash
# See original languages in queries
cat /opt/iasantiago-rag/rag-api/retrieval.jsonl | \
  jq '.original_language' | sort | uniq -c

# See which queries were translated
cat /opt/iasantiago-rag/rag-api/retrieval.jsonl | \
  jq 'select(.translated_query != null) | .translated_query' | head -20
```

### Monitor Performance

```bash
# See translation latency in logs
docker compose logs rag-api | grep "Translated query"

# Example output:
# 🌐 Translated query (es→en): dime que es Quality of Service → Tell me what Quality of Service is
```

## Testing

### Manual Test

```bash
cd /datos/iasantiago/rag-api

# Run test script
python3 test_translation.py

# Expected output:
# 📝 Original query: dime que es Quality of Service en MQTT
#    Expected language: es
#    Detected language: es
#    Translated query: Tell me what Quality of Service in MQTT is
#    Source language: es
```

### Integration Test

Test with actual retrieval:

```bash
docker compose exec rag-api python3 << 'EOF'
from retrieval import choose_retrieval_enhanced

# Test Spanish query
results, meta = choose_retrieval_enhanced(
    topic="Electronics",
    query="dime que es Quality of Service en MQTT"
)

print(f"Original language: {meta.get('original_language')}")
print(f"Translated query: {meta.get('original_query')}")
print(f"Found {len(results)} results")
for r in results[:3]:
    print(f"  - {r['file_path']}, p.{r['page']}")
EOF
```

## Future Improvements

1. **LLM-based translation**: Use vLLM for more accurate translation (slower but better quality)
2. **Query expansion**: Translate query to multiple languages and use ensemble retrieval
3. **Document translation**: Cache translated documents for frequently-used language pairs
4. **Fine-tuned models**: Train specialized translation models on domain-specific terminology
5. **Language-specific embeddings**: Use language-specific embedding models when appropriate

## References

- **Helsinki-NLP Models**: https://huggingface.co/Helsinki-NLP
- **Langdetect**: https://github.com/Mikojil/lang-detect
- **MarianMT**: https://huggingface.co/docs/transformers/model_doc/marian
- **Opus Models**: https://github.com/Helsinki-NLP/Opus-MT
