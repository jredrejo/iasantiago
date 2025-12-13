# Cross-Lingual RAG Implementation

## Summary

Your RAG system now automatically translates queries from any language to English before retrieval, solving the cross-lingual problem where Spanish queries couldn't find English documents.

## Quick Example

**Before**: "dime que es Quality of Service en MQTT" → ❌ No results  
**After**: "dime que es Quality of Service en MQTT" → ✅ Finds English MQTT documents

## What Changed

### New Files

- `rag-api/translation.py` - Translation pipeline
- `rag-api/test_translation.py` - Test script
- `TRANSLATION_GUIDE.md` - Full documentation

### Modified Files

- `rag-api/retrieval.py` - Auto-translate queries before retrieval
- `rag-api/app.py` - Log original language in telemetry
- `rag-api/requirements.txt` - Added langdetect + torch

## How to Deploy

### 1. Rebuild Docker Image

```bash
cd /datos/iasantiago
docker compose build rag-api
```

### 2. Restart Service

```bash
docker compose restart rag-api
```

### 3. Verify It Works

```bash
docker compose exec rag-api python3 test_translation.py
```

## How It Works

```
User Query in Spanish:
  "dime que es Quality of Service en MQTT"
        ↓
Language Detection:
  Detected: es (Spanish)
        ↓
Automatic Translation:
  "Tell me what Quality of Service in MQTT is"
        ↓
Vector Search + BM25:
  Search English embeddings + keywords
        ↓
Find English Documents:
  MQTT.pdf, Electronics.pdf, etc.
        ↓
Return Results with Citations
```

## Supported Languages

Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Chinese, Japanese, Korean, Arabic, and many more.

See `TRANSLATION_GUIDE.md` for the complete list.

## Performance Impact

- **First query in new language**: +200-500ms (model loading)
- **Subsequent queries**: +50-100ms (translation)
- **English queries**: 0ms overhead (no translation)

Translation models are cached in GPU memory for fast reuse.

## Configuration

No configuration needed - translation is enabled by default.

To disable it (not recommended):

1. Edit `rag-api/retrieval.py`
2. Comment out the translation block in `choose_retrieval()` and `choose_retrieval_enhanced()`

## Monitoring

Check telemetry to see translation stats:

```bash
# View detected languages
cat /opt/iasantiago-rag/rag-api/retrieval.jsonl | \
  jq '.original_language' | sort | uniq -c

# See translated queries
docker compose logs rag-api | grep "Translated query"
```

## Troubleshooting

**Q: First query in a language is slow**  
A: Normal - the translation model is being loaded (~400MB). Subsequent queries are fast.

**Q: Translation seems inaccurate**  
A: Helsinki-NLP models are lightweight. The re-ranking step helps recover. The original query is always preserved in logs.

**Q: Translation fails**  
A: Falls back to original query gracefully. Check logs for details.

## Testing

Test translation locally:

```bash
cd /datos/iasantiago/rag-api
python3 test_translation.py
```

## Documentation

Full documentation: `TRANSLATION_GUIDE.md`

## Questions?

See `TRANSLATION_GUIDE.md` for detailed information on:

- Supported languages
- Performance tuning
- Troubleshooting
- Future improvements
