# GenAI Knowledge Explorer — API

FastAPI backend providing RAG-powered chat and topic exploration endpoints.

## Running locally

```bash
# From project root
export OPENAI_API_KEY=sk-...
uv run uvicorn api.index:app --reload --port 8000
```

## Endpoints

- `GET /api/health` — Health check, returns data stats
- `POST /api/chat` — RAG chat with `{message: str, history: [{role, content}]}`
- `GET /api/search?q=...&top_k=10` — Semantic search
- `GET /api/graph` — Topic graph (nodes, edges, hierarchy)
- `GET /api/topics/{topic_id}` — Topic details and resources

## Vector Store Strategy

The API supports two modes:
1. **ChromaDB** (local development) — loaded from `data/chroma_db/`
2. **Numpy fallback** (Vercel/serverless) — cosine similarity over pre-computed embeddings from `data/embeddings.json`

The fallback is automatic — if ChromaDB is unavailable, it uses numpy.
