"""
GenAI Knowledge Explorer — FastAPI Backend

Endpoints:
  POST /api/chat        — RAG-powered conversational Q&A (OpenAI gpt-4.1-mini)
  GET  /api/graph       — Topic graph (nodes + edges + hierarchy)
  GET  /api/search      — Semantic search over knowledge base
  GET  /api/topics/{id} — Resources for a specific topic node

LLM: OpenAI gpt-4.1-mini (chat) + text-embedding-3-small (embeddings)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GenAI Knowledge Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model configuration — cheap OpenAI models
# ---------------------------------------------------------------------------

CHAT_MODEL = "gpt-4.1-mini"          # cheap, fast, good quality
EMBEDDING_MODEL = "text-embedding-3-small"  # cheapest embedding model

# ---------------------------------------------------------------------------
# Data loading — lazy singleton
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_store = {}


def _load_data():
    """Load pre-computed data into memory (once)."""
    if _store:
        return

    # Load chunks
    chunks_path = DATA_DIR / "chunks.json"
    if chunks_path.exists():
        with open(chunks_path) as f:
            _store["chunks"] = {c["id"]: c for c in json.load(f)}
    else:
        _store["chunks"] = {}

    # Load topic graph
    graph_path = DATA_DIR / "topic_graph.json"
    if graph_path.exists():
        with open(graph_path) as f:
            _store["graph"] = json.load(f)
    else:
        _store["graph"] = {"nodes": [], "edges": [], "hierarchy": {}}

    # Load embeddings for lightweight search
    emb_path = DATA_DIR / "embeddings.json"
    if emb_path.exists():
        with open(emb_path) as f:
            raw = json.load(f)
        _store["emb_ids"] = list(raw.keys())
        _store["emb_matrix"] = np.array(list(raw.values()), dtype=np.float32)
        # Pre-normalize for cosine similarity
        norms = np.linalg.norm(_store["emb_matrix"], axis=1, keepdims=True)
        norms[norms == 0] = 1
        _store["emb_matrix_norm"] = _store["emb_matrix"] / norms
    else:
        _store["emb_ids"] = []
        _store["emb_matrix_norm"] = np.array([])

    # Try ChromaDB (local dev)
    try:
        import chromadb
        chroma_dir = DATA_DIR / "chroma_db"
        if chroma_dir.exists():
            client = chromadb.PersistentClient(path=str(chroma_dir))
            _store["chroma"] = client.get_collection("genai_knowledge")
            print("✅ ChromaDB loaded")
        else:
            _store["chroma"] = None
    except Exception:
        _store["chroma"] = None

    print(f"📦 Loaded {len(_store['chunks'])} chunks, "
          f"{len(_store['graph']['nodes'])} topic nodes, "
          f"{len(_store['emb_ids'])} embeddings")


def get_openai() -> OpenAI:
    """Single OpenAI client for both chat and embeddings."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def search_similar(query: str, top_k: int = 8) -> list[dict]:
    """Search for similar chunks using either ChromaDB or numpy fallback."""
    _load_data()

    client = get_openai()
    resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    query_emb = np.array(resp.data[0].embedding, dtype=np.float32)

    # Try ChromaDB first
    if _store.get("chroma"):
        results = _store["chroma"].query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
        )
        found = []
        for i, doc_id in enumerate(results["ids"][0]):
            chunk = _store["chunks"].get(doc_id, {})
            found.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
                **{k: v for k, v in chunk.items() if k not in ("id", "text")},
            })
        return found

    # Numpy fallback
    if len(_store["emb_ids"]) == 0:
        return []

    query_norm = query_emb / (np.linalg.norm(query_emb) or 1)
    scores = _store["emb_matrix_norm"] @ query_norm
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk_id = _store["emb_ids"][idx]
        chunk = _store["chunks"].get(chunk_id, {})
        results.append({
            "id": chunk_id,
            "text": chunk.get("text", ""),
            "score": float(scores[idx]),
            "source_file": chunk.get("source_file", ""),
            "section": chunk.get("section", ""),
            "category": chunk.get("category", ""),
            "title": chunk.get("title", ""),
            "url": chunk.get("url", ""),
            "topics": chunk.get("topics", []),
            "content_type": chunk.get("content_type", "text"),
            "all_urls": chunk.get("all_urls", []),
        })
    return results


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    _load_data()
    return {
        "status": "ok",
        "chunks": len(_store.get("chunks", {})),
        "topics": len(_store.get("graph", {}).get("nodes", [])),
        "has_chroma": _store.get("chroma") is not None,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """RAG-powered chat: retrieve relevant chunks, generate answer with citations."""
    _load_data()

    # 1. Retrieve relevant chunks
    results = search_similar(req.message, top_k=6)

    # 2. Build context — include paper links explicitly
    context_parts = []
    for i, r in enumerate(results):
        source_label = r.get("title", r.get("section", "Unknown"))
        topics = r.get("topics", [])
        topic_str = f" [Topics: {', '.join(topics)}]" if topics else ""
        content_type = r.get("content_type", "text")
        url = r.get("url", "")

        header = f"[Source {i+1}: {source_label}]"
        if content_type == "paper" and url:
            header += f" (Paper link: {url})"
        header += topic_str

        context_parts.append(f"{header}\n{r['text'][:1500]}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Build messages
    system_prompt = """You are the GenAI Knowledge Explorer assistant. You help users navigate
a comprehensive collection of generative AI resources including courses, research papers,
interview prep materials, roadmaps, and tutorials.

When answering:
- Use the provided context to give accurate, specific answers
- Always cite your sources using [Source N] notation
- When mentioning research papers, ALWAYS include their direct link if available in the context
- When listing resources, include direct URLs when available
- If the context doesn't contain enough info, say so honestly
- Suggest related topics the user might want to explore
- Be concise but thorough
- For papers, format them as: **Paper Title** ([link](url)) — brief description"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 6 messages)
    for msg in req.history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message with context
    user_msg = f"""Context from the knowledge base:

{context}

---

User question: {req.message}"""
    messages.append({"role": "user", "content": user_msg})

    # 4. Generate response via OpenAI
    client = get_openai()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )

    answer = response.choices[0].message.content

    # 5. Format sources — include content_type and all_urls
    sources = []
    for r in results:
        sources.append({
            "title": r.get("title", ""),
            "source_file": r.get("source_file", ""),
            "url": r.get("url", ""),
            "category": r.get("category", ""),
            "topics": r.get("topics", []),
            "score": round(r.get("score", 0), 3),
            "content_type": r.get("content_type", "text"),
            "all_urls": r.get("all_urls", []),
        })

    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), top_k: int = Query(10, ge=1, le=50)):
    """Semantic search across the knowledge base."""
    results = search_similar(q, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/api/graph")
def get_graph():
    """Return the full topic graph for visualization."""
    _load_data()
    return _store.get("graph", {"nodes": [], "edges": [], "hierarchy": {}})


@app.get("/api/topics/{topic_id}")
def get_topic(topic_id: str):
    """Get details and resources for a specific topic."""
    _load_data()
    graph = _store.get("graph", {"nodes": [], "edges": []})

    # Find the topic node
    node = None
    for n in graph["nodes"]:
        if n["id"] == topic_id:
            node = n
            break

    if not node:
        return {"error": "Topic not found"}

    # Get connected topics
    connected = []
    for edge in graph["edges"]:
        if edge["source"] == topic_id:
            connected.append({"id": edge["target"], "weight": edge["weight"]})
        elif edge["target"] == topic_id:
            connected.append({"id": edge["source"], "weight": edge["weight"]})

    # Resolve connected topic labels
    node_map = {n["id"]: n["label"] for n in graph["nodes"]}
    for c in connected:
        c["label"] = node_map.get(c["id"], c["id"])
    connected.sort(key=lambda x: x["weight"], reverse=True)

    # Get chunk details for this topic's resources
    resources = []
    for chunk_id in node.get("resources", [])[:20]:
        chunk = _store["chunks"].get(chunk_id)
        if chunk:
            resources.append({
                "title": chunk.get("title", ""),
                "source_file": chunk.get("source_file", ""),
                "category": chunk.get("category", ""),
                "url": chunk.get("url", ""),
                "text_preview": chunk.get("text", "")[:300],
                "content_type": chunk.get("content_type", "text"),
                "all_urls": chunk.get("all_urls", []),
            })

    return {
        "topic": node,
        "connected_topics": connected[:15],
        "resources": resources,
    }


# ---------------------------------------------------------------------------
# Vercel handler
# ---------------------------------------------------------------------------
# For local dev: uvicorn api.index:app --reload
