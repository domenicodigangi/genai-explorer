"""
GenAI Knowledge Explorer — FastAPI Backend

Endpoints:
  POST /api/chat        — RAG-powered conversational Q&A (OpenAI gpt-4.1-mini)
  GET  /api/graph       — Topic graph (nodes + edges + hierarchy)
  GET  /api/search      — Semantic search over knowledge base
  GET  /api/topics/{id} — Resources for a specific topic node

LLM: OpenAI gpt-4.1-mini (chat) + text-embedding-3-small (embeddings)
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import tiktoken
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI, OpenAIError

logger = logging.getLogger("genai_explorer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GenAI Knowledge Explorer API")

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Model configuration — cheap OpenAI models
# ---------------------------------------------------------------------------

CHAT_MODEL = "gpt-4.1-mini"          # cheap, fast, good quality
EMBEDDING_MODEL = "text-embedding-3-small"  # cheapest embedding model
EMBEDDING_DIM = 512  # Matryoshka dims — must match ingestion

# Retrieval configuration
BM25_K1 = 1.2
BM25_B = 0.75
HYBRID_ALPHA = 0.7         # weight for semantic vs BM25 (1.0 = pure semantic)
MIN_SIMILARITY_THRESHOLD = 0.25
RETRIEVAL_TOP_K_INITIAL = 20   # initial candidate set for reranking
RETRIEVAL_TOP_K_FINAL_MIN = 3
RETRIEVAL_TOP_K_FINAL_MAX = 8
CONTEXT_TOKEN_BUDGET = 6000    # max tokens for assembled context
MAX_CHARS_PER_SOURCE = 2000    # raised from 1500 to reduce truncation

# Token counting (approximate for gpt-4.1-mini)
_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model("gpt-4o")  # closest available
        except Exception:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))

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

    # Build BM25 index from chunk texts
    _build_bm25_index()

    # Embedding versioning check
    emb_meta_path = DATA_DIR / "embeddings_meta.json"
    if emb_meta_path.exists():
        with open(emb_meta_path) as f:
            emb_meta = json.load(f)
        stored_model = emb_meta.get("model", "unknown")
        stored_dims = emb_meta.get("dimensions", "unknown")
        if stored_model != EMBEDDING_MODEL:
            logger.warning(
                "Embedding model mismatch: stored=%s, configured=%s. "
                "Re-run ingestion to update embeddings.", stored_model, EMBEDDING_MODEL
            )
        if stored_dims != EMBEDDING_DIM:
            logger.warning(
                "Embedding dimensions mismatch: stored=%s, configured=%s. "
                "Re-run ingestion to update embeddings.", stored_dims, EMBEDDING_DIM
            )
    else:
        logger.info("No embeddings_meta.json found — embedding versioning not tracked")

    logger.info(
        "Loaded %d chunks, %d topic nodes, %d embeddings",
        len(_store['chunks']), len(_store['graph']['nodes']), len(_store['emb_ids'])
    )


# ---------------------------------------------------------------------------
# BM25 sparse retrieval
# ---------------------------------------------------------------------------

_BM25_WORD_RE = re.compile(r'\b[a-zA-Z0-9][\w-]*\b')


def _tokenize_bm25(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return [w.lower() for w in _BM25_WORD_RE.findall(text)]


def _build_bm25_index():
    """Build in-memory BM25 index from loaded chunks."""
    chunks = _store.get("chunks", {})
    if not chunks:
        _store["bm25_ready"] = False
        return

    doc_ids = list(chunks.keys())
    doc_freqs: dict[str, int] = {}  # term -> number of docs containing term
    doc_term_freqs: list[dict[str, int]] = []  # per-doc term frequencies
    doc_lengths: list[int] = []
    total_length = 0

    for cid in doc_ids:
        text = chunks[cid].get("text", "")
        tokens = _tokenize_bm25(text)
        doc_lengths.append(len(tokens))
        total_length += len(tokens)

        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        doc_term_freqs.append(tf)

        for term in tf:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    n_docs = len(doc_ids)
    avg_dl = total_length / n_docs if n_docs else 1

    # Precompute IDF values
    idf: dict[str, float] = {}
    for term, df in doc_freqs.items():
        idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    _store["bm25"] = {
        "doc_ids": doc_ids,
        "idf": idf,
        "doc_term_freqs": doc_term_freqs,
        "doc_lengths": doc_lengths,
        "avg_dl": avg_dl,
        "n_docs": n_docs,
    }
    _store["bm25_ready"] = True
    logger.info("BM25 index built: %d documents, %d unique terms", n_docs, len(idf))


def _bm25_search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Score documents against query using BM25. Returns [(chunk_id, score), ...]."""
    if not _store.get("bm25_ready"):
        return []

    bm25 = _store["bm25"]
    query_tokens = _tokenize_bm25(query)
    if not query_tokens:
        return []

    scores = np.zeros(bm25["n_docs"], dtype=np.float64)
    for term in query_tokens:
        if term not in bm25["idf"]:
            continue
        idf_val = bm25["idf"][term]
        for i, tf_dict in enumerate(bm25["doc_term_freqs"]):
            if term in tf_dict:
                tf = tf_dict[term]
                dl = bm25["doc_lengths"][i]
                numerator = tf * (BM25_K1 + 1)
                denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / bm25["avg_dl"])
                scores[i] += idf_val * numerator / denominator

    top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((bm25["doc_ids"][idx], float(scores[idx])))
    return results


_openai_client: OpenAI | None = None


def get_openai() -> OpenAI:
    """Lazy-singleton OpenAI client — reuses connection pool across requests."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY in environment variables.",
        )
    kwargs: dict = {"api_key": api_key, "timeout": 30.0, "max_retries": 1}
    project_id = (os.environ.get("OPENAI_PROJECT_ID") or "").strip()
    org_id = (os.environ.get("OPENAI_ORG_ID") or "").strip()
    if project_id:
        kwargs["project"] = project_id
    if org_id:
        kwargs["organization"] = org_id
    _openai_client = OpenAI(**kwargs)
    return _openai_client


# ---------------------------------------------------------------------------
# Rate limiting — simple in-memory sliding window (per IP)
# ---------------------------------------------------------------------------

_rate_limits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
RATE_LIMIT_WINDOW = 60             # seconds
RATE_LIMIT_MAX_CHAT = 10           # max chat requests per window per IP
RATE_LIMIT_MAX_SEARCH = 20         # max search requests per window per IP (embeddings are cheap)


def _check_rate_limit(ip: str, endpoint: str = "chat", max_requests: int | None = None) -> bool:
    """Return True if request is allowed, False if rate limited."""
    if max_requests is None:
        max_requests = RATE_LIMIT_MAX_CHAT
    now = time.time()
    bucket = _rate_limits[endpoint]
    bucket[ip] = [t for t in bucket[ip] if t > now - RATE_LIMIT_WINDOW]
    if len(bucket[ip]) >= max_requests:
        return False
    bucket[ip].append(now)
    return True


def _get_client_ip(request: Request) -> str:
    return (request.headers.get("x-forwarded-for", "") or
            (request.client.host if request.client else "unknown")).split(",")[0].strip()


# ---------------------------------------------------------------------------
# Chat response cache — avoid repeated expensive LLM calls
# ---------------------------------------------------------------------------

_chat_cache: dict[str, tuple[float, str, list[dict]]] = {}
CHAT_CACHE_MAX = 128
CHAT_CACHE_TTL = 3600  # 1 hour


def _chat_cache_key(message: str, history: list) -> str:
    """Deterministic hash of chat request (message + last 6 history entries)."""
    history_str = "|".join(f"{m.role}:{m.content}" for m in history[-6:])
    return hashlib.sha256(f"{message}||{history_str}".encode()).hexdigest()


def _get_cached_chat(key: str) -> tuple[str, list[dict]] | None:
    if key in _chat_cache:
        ts, answer, sources = _chat_cache[key]
        if time.time() - ts < CHAT_CACHE_TTL:
            return answer, sources
        del _chat_cache[key]
    return None


def _set_chat_cache(key: str, answer: str, sources: list[dict]):
    if len(_chat_cache) >= CHAT_CACHE_MAX:
        oldest_key = min(_chat_cache, key=lambda k: _chat_cache[k][0])
        del _chat_cache[oldest_key]
    _chat_cache[key] = (time.time(), answer, sources)


# ---------------------------------------------------------------------------
# Query rewriting — resolve conversational references using history
# ---------------------------------------------------------------------------

def _rewrite_query(message: str, history: list) -> str:
    """Rewrite a conversational query into a standalone search query using history context."""
    if not history:
        return message

    # Only rewrite if the query seems to reference prior context
    reference_patterns = r'\b(it|that|this|those|these|they|them|the same|above|previous|mentioned|said)\b'
    if not re.search(reference_patterns, message, re.IGNORECASE) and len(message.split()) > 4:
        return message

    try:
        client = get_openai()
        history_text = "\n".join(
            f"{m.role}: {m.content[:200]}" for m in history[-4:]
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{
                "role": "system",
                "content": "Rewrite the user's latest message as a standalone search query that "
                           "includes necessary context from the conversation history. "
                           "Output ONLY the rewritten query, nothing else. "
                           "If the message is already standalone, return it unchanged."
            }, {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nLatest message: {message}"
            }],
            temperature=0,
            max_tokens=150,
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten and len(rewritten) < 500:
            logger.info("Query rewritten: '%s' -> '%s'", message[:80], rewritten[:80])
            return rewritten
    except Exception as e:
        logger.warning("Query rewriting failed, using original: %s", str(e)[:200])

    return message


# ---------------------------------------------------------------------------
# Metadata filtering — extract intent from query
# ---------------------------------------------------------------------------

_CONTENT_TYPE_PATTERNS = {
    "paper": re.compile(r'\b(paper|papers|research|study|studies|arxiv|publication)\b', re.I),
    "course": re.compile(r'\b(course|courses|tutorial|tutorials|lesson|lessons|learn|training)\b', re.I),
    "interview": re.compile(r'\b(interview|interviews|prep|preparation|hiring|questions)\b', re.I),
}


def _extract_metadata_filters(query: str) -> dict[str, str | None]:
    """Extract metadata filters from query intent."""
    filters: dict[str, str | None] = {"content_type": None}
    for content_type, pattern in _CONTENT_TYPE_PATTERNS.items():
        if pattern.search(query):
            filters["content_type"] = content_type
            break
    return filters


# ---------------------------------------------------------------------------
# Similarity search — hybrid (semantic + BM25), threshold, reranking
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _get_embedding_cached(query: str) -> tuple[float, ...]:
    """Cache embedding results for identical queries (~3MB for 256 entries)."""
    client = get_openai()
    resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
    return tuple(resp.data[0].embedding)


def _deduplicate_results(results: list[dict]) -> list[dict]:
    """Remove near-duplicate chunks from the same source/section."""
    seen: set[str] = set()
    deduped = []
    for r in results:
        # Key on source_file + section to detect adjacent chunks
        dedup_key = f"{r.get('source_file', '')}::{r.get('section', '')}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append(r)
    return deduped


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last sentence-ending punctuation
    last_period = max(truncated.rfind('. '), truncated.rfind('.\n'),
                      truncated.rfind('? '), truncated.rfind('! '))
    if last_period > max_chars * 0.5:  # only use if we keep at least half
        return truncated[:last_period + 1]
    return truncated.rstrip()


def search_similar(query: str, top_k: int = 8, metadata_filters: dict | None = None) -> list[dict]:
    """
    Hybrid search: combine semantic (cosine) + BM25 (keyword) scores.
    Apply similarity threshold, metadata filtering, and deduplication.
    """
    _load_data()
    t0 = time.time()

    # --- Semantic search (large candidate set) ---
    try:
        emb_tuple = _get_embedding_cached(query)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)[:500]}")
    query_emb = np.array(emb_tuple, dtype=np.float32)

    if len(_store["emb_ids"]) == 0:
        return []

    query_norm = query_emb / (np.linalg.norm(query_emb) or 1)
    semantic_scores = _store["emb_matrix_norm"] @ query_norm

    # Build ID-to-index map for merging
    id_to_idx = {cid: i for i, cid in enumerate(_store["emb_ids"])}

    # --- BM25 search ---
    bm25_results = _bm25_search(query, top_k=RETRIEVAL_TOP_K_INITIAL)
    bm25_scores_map: dict[str, float] = {}
    if bm25_results:
        max_bm25 = max(s for _, s in bm25_results) or 1.0
        bm25_scores_map = {cid: score / max_bm25 for cid, score in bm25_results}

    # --- Hybrid fusion: combine scores ---
    # Normalize semantic scores to [0, 1] range for fusion
    sem_min, sem_max = float(semantic_scores.min()), float(semantic_scores.max())
    sem_range = sem_max - sem_min if sem_max > sem_min else 1.0

    combined_scores: dict[str, float] = {}
    # Start with all embedding IDs
    for i, cid in enumerate(_store["emb_ids"]):
        sem_norm = (float(semantic_scores[i]) - sem_min) / sem_range
        bm25_norm = bm25_scores_map.get(cid, 0.0)
        combined_scores[cid] = HYBRID_ALPHA * sem_norm + (1 - HYBRID_ALPHA) * bm25_norm

    # Sort by combined score
    ranked_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)

    # --- Apply similarity threshold and dynamic top_k ---
    results = []
    for cid in ranked_ids:
        if len(results) >= RETRIEVAL_TOP_K_INITIAL:
            break
        # Use raw semantic score for threshold (combined score is normalized differently)
        idx = id_to_idx.get(cid)
        if idx is None:
            continue
        raw_score = float(semantic_scores[idx])
        if raw_score < MIN_SIMILARITY_THRESHOLD:
            continue

        chunk = _store["chunks"].get(cid, {})

        # Metadata filtering
        if metadata_filters and metadata_filters.get("content_type"):
            if chunk.get("content_type") != metadata_filters["content_type"]:
                continue

        results.append({
            "id": cid,
            "text": chunk.get("text", ""),
            "score": raw_score,
            "combined_score": combined_scores[cid],
            "source_file": chunk.get("source_file", ""),
            "section": chunk.get("section", ""),
            "category": chunk.get("category", ""),
            "title": chunk.get("title", ""),
            "url": chunk.get("url", ""),
            "topics": chunk.get("topics", []),
            "content_type": chunk.get("content_type", "text"),
            "all_urls": chunk.get("all_urls", []),
        })

    # --- Deduplicate ---
    results = _deduplicate_results(results)

    # --- Dynamic top_k based on score distribution ---
    if len(results) > RETRIEVAL_TOP_K_FINAL_MIN:
        # Find natural cutoff: if there's a big score drop, cut there
        final_k = min(top_k, RETRIEVAL_TOP_K_FINAL_MAX, len(results))
        for i in range(RETRIEVAL_TOP_K_FINAL_MIN, min(final_k, len(results))):
            score_drop = results[i - 1]["score"] - results[i]["score"]
            if score_drop > 0.08:  # significant gap
                final_k = i
                break
        results = results[:final_k]

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "Search: query='%s' results=%d top_score=%.3f min_score=%.3f elapsed=%.0fms",
        query[:60], len(results),
        results[0]["score"] if results else 0,
        results[-1]["score"] if results else 0,
        elapsed_ms,
    )

    return results


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: str
    content: str = Field(max_length=2000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default=[], max_length=20)
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    _load_data()
    has_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    resp: dict = {
        "status": "ok",
        "data_loaded": bool(_store.get("chunks")),
        "chunks": len(_store.get("chunks", {})),
        "topics": len(_store.get("graph", {}).get("nodes", [])),
        "embeddings": len(_store.get("emb_ids", [])),
        "api_key_set": has_key,
    }
    return resp


def _build_system_prompt() -> str:
    """Build system prompt with metadata legend for content types."""
    return """You are the GenAI Knowledge Explorer assistant. You help users navigate
a comprehensive collection of generative AI resources including courses, research papers,
interview prep materials, roadmaps, and tutorials.

Source types in the context:
- "paper" sources contain research papers — always include their direct link when referencing
- "course" sources contain educational courses and tutorials
- "interview" sources contain interview preparation materials
- "text" sources contain general documentation and guides

When answering:
- Use the provided context to give accurate, specific answers
- Always cite your sources using [Source N] notation (e.g., [Source 1], [Source 2])
- When mentioning research papers, ALWAYS include their direct link if available in the context
- When listing resources, include direct URLs when available
- If the context doesn't contain enough info, say so honestly
- Suggest related topics the user might want to explore
- Be concise but thorough
- For papers, format them as: **Paper Title** ([link](url)) — brief description"""


def _build_context_with_budget(results: list[dict]) -> tuple[str, list[dict]]:
    """
    Build context from results, respecting token budget.
    Returns (context_string, used_results) — may use fewer results if budget is tight.
    """
    context_parts = []
    used_results = []
    total_tokens = 0

    for i, r in enumerate(results):
        source_label = r.get("title", r.get("section", "Unknown"))
        topics = r.get("topics", [])
        topic_str = f" [Topics: {', '.join(topics)}]" if topics else ""
        content_type = r.get("content_type", "text")
        url = r.get("url", "")

        header = f"[Source {i+1}: {source_label}] (type: {content_type})"
        if content_type == "paper" and url:
            header += f" (Paper link: {url})"
        header += topic_str

        text = _truncate_at_sentence(r["text"], MAX_CHARS_PER_SOURCE)
        part = f"{header}\n{text}"

        part_tokens = _count_tokens(part)
        if total_tokens + part_tokens > CONTEXT_TOKEN_BUDGET and used_results:
            break  # budget exceeded, stop adding sources
        total_tokens += part_tokens
        context_parts.append(part)
        used_results.append(r)

    context = "\n\n---\n\n".join(context_parts)
    return context, used_results


def _validate_citations(answer: str, num_sources: int) -> str:
    """Validate [Source N] citations — flag invalid references."""
    citation_pattern = re.compile(r'\[Source (\d+)\]')
    for match in citation_pattern.finditer(answer):
        n = int(match.group(1))
        if n < 1 or n > num_sources:
            # Replace invalid citation with a note
            answer = answer.replace(match.group(0), f"[Source ?]")
    return answer


def _format_sources(results: list[dict]) -> list[dict]:
    """Format source metadata for the response, including text snippet."""
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
            "text_snippet": _truncate_at_sentence(r.get("text", ""), 300),
        })
    return sources


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """RAG-powered chat: retrieve relevant chunks, generate answer with citations."""
    t0 = time.time()
    client_ip = _get_client_ip(request)
    if not _check_rate_limit(client_ip, "chat", RATE_LIMIT_MAX_CHAT):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before sending more messages.")
    _load_data()

    # 0. Check chat cache (skip for streaming requests)
    cache_key = _chat_cache_key(req.message, req.history)
    if not req.stream:
        cached = _get_cached_chat(cache_key)
        if cached:
            logger.info("Chat cache hit for: '%s'", req.message[:60])
            return ChatResponse(answer=cached[0], sources=cached[1])

    # 1. Rewrite query for better retrieval (resolves conversational references)
    search_query = _rewrite_query(req.message, req.history)

    # 2. Extract metadata filters from query intent
    metadata_filters = _extract_metadata_filters(req.message)

    # 3. Retrieve relevant chunks (hybrid search + threshold + reranking)
    results = search_similar(search_query, top_k=RETRIEVAL_TOP_K_FINAL_MAX, metadata_filters=metadata_filters)

    # If metadata filter returned too few results, retry without filter
    if len(results) < RETRIEVAL_TOP_K_FINAL_MIN and metadata_filters.get("content_type"):
        logger.info("Retrying without metadata filter (got %d results)", len(results))
        results = search_similar(search_query, top_k=RETRIEVAL_TOP_K_FINAL_MAX)

    # 4. Build context with token budget management
    context, used_results = _build_context_with_budget(results)

    # 5. Assemble messages
    system_prompt = _build_system_prompt()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 6 messages)
    for msg in req.history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    user_msg = f"""Context from the knowledge base:

{context}

---

User question: {req.message}"""
    messages.append({"role": "user", "content": user_msg})

    # Check total token budget (model context)
    total_tokens = sum(_count_tokens(m["content"]) for m in messages)
    logger.info("Chat: total_prompt_tokens=%d, sources_used=%d", total_tokens, len(used_results))

    # 6. Format sources for response
    sources = _format_sources(used_results)

    # 7. Generate response
    client = get_openai()

    if req.stream:
        # --- Streaming mode: SSE ---
        def _stream_response():
            # First send sources as a JSON event
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            try:
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                    stream=True,
                )
                full_answer = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_answer += token
                        yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

                # Validate citations
                validated = _validate_citations(full_answer, len(used_results))
                if validated != full_answer:
                    yield f"event: corrected\ndata: {json.dumps({'answer': validated})}\n\n"

                # Cache the result
                _set_chat_cache(cache_key, validated, sources)

                elapsed_ms = (time.time() - t0) * 1000
                logger.info("Chat streamed: query='%s' elapsed=%.0fms", req.message[:60], elapsed_ms)

            except Exception as e:
                logger.error("Streaming error: %s", str(e)[:300])
                yield f"event: error\ndata: {json.dumps({'detail': 'LLM service temporarily unavailable'})}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            _stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Non-streaming mode ---
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error("LLM error: %s", str(e)[:300])
        raise HTTPException(status_code=502, detail="LLM service temporarily unavailable")

    # Validate citations
    answer = _validate_citations(answer, len(used_results))

    elapsed_ms = (time.time() - t0) * 1000
    logger.info("Chat: query='%s' elapsed=%.0fms cache=miss", req.message[:60], elapsed_ms)

    _set_chat_cache(cache_key, answer, sources)
    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/search")
def search(request: Request, q: str = Query(..., min_length=2), top_k: int = Query(10, ge=1, le=20)):
    """Semantic search across the knowledge base."""
    client_ip = _get_client_ip(request)
    if not _check_rate_limit(client_ip, "search", RATE_LIMIT_MAX_SEARCH):
        raise HTTPException(status_code=429, detail="Search rate limit exceeded. Please wait before searching again.")
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
        raise HTTPException(status_code=404, detail="Topic not found")

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
