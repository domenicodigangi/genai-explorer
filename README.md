# 🧭 GenAI Knowledge Explorer

> Navigate the generative AI landscape through conversational RAG and interactive topic visualization.

Built on top of the [awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) repository (23k+ stars), this application makes it easy to explore 90+ courses, research papers, roadmaps, and interview materials through two complementary interfaces:

- **💬 Chat Mode** — Ask questions and get RAG-powered answers with source citations
- **🗺️ Explore Mode** — Navigate a visual knowledge graph and treemap to discover topics and their connections

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Chat UI    │  │ Force Graph  │  │  Treemap   │ │
│  │  (RAG Q&A)   │  │   (D3.js)    │  │  (D3.js)   │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
└─────────┼─────────────────┼────────────────┼────────┘
          │                 │                │
          ▼                 ▼                ▼
┌─────────────────────────────────────────────────────┐
│               FastAPI Backend (/api)                 │
│  POST /api/chat    GET /api/graph   GET /api/search  │
│  GET /api/topics/{id}           GET /api/health      │
└────────┬──────────────────┬─────────────────────────┘
         │                  │
         ▼                  ▼
┌────────────────┐  ┌───────────────┐
│ Numpy Cosine   │  │ OpenAI API    │
│ Similarity     │  │ (Embeddings + │
│ (in-memory)    │  │  Chat GPT-4.1)│
└────────────────┘  └───────────────┘
         ▲
         │
┌────────────────────────────────────┐
│     Ingestion Pipeline             │
│  scripts/ingest.py                 │
│  • Clone repo                      │
│  • Parse markdown → chunks         │
│  • Generate embeddings             │
│  • Build topic graph               │
│  • Export to JSON (+ ChromaDB)     │
└────────────────────────────────────┘
```

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | Next.js 14, TypeScript, Tailwind CSS |
| Viz         | D3.js (force graph + treemap)       |
| Backend     | FastAPI (Python)                    |
| Vector Search | Numpy cosine similarity (in-memory) |
| Embeddings  | OpenAI `text-embedding-3-small`     |
| Chat LLM    | OpenAI `gpt-4.1-mini`              |
| Deployment  | Vercel (serverless)                 |

---

## Quick Start

### Prerequisites

- Python 3.11+ and [uv](https://github.com/astral-sh/uv) package manager
- Node.js 18+ and npm
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Git

### 1. Clone and Install

```bash
git clone https://github.com/<YOUR_USERNAME>/genai-knowledge-explorer.git
cd genai-knowledge-explorer

# Python dependencies
pip install uv
uv sync

# Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Run the Ingestion Pipeline

This clones the awesome-generative-ai-guide repo, parses all markdown files,
generates embeddings, and builds the topic graph:

```bash
uv run python scripts/ingest.py
```

This takes ~2-3 minutes (depending on the number of chunks and your API rate).
It creates:
- `data/chunks.json` — All parsed chunks with metadata
- `data/topic_graph.json` — Topic graph (nodes, edges, hierarchy)
- `data/embeddings.json` — Pre-computed embeddings for vector search
- `data/chroma_db/` — ChromaDB store (local dev only, gitignored)
- `data/ingestion_meta.json` — Ingestion metadata (source commit, timestamps)

You can check if the upstream repo has new data without running a full ingestion:

```bash
uv run python scripts/ingest.py --check-only
# Exit code 0 = no changes, 1 = changes detected
```

### 4. Start the Backend

```bash
uv run uvicorn api.index:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/health`

### 5. Start the Frontend

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** and start exploring!

---

## Deploying to Vercel

### Prerequisites

- A [Vercel account](https://vercel.com/signup)
- Your repo pushed to GitHub
- The ingestion pipeline has been run (`data/chunks.json`, `data/topic_graph.json`, and `data/embeddings.json` exist and are committed)

### Step 1: Import Project

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **Import Git Repository** and select your `genai-explorer` repo
3. Vercel auto-detects the monorepo structure from `vercel.json`

### Step 2: Configure Environment Variables

In the Vercel dashboard under **Project Settings > Environment Variables**, add:

| Variable | Value | Required |
|----------|-------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `ALLOWED_ORIGINS` | Your production URL (e.g. `https://genai-explorer.vercel.app`) | After first deploy |

> **Never** put your API key in the repo. Always use the Vercel dashboard.

### Step 3: Deploy

Click **Deploy**. Vercel will:
- Build the Next.js frontend via `@vercel/next`
- Deploy the FastAPI backend via `@vercel/python`
- Apply security headers from `vercel.json` (CSP, HSTS, X-Frame-Options, etc.)
- Route `/api/*` to the Python function, everything else to Next.js

### Step 4: Post-Deploy

1. Copy your production URL from the Vercel dashboard
2. Go to **Settings > Environment Variables** and set `ALLOWED_ORIGINS` to that URL
3. Redeploy (Deployments > latest > Redeploy)
4. Set a monthly spend cap in your [OpenAI dashboard](https://platform.openai.com/settings/organization/limits)

### Verify

```bash
# Health check
curl https://your-app.vercel.app/api/health
# Expected: {"status":"ok","data_loaded":true}

# Security headers
curl -I https://your-app.vercel.app/
# Should include X-Frame-Options, Content-Security-Policy, Strict-Transport-Security, etc.
```

### Troubleshooting

**"Serverless Function exceeds 250 MB unzipped"**

The Python function bundle is too large. Check that `chromadb` is **not** in `requirements.txt` — it pulls ~100+ MB of transitive dependencies (onnxruntime, grpcio, etc.) and is not used on Vercel. The API uses lightweight numpy cosine similarity with pre-computed embeddings instead. Only these packages should be in `requirements.txt`:

```
fastapi>=0.111.0
uvicorn>=0.30.0
openai>=1.30.0
numpy>=1.26.0
pydantic>=2.7.0
python-dotenv>=1.0.0
```

**"uv sync --locked failed"**

The `uv.lock` file is out of sync with `pyproject.toml`. Regenerate it locally and push:

```bash
uv lock
git add uv.lock && git commit -m "Regenerate uv.lock" && git push
```

**"unknown field `python`" in pyproject.toml**

Newer versions of `uv` don't recognize `python = "3.12"` under `[tool.uv]`. Remove the `[tool.uv]` section — use `requires-python = ">=3.11"` in `[project]` instead.

**"routes cannot be present" with headers/rewrites**

Vercel doesn't allow the legacy `routes` field alongside `headers`, `rewrites`, or `redirects`. Use `rewrites` instead of `routes` in `vercel.json` — see the current config for the correct format.

**Chat returns 502 "LLM service temporarily unavailable"**

The `OPENAI_API_KEY` environment variable is missing or invalid in Vercel. Check **Settings > Environment Variables** and redeploy.

**CORS errors in browser console**

Set `ALLOWED_ORIGINS` in Vercel environment variables to your production domain. On Vercel, the frontend and API share the same origin so CORS is rarely triggered, but the explicit allowlist is defense-in-depth.

### How It Works on Vercel

The deployed API uses **numpy cosine similarity** on pre-computed embeddings instead of ChromaDB (which requires persistent disk that Vercel's ephemeral serverless functions don't provide). At ~700 vectors with 1536 dimensions, brute-force cosine similarity takes <1ms — far less than the ~300ms OpenAI API call. This approach requires zero additional infrastructure.

---

## Project Structure

```
genai-knowledge-explorer/
├── api/
│   └── index.py              # FastAPI backend (all endpoints)
├── scripts/
│   └── ingest.py             # Data ingestion pipeline
├── .github/workflows/
│   └── update-data.yml       # Automated weekly data update pipeline
├── data/                     # Generated data (after running ingest.py)
│   ├── chunks.json           # Parsed chunks with metadata
│   ├── topic_graph.json      # Topic graph for visualization
│   ├── embeddings.json       # Pre-computed embeddings for search
│   ├── chroma_db/            # ChromaDB store (local only, gitignored)
│   └── ingestion_meta.json   # Ingestion metadata (source commit, timestamps)
├── frontend/
│   ├── app/
│   │   ├── layout.tsx        # Root layout
│   │   ├── page.tsx          # Main page (tabs, state management)
│   │   └── globals.css       # Global styles + design system
│   ├── components/
│   │   ├── Chat.tsx          # RAG chat interface
│   │   ├── TopicGraph.tsx    # D3.js force-directed graph
│   │   ├── TopicTree.tsx     # D3.js treemap visualization
│   │   └── TopicDetail.tsx   # Topic detail panel
│   ├── lib/
│   │   └── sanitizeUrl.ts    # URL protocol validation
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── tsconfig.json
├── .cursor/rules/
│   └── frontend-rule.mdc     # Cursor AI development rules
├── vercel.json               # Vercel deployment config
├── pyproject.toml            # Python project config (uv)
├── requirements.txt          # Python deps (Vercel)
└── README.md
```

---

## Keeping Data Up-to-Date

The app's data comes from the [awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) repo. A GitHub Actions pipeline keeps it in sync automatically.

### Automated Updates

A [workflow](.github/workflows/update-data.yml) runs **every Monday at 9:00 AM UTC** and:

1. Pulls the latest upstream repo
2. Compares the upstream HEAD commit against `data/ingestion_meta.json`
3. **Skips** if nothing has changed
4. Runs the full ingestion pipeline if changes are detected
5. Verifies output files and runs a sanity check (data must not shrink below 80% of its previous size)
6. Commits and pushes the updated data files to `main`

### Manual Trigger

You can trigger an update at any time from the GitHub Actions tab, or via the CLI:

```bash
gh workflow run update-data.yml
```

### Local Check

To check if the upstream repo has new content without running a full ingestion:

```bash
uv run python scripts/ingest.py --check-only
# Exit code 0 = no changes, 1 = changes detected
```

### Required Secret

The workflow needs an `OPENAI_API_KEY` repository secret (Settings > Secrets and variables > Actions) for embedding generation.

---

## How It Works

### Ingestion Pipeline

1. **Clone** the awesome-generative-ai-guide repo
2. **Parse** all markdown files, splitting by heading into semantic chunks
3. **Extract** topics from each chunk using keyword matching against 50+ canonical GenAI topics
4. **Generate** embeddings via OpenAI `text-embedding-3-small`
5. **Build** a co-occurrence topic graph (topics that appear in the same chunk are connected)
6. **Export** as JSON files (+ ChromaDB for local dev)

### RAG Chat

1. User sends a question
2. Embed the question with OpenAI `text-embedding-3-small`
3. Retrieve top-6 similar chunks from the vector store
4. Pass chunks as context to OpenAI `gpt-4.1-mini` with a system prompt that instructs the model to include paper links
5. Return the answer with source citations, paper links, and "Read Paper" buttons

### Research Paper Ingestion

The ingestion pipeline has a dedicated markdown table parser that:
- Detects tabular content in `research_updates/*.md` files
- Extracts each paper as a separate chunk with its title, URL, summary, and category
- Preserves all links (arxiv, GitHub, etc.) in the `all_urls` field
- Tags chunks with `content_type: "paper"` so the UI can render them with "Read Paper" buttons

### Visual Exploration

- **Knowledge Graph**: Force-directed layout where nodes = topics, sized by frequency, colored by category, edges = co-occurrence. Hover to highlight connections, click to see details.
- **Topic Treemap**: Hierarchical view organized by category (Foundations, Techniques, RAG, Agents, etc.) with rectangles sized by reference count.

---

## API Reference

| Endpoint               | Method | Description                              |
|------------------------|--------|------------------------------------------|
| `/api/health`          | GET    | Health check                             |
| `/api/chat`            | POST   | RAG chat (body: `{message, history}`)    |
| `/api/search?q=...`    | GET    | Semantic search                          |
| `/api/graph`           | GET    | Full topic graph (nodes, edges, hierarchy)|
| `/api/topics/{id}`     | GET    | Topic details + connected topics + resources|

---

## License

MIT — Built for the AI Engineering Challenge.

Data sourced from [awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) (MIT License).
