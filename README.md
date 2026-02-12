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
│   ChromaDB     │  │ OpenAI API    │
│ (Vector Store) │  │ (Embeddings + │
│                │  │  Chat GPT-4.1)│
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
│  • Store in ChromaDB + JSON        │
└────────────────────────────────────┘
```

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | Next.js 14, TypeScript, Tailwind CSS |
| Viz         | D3.js (force graph + treemap)       |
| Backend     | FastAPI (Python)                    |
| Vector DB   | ChromaDB (persistent, file-based)   |
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
- `data/chroma_db/` — ChromaDB persistent vector store
- `data/chunks.json` — All parsed chunks with metadata
- `data/topic_graph.json` — Topic graph (nodes, edges, hierarchy)
- `data/embeddings.json` — Pre-computed embeddings (Vercel fallback)

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

### 1. Prepare

Make sure the ingestion pipeline has been run and `data/` folder contains the
generated files. These will be bundled with the deployment.

### 2. Install Vercel CLI

```bash
npm install -g vercel
```

### 3. Configure Environment Variables

In the Vercel dashboard (or via CLI), set:

```
OPENAI_API_KEY=sk-...
```

### 4. Deploy

```bash
vercel
```

Follow the prompts. The `vercel.json` is pre-configured to route `/api/*`
to the Python serverless function and everything else to the Next.js frontend.

> **Note on Vercel + ChromaDB**: On Vercel serverless, the filesystem is
> ephemeral. The backend automatically falls back to a lightweight numpy-based
> similarity search using the pre-computed `embeddings.json` file. For
> production, consider using a hosted vector DB.

---

## Project Structure

```
genai-knowledge-explorer/
├── api/
│   └── index.py              # FastAPI backend (all endpoints)
├── scripts/
│   └── ingest.py             # Data ingestion pipeline
├── data/                     # Generated data (after running ingest.py)
│   ├── chroma_db/            # ChromaDB persistent store
│   ├── chunks.json           # Parsed chunks with metadata
│   ├── topic_graph.json      # Topic graph for visualization
│   └── embeddings.json       # Pre-computed embeddings
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

## How It Works

### Ingestion Pipeline

1. **Clone** the awesome-generative-ai-guide repo
2. **Parse** all markdown files, splitting by heading into semantic chunks
3. **Extract** topics from each chunk using keyword matching against 50+ canonical GenAI topics
4. **Generate** embeddings via OpenAI `text-embedding-3-small`
5. **Build** a co-occurrence topic graph (topics that appear in the same chunk are connected)
6. **Store** in ChromaDB + export as JSON

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
| `/api/health`          | GET    | Health check with data stats             |
| `/api/chat`            | POST   | RAG chat (body: `{message, history}`)    |
| `/api/search?q=...`    | GET    | Semantic search                          |
| `/api/graph`           | GET    | Full topic graph (nodes, edges, hierarchy)|
| `/api/topics/{id}`     | GET    | Topic details + connected topics + resources|

---

## License

MIT — Built for the AI Engineering Challenge.

Data sourced from [awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) (MIT License).
