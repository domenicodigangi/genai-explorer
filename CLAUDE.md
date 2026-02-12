# GenAI Knowledge Explorer

## Project Overview
A full-stack app for exploring generative AI learning resources. Built with a Python/FastAPI backend and Next.js frontend. Data is ingested from the awesome-generative-ai-guide GitHub repo, chunked, embedded, and stored in ChromaDB for RAG-based chat and a topic knowledge graph.

## Architecture
- **Backend**: `api/index.py` — FastAPI app with `/api/chat`, `/api/graph`, `/api/topics/{id}` endpoints. Uses OpenAI gpt-4.1-mini for chat, ChromaDB for vector search.
- **Frontend**: `frontend/` — Next.js app with two modes: Chat (RAG Q&A) and Explore (knowledge graph + treemap + detail panel).
- **Ingestion**: `scripts/ingest.py` — Parses markdown files, extracts topics/entities, builds graph, generates embeddings.
- **Data**: `data/chunks.json` and `data/topic_graph.json` are checked into git for Vercel deployment.

## Key Commands
- Backend: `cd /home/ddg/repos/genai-explorer && uv run uvicorn api.index:app --reload --port 8000`
- Frontend: `cd /home/ddg/repos/genai-explorer/frontend && npm run dev`
- Type check: `cd /home/ddg/repos/genai-explorer/frontend && npx tsc --noEmit`
- Ingest: `cd /home/ddg/repos/genai-explorer && uv run python scripts/ingest.py`

## Screenshots
**Always save screenshots to the `screenshots/` directory** (not the project root). Use descriptive filenames, e.g. `screenshots/graph-keyboard-nav.png`. This folder is gitignored.

## Code Conventions
- Frontend components use 'use client' directive (Next.js App Router)
- D3.js for visualizations (TopicGraph.tsx, TopicTree.tsx)
- Tailwind CSS with CSS custom properties for theming (globals.css)
- Dark theme only — CSS vars defined in `:root` (--midnight, --surface, --border, etc.)
- TypeScript strict mode
