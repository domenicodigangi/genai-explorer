#!/usr/bin/env python3
"""
Ingestion pipeline for awesome-generative-ai-guide repository.

This script:
1. Clones/pulls the repo
2. Parses all markdown files into structured chunks
3. Generates embeddings via OpenAI
4. Builds a topic graph (nodes = topics, edges = co-occurrence)
5. Persists everything to ChromaDB + JSON files for the API
"""

import argparse
import json
import os
import re

from dotenv import load_dotenv
load_dotenv()
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import chromadb
from chromadb.config import Settings
from openai import OpenAI
import tiktoken

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/aishwaryanr/awesome-generative-ai-guide.git"
REPO_DIR = Path(__file__).resolve().parent.parent / "repo_cache"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512  # Matryoshka embeddings: 1536 → 512 dims (~3x smaller, negligible quality loss)
BATCH_SIZE = 64  # OpenAI embedding batch size

# Chunking configuration
TARGET_CHUNK_TOKENS = 512         # sweet spot for text-embedding-3-small
CHUNK_OVERLAP_TOKENS = 50        # ~200 chars overlap between consecutive chunks
MIN_SECTION_CHARS = 50

# Token counter (lazy-loaded)
_enc: tiktoken.Encoding | None = None


def _get_enc() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _count_tokens(text: str) -> int:
    return len(_get_enc().encode(text))


def _tokens_to_chars_approx(tokens: int) -> int:
    """Approximate character count from token count (avg ~4 chars/token)."""
    return tokens * 4

# Topic extraction – canonical topics we look for in content
CANONICAL_TOPICS = [
    "Large Language Models", "Transformers", "Attention Mechanism",
    "Fine-Tuning", "LoRA", "QLoRA", "PEFT", "RLHF",
    "Prompt Engineering", "Chain-of-Thought", "Few-Shot Learning",
    "RAG", "Retrieval-Augmented Generation", "Vector Databases", "Embeddings",
    "Agents", "Tool Use", "Function Calling", "ReAct", "LangChain", "LlamaIndex",
    "LLMOps", "MLOps", "Evaluation", "Benchmarks", "Red Teaming",
    "Diffusion Models", "Image Generation", "Multimodal", "Vision-Language",
    "GPT", "Claude", "Llama", "Mistral", "Gemini", "BERT",
    "Hugging Face", "OpenAI", "DeepLearning.AI", "Coursera", "Nvidia",
    "Quantization", "Distillation", "Alignment", "Safety",
    "NLP", "Tokenization", "Neural Networks", "Deep Learning",
    "Knowledge Graphs", "Agentic RAG", "Multi-Agent Systems",
    "AutoGen", "crewAI", "Amazon Bedrock", "Google Cloud", "Azure",
]

# Aliases to merge similar topics
TOPIC_ALIASES = {
    "retrieval-augmented generation": "RAG",
    "retrieval augmented generation": "RAG",
    "large language models": "Large Language Models",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "peft": "PEFT",
    "rlhf": "RLHF",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "deeplearning.ai": "DeepLearning.AI",
    "gpt-4": "GPT",
    "gpt-3": "GPT",
    "chatgpt": "GPT",
    "llama 2": "Llama",
    "llama 3": "Llama",
    "chain of thought": "Chain-of-Thought",
    "cot": "Chain-of-Thought",
    "react": "ReAct",
    "multi-agent": "Multi-Agent Systems",
    "multiagent": "Multi-Agent Systems",
    "mlops": "MLOps",
    "llmops": "LLMOps",
    "vector database": "Vector Databases",
    "vector db": "Vector Databases",
    "fine tuning": "Fine-Tuning",
    "fine-tune": "Fine-Tuning",
    "finetuning": "Fine-Tuning",
    "prompt engineering": "Prompt Engineering",
    "image generation": "Image Generation",
    "diffusion model": "Diffusion Models",
    "multimodal": "Multimodal",
    "vision language": "Vision-Language",
    "knowledge graph": "Knowledge Graphs",
    "agentic rag": "Agentic RAG",
    "red teaming": "Red Teaming",
    "amazon bedrock": "Amazon Bedrock",
    "google cloud": "Google Cloud",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    section: str
    category: str  # e.g. "free_courses", "interview_prep", "resources", "research_updates"
    title: str
    url: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    content_type: str = "text"  # "text", "paper", "course", "tool"
    all_urls: list[str] = field(default_factory=list)  # ALL urls found in this chunk


@dataclass
class TopicNode:
    id: str
    label: str
    category: str  # "concept", "tool", "provider", "technique"
    weight: int = 0  # number of chunks referencing this topic
    resources: list[str] = field(default_factory=list)  # chunk IDs


@dataclass
class TopicEdge:
    source: str
    target: str
    weight: int = 0  # co-occurrence count


# ---------------------------------------------------------------------------
# Step 1: Clone / update repo
# ---------------------------------------------------------------------------

def clone_or_pull_repo():
    """Clone the repo if not present, otherwise pull latest."""
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        print("📦 Pulling latest changes...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
    else:
        print("📥 Cloning repository...")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    print(f"✅ Repo ready at {REPO_DIR}")


def get_upstream_head_sha() -> tuple[str, str]:
    """Return (commit_sha, commit_date_iso) of the upstream repo HEAD."""
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "log", "-1", "--format=%H %cI"],
        capture_output=True, text=True, check=True,
    )
    parts = result.stdout.strip().split(" ", 1)
    return parts[0], parts[1]


def check_for_upstream_changes() -> int:
    """
    Check if the upstream repo has changed since last ingestion.
    Returns 0 if no changes (skip), 1 if changes detected (proceed).
    """
    clone_or_pull_repo()
    current_sha, current_date = get_upstream_head_sha()
    meta_path = DATA_DIR / "ingestion_meta.json"

    if not meta_path.exists():
        print(f"No ingestion metadata found. Changes detected (first run).")
        print(f"Upstream HEAD: {current_sha[:12]} ({current_date})")
        return 1

    with open(meta_path) as f:
        meta = json.load(f)

    last_sha = meta.get("source_commit_sha", "")

    if current_sha == last_sha:
        print(f"No upstream changes detected. Last ingested: {current_sha[:12]} ({meta.get('ingested_at', 'unknown')})")
        return 0
    else:
        print(f"Upstream changes detected!")
        print(f"  Last ingested: {last_sha[:12]}")
        print(f"  Upstream HEAD: {current_sha[:12]} ({current_date})")
        return 1


def write_ingestion_metadata(file_count: int, chunk_count: int, topic_count: int):
    """Write ingestion metadata after a successful run."""
    commit_sha, commit_date = get_upstream_head_sha()
    meta = {
        "source_repo": REPO_URL,
        "source_commit_sha": commit_sha,
        "source_commit_date": commit_date,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "file_count": file_count,
        "chunk_count": chunk_count,
        "topic_count": topic_count,
    }
    meta_path = DATA_DIR / "ingestion_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata written to {meta_path}")


# ---------------------------------------------------------------------------
# Step 2: Parse markdown files into chunks
# ---------------------------------------------------------------------------

def find_markdown_files() -> list[Path]:
    """Find all .md files in the repo (excluding LICENSE)."""
    md_files = []
    for p in REPO_DIR.rglob("*.md"):
        if "LICENSE" in p.name or ".git" in str(p):
            continue
        md_files.append(p)
    return sorted(md_files)


def infer_category(filepath: Path) -> str:
    """Infer category from file path."""
    rel = filepath.relative_to(REPO_DIR)
    parts = rel.parts
    if len(parts) > 1:
        return parts[0]
    return "root"


def extract_urls(text: str) -> list[str]:
    """Extract URLs from markdown text."""
    url_pattern = r'https?://[^\s\)\]>\"\'`]+'
    return re.findall(url_pattern, text)


def extract_markdown_links(text: str) -> list[dict]:
    """Extract named markdown links: [title](url)."""
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    return [{"title": m[0].strip(), "url": m[1].strip()} for m in re.findall(pattern, text)]


def parse_markdown_table(text: str) -> list[dict]:
    """
    Parse a markdown table into a list of row dicts.
    Returns list of {col_name: value} for each row.
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 3:
        return []

    # Find header row and separator
    header_idx = None
    for i, line in enumerate(lines):
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1]):
            header_idx = i
            break

    if header_idx is None:
        return []

    headers = [h.strip() for h in lines[header_idx].split('|') if h.strip()]
    rows = []
    for line in lines[header_idx + 2:]:
        if not line.startswith('|') and '|' not in line:
            continue
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if len(cells) >= len(headers):
            row = dict(zip(headers, cells[:len(headers)]))
            rows.append(row)
        elif cells:
            # Partial row — pad
            row = dict(zip(headers, cells + [''] * (len(headers) - len(cells))))
            rows.append(row)
    return rows


def parse_research_table_to_chunks(filepath: Path) -> list[Chunk]:
    """
    Special parser for research table files (e.g. rag_research_table.md).
    Each table row becomes its own chunk with paper title, URL, and summary.
    """
    content = filepath.read_text(encoding="utf-8", errors="replace")
    category = infer_category(filepath)
    rel_path = str(filepath.relative_to(REPO_DIR))
    chunks = []

    # Split content into sections (intro text + table blocks)
    # First, try to parse tables from the whole content
    rows = parse_markdown_table(content)

    if rows:
        for i, row in enumerate(rows):
            # Try to find the paper name/title column
            paper_title = ""
            paper_url = ""
            summary = ""

            for key, val in row.items():
                key_lower = key.lower()
                # Detect title/paper column
                if any(k in key_lower for k in ['paper', 'title', 'name', 'topic']):
                    links = extract_markdown_links(val)
                    if links:
                        paper_title = links[0]["title"]
                        paper_url = links[0]["url"]
                    else:
                        paper_title = re.sub(r'[*_`\[\]]', '', val).strip()
                        urls = extract_urls(val)
                        if urls:
                            paper_url = urls[0]
                # Detect summary/description column
                elif any(k in key_lower for k in ['summary', 'description', 'details', 'overview', 'contribution']):
                    summary = re.sub(r'[*_`]', '', val).strip()
                # Detect category/type column
                elif any(k in key_lower for k in ['category', 'type', 'area', 'topic']):
                    pass  # captured in topics

            # If no clear title found, use first column
            if not paper_title:
                first_val = list(row.values())[0]
                links = extract_markdown_links(first_val)
                if links:
                    paper_title = links[0]["title"]
                    paper_url = links[0]["url"]
                else:
                    paper_title = re.sub(r'[*_`\[\]]', '', first_val).strip()

            if not paper_title or len(paper_title) < 5:
                continue

            # Build chunk text from all columns
            text_parts = [f"Paper: {paper_title}"]
            if paper_url:
                text_parts.append(f"Link: {paper_url}")
            for key, val in row.items():
                clean_val = re.sub(r'[*_`]', '', val).strip()
                if clean_val and clean_val != paper_title:
                    text_parts.append(f"{key}: {clean_val}")
            text = "\n".join(text_parts)

            # Collect ALL URLs from row
            all_urls = extract_urls(text)

            topics = extract_topics(text)

            chunk_id = f"{rel_path}::paper::{i}"
            chunks.append(Chunk(
                id=chunk_id,
                text=text,
                source_file=rel_path,
                section=paper_title[:80],
                category=category,
                title=paper_title,
                url=paper_url or (all_urls[0] if all_urls else None),
                topics=topics,
                content_type="paper",
                all_urls=all_urls,
            ))
    else:
        # Fallback: also try to find inline paper references [Title](url) in non-table content
        pass

    # Also parse non-table sections normally
    sections = re.split(r'\n(?=#{1,3}\s)', content)
    for i, section in enumerate(sections):
        section = section.strip()
        # Skip if it looks like a pure table
        if not section or len(section) < 50:
            continue
        lines = section.split('\n')
        table_lines = sum(1 for l in lines if '|' in l and l.strip().startswith('|'))
        if table_lines > len(lines) * 0.5:
            continue  # Skip table-heavy sections (already parsed above)

        title_match = re.match(r'^(#{1,3})\s+(.+)', section)
        title = title_match.group(2).strip() if title_match else f"Section {i}"
        title = re.sub(r'[*_`\[\]]', '', title).strip()

        all_urls = extract_urls(section)
        named_links = extract_markdown_links(section)
        topics = extract_topics(section)

        chunk_id = f"{rel_path}::{title}::0"
        chunks.append(Chunk(
            id=chunk_id,
            text=section,
            source_file=rel_path,
            section=title,
            category=category,
            title=title,
            url=all_urls[0] if all_urls else None,
            topics=topics,
            content_type="text",
            all_urls=all_urls,
        ))

    return chunks


def normalize_topic(topic: str) -> Optional[str]:
    """Normalize a topic string to its canonical form."""
    lower = topic.lower().strip()
    if lower in TOPIC_ALIASES:
        return TOPIC_ALIASES[lower]
    for canonical in CANONICAL_TOPICS:
        if canonical.lower() == lower:
            return canonical
    return None


_TOPIC_PATTERNS: dict[str, re.Pattern] | None = None
_ALIAS_PATTERNS: dict[str, re.Pattern] | None = None


def _build_topic_patterns():
    """Pre-compile word-boundary regex patterns for topics and aliases."""
    global _TOPIC_PATTERNS, _ALIAS_PATTERNS
    if _TOPIC_PATTERNS is not None:
        return

    _TOPIC_PATTERNS = {}
    for topic in CANONICAL_TOPICS:
        escaped = re.escape(topic)
        _TOPIC_PATTERNS[topic] = re.compile(r'(?<!\w)' + escaped + r'(?!\w)', re.IGNORECASE)

    _ALIAS_PATTERNS = {}
    for alias, canonical in TOPIC_ALIASES.items():
        escaped = re.escape(alias)
        _ALIAS_PATTERNS[(alias, canonical)] = re.compile(r'(?<!\w)' + escaped + r'(?!\w)', re.IGNORECASE)


def extract_topics(text: str) -> list[str]:
    """Extract canonical topics mentioned in a text chunk using word-boundary matching."""
    _build_topic_patterns()
    found = set()

    # Check canonical topics with word boundaries
    for topic, pattern in _TOPIC_PATTERNS.items():
        if pattern.search(text):
            found.add(topic)

    # Check aliases with word boundaries
    for (alias, canonical), pattern in _ALIAS_PATTERNS.items():
        if pattern.search(text):
            found.add(canonical)

    return sorted(found)


def _compute_quality_score(text: str) -> float:
    """
    Score a chunk by information density. Returns 0.0-1.0.
    Low scores indicate boilerplate (pure link lists, tables of contents, etc.).
    """
    lines = text.strip().split('\n')
    if not lines:
        return 0.0

    total_chars = len(text)
    if total_chars == 0:
        return 0.0

    # Count prose lines (not headings, links-only, separators, or table rows)
    prose_chars = 0
    for line in lines:
        stripped = line.strip()
        # Skip headings, separators, empty lines, table rows
        if (not stripped or stripped.startswith('#') or stripped.startswith('|')
                or stripped.startswith('---') or stripped.startswith('===')):
            continue
        # Skip lines that are purely links or bullets with just a link
        if re.match(r'^[-*]\s*\[.+\]\(.+\)\s*$', stripped):
            continue
        prose_chars += len(stripped)

    return min(prose_chars / total_chars, 1.0) if total_chars > 0 else 0.0


def _split_with_overlap(paragraphs: list[str], title: str, rel_path: str,
                         category: str, content_type: str) -> list[Chunk]:
    """
    Split paragraphs into chunks using token counting with overlap.
    Target: ~TARGET_CHUNK_TOKENS tokens per chunk with CHUNK_OVERLAP_TOKENS overlap.
    """
    chunks = []
    current_paras: list[str] = []
    current_tokens = 0
    chunk_idx = 0
    target_chars = _tokens_to_chars_approx(TARGET_CHUNK_TOKENS)
    overlap_chars = _tokens_to_chars_approx(CHUNK_OVERLAP_TOKENS)

    def _emit_chunk(paras: list[str]) -> Chunk | None:
        nonlocal chunk_idx
        text = "\n\n".join(paras).strip()
        if not text or len(text) < MIN_SECTION_CHARS:
            return None
        all_urls = extract_urls(text)
        topics = extract_topics(text)
        chunk_id = f"{rel_path}::{title}::{chunk_idx}"
        chunk_idx += 1
        return Chunk(
            id=chunk_id,
            text=text,
            source_file=rel_path,
            section=title,
            category=category,
            title=title,
            url=all_urls[0] if all_urls else None,
            topics=topics,
            content_type=content_type,
            all_urls=all_urls,
        )

    for para in paragraphs:
        para_len = len(para)
        if current_tokens + para_len > target_chars and current_paras:
            chunk = _emit_chunk(current_paras)
            if chunk:
                chunks.append(chunk)

            # Overlap: keep last paragraph(s) up to overlap_chars
            overlap_paras: list[str] = []
            overlap_len = 0
            for p in reversed(current_paras):
                if overlap_len + len(p) > overlap_chars:
                    break
                overlap_paras.insert(0, p)
                overlap_len += len(p)

            current_paras = overlap_paras
            current_tokens = overlap_len

        current_paras.append(para)
        current_tokens += para_len

    # Emit final chunk
    if current_paras:
        chunk = _emit_chunk(current_paras)
        if chunk:
            chunks.append(chunk)

    return chunks


def parse_markdown_to_chunks(filepath: Path) -> list[Chunk]:
    """Parse a markdown file into semantic chunks (by section)."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    category = infer_category(filepath)
    rel_path = str(filepath.relative_to(REPO_DIR))

    # Detect research table files — they need special handling
    is_research_table = (
        'research_updates' in rel_path
        and ('table' in rel_path.lower() or 'list' in rel_path.lower())
    )
    # Also detect if file has significant table content
    table_lines = sum(1 for l in content.split('\n') if l.strip().startswith('|'))
    if table_lines > 10:
        is_research_table = True

    if is_research_table:
        table_chunks = parse_research_table_to_chunks(filepath)
        if table_chunks:
            return table_chunks

    # Detect content type from path
    content_type = "text"
    if "free_courses" in rel_path:
        content_type = "course"
    elif "research_updates" in rel_path:
        content_type = "paper"
    elif "interview_prep" in rel_path:
        content_type = "interview"

    chunks = []
    # Split by headings (## or ###)
    sections = re.split(r'\n(?=#{1,3}\s)', content)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < MIN_SECTION_CHARS:
            continue

        # Quality filter: skip low-information-density sections
        quality = _compute_quality_score(section)
        if quality < 0.15:
            continue

        # Extract section title
        title_match = re.match(r'^(#{1,3})\s+(.+)', section)
        title = title_match.group(2).strip() if title_match else f"Section {i}"
        # Clean markdown artifacts from title
        title = re.sub(r'[*_`\[\]]', '', title).strip()

        # Token-based chunking with overlap
        target_chars = _tokens_to_chars_approx(TARGET_CHUNK_TOKENS)
        if len(section) > target_chars * 1.5:
            paragraphs = section.split('\n\n')
            sub_chunks = _split_with_overlap(
                paragraphs, title, rel_path, category, content_type
            )
            chunks.extend(sub_chunks)
        else:
            all_urls = extract_urls(section)
            topics = extract_topics(section)
            chunk_id = f"{rel_path}::{title}::0"
            chunks.append(Chunk(
                id=chunk_id,
                text=section,
                source_file=rel_path,
                section=title,
                category=category,
                title=title,
                url=all_urls[0] if all_urls else None,
                topics=topics,
                content_type=content_type,
                all_urls=all_urls,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Step 3: Generate embeddings & store in ChromaDB
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Please export it.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


MAX_BATCH_TOKENS = 7500  # stay safely under the 8192 per-request limit


def generate_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings in token-aware batches."""
    all_embeddings = []
    batch: list[str] = []
    batch_tokens = 0
    batch_num = 0
    total_batches_est = max(1, sum(_count_tokens(t) for t in texts) // MAX_BATCH_TOKENS)

    for text in texts:
        text_tokens = _count_tokens(text)
        # If adding this text would exceed the budget, flush current batch
        if batch and (batch_tokens + text_tokens > MAX_BATCH_TOKENS or len(batch) >= BATCH_SIZE):
            batch_num += 1
            print(f"  ⚡ Embedding batch {batch_num}/~{total_batches_est} ({len(batch)} texts, {batch_tokens} tokens)")
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
            all_embeddings.extend([e.embedding for e in response.data])
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += text_tokens

    # Flush remaining
    if batch:
        batch_num += 1
        print(f"  ⚡ Embedding batch {batch_num}/~{total_batches_est} ({len(batch)} texts, {batch_tokens} tokens)")
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
        all_embeddings.extend([e.embedding for e in response.data])

    return all_embeddings


def store_in_chromadb(chunks: list[Chunk], embeddings: list[list[float]]):
    """Persist chunks and embeddings to ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if present
    try:
        client.delete_collection("genai_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="genai_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    # Deduplicate by ID (keep first occurrence)
    seen_ids: set[str] = set()
    unique_indices: list[int] = []
    for idx, c in enumerate(chunks):
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            unique_indices.append(idx)
    if len(unique_indices) < len(chunks):
        print(f"  ⚠️  Removed {len(chunks) - len(unique_indices)} duplicate chunk IDs")
        chunks = [chunks[i] for i in unique_indices]
        embeddings = [embeddings[i] for i in unique_indices]

    # Add in batches (ChromaDB limit)
    batch = 100
    for i in range(0, len(chunks), batch):
        end = min(i + batch, len(chunks))
        collection.add(
            ids=[c.id for c in chunks[i:end]],
            embeddings=embeddings[i:end],
            documents=[c.text for c in chunks[i:end]],
            metadatas=[{
                "source_file": c.source_file,
                "section": c.section,
                "category": c.category,
                "title": c.title,
                "url": c.url or "",
                "topics": json.dumps(c.topics),
                "content_type": c.content_type,
                "all_urls": json.dumps(c.all_urls),
            } for c in chunks[i:end]],
        )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")


# ---------------------------------------------------------------------------
# Step 4: Build topic graph
# ---------------------------------------------------------------------------

def build_topic_graph(chunks: list[Chunk]) -> dict:
    """
    Build a topic graph from chunk topics.
    Nodes = unique topics, sized by frequency.
    Edges = co-occurrence within the same chunk.
    """
    topic_nodes: dict[str, TopicNode] = {}
    edge_counts: dict[tuple[str, str], int] = {}

    # Categorize topics
    concept_keywords = {"Large Language Models", "Transformers", "Attention Mechanism",
                        "NLP", "Tokenization", "Neural Networks", "Deep Learning",
                        "Embeddings", "Knowledge Graphs", "Diffusion Models"}
    technique_keywords = {"Fine-Tuning", "LoRA", "QLoRA", "PEFT", "RLHF",
                          "Prompt Engineering", "Chain-of-Thought", "Few-Shot Learning",
                          "RAG", "Retrieval-Augmented Generation", "Quantization",
                          "Distillation", "Alignment", "Safety", "Red Teaming",
                          "Agentic RAG", "Evaluation", "Benchmarks"}
    tool_keywords = {"LangChain", "LlamaIndex", "AutoGen", "crewAI",
                     "Hugging Face", "Amazon Bedrock", "Vector Databases"}
    provider_keywords = {"OpenAI", "DeepLearning.AI", "Coursera", "Nvidia",
                         "Google Cloud", "Azure", "GPT", "Claude", "Llama",
                         "Mistral", "Gemini", "BERT"}

    def categorize(topic: str) -> str:
        if topic in concept_keywords:
            return "concept"
        if topic in technique_keywords:
            return "technique"
        if topic in tool_keywords:
            return "tool"
        if topic in provider_keywords:
            return "provider"
        return "concept"

    for chunk in chunks:
        for topic in chunk.topics:
            if topic not in topic_nodes:
                topic_nodes[topic] = TopicNode(
                    id=topic.lower().replace(" ", "_").replace("-", "_"),
                    label=topic,
                    category=categorize(topic),
                )
            topic_nodes[topic].weight += 1
            topic_nodes[topic].resources.append(chunk.id)

        # Co-occurrence edges
        sorted_topics = sorted(chunk.topics)
        for i in range(len(sorted_topics)):
            for j in range(i + 1, len(sorted_topics)):
                key = (sorted_topics[i], sorted_topics[j])
                edge_counts[key] = edge_counts.get(key, 0) + 1

    nodes = [asdict(n) for n in topic_nodes.values()]
    # Limit resources list to avoid huge JSON
    for n in nodes:
        n["resources"] = n["resources"][:50]

    edges = [
        {"source": topic_nodes[s].id, "target": topic_nodes[t].id, "weight": w}
        for (s, t), w in edge_counts.items()
        if w >= 2 and s in topic_nodes and t in topic_nodes  # filter noise
    ]

    # Build hierarchy for tree view
    hierarchy = build_topic_hierarchy(topic_nodes)

    return {
        "nodes": nodes,
        "edges": edges,
        "hierarchy": hierarchy,
    }


def build_topic_hierarchy(topic_nodes: dict[str, TopicNode]) -> dict:
    """Build a hierarchical tree for the sunburst/treemap view."""
    categories = {
        "Foundations": ["Large Language Models", "Transformers", "Attention Mechanism",
                        "NLP", "Tokenization", "Neural Networks", "Deep Learning",
                        "Embeddings", "BERT", "GPT"],
        "Techniques": ["Fine-Tuning", "LoRA", "QLoRA", "PEFT", "RLHF",
                        "Prompt Engineering", "Chain-of-Thought", "Few-Shot Learning",
                        "Quantization", "Distillation", "Alignment", "Safety"],
        "RAG & Retrieval": ["RAG", "Retrieval-Augmented Generation", "Vector Databases",
                             "Knowledge Graphs", "Agentic RAG"],
        "Agents": ["Agents", "Tool Use", "Function Calling", "ReAct",
                    "Multi-Agent Systems", "AutoGen", "crewAI"],
        "Frameworks & Tools": ["LangChain", "LlamaIndex", "Hugging Face",
                                "Amazon Bedrock"],
        "Evaluation & Ops": ["Evaluation", "Benchmarks", "Red Teaming",
                              "LLMOps", "MLOps"],
        "Models": ["GPT", "Claude", "Llama", "Mistral", "Gemini"],
        "Multimodal": ["Diffusion Models", "Image Generation", "Multimodal",
                        "Vision-Language"],
        "Providers & Platforms": ["OpenAI", "DeepLearning.AI", "Coursera",
                                   "Nvidia", "Google Cloud", "Azure"],
    }

    hierarchy = {
        "name": "GenAI Knowledge",
        "children": []
    }

    for cat_name, topics in categories.items():
        children = []
        for topic in topics:
            if topic in topic_nodes:
                node = topic_nodes[topic]
                children.append({
                    "name": node.label,
                    "id": node.id,
                    "value": node.weight,
                    "category": node.category,
                })
        if children:
            hierarchy["children"].append({
                "name": cat_name,
                "children": children,
            })

    return hierarchy


# ---------------------------------------------------------------------------
# Step 5: Export data for the API
# ---------------------------------------------------------------------------

def export_data(chunks: list[Chunk], topic_graph: dict, embeddings: list[list[float]]):
    """Export chunks and graph as JSON for the API to load."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Chunks metadata (without embeddings, those are in ChromaDB)
    chunks_data = [asdict(c) for c in chunks]
    with open(DATA_DIR / "chunks.json", "w") as f:
        json.dump(chunks_data, f, indent=2)
    print(f"✅ Exported {len(chunks_data)} chunks to {DATA_DIR / 'chunks.json'}")

    # Topic graph
    with open(DATA_DIR / "topic_graph.json", "w") as f:
        json.dump(topic_graph, f, indent=2)
    print(f"✅ Exported topic graph ({len(topic_graph['nodes'])} nodes, {len(topic_graph['edges'])} edges)")

    # Lightweight embeddings export for Vercel (fallback if ChromaDB unavailable)
    lightweight = {}
    for chunk, emb in zip(chunks, embeddings):
        lightweight[chunk.id] = emb  # full embeddings for accuracy
    with open(DATA_DIR / "embeddings.json", "w") as f:
        json.dump(lightweight, f)
    print(f"✅ Exported embeddings to {DATA_DIR / 'embeddings.json'}")

    # Embedding versioning metadata
    emb_meta = {
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIM,
        "chunk_count": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(DATA_DIR / "embeddings_meta.json", "w") as f:
        json.dump(emb_meta, f, indent=2)
    print(f"✅ Exported embeddings metadata")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("🚀 Starting ingestion pipeline\n")

    # 1. Clone/pull repo
    clone_or_pull_repo()

    # 2. Parse markdown files
    print("\n📄 Parsing markdown files...")
    md_files = find_markdown_files()
    print(f"   Found {len(md_files)} markdown files")

    all_chunks: list[Chunk] = []
    for filepath in md_files:
        chunks = parse_markdown_to_chunks(filepath)
        all_chunks.extend(chunks)
    print(f"   Parsed into {len(all_chunks)} chunks")
    content_types = {}
    for c in all_chunks:
        content_types[c.content_type] = content_types.get(c.content_type, 0) + 1
    for ct, count in sorted(content_types.items()):
        print(f"     → {ct}: {count}")
    papers_with_urls = sum(1 for c in all_chunks if c.content_type == "paper" and c.url)
    print(f"   Papers with direct links: {papers_with_urls}")

    # 3. Generate embeddings
    print("\n🧠 Generating embeddings...")
    client = get_openai_client()
    texts = [c.text for c in all_chunks]
    embeddings = generate_embeddings(client, texts)

    # 4. Store in ChromaDB
    print("\n💾 Storing in ChromaDB...")
    store_in_chromadb(all_chunks, embeddings)

    # 5. Build topic graph
    print("\n🕸️ Building topic graph...")
    topic_graph = build_topic_graph(all_chunks)
    print(f"   {len(topic_graph['nodes'])} topic nodes, {len(topic_graph['edges'])} edges")

    # 6. Export data
    print("\n📤 Exporting data...")
    export_data(all_chunks, topic_graph, embeddings)

    # 7. Write ingestion metadata
    print("\n📋 Writing ingestion metadata...")
    write_ingestion_metadata(
        file_count=len(md_files),
        chunk_count=len(all_chunks),
        topic_count=len(topic_graph['nodes']),
    )

    print("\n✅ Ingestion complete!")
    print(f"   📁 ChromaDB: {CHROMA_DIR}")
    print(f"   📁 Data exports: {DATA_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest upstream repo into GenAI Explorer data files."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if upstream has changed. Exit 0 = no changes, exit 1 = changes detected.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        sys.exit(check_for_upstream_changes())
    else:
        main()
