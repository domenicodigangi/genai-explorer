import os

import pytest


# --- Health endpoint ---

class TestHealth:

    def test_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert isinstance(data["chunks"], int)
        assert isinstance(data["topics"], int)
        assert isinstance(data["has_chroma"], bool)
        assert data["chat_model"] == "gpt-4.1-mini"
        assert data["embedding_model"] == "text-embedding-3-small"

    def test_has_data(self, client):
        data = client.get("/api/health").json()
        assert data["chunks"] > 0, "No chunks loaded — run ingest.py first"
        assert data["topics"] > 0, "No topics loaded — run ingest.py first"


# --- Graph endpoint ---

class TestGraph:

    def test_returns_200(self, client):
        response = client.get("/api/graph")
        assert response.status_code == 200

    def test_structure(self, client):
        data = client.get("/api/graph").json()
        assert "nodes" in data
        assert "edges" in data
        assert "hierarchy" in data
        assert len(data["nodes"]) > 0

    def test_node_fields(self, client):
        data = client.get("/api/graph").json()
        node = data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "category" in node
        assert "weight" in node

    def test_hierarchy_has_categories(self, client):
        data = client.get("/api/graph").json()
        hierarchy = data["hierarchy"]
        assert hierarchy["name"] == "GenAI Knowledge"
        assert len(hierarchy["children"]) == 9


# --- Topics endpoint ---

class TestTopics:

    def test_known_topic(self, client):
        response = client.get("/api/topics/rag")
        assert response.status_code == 200
        data = response.json()
        assert "topic" in data
        assert data["topic"]["id"] == "rag"

    def test_unknown_topic(self, client):
        data = client.get("/api/topics/nonexistent_xyz").json()
        assert "error" in data

    def test_has_connected_topics(self, client):
        data = client.get("/api/topics/rag").json()
        assert "connected_topics" in data
        assert isinstance(data["connected_topics"], list)
        assert len(data["connected_topics"]) > 0

    def test_has_resources(self, client):
        data = client.get("/api/topics/rag").json()
        assert "resources" in data
        assert isinstance(data["resources"], list)
        assert len(data["resources"]) > 0


# --- Search endpoint (requires OPENAI_API_KEY) ---

needs_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class TestSearch:

    def test_rejects_short_query(self, client):
        response = client.get("/api/search", params={"q": "a"})
        assert response.status_code == 422

    @needs_api_key
    def test_returns_results(self, client):
        data = client.get("/api/search", params={"q": "transformers", "top_k": 3}).json()
        assert data["query"] == "transformers"
        assert len(data["results"]) > 0
        assert len(data["results"]) <= 3

    @needs_api_key
    def test_result_has_score(self, client):
        data = client.get("/api/search", params={"q": "fine tuning", "top_k": 1}).json()
        result = data["results"][0]
        assert "score" in result
        assert result["score"] > 0


# --- Chat endpoint (requires OPENAI_API_KEY) ---

class TestChat:

    @needs_api_key
    def test_returns_answer_and_sources(self, client):
        data = client.post("/api/chat", json={
            "message": "What is RAG?",
            "history": [],
        }).json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    @needs_api_key
    def test_source_fields(self, client):
        data = client.post("/api/chat", json={
            "message": "What is RAG?",
            "history": [],
        }).json()
        source = data["sources"][0]
        assert "title" in source
        assert "score" in source
        assert "content_type" in source
        assert "category" in source
