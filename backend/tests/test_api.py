"""
Integration tests for FastAPI endpoints.
Uses TestClient so no real server is needed.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ.setdefault("GROQ_API_KEY", "test-key-placeholder")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma_db")
os.environ.setdefault("CHROMA_COLLECTION", "test_collection")


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with mocked external services."""
    with (
        patch("app.services.llm_service.ChatGroq") as mock_llm_cls,
        patch("app.services.embedding_service.HuggingFaceEmbeddings") as mock_emb_cls,
    ):
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"rewritten_query": "test query", "query_type": "factual", "key_concepts": []}'
        )
        mock_llm_cls.return_value = mock_llm

        # Mock embeddings
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 384
        mock_emb.embed_documents.return_value = [[0.1] * 384]
        mock_emb_cls.return_value = mock_emb

        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        required = ["status", "version", "chroma_connected", "llm_available",
                    "embeddings_loaded", "document_count", "uptime_seconds"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_health_version(self, client):
        response = client.get("/api/v1/health")
        assert response.json()["version"] == "1.0.0"


class TestQueryEndpoint:
    def test_query_requires_body(self, client):
        response = client.post("/api/v1/query")
        assert response.status_code == 422  # Validation error

    def test_query_empty_string_rejected(self, client):
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_too_long_rejected(self, client):
        response = client.post("/api/v1/query", json={"query": "x" * 2001})
        assert response.status_code == 422

    def test_query_response_schema(self, client):
        """Response must contain all required fields."""
        with patch("app.api.routes.query.build_rag_graph") as mock_graph:
            mock_rag = MagicMock()
            mock_rag.run.return_value = {
                "original_query": "test",
                "rewritten_query": "test query improved",
                "retrieved_docs": [{"content": "doc", "metadata": {}, "score": 0.9, "chunk_id": "abc"}],
                "filtered_docs": [{"content": "doc", "metadata": {}, "score": 0.9, "chunk_id": "abc"}],
                "generated_answer": "This is the answer.",
                "retry_count": 0,
                "max_retries": 2,
                "citations": [],
                "query_type": "factual",
                "is_hallucination": False,
                "hallucination_score": 0.05,
                "session_id": None,
                "error": None,
            }
            mock_graph.return_value = mock_rag

            response = client.post("/api/v1/query", json={"query": "What is FastAPI?"})
            assert response.status_code == 200

            data = response.json()
            required_fields = [
                "query", "rewritten_query", "answer", "citations",
                "query_type", "retry_count", "retrieved_doc_count",
                "filtered_doc_count", "is_hallucination", "hallucination_score",
                "processing_time_ms",
            ]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"


class TestIngestEndpoint:
    def test_ingest_nonexistent_path(self, client):
        response = client.post("/api/v1/ingest", json={
            "source_path": "/nonexistent/path/that/does/not/exist"
        })
        assert response.status_code in (404, 500)

    def test_ingest_requires_source_path(self, client):
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422

    def test_ingest_chunk_size_validation(self, client):
        """chunk_size must be 100-4000."""
        response = client.post("/api/v1/ingest", json={
            "source_path": "./data",
            "chunk_size": 50  # Below minimum
        })
        assert response.status_code == 422


class TestDocumentsEndpoint:
    def test_list_documents_returns_200(self, client):
        response = client.get("/api/v1/documents")
        assert response.status_code == 200

    def test_list_documents_returns_list(self, client):
        response = client.get("/api/v1/documents")
        assert isinstance(response.json(), list)

    def test_document_count_endpoint(self, client):
        response = client.get("/api/v1/documents/count")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "collection" in data


class TestFeedbackEndpoint:
    def test_feedback_valid_request(self, client):
        response = client.post("/api/v1/feedback", json={
            "session_id": "test-session-001",
            "query": "What is FastAPI?",
            "answer": "FastAPI is a web framework...",
            "rating": 5,
            "is_helpful": True,
            "comment": "Very helpful!",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "feedback_id" in data

    def test_feedback_rating_validation(self, client):
        """Rating must be 1-5."""
        response = client.post("/api/v1/feedback", json={
            "session_id": "test-session-001",
            "query": "test",
            "answer": "test",
            "rating": 6,  # Invalid
            "is_helpful": True,
        })
        assert response.status_code == 422

    def test_feedback_stats_endpoint(self, client):
        response = client.get("/api/v1/feedback/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "average_rating" in data
        assert "helpful_pct" in data
