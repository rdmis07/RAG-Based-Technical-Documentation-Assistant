"""
Unit tests for individual LangGraph nodes.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-key-placeholder")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma_db")


class TestQueryAnalysisNode:
    def test_node_returns_required_keys(self):
        """query_analysis_node must return rewritten_query, query_type, retry_count."""
        from app.graph.nodes.query_analysis import query_analysis_node

        with patch("app.graph.nodes.query_analysis.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = (
                '{"rewritten_query": "better query", "query_type": "conceptual", "key_concepts": ["test"]}'
            )
            mock_llm_fn.return_value = mock_service

            state = {
                "original_query": "What is Python?",
                "max_retries": 2,
                "conversation_history": [],
            }
            result = query_analysis_node(state)

        assert "rewritten_query" in result
        assert "query_type" in result
        assert "retry_count" in result
        assert result["retry_count"] == 0

    def test_node_falls_back_on_json_error(self):
        """Should use original query if LLM returns invalid JSON."""
        from app.graph.nodes.query_analysis import query_analysis_node

        with patch("app.graph.nodes.query_analysis.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = "not valid json at all"
            mock_llm_fn.return_value = mock_service

            state = {
                "original_query": "fallback query",
                "max_retries": 2,
                "conversation_history": [],
            }
            result = query_analysis_node(state)

        assert result["rewritten_query"] == "fallback query"
        assert result["query_type"] == "factual"


class TestDocumentGradingNode:
    def test_high_score_docs_auto_pass(self):
        """Docs with score ≥ 0.85 should be kept without LLM call."""
        from app.graph.nodes.grading import document_grading_node

        state = {
            "retrieved_docs": [
                {"content": "FastAPI is a web framework", "metadata": {}, "score": 0.92, "chunk_id": "a1"},
                {"content": "Another relevant doc", "metadata": {}, "score": 0.87, "chunk_id": "a2"},
            ],
            "rewritten_query": "What is FastAPI?",
            "original_query": "What is FastAPI?",
        }

        # No LLM calls should be made for high-score docs
        with patch("app.graph.nodes.grading.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_llm_fn.return_value = mock_service

            result = document_grading_node(state)

            # LLM should NOT have been called for high-score docs
            mock_service.invoke.assert_not_called()

        assert len(result["filtered_docs"]) == 2

    def test_low_score_docs_auto_reject(self):
        """Docs with score < 0.20 should be rejected without LLM call."""
        from app.graph.nodes.grading import document_grading_node

        state = {
            "retrieved_docs": [
                {"content": "Unrelated content", "metadata": {}, "score": 0.10, "chunk_id": "b1"},
            ],
            "rewritten_query": "What is FastAPI?",
            "original_query": "What is FastAPI?",
        }

        with patch("app.graph.nodes.grading.get_llm_service"):
            result = document_grading_node(state)

        assert len(result["filtered_docs"]) == 0

    def test_empty_docs_returns_empty(self):
        """Empty retrieved_docs should produce empty filtered_docs."""
        from app.graph.nodes.grading import document_grading_node

        state = {
            "retrieved_docs": [],
            "rewritten_query": "test",
            "original_query": "test",
        }

        result = document_grading_node(state)
        assert result["filtered_docs"] == []


class TestQueryRewriteNode:
    def test_increments_retry_count(self):
        from app.graph.nodes.query_rewrite import query_rewrite_node

        with patch("app.graph.nodes.query_rewrite.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = (
                '{"rewritten_query": "new variant", "strategy_used": "synonym expansion"}'
            )
            mock_llm_fn.return_value = mock_service

            state = {
                "original_query": "How to use FastAPI?",
                "rewritten_query": "FastAPI usage guide",
                "retry_count": 0,
            }
            result = query_rewrite_node(state)

        assert result["retry_count"] == 1
        assert result["retrieved_docs"] == []
        assert result["filtered_docs"] == []

    def test_clears_previous_docs(self):
        from app.graph.nodes.query_rewrite import query_rewrite_node

        with patch("app.graph.nodes.query_rewrite.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = (
                '{"rewritten_query": "refreshed query", "strategy_used": "test"}'
            )
            mock_llm_fn.return_value = mock_service

            state = {
                "original_query": "original",
                "rewritten_query": "first attempt",
                "retry_count": 1,
            }
            result = query_rewrite_node(state)

        assert result["retrieved_docs"] == []
        assert result["filtered_docs"] == []


class TestGenerationNode:
    def test_generates_with_docs(self):
        from app.graph.nodes.generation import generation_node

        with patch("app.graph.nodes.generation.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = "FastAPI is a modern Python web framework."
            mock_llm_fn.return_value = mock_service

            state = {
                "filtered_docs": [
                    {
                        "content": "FastAPI is a modern Python web framework.",
                        "metadata": {"source": "fastapi.md", "title": "FastAPI Guide"},
                        "score": 0.95,
                        "chunk_id": "c1",
                    }
                ],
                "rewritten_query": "What is FastAPI?",
                "original_query": "What is FastAPI?",
                "query_type": "conceptual",
                "conversation_history": [],
            }

            result = generation_node(state)

        assert "generated_answer" in result
        assert len(result["generated_answer"]) > 0
        assert "citations" in result

    def test_fallback_when_no_docs(self):
        from app.graph.nodes.generation import generation_node

        with patch("app.graph.nodes.generation.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = "Based on general knowledge..."
            mock_llm_fn.return_value = mock_service

            state = {
                "filtered_docs": [],
                "rewritten_query": "obscure question",
                "original_query": "obscure question",
                "query_type": "factual",
                "conversation_history": [],
            }

            result = generation_node(state)

        assert "generated_answer" in result
        assert result["citations"] == []


class TestHallucinationCheckNode:
    def test_no_docs_skips_check(self):
        from app.graph.nodes.hallucination_check import hallucination_check_node

        state = {
            "generated_answer": "Some answer",
            "filtered_docs": [],
            "original_query": "test",
        }

        result = hallucination_check_node(state)
        assert result["is_hallucination"] is False
        assert result["hallucination_score"] == 0.0

    def test_grounded_answer_detected(self):
        from app.graph.nodes.hallucination_check import hallucination_check_node

        with patch("app.graph.nodes.hallucination_check.get_llm_service") as mock_llm_fn:
            mock_service = MagicMock()
            mock_service.invoke.return_value = (
                '{"is_hallucination": false, "hallucination_score": 0.05, '
                '"grounded_claims": ["FastAPI uses asyncio"], '
                '"unsupported_claims": [], "reasoning": "Fully grounded"}'
            )
            mock_llm_fn.return_value = mock_service

            state = {
                "generated_answer": "FastAPI uses asyncio for async support.",
                "filtered_docs": [
                    {"content": "FastAPI uses asyncio under the hood.", "metadata": {}, "score": 0.9, "chunk_id": "d1"}
                ],
                "original_query": "How does FastAPI handle async?",
            }

            result = hallucination_check_node(state)

        assert result["is_hallucination"] is False
        assert result["hallucination_score"] < 0.5


class TestUtilityFunctions:
    def test_generate_chunk_id_deterministic(self):
        from app.utils.helpers import generate_chunk_id

        id1 = generate_chunk_id("file.md", 0, "content")
        id2 = generate_chunk_id("file.md", 0, "content")
        assert id1 == id2

    def test_generate_chunk_id_unique_per_chunk(self):
        from app.utils.helpers import generate_chunk_id

        id1 = generate_chunk_id("file.md", 0, "content")
        id2 = generate_chunk_id("file.md", 1, "content")
        assert id1 != id2

    def test_truncate_text_short_text(self):
        from app.utils.helpers import truncate_text

        text = "Short text"
        assert truncate_text(text, max_length=100) == text

    def test_truncate_text_long_text(self):
        from app.utils.helpers import truncate_text

        text = "x" * 500
        result = truncate_text(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_format_docs_empty(self):
        from app.utils.helpers import format_docs_for_prompt

        result = format_docs_for_prompt([])
        assert "No relevant documents" in result

    def test_extract_citations_deduplication(self):
        from app.utils.helpers import extract_citations

        docs = [
            {"content": "content A", "metadata": {"source": "file.md"}, "score": 0.9, "chunk_id": "x1"},
            {"content": "content A", "metadata": {"source": "file.md"}, "score": 0.9, "chunk_id": "x1"},  # Duplicate
        ]
        citations = extract_citations(docs)
        assert len(citations) == 1

    def test_sanitize_collection_name(self):
        from app.utils.helpers import sanitize_collection_name

        assert sanitize_collection_name("My Collection!") == "My_Collection"
        assert len(sanitize_collection_name("a")) >= 3
        assert len(sanitize_collection_name("x" * 100)) <= 63
