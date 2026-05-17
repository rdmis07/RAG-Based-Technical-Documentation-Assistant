"""
RAG Graph — assembles all nodes into a self-corrective LangGraph StateGraph.

Pipeline flow:
  query_analysis
      ↓
  retrieval
      ↓
  document_grading
      ↓ (conditional)
  ┌── relevant? ──────────────────────────────┐
  │   YES → generation → hallucination_check → END
  │   NO  → retry_count < max_retries?
  │             YES → query_rewrite → retrieval (loop)
  └─────────── NO  → generation (fallback) → END
"""
from __future__ import annotations

import os
from typing import Any, Dict, Literal

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.generation import generation_node
from app.graph.nodes.grading import document_grading_node
from app.graph.nodes.hallucination_check import hallucination_check_node
from app.graph.nodes.query_analysis import query_analysis_node
from app.graph.nodes.query_rewrite import query_rewrite_node
from app.graph.nodes.retrieval import retrieval_node
from app.models.schemas import RAGState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────────────────────


def route_after_grading(
    state: RAGState,
) -> Literal["generation", "query_rewrite", "generation_fallback"]:
    """
    Decide what happens after document grading:

    - If filtered_docs is non-empty  → go to generation
    - If retry_count < max_retries   → rewrite query and retry retrieval
    - Otherwise                      → generate with no context (fallback)
    """
    filtered_docs = state.get("filtered_docs", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if filtered_docs:
        logger.debug("Routing → generation (%d relevant docs)", len(filtered_docs))
        return "generation"

    if retry_count < max_retries:
        logger.debug(
            "Routing → query_rewrite (retry %d/%d)", retry_count + 1, max_retries
        )
        return "query_rewrite"

    logger.warning(
        "Max retries reached (%d) with no relevant docs — routing to fallback generation.",
        max_retries,
    )
    return "generation_fallback"


# ──────────────────────────────────────────────────────────────
# Fallback generation node (no context)
# ──────────────────────────────────────────────────────────────


def generation_fallback_node(state: RAGState) -> Dict[str, Any]:
    """
    Thin wrapper: clears filtered_docs so generation_node uses its fallback path.
    """
    return {**generation_node({**state, "filtered_docs": []})}


# ──────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────


class RAGGraph:
    """
    Compiled self-corrective RAG StateGraph.
    Call `.run(query, **kwargs)` to execute the full pipeline.
    """

    def __init__(self) -> None:
        self._graph = None

    def build(self) -> "RAGGraph":
        """Construct and compile the LangGraph StateGraph."""
        logger.info("Building RAG StateGraph…")

        workflow = StateGraph(RAGState)

        # ── Register nodes ────────────────────────────────────
        workflow.add_node("query_analysis", query_analysis_node)
        workflow.add_node("retrieval", retrieval_node)
        workflow.add_node("document_grading", document_grading_node)
        workflow.add_node("generation", generation_node)
        workflow.add_node("generation_fallback", generation_fallback_node)
        workflow.add_node("query_rewrite", query_rewrite_node)
        workflow.add_node("hallucination_check", hallucination_check_node)

        # ── Unconditional edges ───────────────────────────────
        workflow.add_edge(START, "query_analysis")
        workflow.add_edge("query_analysis", "retrieval")
        workflow.add_edge("retrieval", "document_grading")

        # ── Conditional routing after grading ─────────────────
        workflow.add_conditional_edges(
            "document_grading",
            route_after_grading,
            {
                "generation": "generation",
                "query_rewrite": "query_rewrite",
                "generation_fallback": "generation_fallback",
            },
        )

        # ── Retry loop: rewrite → retrieval ───────────────────
        workflow.add_edge("query_rewrite", "retrieval")

        # ── After generation → hallucination check → END ──────
        workflow.add_edge("generation", "hallucination_check")
        workflow.add_edge("generation_fallback", "hallucination_check")
        workflow.add_edge("hallucination_check", END)

        self._graph = workflow.compile()
        logger.info("RAG StateGraph compiled successfully.")
        return self

    def run(
        self,
        query: str,
        session_id: str | None = None,
        max_retries: int = 2,
        conversation_history: list | None = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a single query.

        Args:
            query: User's natural language question.
            session_id: Optional session identifier for memory.
            max_retries: Maximum retrieval retry attempts.
            conversation_history: List of {'role', 'content'} dicts.

        Returns:
            Final RAGState dict with generated_answer, citations, etc.
        """
        if self._graph is None:
            self.build()

        initial_state: RAGState = {
            "original_query": query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "filtered_docs": [],
            "generated_answer": "",
            "retry_count": 0,
            "max_retries": max_retries,
            "citations": [],
            "query_type": "factual",
            "is_hallucination": False,
            "hallucination_score": 0.0,
            "conversation_history": conversation_history or [],
            "session_id": session_id,
            "error": None,
        }

        logger.info("RAG pipeline started | query=%.80s | session=%s", query, session_id)

        try:
            final_state = self._graph.invoke(initial_state)
            logger.info(
                "RAG pipeline complete | retry=%d | docs=%d | hallucination=%s",
                final_state.get("retry_count", 0),
                len(final_state.get("filtered_docs", [])),
                final_state.get("is_hallucination"),
            )
            return final_state

        except Exception as exc:
            logger.error("RAG pipeline error: %s", exc, exc_info=True)
            return {
                **initial_state,
                "generated_answer": "An internal error occurred. Please try again.",
                "error": str(exc),
            }

    def get_mermaid_diagram(self) -> str:
        """Return a Mermaid diagram of the compiled graph (for debugging)."""
        if self._graph is None:
            self.build()
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return "Unable to generate diagram."


def build_rag_graph() -> RAGGraph:
    """Factory function — builds and returns a compiled RAGGraph."""
    return RAGGraph().build()
