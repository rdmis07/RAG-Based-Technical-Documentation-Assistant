"""
Retrieval Node — searches ChromaDB for relevant document chunks.

Responsibilities:
- Use the (possibly rewritten) query for semantic similarity search
- Return top-k chunks with metadata and scores
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from app.models.schemas import RAGState
from app.services.vector_store_service import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton so we don't rebuild on every invocation
_vector_store_service: VectorStoreService | None = None


def _get_vector_store() -> VectorStoreService:
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
        _vector_store_service.initialize()
    return _vector_store_service


def retrieval_node(state: RAGState) -> Dict[str, Any]:
    """
    Retrieve relevant document chunks from ChromaDB.

    Reads:
        state["rewritten_query"]
        state["original_query"]   (fallback)

    Writes:
        state["retrieved_docs"]
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    logger.info("Retrieval node | query=%.80s | top_k=%d", query, top_k)

    try:
        vs = _get_vector_store()
        docs = vs.similarity_search_with_scores(query=query, k=top_k)
        logger.info("Retrieved %d document chunks.", len(docs))
        return {"retrieved_docs": docs}

    except Exception as exc:
        logger.error("Retrieval node failed: %s", exc, exc_info=True)
        return {"retrieved_docs": []}
