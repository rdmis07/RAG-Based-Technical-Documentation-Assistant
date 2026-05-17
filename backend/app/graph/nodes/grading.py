"""
Document Grading Node — LLM-based relevance filter for retrieved chunks.

Responsibilities:
- Assess whether each retrieved chunk is relevant to the user query
- Filter out irrelevant chunks
- Pass filtered_docs downstream
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.models.schemas import RAGState
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

GRADER_SYSTEM_PROMPT = """You are a strict relevance grader for a technical documentation RAG system.
Your job is to decide whether a retrieved document chunk is relevant to the user's query.

Respond ONLY with a JSON object (no markdown):
{
  "relevant": true | false,
  "confidence": 0.0–1.0,
  "reason": "<one sentence explanation>"
}

Be strict: only mark as relevant if the chunk directly addresses the query topic.
Score 'relevant: true' only when confidence ≥ 0.6.
"""


def _grade_single_doc(
    query: str,
    doc_content: str,
    llm_service,
) -> Dict[str, Any]:
    """Grade a single document chunk against the query."""
    human_message = f"""Query: {query}

Document chunk:
\"\"\"
{doc_content[:1500]}
\"\"\"

Is this document chunk relevant to the query? Respond with JSON only."""

    try:
        raw = llm_service.invoke(
            system_prompt=GRADER_SYSTEM_PROMPT,
            human_message=human_message,
            temperature_override=0.0,
        )
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(cleaned)
        return {
            "relevant": bool(result.get("relevant", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", ""),
        }
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Grading LLM call failed (%s) — defaulting to relevant=True.", exc)
        # On error, keep the doc to avoid losing context
        return {"relevant": True, "confidence": 0.5, "reason": "Grading error — kept by default"}


def document_grading_node(state: RAGState) -> Dict[str, Any]:
    """
    Grade each retrieved document chunk for relevance.

    Reads:
        state["retrieved_docs"]
        state["rewritten_query"]  (primary)
        state["original_query"]   (fallback)

    Writes:
        state["filtered_docs"]
    """
    retrieved_docs: List[Dict[str, Any]] = state.get("retrieved_docs", [])
    query = state.get("rewritten_query") or state.get("original_query", "")

    logger.info(
        "Document grading node | query=%.60s | docs_to_grade=%d",
        query,
        len(retrieved_docs),
    )

    if not retrieved_docs:
        logger.warning("No documents to grade.")
        return {"filtered_docs": []}

    llm_service = get_llm_service()
    filtered: List[Dict[str, Any]] = []

    for i, doc in enumerate(retrieved_docs):
        content = doc.get("content", "")
        score = doc.get("score", 0.0)

        # Fast-path: skip grading for very high similarity scores
        if score >= 0.85:
            logger.debug("Doc %d auto-passed (score=%.4f)", i, score)
            filtered.append(doc)
            continue

        # Fast-path: skip grading for very low similarity scores
        if score < 0.20:
            logger.debug("Doc %d auto-rejected (score=%.4f)", i, score)
            continue

        grade = _grade_single_doc(query, content, llm_service)
        logger.debug(
            "Doc %d | relevant=%s | confidence=%.2f | reason=%s",
            i,
            grade["relevant"],
            grade["confidence"],
            grade["reason"],
        )

        if grade["relevant"]:
            doc_copy = dict(doc)
            doc_copy["grade"] = grade
            filtered.append(doc_copy)

    logger.info(
        "Grading complete | kept=%d / %d",
        len(filtered),
        len(retrieved_docs),
    )
    return {"filtered_docs": filtered}
