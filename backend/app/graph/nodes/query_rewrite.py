"""
Query Rewrite Node — triggered when retrieved docs are irrelevant.

Responsibilities:
- Generate a semantically different, more specific rewrite of the query
- Increment retry_count
- The graph will loop back to the retrieval node after this
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from app.models.schemas import RAGState
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a query optimization expert for technical documentation search.
The previous retrieval attempt returned no relevant results.

Your goal is to reformulate the query to improve retrieval.

Strategies:
1. Use different technical terminology or synonyms
2. Break down compound questions into the most specific sub-question
3. Add domain context (e.g. "in Python FastAPI" or "in Docker")
4. Focus on the core concept rather than the surrounding context

Respond ONLY with a JSON object (no markdown):
{
  "rewritten_query": "<new query variant>",
  "strategy_used": "<brief description of reformulation strategy>"
}
"""


def query_rewrite_node(state: RAGState) -> Dict[str, Any]:
    """
    Rewrite the query after a failed retrieval attempt.

    Reads:
        state["original_query"]
        state["rewritten_query"]
        state["retry_count"]

    Writes:
        state["rewritten_query"]   (new variant)
        state["retry_count"]       (incremented)
        state["retrieved_docs"]    (cleared for fresh retrieval)
        state["filtered_docs"]     (cleared)
    """
    original_query = state.get("original_query", "")
    current_rewrite = state.get("rewritten_query", original_query)
    retry_count = state.get("retry_count", 0) + 1

    logger.info(
        "Query rewrite node | attempt=%d | current=%.80s",
        retry_count,
        current_rewrite,
    )

    human_message = f"""Original query: {original_query}

Previous (failed) query variant: {current_rewrite}
Retry attempt: {retry_count}

Generate a better query variant for retrieval. Respond with JSON only."""

    try:
        llm_service = get_llm_service()
        raw = llm_service.invoke(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            human_message=human_message,
            temperature_override=0.3,
        )
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(cleaned)
        new_query = parsed.get("rewritten_query", current_rewrite)
        strategy = parsed.get("strategy_used", "unknown")

        logger.info(
            "Query rewritten | new=%.80s | strategy=%s", new_query, strategy
        )

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Query rewrite LLM call failed (%s) — using fallback.", exc)
        # Simple fallback: append context keywords
        new_query = f"{original_query} technical documentation explanation"

    return {
        "rewritten_query": new_query,
        "retry_count": retry_count,
        "retrieved_docs": [],
        "filtered_docs": [],
    }
