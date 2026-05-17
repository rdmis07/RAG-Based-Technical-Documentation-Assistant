"""
Query Analysis Node — the entry point of the RAG graph.

Responsibilities:
- Detect query type (factual, conceptual, procedural, comparison, troubleshooting)
- Rewrite/expand the query for better retrieval
- Initialise retry tracking fields
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from app.models.schemas import RAGState
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert technical documentation analyst.
Your task is to analyze a user query and prepare it for optimal retrieval from a technical documentation database.

You must respond with ONLY a valid JSON object (no markdown, no explanation) with exactly these fields:
{
  "rewritten_query": "<improved version of the query optimized for semantic similarity search>",
  "query_type": "<one of: factual | conceptual | procedural | comparison | troubleshooting>",
  "key_concepts": ["<concept1>", "<concept2>"]
}

Query type definitions:
- factual: Asking for a specific fact or value (e.g. "What is the default port?")
- conceptual: Asking to understand a concept (e.g. "What is dependency injection?")
- procedural: Asking how to do something step-by-step (e.g. "How do I install X?")
- comparison: Asking to compare two or more things (e.g. "What is the difference between X and Y?")
- troubleshooting: Asking about an error or debugging (e.g. "Why does X fail?")
"""


def query_analysis_node(state: RAGState) -> Dict[str, Any]:
    """
    Analyse and rewrite the user query.

    Reads:
        state["original_query"]
        state["max_retries"]      (default 2)
        state["conversation_history"]  (optional)

    Writes:
        state["rewritten_query"]
        state["query_type"]
        state["retry_count"]      (reset to 0)
    """
    original_query = state.get("original_query", "")
    max_retries = state.get("max_retries", 2)
    conversation_history = state.get("conversation_history", [])

    logger.info("Query analysis node | query=%.80s", original_query)

    # Build context from conversation history
    history_context = ""
    if conversation_history:
        last_turns = conversation_history[-4:]  # last 2 exchanges
        history_context = "\n\nConversation history (most recent):\n"
        for turn in last_turns:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_context += f"{role}: {content}\n"

    human_message = f"""Analyze this query and respond with JSON only:

Query: {original_query}{history_context}"""

    try:
        llm_service = get_llm_service()
        raw_response = llm_service.invoke(
            system_prompt=SYSTEM_PROMPT,
            human_message=human_message,
            temperature_override=0.0,
        )

        # Strip any accidental markdown fences
        cleaned = re.sub(r"```(?:json)?|```", "", raw_response).strip()
        parsed = json.loads(cleaned)

        rewritten_query = parsed.get("rewritten_query", original_query)
        query_type = parsed.get("query_type", "factual")
        key_concepts = parsed.get("key_concepts", [])

        logger.info(
            "Query analysed | type=%s | rewritten=%.80s | concepts=%s",
            query_type,
            rewritten_query,
            key_concepts,
        )

    except (json.JSONDecodeError, KeyError, Exception) as exc:
        logger.warning("Query analysis LLM call failed (%s) — using original query.", exc)
        rewritten_query = original_query
        query_type = "factual"

    return {
        "rewritten_query": rewritten_query,
        "query_type": query_type,
        "retry_count": 0,
        "max_retries": max_retries,
    }
