"""
Hallucination Check Node (Bonus feature) — verifies the generated answer
is grounded in the retrieved documents rather than fabricated.

Responsibilities:
- Compare the generated answer against the filtered docs
- Score the likelihood of hallucination (0=grounded, 1=hallucinated)
- Flag the response if hallucination is detected
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.models.schemas import RAGState
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

HALLUCINATION_SYSTEM_PROMPT = """You are a factual accuracy auditor for an AI assistant.
Your task: determine whether the AI-generated answer is GROUNDED in the provided source documents, or contains hallucinated/fabricated information.

Respond ONLY with a JSON object (no markdown):
{
  "is_hallucination": true | false,
  "hallucination_score": 0.0–1.0,
  "grounded_claims": ["<claim supported by docs>"],
  "unsupported_claims": ["<claim NOT found in docs>"],
  "reasoning": "<brief explanation>"
}

Scoring guide:
- 0.0: Fully grounded — every claim is supported by the source docs
- 0.3: Mostly grounded — minor extrapolations
- 0.6: Partially hallucinated — several unsupported claims
- 1.0: Fully hallucinated — answer not supported by docs

Only flag is_hallucination=true when hallucination_score ≥ 0.5.
"""

# Threshold above which we flag as hallucination
HALLUCINATION_THRESHOLD = float("0.5")


def hallucination_check_node(state: RAGState) -> Dict[str, Any]:
    """
    Check the generated answer for hallucinations against the source docs.

    Reads:
        state["generated_answer"]
        state["filtered_docs"]
        state["original_query"]

    Writes:
        state["is_hallucination"]
        state["hallucination_score"]
    """
    generated_answer = state.get("generated_answer", "")
    filtered_docs: List[Dict[str, Any]] = state.get("filtered_docs", [])
    original_query = state.get("original_query", "")

    logger.info("Hallucination check node | answer_length=%d", len(generated_answer))

    # If no docs were used, the answer is from general knowledge — not a hallucination in the RAG sense
    if not filtered_docs:
        logger.info("No source docs — skipping hallucination check.")
        return {"is_hallucination": False, "hallucination_score": 0.0}

    # Build condensed context from source docs
    doc_snippets = "\n\n".join(
        f"[Doc {i+1}]: {doc.get('content', '')[:600]}"
        for i, doc in enumerate(filtered_docs[:5])
    )

    human_message = f"""Original question: {original_query}

Source documents:
{doc_snippets}

Generated answer:
\"\"\"
{generated_answer[:2000]}
\"\"\"

Is this answer grounded in the source documents? Respond with JSON only."""

    try:
        llm_service = get_llm_service()
        raw = llm_service.invoke(
            system_prompt=HALLUCINATION_SYSTEM_PROMPT,
            human_message=human_message,
            temperature_override=0.0,
        )
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(cleaned)

        is_hallucination = bool(result.get("is_hallucination", False))
        hallucination_score = float(result.get("hallucination_score", 0.0))
        unsupported = result.get("unsupported_claims", [])
        reasoning = result.get("reasoning", "")

        logger.info(
            "Hallucination check | is_hallucination=%s | score=%.2f | unsupported=%d | reason=%s",
            is_hallucination,
            hallucination_score,
            len(unsupported),
            reasoning,
        )

        return {
            "is_hallucination": is_hallucination,
            "hallucination_score": hallucination_score,
        }

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Hallucination check failed (%s) — defaulting to grounded.", exc)
        return {"is_hallucination": False, "hallucination_score": 0.0}
