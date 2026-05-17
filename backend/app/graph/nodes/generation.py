"""
Generation Node — synthesises a grounded answer from filtered document chunks.

Responsibilities:
- Build a context-rich prompt from filtered docs
- Generate an answer citing source documents
- Extract citation metadata
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.models.schemas import RAGState
from app.services.llm_service import get_llm_service
from app.utils.helpers import extract_citations, format_docs_for_prompt
from app.utils.logger import get_logger

logger = get_logger(__name__)

GENERATION_SYSTEM_PROMPT = """You are an expert technical documentation assistant.
Your role is to provide accurate, well-structured answers based ONLY on the provided documentation excerpts.

Guidelines:
1. Answer the question directly and comprehensively using information from the provided documents.
2. Use technical precision — preserve code syntax, commands, configuration values exactly.
3. Structure complex answers with clear sections, bullet points, or numbered steps when appropriate.
4. Always cite which document(s) your answer is based on using [Document N] notation.
5. If the documents don't fully answer the question, say what IS covered and what's missing.
6. Never fabricate information not present in the documents.
7. If code examples exist in the documents, include them in your answer.

Format your answer in Markdown for readability.
"""

FALLBACK_GENERATION_SYSTEM_PROMPT = """You are a helpful technical assistant.
The documentation search did not return relevant results for this query.
Provide a helpful response based on your general technical knowledge, but clearly state:
'Note: This answer is based on general knowledge as no matching documentation was found.'
Keep the answer concise and accurate.
"""


def generation_node(state: RAGState) -> Dict[str, Any]:
    """
    Generate a grounded answer from filtered documents.

    Reads:
        state["filtered_docs"]
        state["rewritten_query"]
        state["original_query"]
        state["query_type"]
        state["conversation_history"]  (optional)

    Writes:
        state["generated_answer"]
        state["citations"]
    """
    filtered_docs: List[Dict[str, Any]] = state.get("filtered_docs", [])
    query = state.get("rewritten_query") or state.get("original_query", "")
    original_query = state.get("original_query", query)
    query_type = state.get("query_type", "factual")
    conversation_history = state.get("conversation_history", [])

    logger.info(
        "Generation node | query=%.60s | docs=%d | type=%s",
        query,
        len(filtered_docs),
        query_type,
    )

    llm_service = get_llm_service()
    citations: List[Dict[str, Any]] = []

    if not filtered_docs:
        # Fallback: generate without context
        logger.warning("No filtered docs — generating fallback response.")
        human_message = f"Question: {original_query}"
        answer = llm_service.invoke(
            system_prompt=FALLBACK_GENERATION_SYSTEM_PROMPT,
            human_message=human_message,
        )
        return {
            "generated_answer": answer,
            "citations": [],
        }

    # Format documents for the prompt
    formatted_docs = format_docs_for_prompt(filtered_docs)
    citations = extract_citations(filtered_docs)

    # Build type-specific instruction
    type_hint = {
        "factual": "Provide a direct, precise answer.",
        "conceptual": "Explain the concept clearly with examples from the docs.",
        "procedural": "Provide step-by-step instructions based on the documentation.",
        "comparison": "Compare the items systematically using a table or structured list.",
        "troubleshooting": "Diagnose the issue and provide solution steps from the docs.",
    }.get(query_type, "Answer the question comprehensively.")

    human_message = f"""Question: {original_query}

{type_hint}

Documentation context:
{formatted_docs}

Remember to cite [Document N] for each piece of information you use.
"""

    # Use conversation-aware invocation when history exists
    try:
        if conversation_history:
            answer = llm_service.invoke_with_history(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                conversation_history=conversation_history,
                current_message=human_message,
            )
        else:
            answer = llm_service.invoke(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                human_message=human_message,
            )
    except Exception as exc:
        logger.error("Generation LLM call failed: %s", exc, exc_info=True)
        answer = (
            "I encountered an error while generating the answer. "
            "Please try again or rephrase your question."
        )

    logger.info("Answer generated | length=%d chars | citations=%d", len(answer), len(citations))

    return {
        "generated_answer": answer,
        "citations": citations,
    }
