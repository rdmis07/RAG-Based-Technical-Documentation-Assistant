"""
POST /query — main RAG query endpoint.
Supports both standard and streaming responses.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import Citation, QueryRequest, QueryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Conversation memory store (session_id → history list)
# In production, replace with Redis or a database
_session_memory: dict[str, list] = {}
MAX_HISTORY_TURNS = 10  # Keep last N turns per session


def _get_or_create_session(session_id: str | None) -> tuple[str, list]:
    """Return (session_id, conversation_history)."""
    sid = session_id or str(uuid.uuid4())
    history = _session_memory.get(sid, [])
    return sid, history


def _update_session_memory(
    session_id: str,
    user_query: str,
    assistant_answer: str,
) -> None:
    """Append the latest exchange to the session history."""
    if session_id not in _session_memory:
        _session_memory[session_id] = []

    history = _session_memory[session_id]
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": assistant_answer})

    # Trim to last MAX_HISTORY_TURNS exchanges (each exchange = 2 entries)
    if len(history) > MAX_HISTORY_TURNS * 2:
        _session_memory[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


@router.post("/query", response_model=QueryResponse, summary="Query the RAG pipeline")
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
) -> Any:
    """
    Submit a natural-language question to the RAG pipeline.

    The pipeline:
    1. Analyses and rewrites your query
    2. Retrieves relevant documentation chunks from ChromaDB
    3. Grades chunks for relevance (LLM-based filter)
    4. If no relevant chunks → rewrites query and retries (up to max_retries)
    5. Generates a grounded answer with citations
    6. Checks for hallucinations
    7. Returns the answer with source citations

    Set `stream: true` to receive a streaming response instead.
    """
    if request.stream:
        return await _stream_query(request)

    return await _standard_query(request, background_tasks)


async def _standard_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
) -> QueryResponse:
    """Execute the RAG pipeline and return a complete QueryResponse."""
    from app.graph.rag_graph import build_rag_graph

    start_time = time.time()
    session_id, conversation_history = _get_or_create_session(request.session_id)

    logger.info(
        "POST /query | session=%s | query=%.80s",
        session_id,
        request.query,
    )

    try:
        # Build and run the RAG graph
        rag = build_rag_graph()
        final_state = rag.run(
            query=request.query,
            session_id=session_id,
            max_retries=request.max_retries,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        logger.error("RAG pipeline exception: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}")

    processing_ms = round((time.time() - start_time) * 1000, 2)

    # Persist conversation turn in background
    background_tasks.add_task(
        _update_session_memory,
        session_id,
        request.query,
        final_state.get("generated_answer", ""),
    )

    # Build Citation objects
    raw_citations = final_state.get("citations", [])
    citations = [Citation(**c) for c in raw_citations]

    return QueryResponse(
        query=request.query,
        rewritten_query=final_state.get("rewritten_query", request.query),
        answer=final_state.get("generated_answer", "No answer generated."),
        citations=citations,
        query_type=final_state.get("query_type", "factual"),
        retry_count=final_state.get("retry_count", 0),
        retrieved_doc_count=len(final_state.get("retrieved_docs", [])),
        filtered_doc_count=len(final_state.get("filtered_docs", [])),
        is_hallucination=final_state.get("is_hallucination", False),
        hallucination_score=final_state.get("hallucination_score", 0.0),
        session_id=session_id,
        processing_time_ms=processing_ms,
    )


async def _stream_query(request: QueryRequest) -> StreamingResponse:
    """
    Execute the RAG pipeline and stream the generation output token-by-token.

    The pipeline runs normally up to the generation step, then streams
    the LLM output as Server-Sent Events (SSE).
    """
    from app.graph.nodes.query_analysis import query_analysis_node
    from app.graph.nodes.retrieval import retrieval_node
    from app.graph.nodes.grading import document_grading_node
    from app.graph.nodes.query_rewrite import query_rewrite_node
    from app.services.llm_service import get_llm_service
    from app.utils.helpers import format_docs_for_prompt
    from app.models.schemas import RAGState

    session_id, conversation_history = _get_or_create_session(request.session_id)

    async def generate() -> AsyncIterator[str]:
        # Run pre-generation pipeline steps
        state: RAGState = {
            "original_query": request.query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "filtered_docs": [],
            "generated_answer": "",
            "retry_count": 0,
            "max_retries": request.max_retries,
            "citations": [],
            "query_type": "factual",
            "is_hallucination": False,
            "hallucination_score": 0.0,
            "conversation_history": conversation_history,
            "session_id": session_id,
            "error": None,
        }

        state.update(query_analysis_node(state))
        state.update(retrieval_node(state))
        state.update(document_grading_node(state))

        # Retry loop
        retry = 0
        while not state.get("filtered_docs") and retry < request.max_retries:
            state.update(query_rewrite_node(state))
            state.update(retrieval_node(state))
            state.update(document_grading_node(state))
            retry += 1

        filtered_docs = state.get("filtered_docs", [])
        query = state.get("rewritten_query") or request.query

        from app.graph.nodes.generation import (
            GENERATION_SYSTEM_PROMPT,
            FALLBACK_GENERATION_SYSTEM_PROMPT,
        )

        llm = get_llm_service()

        if filtered_docs:
            formatted = format_docs_for_prompt(filtered_docs)
            human_msg = f"Question: {request.query}\n\nDocumentation:\n{formatted}"
            system_prompt = GENERATION_SYSTEM_PROMPT
        else:
            human_msg = f"Question: {request.query}"
            system_prompt = FALLBACK_GENERATION_SYSTEM_PROMPT

        yield f"data: {{'event': 'start', 'session_id': '{session_id}'}}\n\n"

        full_answer = ""
        for chunk in llm.stream(system_prompt, human_msg):
            full_answer += chunk
            # Escape newlines for SSE
            safe_chunk = chunk.replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"event": "token", "content": "{safe_chunk}"}}\n\n'

        # Save to memory
        _update_session_memory(session_id, request.query, full_answer)

        yield f"data: {{'event': 'end', 'retry_count': {state.get('retry_count', 0)}}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
