"""
GET /health — system health check endpoint.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Track application start time for uptime calculation
_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> Any:
    """
    Returns the operational status of all system components:
    - ChromaDB connectivity
    - LLM availability
    - Embedding model status
    - Document count in the active collection
    """
    from app.services.embedding_service import get_embedding_service
    from app.services.llm_service import LLMService
    from app.services.vector_store_service import VectorStoreService
    import os

    # Embedding service
    try:
        emb_service = get_embedding_service()
        embeddings_loaded = emb_service.is_loaded()
    except Exception:
        embeddings_loaded = False

    # Vector store
    try:
        vs = VectorStoreService()
        if not vs._chroma_client:
            vs.initialize()
        chroma_connected = vs.is_connected()
        document_count = vs.get_document_count()
        collection_name = vs._collection_name
    except Exception:
        chroma_connected = False
        document_count = 0
        collection_name = os.getenv("CHROMA_COLLECTION", "technical_docs")

    # LLM
    llm_service = LLMService()
    llm_available = llm_service.is_available()

    uptime = round(time.time() - _START_TIME, 2)

    status = "healthy" if (chroma_connected and embeddings_loaded) else "degraded"

    logger.info(
        "Health check | status=%s | chroma=%s | llm=%s | embeddings=%s | docs=%d",
        status,
        chroma_connected,
        llm_available,
        embeddings_loaded,
        document_count,
    )

    return HealthResponse(
        status=status,
        version="1.0.0",
        chroma_connected=chroma_connected,
        llm_available=llm_available,
        embeddings_loaded=embeddings_loaded,
        collection_name=collection_name,
        document_count=document_count,
        uptime_seconds=uptime,
    )
