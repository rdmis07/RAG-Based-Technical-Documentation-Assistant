"""
GET /documents — list stored document chunks with metadata.
"""
from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import DocumentInfo
from app.services.vector_store_service import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List stored document chunks",
)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
) -> Any:
    """
    Returns metadata for all document chunks stored in ChromaDB.

    Useful for inspecting what's been ingested and verifying the collection contents.
    Supports pagination via `limit` and `offset` query parameters.
    """
    logger.info("GET /documents | limit=%d | offset=%d", limit, offset)

    try:
        vs = VectorStoreService()
        if not vs._chroma_client:
            vs.initialize()

        docs = vs.get_all_documents(limit=limit, offset=offset)
        total = vs.get_document_count()

        logger.info("Returning %d/%d document chunks", len(docs), total)
        return [DocumentInfo(**d) for d in docs]

    except Exception as exc:
        logger.error("Failed to list documents: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {exc}")


@router.get(
    "/documents/count",
    summary="Get total document count",
)
async def document_count() -> Any:
    """Returns the total number of document chunks in the active collection."""
    try:
        vs = VectorStoreService()
        if not vs._chroma_client:
            vs.initialize()
        count = vs.get_document_count()
        return {"count": count, "collection": vs._collection_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
