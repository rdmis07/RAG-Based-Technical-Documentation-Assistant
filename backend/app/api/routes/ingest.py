"""
POST /ingest — document ingestion endpoint.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.models.schemas import IngestRequest, IngestResponse
from app.services.vector_store_service import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into ChromaDB",
)
async def ingest_documents(request: IngestRequest) -> Any:
    """
    Load and index documents from a file or directory path.

    Supported file types: `.md`, `.markdown`, `.txt`, `.html`, `.htm`

    The pipeline:
    1. Loads files using appropriate LangChain document loaders
    2. Splits into chunks (RecursiveCharacterTextSplitter)
    3. Generates embeddings (sentence-transformers/all-MiniLM-L6-v2)
    4. Stores chunks + embeddings in ChromaDB

    **Note:** `source_path` must be accessible from the server's filesystem.
    """
    start_time = time.time()

    logger.info(
        "POST /ingest | path=%s | collection=%s | chunk_size=%d",
        request.source_path,
        request.collection_name,
        request.chunk_size,
    )

    try:
        vs = VectorStoreService(collection_name=request.collection_name)
        vs.initialize()

        pipeline = IngestionPipeline(
            vector_store_service=vs,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        result = pipeline.ingest(
            source_path=request.source_path,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            recursive=request.recursive,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    processing_ms = round((time.time() - start_time) * 1000, 2)

    status = "success" if result["chunks_created"] > 0 else "no_content"
    if result["errors"]:
        status = "partial_success" if result["chunks_created"] > 0 else "failed"

    return IngestResponse(
        status=status,
        files_processed=result["files_processed"],
        chunks_created=result["chunks_created"],
        collection_name=result["collection_name"],
        processing_time_ms=processing_ms,
        errors=result["errors"],
    )
