"""
POST /ingest — document ingestion endpoint with file upload support.
"""
from __future__ import annotations

import time
import os
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.models.schemas import IngestResponse
from app.services.vector_store_service import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Ek temporary folder jahan upload ki hui file save hogi processing ke liye
UPLOAD_DIR = "data/sample_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest uploaded document into ChromaDB",
)
async def ingest_documents(
    file: UploadFile = File(...),
    collection_name: str = Form("rag-collection"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
) -> Any:
    """
    Upload a document (PDF, TXT, MD, etc.) and index it into ChromaDB.
    """
    start_time = time.time()

    logger.info(
        "POST /ingest | filename=%s | collection=%s | chunk_size=%d",
        file.filename,
        collection_name,
        chunk_size,
    )

    # 1. File ko binary mode me temporary save karo
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail=f"File upload failed: {exc}")

    # 2. Ingestion pipeline chalao saved file par
    try:
        vs = VectorStoreService(collection_name=collection_name)
        vs.initialize()

        pipeline = IngestionPipeline(
            vector_store_service=vs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        result = pipeline.ingest(
            source_path=file_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=False,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")
    finally:
        # Optional: Kaam hone ke baad temporary file delete karna chaho toh kar sakte ho
        pass

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