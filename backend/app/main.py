"""
RAG Technical Documentation Assistant — FastAPI application entry point.

Self-corrective RAG pipeline built with:
- LangGraph StateGraph
- LangChain + ChromaDB
- Groq Llama 3 (via langchain-groq)
- sentence-transformers/all-MiniLM-L6-v2 embeddings
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Application lifecycle
# ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.
    Runs startup logic before yield and shutdown logic after.
    """
    logger.info("=" * 60)
    logger.info("  RAG Technical Documentation Assistant  ")
    logger.info("  Starting up...                         ")
    logger.info("=" * 60)

    # 1. Load embedding model (CPU warm-up)
    try:
        from app.services.embedding_service import get_embedding_service
        emb = get_embedding_service()
        logger.info("✓ Embedding model loaded: %s", os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    except Exception as exc:
        logger.error("✗ Embedding model failed to load: %s", exc)

    # 2. Initialise vector store
    try:
        from app.services.vector_store_service import VectorStoreService
        vs = VectorStoreService()
        vs.initialize()
        count = vs.get_document_count()
        logger.info(
            "✓ ChromaDB connected | collection=%s | docs=%d",
            vs._collection_name,
            count,
        )
    except Exception as exc:
        logger.error("✗ ChromaDB failed to initialise: %s", exc)

    # 3. Verify LLM availability
    try:
        from app.services.llm_service import get_llm_service
        llm = get_llm_service()
        logger.info("✓ Groq LLM initialised | model=%s", os.getenv("GROQ_MODEL", "llama3-70b-8192"))
    except Exception as exc:
        logger.warning("⚠ LLM not available at startup: %s", exc)

    # 4. Pre-ingest sample docs if the collection is empty
    try:
        from app.services.vector_store_service import VectorStoreService
        vs2 = VectorStoreService()
        vs2.initialize()
        if vs2.get_document_count() == 0:
            _seed_sample_documents()
    except Exception as exc:
        logger.warning("Could not seed sample documents: %s", exc)

    logger.info("=" * 60)
    logger.info("  API ready — http://0.0.0.0:%s", os.getenv("PORT", "8000"))
    logger.info("  Docs — http://0.0.0.0:%s/docs", os.getenv("PORT", "8000"))
    logger.info("=" * 60)

    yield

    logger.info("Shutting down RAG Assistant…")


def _seed_sample_documents() -> None:
    """
    Ingest built-in sample technical documentation if the DB is empty.
    Provides out-of-the-box demo capability.
    """
    from app.ingestion.ingestion_pipeline import IngestionPipeline

    sample_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    if not os.path.exists(sample_data_path):
        logger.info("No sample docs directory found — skipping seed.")
        return

    pipeline = IngestionPipeline()
    result = pipeline.ingest(source_path=sample_data_path)
    logger.info(
        "Seeded sample documents | files=%d | chunks=%d",
        result["files_processed"],
        result["chunks_created"],
    )


# ──────────────────────────────────────────────────────────────
# FastAPI app factory
# ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Technical Documentation Assistant",
        description=(
            "A self-corrective RAG pipeline for answering questions from technical documentation.\n\n"
            "## Features\n"
            "- **Self-corrective retrieval** with up to N retry attempts\n"
            "- **LLM-based document grading** (relevance filter)\n"
            "- **Query rewriting** when initial retrieval fails\n"
            "- **Hallucination detection** on generated answers\n"
            "- **Conversation memory** (session-based)\n"
            "- **Streaming responses** (SSE)\n"
            "- **Source citations** in every answer\n\n"
            "## Tech Stack\n"
            "LangGraph · LangChain · ChromaDB · Groq Llama 3 · sentence-transformers"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8080",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next: Any) -> Any:
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        return response

    # ── Global exception handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    # ── Mount API router ──────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Root redirect ─────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> Any:
        return {
            "message": "RAG Technical Documentation Assistant",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
