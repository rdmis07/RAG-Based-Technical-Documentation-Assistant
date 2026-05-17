"""
Pydantic schemas for request/response models and RAG pipeline state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────
# LangGraph State
# ──────────────────────────────────────────────────────────────

class RAGState(TypedDict, total=False):
    """
    Shared state object that flows through the LangGraph StateGraph.
    Every node reads from and writes to this dict.
    """
    original_query: str
    rewritten_query: str
    retrieved_docs: List[Dict[str, Any]]
    filtered_docs: List[Dict[str, Any]]
    generated_answer: str
    retry_count: int
    max_retries: int
    citations: List[Dict[str, Any]]
    query_type: str
    is_hallucination: bool
    hallucination_score: float
    conversation_history: List[Dict[str, str]]
    session_id: Optional[str]
    error: Optional[str]


# ──────────────────────────────────────────────────────────────
# API Models
# ──────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """Source citation for a generated answer."""
    source: str = Field(..., description="Document source path or URL")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    page: Optional[int] = Field(None, description="Page number if applicable")
    relevance_score: float = Field(..., description="Similarity score (0-1)")
    excerpt: str = Field(..., description="Relevant excerpt from the document")
    title: Optional[str] = Field(None, description="Document title if available")


class DocumentChunk(BaseModel):
    """A single chunk retrieved from the vector store."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class QueryRequest(BaseModel):
    """Incoming query request."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of docs to retrieve")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max retrieval retry attempts")
    stream: bool = Field(default=False, description="Stream the response tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does FastAPI handle async requests?",
                "session_id": "session-abc-123",
                "top_k": 5,
                "max_retries": 2,
                "stream": False,
            }
        }


class QueryResponse(BaseModel):
    """Response to a user query."""
    query: str
    rewritten_query: str
    answer: str
    citations: List[Citation]
    query_type: str
    retry_count: int
    retrieved_doc_count: int
    filtered_doc_count: int
    is_hallucination: bool
    hallucination_score: float
    session_id: Optional[str] = None
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does FastAPI handle async requests?",
                "rewritten_query": "What is FastAPI's asynchronous request handling mechanism?",
                "answer": "FastAPI handles async requests using Python's asyncio...",
                "citations": [],
                "query_type": "technical",
                "retry_count": 0,
                "retrieved_doc_count": 5,
                "filtered_doc_count": 3,
                "is_hallucination": False,
                "hallucination_score": 0.05,
                "session_id": "session-abc-123",
                "processing_time_ms": 1234.5,
            }
        }


class IngestRequest(BaseModel):
    """Request to ingest documents."""
    source_path: str = Field(..., description="Path to file or directory to ingest")
    collection_name: Optional[str] = Field(
        default="technical_docs",
        description="ChromaDB collection name",
    )
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    recursive: bool = Field(default=True, description="Recursively scan directories")

    class Config:
        json_schema_extra = {
            "example": {
                "source_path": "./data/docs",
                "collection_name": "technical_docs",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "recursive": True,
            }
        }


class IngestResponse(BaseModel):
    """Response after document ingestion."""
    status: str
    files_processed: int
    chunks_created: int
    collection_name: str
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "files_processed": 12,
                "chunks_created": 148,
                "collection_name": "technical_docs",
                "processing_time_ms": 8432.1,
                "errors": [],
            }
        }


class DocumentInfo(BaseModel):
    """Metadata for a stored document."""
    chunk_id: str
    source: str
    title: Optional[str]
    chunk_index: int
    total_chunks: int
    content_preview: str
    created_at: Optional[str]


class FeedbackRequest(BaseModel):
    """User feedback on a generated answer."""
    session_id: str
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (bad) to 5 (excellent)")
    comment: Optional[str] = Field(None, max_length=1000)
    is_helpful: bool = Field(..., description="Was the answer helpful?")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc-123",
                "query": "How does FastAPI handle async?",
                "answer": "FastAPI uses asyncio...",
                "rating": 5,
                "comment": "Very clear explanation!",
                "is_helpful": True,
            }
        }


class FeedbackResponse(BaseModel):
    """Response after storing feedback."""
    status: str
    feedback_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    chroma_connected: bool
    llm_available: bool
    embeddings_loaded: bool
    collection_name: str
    document_count: int
    uptime_seconds: float
