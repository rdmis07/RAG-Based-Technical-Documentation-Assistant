"""
Ingestion Pipeline — orchestrates document loading, chunking, embedding, and storage.

Flow:
  load documents → split into chunks → generate embeddings → store in ChromaDB
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.ingestion.document_loader import DocumentLoader
from app.services.vector_store_service import VectorStoreService
from app.utils.helpers import generate_chunk_id, parse_file_extension
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    1. Loads files from a path (file or directory).
    2. Splits documents into overlapping chunks.
    3. Attaches rich metadata to each chunk.
    4. Stores chunks in ChromaDB via VectorStoreService.
    """

    def __init__(
        self,
        vector_store_service: Optional[VectorStoreService] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._loader = DocumentLoader()
        self._vs = vector_store_service or VectorStoreService()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def ingest(
        self,
        source_path: str,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Full ingestion run for a file or directory.

        Args:
            source_path: File path or directory path to ingest.
            collection_name: Target ChromaDB collection (overrides env var).
            chunk_size: Characters per chunk (default: 1000).
            chunk_overlap: Overlapping characters between chunks (default: 200).
            recursive: Scan directories recursively.

        Returns:
            Dict with 'files_processed', 'chunks_created', 'errors'.
        """
        cs = chunk_size or self._chunk_size
        co = chunk_overlap or self._chunk_overlap

        logger.info(
            "Ingestion started | path=%s | chunk_size=%d | overlap=%d",
            source_path,
            cs,
            co,
        )

        path = Path(source_path)
        errors: List[str] = []
        all_docs: List[Document] = []
        files_processed = 0

        # ── Load documents ────────────────────────────────────
        if path.is_file():
            try:
                docs = self._loader.load_file(source_path)
                all_docs.extend(docs)
                files_processed = 1
            except Exception as exc:
                errors.append(f"{source_path}: {exc}")
                logger.error("Failed to load file: %s", exc)
        elif path.is_dir():
            try:
                docs = self._loader.load_directory(source_path, recursive=recursive)
                all_docs.extend(docs)
                # Count unique source files
                sources = {d.metadata.get("source", "") for d in docs}
                files_processed = len(sources)
            except Exception as exc:
                errors.append(f"{source_path}: {exc}")
                logger.error("Failed to load directory: %s", exc)
        else:
            raise FileNotFoundError(f"Path does not exist: {source_path}")

        if not all_docs:
            logger.warning("No documents loaded from %s", source_path)
            return {
                "files_processed": files_processed,
                "chunks_created": 0,
                "errors": errors,
                "collection_name": collection_name or "rag-collection",
            }

        # ── Split into chunks ─────────────────────────────────
        chunks = self._split_documents(all_docs, cs, co)
        logger.info("Split %d docs into %d chunks", len(all_docs), len(chunks))

        # ── Prepare for storage ───────────────────────────────
        chunk_dicts = self._prepare_chunk_dicts(chunks)

        # ── Store in ChromaDB ─────────────────────────────────
        try:
            if not self._vs._chroma_client:
                self._vs.initialize()

            added = self._vs.add_documents(chunk_dicts, collection_name=collection_name)
        except Exception as exc:
            errors.append(f"ChromaDB storage error: {exc}")
            logger.error("Failed to store chunks: %s", exc, exc_info=True)
            added = 0

        result = {
            "files_processed": files_processed,
            "chunks_created": added,
            "errors": errors,
            "collection_name": collection_name or "rag-collection",
        }
        logger.info(
            "Ingestion complete | files=%d | chunks=%d | errors=%d",
            files_processed,
            added,
            len(errors),
        )
        return result

    def ingest_text(
        self,
        content: str,
        source: str = "manual_input",
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest raw text content directly (without loading from disk).

        Args:
            content: Raw document text.
            source: Source identifier.
            metadata: Optional metadata dict.
            collection_name: Target collection.

        Returns:
            Ingestion result dict.
        """
        docs = self._loader.load_from_text(content, metadata=metadata, source=source)
        chunks = self._split_documents(docs, self._chunk_size, self._chunk_overlap)
        chunk_dicts = self._prepare_chunk_dicts(chunks)

        if not self._vs._chroma_client:
            self._vs.initialize()

        added = self._vs.add_documents(chunk_dicts, collection_name=collection_name)
        return {
            "files_processed": 1,
            "chunks_created": added,
            "errors": [],
            "collection_name": collection_name or os.getenv("CHROMA_COLLECTION", "technical_docs"),
        }

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _split_documents(
        self,
        docs: List[Document],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        """
        Split documents using RecursiveCharacterTextSplitter.

        Separators prioritise semantic boundaries:
        paragraph → sentence → word → character.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(docs)

    def _prepare_chunk_dicts(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Convert LangChain Document objects to storage-ready dicts with enriched metadata.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Group chunks by source to track total_chunks per document
        source_counts: Dict[str, int] = {}
        source_indices: Dict[str, int] = {}

        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        result: List[Dict[str, Any]] = []
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            idx = source_indices.get(src, 0)
            source_indices[src] = idx + 1

            chunk_id = generate_chunk_id(src, idx, chunk.page_content)

            metadata = {
                **chunk.metadata,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": source_counts[src],
                "created_at": now_iso,
                "char_count": len(chunk.page_content),
            }

            result.append(
                {
                    "content": chunk.page_content,
                    "metadata": metadata,
                    "chunk_id": chunk_id,
                }
            )

        return result
