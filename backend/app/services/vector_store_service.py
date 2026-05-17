"""
Vector store service wrapping ChromaDB with a persistent client.
Handles all CRUD operations on the vector collection.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Manages ChromaDB persistent vector store interactions.

    Provides similarity search, document insertion, and metadata retrieval.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._embedding_service = embedding_service or get_embedding_service()
        self._persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma_db"
        )
        self._collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "technical_docs"
        )
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._vector_store: Optional[Chroma] = None

    # ──────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Set up the ChromaDB persistent client and LangChain Chroma wrapper."""
        os.makedirs(self._persist_directory, exist_ok=True)
        logger.info(
            "Initializing ChromaDB | dir=%s | collection=%s",
            self._persist_directory,
            self._collection_name,
        )

        self._chroma_client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._vector_store = Chroma(
            client=self._chroma_client,
            collection_name=self._collection_name,
            embedding_function=self._embedding_service.embeddings,
        )
        logger.info("ChromaDB initialized successfully.")

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self.initialize()
        return self._vector_store  # type: ignore[return-value]

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        if self._chroma_client is None:
            self.initialize()
        return self._chroma_client  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return documents with relevance scores.

        Args:
            query: The search query string.
            k: Number of top results to return.
            filter_metadata: Optional metadata filter dict for ChromaDB.

        Returns:
            List of dicts with 'content', 'metadata', 'score', 'chunk_id'.
        """
        try:
            kwargs: Dict[str, Any] = {}
            if filter_metadata:
                kwargs["filter"] = filter_metadata

            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                **kwargs,
            )

            docs: List[Dict[str, Any]] = []
            for doc, score in results:
                docs.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    }
                )

            logger.debug("Similarity search returned %d results for query: %.80s", len(docs), query)
            return docs

        except Exception as exc:
            logger.error("Similarity search failed: %s", exc, exc_info=True)
            return []

    # ──────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            documents: List of dicts with 'content', 'metadata', 'chunk_id'.
            collection_name: Optional override for collection name.

        Returns:
            Number of documents successfully added.
        """
        if collection_name and collection_name != self._collection_name:
            # Switch to different collection temporarily
            target_store = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self._embedding_service.embeddings,
            )
        else:
            target_store = self.vector_store

        texts = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = [doc.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]

        try:
            target_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Added %d document chunks to ChromaDB.", len(texts))
            return len(texts)
        except Exception as exc:
            logger.error("Failed to add documents: %s", exc, exc_info=True)
            raise

    # ──────────────────────────────────────────────────────────
    # Metadata / listing
    # ──────────────────────────────────────────────────────────

    def get_all_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve document metadata from the collection.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of document metadata dicts.
        """
        try:
            collection = self.chroma_client.get_collection(self._collection_name)
            result = collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"],
            )

            docs = []
            ids = result.get("ids", [])
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])

            for i, doc_id in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                docs.append(
                    {
                        "chunk_id": doc_id,
                        "source": meta.get("source", "unknown"),
                        "title": meta.get("title"),
                        "chunk_index": meta.get("chunk_index", 0),
                        "total_chunks": meta.get("total_chunks", 1),
                        "content_preview": content[:150] + "..." if len(content) > 150 else content,
                        "created_at": meta.get("created_at"),
                    }
                )
            return docs

        except Exception as exc:
            logger.error("Failed to retrieve documents: %s", exc, exc_info=True)
            return []

    def get_document_count(self) -> int:
        """Return total number of chunks in the active collection."""
        try:
            collection = self.chroma_client.get_collection(self._collection_name)
            return collection.count()
        except Exception:
            return 0

    def collection_exists(self) -> bool:
        """Check whether the active collection exists in ChromaDB."""
        try:
            collections = self.chroma_client.list_collections()
            names = [c.name for c in collections]
            return self._collection_name in names
        except Exception:
            return False

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete a collection from ChromaDB."""
        target = collection_name or self._collection_name
        try:
            self.chroma_client.delete_collection(target)
            logger.warning("Deleted ChromaDB collection: %s", target)
        except Exception as exc:
            logger.error("Failed to delete collection %s: %s", target, exc)

    def is_connected(self) -> bool:
        """Health-check: verify ChromaDB connectivity."""
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception:
            return False
