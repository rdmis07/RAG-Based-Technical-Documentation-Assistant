"""
Embedding service using sentence-transformers/all-MiniLM-L6-v2.
Provides a singleton embedding model for the entire application.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Manages the HuggingFace sentence-transformer embedding model.
    Uses a singleton pattern to avoid reloading the model on every request.
    """

    _instance: "EmbeddingService | None" = None
    _embeddings: HuggingFaceEmbeddings | None = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        """Load and cache the embedding model."""
        if self._embeddings is not None:
            logger.debug("Embedding model already loaded — skipping initialization.")
            return

        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Loading embedding model: %s", model_name)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            },
        )
        logger.info("Embedding model loaded successfully.")

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Return the embedding model, initializing if necessary."""
        if self._embeddings is None:
            self.initialize()
        return self._embeddings  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single query string.

        Args:
            text: Input text.

        Returns:
            List of floats representing the embedding vector.
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple documents.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def is_loaded(self) -> bool:
        """Check whether the model has been loaded."""
        return self._embeddings is not None


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Dependency-injection helper — returns the singleton EmbeddingService."""
    service = EmbeddingService()
    service.initialize()
    return service
