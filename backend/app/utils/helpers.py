"""
Utility helper functions for the RAG pipeline.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """
    Generate a deterministic unique ID for a document chunk.

    Args:
        source: Source file path or URL.
        chunk_index: Index of the chunk within the document.
        content: The actual chunk text (first 100 chars used for hash).

    Returns:
        A hex string chunk ID.
    """
    raw = f"{source}::{chunk_index}::{content[:100]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def truncate_text(text: str, max_length: int = 300, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.

    Args:
        text: Input text.
        max_length: Maximum character length.
        suffix: String appended when truncation occurs.

    Returns:
        Possibly truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)].rstrip() + suffix


def format_docs_for_prompt(docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved document chunks into a structured string for LLM prompts.

    Args:
        docs: List of document dicts with 'content', 'metadata', and 'score' keys.

    Returns:
        Formatted multi-document string.
    """
    if not docs:
        return "No relevant documents found."

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Unknown")
        title = metadata.get("title", source)
        chunk_idx = metadata.get("chunk_index", "?")
        score = doc.get("score", 0.0)
        content = doc.get("content", "").strip()

        parts.append(
            f"[Document {i}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Chunk: {chunk_idx}\n"
            f"Relevance Score: {score:.4f}\n"
            f"Content:\n{content}\n"
            f"{'─' * 60}"
        )
    return "\n\n".join(parts)


def extract_citations(
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build citation objects from filtered document chunks.

    Args:
        docs: Filtered document dicts.

    Returns:
        List of citation dicts compatible with the Citation pydantic model.
    """
    citations: List[Dict[str, Any]] = []
    seen_sources: set = set()

    for doc in docs:
        metadata = doc.get("metadata", {})
        chunk_id = doc.get("chunk_id", generate_chunk_id(
            metadata.get("source", "unknown"), 0, doc.get("content", "")
        ))
        source = metadata.get("source", "unknown")
        title = metadata.get("title", source)
        page = metadata.get("page", None)
        score = doc.get("score", 0.0)
        content = doc.get("content", "")

        # Deduplicate by source + chunk_id
        dedup_key = f"{source}::{chunk_id}"
        if dedup_key in seen_sources:
            continue
        seen_sources.add(dedup_key)

        citations.append(
            {
                "source": source,
                "chunk_id": chunk_id,
                "page": page,
                "relevance_score": round(score, 4),
                "excerpt": truncate_text(content, max_length=200),
                "title": title,
            }
        )

    return citations


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension.")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize a string to be a valid ChromaDB collection name.

    Collection names must:
    - Be 3–63 characters
    - Start/end with alphanumeric chars
    - Contain only alphanumeric chars, hyphens, or underscores
    - Not contain consecutive periods

    Args:
        name: Raw collection name.

    Returns:
        Sanitized collection name.
    """
    # Replace spaces and invalid chars with underscores
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Remove leading/trailing underscores/hyphens
    name = name.strip("_-")
    # Ensure minimum length
    if len(name) < 3:
        name = name + "_db"
    # Truncate to 63 characters
    return name[:63]


def parse_file_extension(path: str) -> str:
    """
    Extract the file extension from a path.

    Args:
        path: File path string.

    Returns:
        Lowercase extension without dot, or empty string.
    """
    import os
    _, ext = os.path.splitext(path)
    return ext.lower().lstrip(".")
