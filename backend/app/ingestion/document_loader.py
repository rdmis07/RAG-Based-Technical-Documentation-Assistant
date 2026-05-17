"""
Document Loader — supports Markdown, plain text, HTML, and PDF files.
Uses LangChain document loaders under the hood.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from file extension to LangChain loader class
LOADER_MAP = {
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}

# Glob patterns for DirectoryLoader
GLOB_PATTERNS = ["**/*.md", "**/*.markdown", "**/*.txt", "**/*.html", "**/*.htm"]


class DocumentLoader:
    """
    Loads documents from files or directories into LangChain Document objects.

    Supports: .md, .markdown, .txt, .html, .htm
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file into LangChain Document objects.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List of Document objects (usually one per file, more for PDFs).

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in LOADER_MAP:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {list(LOADER_MAP.keys())}"
            )

        loader_cls = LOADER_MAP[ext]
        try:
            if ext in (".txt", ".md", ".markdown"):
                loader = loader_cls(str(path), encoding=self._encoding)
            else:
                loader = loader_cls(str(path))

            docs = loader.load()

            # Enrich metadata
            for doc in docs:
                doc.metadata.setdefault("source", str(path))
                doc.metadata.setdefault("title", path.stem.replace("_", " ").title())
                doc.metadata.setdefault("file_type", ext.lstrip("."))

            logger.info("Loaded %d document(s) from %s", len(docs), file_path)
            return docs

        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc, exc_info=True)
            raise

    def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            dir_path: Path to the directory.
            recursive: Whether to scan subdirectories.

        Returns:
            Combined list of Document objects from all loaded files.
        """
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_docs: List[Document] = []
        errors: List[str] = []

        # Collect all supported files
        if recursive:
            files = [
                f
                for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in LOADER_MAP
            ]
        else:
            files = [
                f
                for f in path.glob("*")
                if f.is_file() and f.suffix.lower() in LOADER_MAP
            ]

        logger.info("Found %d supported files in %s", len(files), dir_path)

        for file_path in files:
            try:
                docs = self.load_file(str(file_path))
                all_docs.extend(docs)
            except Exception as exc:
                error_msg = f"{file_path}: {exc}"
                logger.warning("Skipped file due to error — %s", error_msg)
                errors.append(error_msg)

        logger.info(
            "Directory load complete | loaded=%d docs | errors=%d",
            len(all_docs),
            len(errors),
        )
        return all_docs

    def load_from_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "manual_input",
    ) -> List[Document]:
        """
        Create Document objects directly from in-memory text.

        Args:
            content: The raw text content.
            metadata: Optional metadata dict.
            source: Source identifier string.

        Returns:
            List containing a single Document.
        """
        doc = Document(
            page_content=content,
            metadata={
                "source": source,
                "title": source,
                "file_type": "text",
                **(metadata or {}),
            },
        )
        return [doc]
