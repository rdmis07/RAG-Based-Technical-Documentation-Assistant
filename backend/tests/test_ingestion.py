"""
Tests for the document ingestion pipeline.
"""
from __future__ import annotations

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-key-placeholder")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma_db")


class TestDocumentLoader:
    def test_load_markdown_file(self, tmp_path):
        from app.ingestion.document_loader import DocumentLoader

        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Document\n\nThis is test content.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_file(str(md_file))

        assert len(docs) >= 1
        assert any("Test Document" in d.page_content or "test content" in d.page_content for d in docs)

    def test_load_text_file(self, tmp_path):
        from app.ingestion.document_loader import DocumentLoader

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text document content.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_file(str(txt_file))

        assert len(docs) >= 1
        assert "Plain text document content." in docs[0].page_content

    def test_load_nonexistent_file_raises(self):
        from app.ingestion.document_loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.md")

    def test_unsupported_extension_raises(self, tmp_path):
        from app.ingestion.document_loader import DocumentLoader

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            loader.load_file(str(pdf_file))

    def test_load_directory(self, tmp_path):
        from app.ingestion.document_loader import DocumentLoader

        (tmp_path / "doc1.md").write_text("# Doc 1\nContent 1", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("Doc 2 content", encoding="utf-8")
        (tmp_path / "ignored.pdf").write_bytes(b"fake pdf")

        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))

        # Should load .md and .txt but skip .pdf
        sources = {d.metadata.get("source", "") for d in docs}
        assert any("doc1.md" in s for s in sources)
        assert any("doc2.txt" in s for s in sources)

    def test_load_from_text(self):
        from app.ingestion.document_loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load_from_text("Sample content", source="test_source")

        assert len(docs) == 1
        assert docs[0].page_content == "Sample content"
        assert docs[0].metadata["source"] == "test_source"


class TestIngestionPipeline:
    def test_ingest_single_file(self, tmp_path):
        from app.ingestion.ingestion_pipeline import IngestionPipeline

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# FastAPI Guide\n\n" + "FastAPI is a modern web framework. " * 50,
            encoding="utf-8",
        )

        mock_vs = MagicMock()
        mock_vs._chroma_client = MagicMock()
        mock_vs.add_documents.return_value = 3

        pipeline = IngestionPipeline(
            vector_store_service=mock_vs,
            chunk_size=200,
            chunk_overlap=50,
        )
        result = pipeline.ingest(str(md_file))

        assert result["files_processed"] == 1
        assert result["chunks_created"] > 0
        assert result["errors"] == []

    def test_ingest_nonexistent_path_raises(self):
        from app.ingestion.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.ingest("/nonexistent/path")

    def test_ingest_text_directly(self):
        from app.ingestion.ingestion_pipeline import IngestionPipeline

        mock_vs = MagicMock()
        mock_vs._chroma_client = MagicMock()
        mock_vs.add_documents.return_value = 1

        pipeline = IngestionPipeline(vector_store_service=mock_vs)
        result = pipeline.ingest_text("Hello, world! " * 20, source="test_input")

        assert result["files_processed"] == 1
        assert result["chunks_created"] > 0

    def test_chunk_metadata_is_attached(self, tmp_path):
        from app.ingestion.ingestion_pipeline import IngestionPipeline

        md_file = tmp_path / "guide.md"
        md_file.write_text("# Guide\n\n" + "Content here. " * 100, encoding="utf-8")

        captured_chunks = []

        mock_vs = MagicMock()
        mock_vs._chroma_client = MagicMock()

        def capture_add(docs, **kwargs):
            captured_chunks.extend(docs)
            return len(docs)

        mock_vs.add_documents.side_effect = capture_add

        pipeline = IngestionPipeline(
            vector_store_service=mock_vs,
            chunk_size=300,
            chunk_overlap=50,
        )
        pipeline.ingest(str(md_file))

        assert len(captured_chunks) > 0
        for chunk in captured_chunks:
            assert "chunk_id" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
            assert "total_chunks" in chunk["metadata"]
            assert "created_at" in chunk["metadata"]
            assert "char_count" in chunk["metadata"]
