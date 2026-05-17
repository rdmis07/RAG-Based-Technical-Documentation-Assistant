"""
Feedback service — stores user ratings and comments in a JSON-backed store.
In production this would be replaced with a database (PostgreSQL, etc.).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

FEEDBACK_FILE = os.getenv("FEEDBACK_STORE_PATH", "./data/feedback.json")


class FeedbackService:
    """Simple JSON-file-backed feedback store."""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._path = Path(store_path or FEEDBACK_FILE)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(json.dumps([]), encoding="utf-8")

    # ──────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────

    def save_feedback(
        self,
        session_id: str,
        query: str,
        answer: str,
        rating: int,
        is_helpful: bool,
        comment: Optional[str] = None,
    ) -> str:
        """
        Persist a feedback record and return its unique ID.

        Args:
            session_id: Caller's session identifier.
            query: Original user query.
            answer: Generated answer that was rated.
            rating: Integer from 1 to 5.
            is_helpful: Whether the user found the answer helpful.
            comment: Optional free-text comment.

        Returns:
            Generated feedback_id UUID string.
        """
        feedback_id = str(uuid.uuid4())
        record: Dict[str, Any] = {
            "feedback_id": feedback_id,
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "rating": rating,
            "is_helpful": is_helpful,
            "comment": comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        existing = self._load_all()
        existing.append(record)
        self._save_all(existing)

        logger.info(
            "Feedback saved | id=%s | session=%s | rating=%d",
            feedback_id,
            session_id,
            rating,
        )
        return feedback_id

    # ──────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────

    def get_feedback_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all feedback records for a given session."""
        return [r for r in self._load_all() if r.get("session_id") == session_id]

    def get_all_feedback(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Return paginated feedback records."""
        all_records = self._load_all()
        return all_records[offset : offset + limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all feedback."""
        records = self._load_all()
        if not records:
            return {"total": 0, "average_rating": 0.0, "helpful_pct": 0.0}

        total = len(records)
        avg_rating = sum(r.get("rating", 0) for r in records) / total
        helpful = sum(1 for r in records if r.get("is_helpful", False))

        return {
            "total": total,
            "average_rating": round(avg_rating, 2),
            "helpful_pct": round((helpful / total) * 100, 1),
        }

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_all(self, records: List[Dict[str, Any]]) -> None:
        self._path.write_text(json.dumps(records, indent=2), encoding="utf-8")
