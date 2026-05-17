"""
POST /feedback — store user feedback on generated answers.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.services.feedback_service import FeedbackService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_feedback_service = FeedbackService()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback on a RAG answer",
)
async def submit_feedback(request: FeedbackRequest) -> Any:
    """
    Record user feedback (rating + comment) for a generated answer.

    Feedback is used to:
    - Track answer quality over time
    - Identify poorly-performing query patterns
    - Guide future fine-tuning or prompt improvements

    Ratings: 1 = Very Bad, 2 = Bad, 3 = OK, 4 = Good, 5 = Excellent
    """
    logger.info(
        "POST /feedback | session=%s | rating=%d | helpful=%s",
        request.session_id,
        request.rating,
        request.is_helpful,
    )

    try:
        feedback_id = _feedback_service.save_feedback(
            session_id=request.session_id,
            query=request.query,
            answer=request.answer,
            rating=request.rating,
            is_helpful=request.is_helpful,
            comment=request.comment,
        )
    except Exception as exc:
        logger.error("Failed to save feedback: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {exc}")

    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        message="Thank you for your feedback! It helps us improve the system.",
    )


@router.get("/feedback/stats", summary="Aggregate feedback statistics")
async def feedback_stats() -> Any:
    """Return aggregate statistics (total count, average rating, helpfulness %)."""
    try:
        return _feedback_service.get_statistics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
