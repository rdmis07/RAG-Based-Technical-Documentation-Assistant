"""
Central API router that mounts all route modules.
"""
from fastapi import APIRouter

from app.api.routes import query, ingest, documents, feedback, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(query.router, tags=["Query"])
api_router.include_router(ingest.router, tags=["Ingestion"])
api_router.include_router(documents.router, tags=["Documents"])
api_router.include_router(feedback.router, tags=["Feedback"])
