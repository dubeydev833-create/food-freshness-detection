"""Health check and application status endpoints (Phase 2)."""

import os
from fastapi import APIRouter, Depends, status
import sqlite3

from backend.config import settings
from backend.database import check_db_health, get_db
from backend.schemas.health import HealthResponse, RootResponse

router = APIRouter(tags=["System & Health"])


def is_model_available() -> bool:
    """Check if the trained model file exists on disk."""
    return settings.MODEL_PATH.exists() and settings.MODEL_PATH.is_file()


@router.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Status",
    description="Returns the status of the FreshCheck AI backend service.",
)
async def get_root() -> RootResponse:
    """Return backend application status and link to docs."""
    return RootResponse(
        app=settings.PROJECT_NAME,
        version=settings.VERSION,
        status="online",
        docs_url="/docs",
        health_url="/health",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check whether the backend is running and whether the deep-learning model is available.",
)
async def get_health() -> HealthResponse:
    """Check backend operational health and model readiness."""
    model_exists = is_model_available()
    db_ok = check_db_health(settings.DB_PATH)

    is_healthy = model_exists and db_ok
    status_str = "healthy" if is_healthy else "degraded"

    message = None
    if not model_exists:
        message = (
            f"Model not found at '{settings.MODEL_PATH}'. "
            "Please ensure Member 1's trained .keras model is in the models/ directory."
        )
    elif not db_ok:
        message = "SQLite database connectivity issue detected."
    else:
        message = "All FreshCheck AI backend systems and models are fully operational."

    return HealthResponse(
        status=status_str,
        backend_running=True,
        model_available=model_exists,
        model_path=str(settings.MODEL_PATH),
        database_connected=db_ok,
        message=message,
    )
