"""History API endpoints for querying and clearing prediction records (Phase 2 & Phase 5)."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
import sqlite3

from backend.database import get_db
from backend.schemas.history import ClearHistoryResponse, HistoryListResponse
from backend.services.history_service import (
    clear_prediction_history,
    get_prediction_history,
)

logger = logging.getLogger("FreshCheckAI.HistoryRoute")
router = APIRouter(prefix="/history", tags=["Prediction History"])


@router.get(
    "",
    response_model=HistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Prediction History",
    description=(
        "Retrieve previously stored food freshness predictions. Supports pagination "
        "and optional category filtering. Images are never stored in history."
    ),
)
async def get_history(
    limit: int = Query(50, ge=1, le=500, description="Max number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    category: Optional[str] = Query(None, description="Optional food category filter"),
    db: sqlite3.Connection = Depends(get_db),
) -> HistoryListResponse:
    """Return previously stored food freshness predictions."""
    items, total_count = get_prediction_history(
        conn=db,
        limit=limit,
        offset=offset,
        category=category,
    )
    return HistoryListResponse(
        success=True,
        total_count=total_count,
        limit=limit,
        offset=offset,
        items=items,
    )


@router.delete(
    "",
    response_model=ClearHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear Prediction History",
    description="Clear all stored prediction history records upon user request.",
)
async def clear_history(
    db: sqlite3.Connection = Depends(get_db),
) -> ClearHistoryResponse:
    """Clear all historical prediction records."""
    deleted_count = clear_prediction_history(conn=db)
    return ClearHistoryResponse(
        success=True,
        message="Prediction history cleared successfully.",
        deleted_count=deleted_count,
    )
