"""Pydantic schemas for prediction history (Phase 5)."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HistoryRecord(BaseModel):
    """A single stored prediction event.

    Strict privacy compliance: uploaded personal images are NEVER stored in the database.
    """

    id: int = Field(..., description="Internal database ID")
    prediction_id: str = Field(..., description="Unique UUID for this prediction event")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    category: str = Field(..., description="Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy')")
    predicted_class: str = Field(..., description="Predicted freshness class ('Fresh', 'Moderate', 'Spoiled')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    temperature_celsius: Optional[float] = Field(
        None,
        description="Optional observed storage temperature in Celsius",
    )
    storage_condition: Optional[str] = Field(
        None,
        description="Optional storage condition setting",
    )
    storage_duration: Optional[str] = Field(
        None,
        description="Optional storage duration text",
    )
    message: Optional[str] = Field(None, description="Status note")


class HistoryListResponse(BaseModel):
    """Response containing list of stored prediction records with metadata."""

    success: bool = Field(True, description="Whether request succeeded")
    total_count: int = Field(..., description="Total count of historical records matching filter")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    items: List[HistoryRecord] = Field(..., description="List of historical prediction records")


class ClearHistoryResponse(BaseModel):
    """Response returned upon clearing prediction history."""

    success: bool = Field(True, description="Whether deletion was successful")
    message: str = Field("Prediction history cleared successfully.", description="Status message")
    deleted_count: int = Field(..., description="Number of prediction records removed")
