"""Pydantic models for FreshCheck AI prediction requests and responses."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class Probabilities(BaseModel):
    """Class probability distribution for the three freshness classes."""

    Fresh: float = Field(..., ge=0.0, le=1.0, description="Fresh class probability [0.0 - 1.0]")
    Moderate: float = Field(..., ge=0.0, le=1.0, description="Moderate class probability [0.0 - 1.0]")
    Spoiled: float = Field(..., ge=0.0, le=1.0, description="Spoiled class probability [0.0 - 1.0]")


class FoodMetadataResponse(BaseModel):
    """Environmental and storage metadata associated with the food item.

    Strict rule: Fields MUST be null when not provided by the user.
    Never invent temperature or environmental sensor readings.
    """

    temperature_celsius: Optional[float] = Field(
        None,
        description="Observed temperature in Celsius, or null if not provided",
    )
    storage_condition: Optional[str] = Field(
        None,
        description="Storage condition setting (e.g. 'Refrigerated'), or null if not provided",
    )


class PredictionResponse(BaseModel):
    """Standard prediction response contract (Phase 3)."""

    success: bool = Field(True, description="Whether prediction succeeded")
    category: str = Field(..., description="Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy')")
    predicted_class: str = Field(..., description="Predicted freshness class ('Fresh', 'Moderate', 'Spoiled')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0]")
    probabilities: Probabilities = Field(..., description="Softmax probabilities for each freshness class")
    metadata: FoodMetadataResponse = Field(..., description="Storage metadata (null when omitted)")
    message: str = Field(
        "Image-based freshness prediction completed.",
        description="User-facing summary message",
    )
