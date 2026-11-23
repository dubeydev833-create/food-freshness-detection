"""Prediction API route for FreshCheck AI (Phase 2 & Phase 3)."""

import logging
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.config import settings
from backend.schemas.prediction import (
    FoodMetadataResponse,
    PredictionResponse,
    Probabilities,
)
from backend.services.inference_service import (
    InferenceService,
    get_inference_service,
)
from image_processing.exceptions import (
    ImageFormatError,
    ImageReadError,
    ImageSizeError,
    ImageValidationError,
    MetadataValidationError,
    ModelInferenceError,
    ModelNotFoundError,
)

logger = logging.getLogger("FreshCheckAI.PredictRoute")
router = APIRouter(tags=["Prediction"])


def parse_optional_duration(duration_str: Optional[str]) -> Optional[float]:
    """Parse human-friendly storage duration string into days (float).

    Examples:
        - "2" -> 2.0
        - "2 days" -> 2.0
        - "48 hours" -> 2.0
        - "1 week" -> 7.0
    """
    if not duration_str or not duration_str.strip():
        return None

    clean = duration_str.strip().lower()
    try:
        parts = clean.split()
        if len(parts) == 1:
            return float(parts[0])
        val = float(parts[0])
        unit = parts[1]
        if "hour" in unit:
            return round(val / 24.0, 2)
        if "day" in unit:
            return round(val, 2)
        if "week" in unit:
            return round(val * 7.0, 2)
        if "month" in unit:
            return round(val * 30.0, 2)
        return round(val, 2)
    except Exception:
        # Fallback to None if not parsable as numeric duration
        return None


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Food Freshness",
    description=(
        "Accepts a food image and optional storage metadata, evaluates freshness using "
        "Member 2's inference engine and Member 1's deep-learning model, and returns "
        "predictions across Fresh, Moderate, and Spoiled classes."
    ),
)
async def predict_freshness(
    file: UploadFile = File(
        ...,
        description="Food image file (JPEG, PNG, WEBP, max 10MB)",
    ),
    category: str = Form(
        ...,
        description="Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy')",
    ),
    temperature: Optional[float] = Form(
        None,
        description="Optional storage/ambient temperature in Celsius",
    ),
    storage_condition: Optional[str] = Form(
        None,
        description="Optional storage condition (e.g. 'Refrigerated', 'Room Temperature')",
    ),
    storage_duration: Optional[str] = Form(
        None,
        description="Optional storage duration (e.g. '2 days', '24 hours')",
    ),
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Predict food freshness from uploaded image and optional metadata."""
    # Read uploaded file content
    contents = await file.read()

    # Parse optional duration into days
    duration_days = parse_optional_duration(storage_duration)

    try:
        analysis_result = inference_service.predict(
            image_bytes=contents,
            category=category,
            temperature_celsius=temperature,
            storage_condition=storage_condition,
            storage_duration_days=duration_days,
        )
    except ModelNotFoundError as exc:
        logger.error(f"Model missing: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference model is currently unavailable: {exc}",
        ) from exc
    except (ImageFormatError, ImageSizeError, ImageValidationError, ImageReadError) as exc:
        logger.warning(f"Image validation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {exc}",
        ) from exc
    except MetadataValidationError as exc:
        logger.warning(f"Metadata validation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid food metadata: {exc}",
        ) from exc
    except ModelInferenceError as exc:
        logger.error(f"Inference failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference execution failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected prediction failure: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed due to an unexpected error: {str(exc)}",
        ) from exc

    # Format strictly according to Phase 3 JSON contract
    meta_dict = analysis_result.get("metadata") or {}
    metadata_response = FoodMetadataResponse(
        temperature_celsius=meta_dict.get("temperature_celsius"),
        storage_condition=meta_dict.get("storage_condition"),
    )

    probs = analysis_result["probabilities"]
    probabilities_response = Probabilities(
        Fresh=probs.get("Fresh", 0.0),
        Moderate=probs.get("Moderate", 0.0),
        Spoiled=probs.get("Spoiled", 0.0),
    )

    return PredictionResponse(
        success=True,
        category=analysis_result["category"],
        predicted_class=analysis_result["predicted_class"],
        confidence=analysis_result["confidence"],
        probabilities=probabilities_response,
        metadata=metadata_response,
        message="Image-based freshness prediction completed.",
    )
