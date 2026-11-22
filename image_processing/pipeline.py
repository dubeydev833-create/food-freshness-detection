"""Integrated food freshness analysis pipeline for FreshCheck AI.

Provides high-level orchestration across image preprocessing, model inference,
feature extraction, and storage metadata verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from image_processing.feature_extraction import extract_visual_features
from image_processing.inference import (
    CLASS_NAMES,
    FoodFreshnessInferenceEngine,
    format_freshness_output,
)
from image_processing.metadata import process_food_metadata
from image_processing.preprocessing import (
    DEFAULT_TARGET_SIZE,
    NormalizationMode,
    preprocess_image,
)


def run_freshness_analysis(
    image_input: Any,
    category: str,
    engine: FoodFreshnessInferenceEngine,
    temperature_celsius: Optional[Union[int, float]] = None,
    storage_condition: Optional[str] = None,
    storage_duration_days: Optional[Union[int, float]] = None,
    storage_duration_hours: Optional[Union[int, float]] = None,
    include_visual_features: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end freshness analysis with clean separation of concerns.

    Maintains complete separation between the trained model's deep-learning
    prediction and the auxiliary environmental metadata/visual features.

    Args:
        image_input: Image file path, raw bytes, stream, PIL Image, or array.
        category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').
        engine: Loaded FoodFreshnessInferenceEngine instance.
        temperature_celsius: Optional ambient/storage temperature in C.
        storage_condition: Optional storage setting (e.g. 'Refrigerated').
        storage_duration_days: Optional duration in days.
        storage_duration_hours: Optional duration in hours.
        include_visual_features: Whether to extract and include visual descriptors.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'prediction': Exact standard contract output from the trained model.
            - 'metadata': Validated metadata (returned separately per Phase 5).
            - 'visual_features': Optical feature metrics (when requested).
    """
    # 1. Preprocess image
    tensor_input, rgb_clean = preprocess_image(
        image_input=image_input,
        target_size=engine.target_size,
        normalization_mode=engine.normalization_mode,
    )

    # 2. Run inference on image tensor (strictly model-based prediction)
    prediction = engine.predict_tensor(tensor_input, category=category)

    # 3. Process metadata separately without biasing model prediction (Phase 5)
    metadata = process_food_metadata(
        category=category,
        temperature_celsius=temperature_celsius,
        storage_condition=storage_condition,
        storage_duration_days=storage_duration_days,
        storage_duration_hours=storage_duration_hours,
    )

    # 4. Optional visual feature extraction
    visual_features = None
    if include_visual_features:
        visual_features = extract_visual_features(rgb_clean)

    return {
        "prediction": prediction,
        "metadata": metadata,
        "visual_features": visual_features,
    }
