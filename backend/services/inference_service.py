"""Inference service connecting FastAPI backend to Member 2's inference engine.

Strictly encapsulates production model execution and provides clean abstraction
for dependency injection in tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from backend.config import settings
from image_processing.exceptions import (
    FreshCheckError,
    ModelInferenceError,
    ModelNotFoundError,
)

logger = logging.getLogger("FreshCheckAI.InferenceService")


class InferenceService:
    """Service managing deep-learning model inference for food freshness assessment."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        engine: Optional[Any] = None,
    ) -> None:
        """Initialize inference service.

        Args:
            model_path: Path to Member 1's trained .keras model file.
            engine: Optional pre-loaded engine (used for testing / dependency injection).
        """
        self.model_path = Path(model_path) if model_path else settings.MODEL_PATH
        self._engine = engine

    def is_model_available(self) -> bool:
        """Check whether the model file or pre-loaded engine is available."""
        if self._engine is not None:
            return True
        return self.model_path.exists() and self.model_path.is_file()

    def get_engine(self) -> Any:
        """Get or lazily instantiate Member 2's FoodFreshnessInferenceEngine.

        Returns:
            FoodFreshnessInferenceEngine: Cached singleton inference engine.

        Raises:
            ModelNotFoundError: If the trained model file is missing on disk.
        """
        if self._engine is not None:
            return self._engine

        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Trained model not found at specified path: '{self.model_path}'. "
                f"Please ensure Member 1's trained .keras model is placed in the designated directory."
            )

        from image_processing.inference import FoodFreshnessInferenceEngine

        self._engine = FoodFreshnessInferenceEngine.get_instance(self.model_path)
        return self._engine

    def predict(
        self,
        image_bytes: bytes,
        category: str,
        temperature_celsius: Optional[Union[int, float]] = None,
        storage_condition: Optional[str] = None,
        storage_duration_days: Optional[Union[int, float]] = None,
    ) -> Dict[str, Any]:
        """Execute freshness analysis using Member 2's pipeline and Member 1's model.

        Args:
            image_bytes: Raw bytes of the uploaded food image.
            category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').
            temperature_celsius: Optional observed temperature.
            storage_condition: Optional storage condition string.
            storage_duration_days: Optional storage duration.

        Returns:
            Dict[str, Any]: Formatted result matching the standard contract.

        Raises:
            ModelNotFoundError: If the model weights are not found on disk.
            FreshCheckError: On validation or inference failure.
        """
        engine = self.get_engine()
        from image_processing.pipeline import run_freshness_analysis

        return run_freshness_analysis(
            image_input=image_bytes,
            category=category,
            engine=engine,
            temperature_celsius=temperature_celsius,
            storage_condition=storage_condition,
            storage_duration_days=storage_duration_days,
            include_visual_features=False,
        )


# Singleton service instance for FastAPI dependency injection
_inference_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """FastAPI dependency for accessing the inference service."""
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService(model_path=settings.MODEL_PATH)
    return _inference_service_instance
