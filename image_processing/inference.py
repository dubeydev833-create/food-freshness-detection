"""Food Freshness Inference Engine for FreshCheck AI.

Loads Member 1's trained .keras deep-learning model, runs inference on
preprocessed images, and formats predictions according to the project's
standard output contract.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logger
logger = logging.getLogger("FreshCheckAI.Inference")

from image_processing.exceptions import (
    ImageValidationError,
    ModelInferenceError,
    ModelNotFoundError,
)
from image_processing.metadata import process_food_metadata, validate_category
from image_processing.preprocessing import (
    DEFAULT_TARGET_SIZE,
    NormalizationMode,
    preprocess_image,
)

# Standard class order required by FreshCheck AI specification
CLASS_NAMES: Tuple[str, str, str] = ("Fresh", "Moderate", "Spoiled")

# Supported model formats
SUPPORTED_MODEL_EXTENSIONS = frozenset({".keras", ".h5"})


def format_freshness_output(
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    category: str,
) -> Dict[str, Any]:
    """Build the standard contract output dictionary for FreshCheck AI.

    Output Contract Structure:
        {
            "predicted_class": "Fresh",
            "confidence": 0.94,
            "probabilities": {
                "Fresh": 0.94,
                "Moderate": 0.04,
                "Spoiled": 0.02
            },
            "category": "Fruits"
        }

    Args:
        predicted_class: One of 'Fresh', 'Moderate', 'Spoiled'.
        confidence: Prediction confidence score between 0.0 and 1.0.
        probabilities: Dictionary mapping each class name to its probability.
        category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').

    Returns:
        Dict[str, Any]: Exact standard contract dictionary.
    """
    canonical_cat = validate_category(category)
    return {
        "predicted_class": str(predicted_class),
        "confidence": round(float(confidence), 4),
        "probabilities": {
            cls: round(float(probabilities[cls]), 4) for cls in CLASS_NAMES
        },
        "category": canonical_cat,
    }


class FoodFreshnessInferenceEngine:
    """Production inference engine for food freshness classification.

    Loads and caches Member 1's trained Keras model, avoiding repetitive
    disk reads and model instantiation overhead across requests.
    """

    _cached_instance: Optional["FoodFreshnessInferenceEngine"] = None
    _cached_model_path: Optional[str] = None

    def __init__(
        self,
        model_path: Union[str, Path],
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        normalization_mode: NormalizationMode = "rescale",
        compile_model: bool = False,
    ) -> None:
        """Initialize the inference engine by loading the trained model.

        Args:
            model_path: File path to Member 1's trained .keras or .h5 model.
            target_size: Expected input image dimensions (height, width).
            normalization_mode: Pixel normalization mode:
                - 'rescale' (default): Scales [0, 255] -> [0.0, 1.0] (x / 255.0).
                  Standard default for baseline CNNs trained on normalized floats.
                - 'centered': Scales [0, 255] -> [-1.0, 1.0] ((x / 127.5) - 1.0).
                  REQUIRED if Member 1 used a MobileNetV2 backbone trained with
                  tf.keras.applications.mobilenet_v2.preprocess_input.
                - 'raw': Leaves pixel values as float32 in [0.0, 255.0].
                  REQUIRED if Member 1 embedded a `layers.Rescaling(1./255)` layer
                  inside the Keras model architecture to prevent double normalization.
                - 'standardize': Applies ImageNet RGB channel mean and standard deviation.
            compile_model: Whether to compile the model on load (False for pure inference).

        Raises:
            ModelNotFoundError: If the model file is missing or invalid.
        """
        self.model_path = Path(model_path)
        self.target_size = target_size
        self.normalization_mode = normalization_mode
        self.model = None

        self._load_model(compile_model=compile_model)

    def _load_model(self, compile_model: bool = False) -> None:
        """Load the Keras model weights and architecture from disk."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Trained model not found at specified path: '{self.model_path}'. "
                f"Please ensure Member 1's trained .keras model is placed in the designated directory."
            )

        if not self.model_path.is_file():
            raise ModelNotFoundError(
                f"Model path exists but is not a file: '{self.model_path}'"
            )

        if self.model_path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
            logger.warning(
                f"Model file extension '{self.model_path.suffix}' is non-standard. "
                f"Expected .keras or .h5."
            )

        logger.info(f"Loading trained Keras model from: {self.model_path}")
        try:
            # Modern Keras 3 / TensorFlow Keras loader
            import keras
            self.model = keras.models.load_model(
                str(self.model_path), compile=compile_model
            )
            logger.info("Trained model successfully loaded into memory.")
        except Exception as exc:
            # Fallback to tf.keras if top-level keras throws
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(
                    str(self.model_path), compile=compile_model
                )
                logger.info("Trained model loaded via tf.keras fallback.")
            except Exception as inner_exc:
                raise ModelInferenceError(
                    f"Failed to load trained model from '{self.model_path}': {inner_exc}"
                ) from exc

    @classmethod
    def get_instance(
        cls,
        model_path: Union[str, Path],
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        normalization_mode: NormalizationMode = "rescale",
    ) -> "FoodFreshnessInferenceEngine":
        """Get or initialize singleton instance of the inference engine.

        Ensures the model is loaded only once across the application lifecycle.

        Args:
            model_path: Path to the trained model file.
            target_size: Model input dimensions.
            normalization_mode: Normalization mode.

        Returns:
            FoodFreshnessInferenceEngine: Reusable engine instance.
        """
        path_str = str(Path(model_path).resolve())
        if cls._cached_instance is None or cls._cached_model_path != path_str:
            cls._cached_instance = cls(
                model_path=model_path,
                target_size=target_size,
                normalization_mode=normalization_mode,
            )
            cls._cached_model_path = path_str
        return cls._cached_instance

    def predict_tensor(
        self,
        input_tensor: np.ndarray,
        category: str = "Fruits",
    ) -> Dict[str, Any]:
        """Run inference on an already preprocessed batch tensor.

        Args:
            input_tensor: Preprocessed NumPy array with shape (1, H, W, 3).
            category: Food category string ('Fruits', 'Vegetables', 'Meat', 'Dairy').

        Returns:
            Dict[str, Any]: Formatted result dictionary conforming to standard output contract.

        Raises:
            ModelInferenceError: If inference execution fails.
            ImageValidationError: If tensor shape or values are invalid.
        """
        if self.model is None:
            raise ModelInferenceError("Model is not loaded.")

        if not isinstance(input_tensor, np.ndarray):
            raise ImageValidationError(f"Expected numpy ndarray, got {type(input_tensor)}")

        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1 or input_tensor.shape[3] != 3:
            raise ImageValidationError(
                f"Expected tensor of shape (1, H, W, 3), got {input_tensor.shape}"
            )

        try:
            raw_predictions = self.model(input_tensor, training=False)
            if hasattr(raw_predictions, "numpy"):
                raw_predictions = raw_predictions.numpy()
            raw_predictions = np.asarray(raw_predictions).squeeze()
        except Exception as exc:
            raise ModelInferenceError(f"Model prediction failed during execution: {exc}") from exc

        if raw_predictions.size != len(CLASS_NAMES):
            raise ModelInferenceError(
                f"Model output size ({raw_predictions.size}) does not match "
                f"expected freshness classes ({len(CLASS_NAMES)}: {CLASS_NAMES})"
            )

        # Check if output is raw logits (sum not close to 1.0 or any < 0)
        # Apply numerically stable softmax if needed
        is_probability_dist = (
            np.all(raw_predictions >= 0.0)
            and np.isclose(np.sum(raw_predictions), 1.0, atol=1e-2)
        )
        if not is_probability_dist:
            # Softmax transform
            shift_x = raw_predictions - np.max(raw_predictions)
            exps = np.exp(shift_x)
            probabilities_arr = exps / np.sum(exps)
        else:
            probabilities_arr = raw_predictions

        # Map to class names in strict order: ["Fresh", "Moderate", "Spoiled"]
        probs_dict: Dict[str, float] = {
            CLASS_NAMES[i]: float(probabilities_arr[i]) for i in range(len(CLASS_NAMES))
        }

        # Identify highest-probability class
        top_idx = int(np.argmax(probabilities_arr))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(probabilities_arr[top_idx])

        return format_freshness_output(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probs_dict,
            category=category,
        )

    def predict(
        self,
        image_input: Any,
        category: str = "Fruits",
    ) -> Dict[str, Any]:
        """Run complete end-to-end inference on an uploaded food image.

        Preprocesses the raw image, runs model inference, and packages
        the output according to the standard contract.

        Note:
            Environmental and storage metadata (e.g. temperature, storage condition,
            duration) is intentionally managed at the pipeline orchestration layer
            via `run_freshness_analysis()` to keep model inference strictly decoupled
            from auxiliary telemetry.

        Args:
            image_input: File path, bytes, file stream, or array.
            category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').

        Returns:
            Dict[str, Any]: FreshCheck AI standard contract dictionary.
        """
        # Run preprocessing
        tensor_input, _ = preprocess_image(
            image_input=image_input,
            target_size=self.target_size,
            normalization_mode=self.normalization_mode,
        )

        # Predict using model
        return self.predict_tensor(tensor_input, category=category)


class MockInferenceEngine(FoodFreshnessInferenceEngine):
    """Temporary mock inference engine for integration testing ONLY.

    IMPORTANT NOTICE:
        Per project requirements, this mock engine must ONLY be used for
        integration testing and test harness verification before Member 1
        supplies the trained weights. It must NEVER be presented as real
        predictions in production or college presentations.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        normalization_mode: NormalizationMode = "rescale",
        simulated_class: str = "Fresh",
    ) -> None:
        """Initialize temporary mock engine with predetermined or random weights."""
        self.model_path = Path("MOCK_MODEL_FOR_TESTING_ONLY")
        self.target_size = target_size
        self.normalization_mode = normalization_mode
        self.simulated_class = simulated_class
        self.model = "MOCK_INTEGRATION_TEST_OBJECT"
        logger.warning(
            "MockInferenceEngine initialized. DO NOT USE FOR ACTUAL PREDICTIONS."
        )

    def _load_model(self, compile_model: bool = False) -> None:
        """Override to bypass disk loading for integration tests."""
        pass

    def predict_tensor(
        self,
        input_tensor: np.ndarray,
        category: str = "Fruits",
    ) -> Dict[str, Any]:
        """Generate deterministic test probabilities for pipeline verification."""
        if self.simulated_class == "Fresh":
            probs = {"Fresh": 0.94, "Moderate": 0.04, "Spoiled": 0.02}
        elif self.simulated_class == "Moderate":
            probs = {"Fresh": 0.15, "Moderate": 0.80, "Spoiled": 0.05}
        else:
            probs = {"Fresh": 0.02, "Moderate": 0.08, "Spoiled": 0.90}

        top_class = self.simulated_class
        confidence = probs[top_class]

        result = format_freshness_output(
            predicted_class=top_class,
            confidence=confidence,
            probabilities=probs,
            category=category,
        )
        # Explicit tag marking mock output for developer safety
        result["_is_mock_test"] = True
        return result
