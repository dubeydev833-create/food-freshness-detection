"""FreshCheck AI - Image Processing and Inference Engine (Member 2).

Modules:
    - preprocessing: Format validation, corruption checking, resizing, normalization.
    - feature_extraction: Color distribution, brightness, contrast, texture analysis.
    - metadata: Validation of storage temperature, conditions, and duration.
    - inference: Model caching, prediction execution, and standard contract formatting.
    - pipeline: High-level orchestration for Member 3's backend API.
"""

from image_processing.exceptions import (
    FreshCheckError,
    ImageFormatError,
    ImageReadError,
    ImageSizeError,
    ImageValidationError,
    MetadataValidationError,
    ModelError,
    ModelInferenceError,
    ModelNotFoundError,
)
from image_processing.feature_extraction import (
    analyze_brightness,
    analyze_color_distribution,
    analyze_contrast,
    analyze_texture,
    extract_visual_features,
)
from image_processing.inference import (
    CLASS_NAMES,
    FoodFreshnessInferenceEngine,
    MockInferenceEngine,
    format_freshness_output,
)
from image_processing.metadata import (
    VALID_CATEGORIES,
    process_food_metadata,
    validate_category,
)
from image_processing.pipeline import run_freshness_analysis
from image_processing.preprocessing import (
    DEFAULT_TARGET_SIZE,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
    load_image,
    normalize_pixels,
    prepare_tensor_input,
    preprocess_image,
    resize_image,
    validate_image_format,
)

__all__ = [
    # Constants
    "CLASS_NAMES",
    "VALID_CATEGORIES",
    "SUPPORTED_FORMATS",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_TARGET_SIZE",
    # Preprocessing
    "validate_image_format",
    "load_image",
    "resize_image",
    "normalize_pixels",
    "prepare_tensor_input",
    "preprocess_image",
    # Feature Extraction
    "analyze_color_distribution",
    "analyze_brightness",
    "analyze_contrast",
    "analyze_texture",
    "extract_visual_features",
    # Metadata
    "validate_category",
    "process_food_metadata",
    # Inference & Output Contract
    "FoodFreshnessInferenceEngine",
    "MockInferenceEngine",
    "format_freshness_output",
    "run_freshness_analysis",
    # Exceptions
    "FreshCheckError",
    "ImageValidationError",
    "ImageFormatError",
    "ImageSizeError",
    "ImageReadError",
    "ModelError",
    "ModelNotFoundError",
    "ModelInferenceError",
    "MetadataValidationError",
]
