"""Custom exceptions for FreshCheck AI image processing and inference pipeline."""


class FreshCheckError(Exception):
    """Base exception for all FreshCheck AI image processing errors."""
    pass


class ImageValidationError(FreshCheckError):
    """Raised when an image fails format, size, or dimensional validation."""
    pass


class ImageFormatError(ImageValidationError):
    """Raised when an unsupported image format or extension is provided."""
    pass


class ImageSizeError(ImageValidationError):
    """Raised when an image exceeds maximum allowed byte size or dimensions."""
    pass


class ImageReadError(FreshCheckError):
    """Raised when an image file cannot be opened, decoded, or is corrupted."""
    pass


class ModelError(FreshCheckError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when the specified trained .keras model file does not exist."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference execution fails."""
    pass


class MetadataValidationError(FreshCheckError):
    """Raised when supplied food metadata fails validation checks."""
    pass
