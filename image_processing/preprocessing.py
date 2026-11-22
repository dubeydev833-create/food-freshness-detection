"""Image preprocessing pipeline for FreshCheck AI.

Handles image loading, format validation, corruption checking, resizing,
color normalization, and tensor batch preparation for deep learning inference.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Literal, Tuple, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from image_processing.exceptions import (
    ImageFormatError,
    ImageReadError,
    ImageSizeError,
    ImageValidationError,
)

# Supported image formats
SUPPORTED_FORMATS = frozenset({"JPEG", "JPG", "PNG"})
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

# Safety thresholds
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_IMAGE_DIMENSION = 8192  # 8192 pixels in either axis
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)

# Set PIL safety limit for decompression bombs
Image.MAX_IMAGE_PIXELS = 64_000_000

NormalizationMode = Literal["rescale", "centered", "standardize", "raw"]


def validate_image_format(
    file_source: Union[str, Path, bytes, BinaryIO],
) -> str:
    """Validate that the given image is in an authorized format (JPG, JPEG, PNG).

    Inspects both file extension (when available) and binary image headers
    via Pillow to ensure the file format is authentic and supported.

    Args:
        file_source: File path, raw bytes, or file-like binary stream.

    Returns:
        The detected image format in uppercase (e.g. 'JPEG' or 'PNG').

    Raises:
        ImageFormatError: If the image format is unsupported or invalid.
        ImageValidationError: If the input cannot be verified.
    """
    # Check extension if a path is supplied
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageFormatError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    # Validate actual file header using PIL
    try:
        stream: BinaryIO
        if isinstance(file_source, (str, Path)):
            with open(file_source, "rb") as f:
                header_bytes = f.read(2048)
                stream = io.BytesIO(header_bytes)
                with Image.open(stream) as img:
                    img_format = img.format
        elif isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
            with Image.open(stream) as img:
                img_format = img.format
        elif hasattr(file_source, "read") and hasattr(file_source, "seek"):
            pos = file_source.tell()
            with Image.open(file_source) as img:
                img_format = img.format
            file_source.seek(pos)
        else:
            raise ImageValidationError(f"Unsupported file source type: {type(file_source)}")

        if not img_format or img_format.upper() not in SUPPORTED_FORMATS:
            raise ImageFormatError(
                f"Unsupported image format '{img_format}'. "
                f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
            )

        return img_format.upper()

    except (UnidentifiedImageError, OSError) as exc:
        raise ImageFormatError(f"Could not identify valid image format: {exc}") from exc


def load_image(
    image_input: Union[str, Path, bytes, BinaryIO, np.ndarray, Image.Image],
    max_file_size: int = MAX_FILE_SIZE_BYTES,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """Load and validate an uploaded food image into an RGB NumPy array.

    Supports file paths, byte buffers, streams, existing PIL images, or arrays.
    Performs integrity, size, and corruption validation checks.

    Args:
        image_input: File path, raw bytes, file-like stream, PIL Image, or ndarray.
        max_file_size: Maximum allowable file size in bytes (default 20MB).
        max_dimension: Maximum allowable dimension in either axis (default 8192px).

    Returns:
        np.ndarray: Decoded image in RGB channel order with shape (H, W, 3) and uint8 dtype.

    Raises:
        ImageSizeError: If the image exceeds file size or dimension limits.
        ImageFormatError: If the format is not JPG/JPEG/PNG.
        ImageReadError: If the image cannot be read or is corrupted.
    """
    # 1. Handle existing NumPy ndarray
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ImageReadError("Provided NumPy array is empty.")
        _check_dimensions(image_input, max_dimension)
        if image_input.ndim == 2:
            # Grayscale to RGB
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        if image_input.ndim == 3:
            if image_input.shape[2] == 4:
                # RGBA to RGB
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            if image_input.shape[2] == 3:
                return image_input.copy()
        raise ImageReadError(f"Unexpected array shape for image: {image_input.shape}")

    # 2. Handle existing PIL Image
    if isinstance(image_input, Image.Image):
        try:
            rgb_image = image_input.convert("RGB")
            arr = np.array(rgb_image, dtype=np.uint8)
            _check_dimensions(arr, max_dimension)
            return arr
        except Exception as exc:
            raise ImageReadError(f"Failed to process PIL Image: {exc}") from exc

    # 3. Handle file path, bytes, or stream
    raw_bytes: bytes
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise ImageReadError(f"Image file does not exist: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise ImageReadError(f"Image file is empty (0 bytes): {path}")
        if file_size > max_file_size:
            raise ImageSizeError(
                f"Image file size ({file_size / (1024 * 1024):.2f} MB) "
                f"exceeds maximum allowed {max_file_size / (1024 * 1024):.2f} MB."
            )
        validate_image_format(path)
        with open(path, "rb") as f:
            raw_bytes = f.read()

    elif isinstance(image_input, bytes):
        if len(image_input) == 0:
            raise ImageReadError("Image byte sequence is empty (0 bytes).")
        if len(image_input) > max_file_size:
            raise ImageSizeError(
                f"Image payload ({len(image_input) / (1024 * 1024):.2f} MB) "
                f"exceeds maximum allowed {max_file_size / (1024 * 1024):.2f} MB."
            )
        validate_image_format(image_input)
        raw_bytes = image_input

    elif hasattr(image_input, "read"):
        content = image_input.read()
        if len(content) == 0:
            raise ImageReadError("Provided stream is empty (0 bytes).")
        if len(content) > max_file_size:
            raise ImageSizeError(
                f"Stream content size ({len(content) / (1024 * 1024):.2f} MB) "
                f"exceeds maximum allowed {max_file_size / (1024 * 1024):.2f} MB."
            )
        validate_image_format(content)
        raw_bytes = content

    else:
        raise ImageValidationError(f"Unsupported image input type: {type(image_input)}")

    # Decode bytes using OpenCV
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if bgr_img is None or bgr_img.size == 0:
        # Fallback to PIL in case cv2 failed to decode a valid variant
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                pil_img.verify()  # Verify integrity
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                rgb_img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                _check_dimensions(rgb_img, max_dimension)
                return rgb_img
        except Exception as exc:
            raise ImageReadError(f"Corrupted or unreadable image data: {exc}") from exc

    _check_dimensions(bgr_img, max_dimension)
    # Convert OpenCV BGR to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def _check_dimensions(img_array: np.ndarray, max_dimension: int) -> None:
    """Ensure image dimensions do not exceed safe operational bounds."""
    height, width = img_array.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageReadError(f"Invalid image dimensions: {width}x{height}")
    if height > max_dimension or width > max_dimension:
        raise ImageSizeError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension "
            f"of {max_dimension}px."
        )


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    keep_aspect_ratio: bool = False,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize an RGB image to the model's required input dimensions.

    Args:
        image: RGB NumPy array of shape (H, W, 3).
        target_size: Tuple of (height, width) required by the model.
        keep_aspect_ratio: If True, scales while preserving aspect ratio and pads.
        pad_color: RGB fill color when keep_aspect_ratio is True.

    Returns:
        np.ndarray: Resized image with shape (target_height, target_width, 3).
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    if orig_h == target_h and orig_w == target_w:
        return image.copy()

    # Determine interpolation method
    # INTER_AREA is best for shrinking; INTER_CUBIC is best for enlarging
    is_downscaling = (target_w < orig_w) and (target_h < orig_h)
    interpolation = cv2.INTER_AREA if is_downscaling else cv2.INTER_CUBIC

    if not keep_aspect_ratio:
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    # Aspect-ratio preserving resize with letterbox padding
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create target canvas and center the resized image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=image.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return padded


def normalize_pixels(
    image: np.ndarray,
    mode: NormalizationMode = "rescale",
) -> np.ndarray:
    """Normalize pixel intensities according to the trained model's specification.

    Supported modes:
        - 'rescale': Scales pixel values from [0, 255] to [0.0, 1.0] (Keras standard).
        - 'centered': Scales pixel values to [-1.0, 1.0] (e.g. MobileNetV2 / Inception).
        - 'standardize': Applies ImageNet mean and standard deviation normalization.
        - 'raw': Leaves values as float32 [0.0, 255.0] (models with internal Rescaling layer).

    Args:
        image: RGB NumPy array with values in [0, 255].
        mode: Normalization mode string.

    Returns:
        np.ndarray: Normalized image as float32 array.

    Raises:
        ValueError: If an unsupported normalization mode is specified.
    """
    img_float = image.astype(np.float32)

    if mode == "rescale":
        return img_float / 255.0

    if mode == "centered":
        return (img_float / 127.5) - 1.0

    if mode == "standardize":
        # ImageNet channel-wise mean and std (RGB order)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        scaled = img_float / 255.0
        return (scaled - mean) / std

    if mode == "raw":
        return img_float

    raise ValueError(
        f"Unsupported normalization mode '{mode}'. "
        f"Must be one of ['rescale', 'centered', 'standardize', 'raw']."
    )


def prepare_tensor_input(
    image: np.ndarray,
) -> np.ndarray:
    """Convert an image array into the correct tensor batch format for inference.

    Ensures the array has shape (1, height, width, 3) and dtype float32.

    Args:
        image: Normalized RGB image array of shape (H, W, 3).

    Returns:
        np.ndarray: Batch tensor with shape (1, H, W, 3) and dtype float32.
    """
    if image.ndim == 3:
        batch = np.expand_dims(image, axis=0)
    elif image.ndim == 4:
        batch = image
    else:
        raise ImageValidationError(
            f"Expected image with 3 or 4 dimensions, got shape {image.shape}"
        )

    return batch.astype(np.float32)


def preprocess_image(
    image_input: Union[str, Path, bytes, BinaryIO, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    normalization_mode: NormalizationMode = "rescale",
    keep_aspect_ratio: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute full preprocessing pipeline on an input food image.

    Combines loading, validation, resizing, normalization, and tensor formatting.

    Args:
        image_input: File path, bytes, stream, PIL Image, or ndarray.
        target_size: Desired model input dimensions (height, width).
        normalization_mode: Normalization strategy ('rescale', 'centered', etc.).
        keep_aspect_ratio: Whether to preserve aspect ratio via padding.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - tensor_input: Batch tensor with shape (1, target_h, target_w, 3) for the model.
            - rgb_original: Clean decoded RGB image (H, W, 3) uint8 suitable for feature analysis.
    """
    # 1. Load and validate
    rgb_clean = load_image(image_input)

    # 2. Resize
    resized = resize_image(
        rgb_clean,
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
    )

    # 3. Normalize
    normalized = normalize_pixels(resized, mode=normalization_mode)

    # 4. Convert to batch tensor
    tensor_input = prepare_tensor_input(normalized)

    return tensor_input, rgb_clean
