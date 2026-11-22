"""Unit tests for image loading, validation, resizing, normalization, and tensor formatting."""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from image_processing.exceptions import (
    ImageFormatError,
    ImageReadError,
    ImageSizeError,
    ImageValidationError,
)
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


class TestImageFormatValidation:
    """Tests for image format and header validation (JPG, JPEG, PNG)."""

    def test_validate_jpeg_file(self, sample_jpeg_file: Path):
        fmt = validate_image_format(sample_jpeg_file)
        assert fmt == "JPEG"

    def test_validate_png_file(self, sample_png_file: Path):
        fmt = validate_image_format(sample_png_file)
        assert fmt == "PNG"

    def test_validate_jpeg_bytes(self, sample_jpeg_bytes: bytes):
        fmt = validate_image_format(sample_jpeg_bytes)
        assert fmt == "JPEG"

    def test_validate_png_bytes(self, sample_png_bytes: bytes):
        fmt = validate_image_format(sample_png_bytes)
        assert fmt == "PNG"

    def test_reject_invalid_extension(self, invalid_extension_file: Path):
        with pytest.raises(ImageFormatError, match="Unsupported file extension"):
            validate_image_format(invalid_extension_file)

    def test_reject_corrupted_header(self, corrupted_image_file: Path):
        with pytest.raises(ImageFormatError):
            validate_image_format(corrupted_image_file)

    def test_reject_unsupported_format(self, tmp_path: Path):
        # Create a GIF file
        gif_path = tmp_path / "animation.gif"
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(gif_path, format="GIF")

        with pytest.raises(ImageFormatError, match="Unsupported file extension"):
            validate_image_format(gif_path)


class TestImageLoading:
    """Tests for loading and decoding images from multiple source types."""

    def test_load_from_jpeg_path(self, sample_jpeg_file: Path):
        img = load_image(sample_jpeg_file)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_load_from_png_path(self, sample_png_file: Path):
        img = load_image(sample_png_file)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_from_bytes(self, sample_jpeg_bytes: bytes):
        img = load_image(sample_jpeg_bytes)
        assert isinstance(img, np.ndarray)
        assert img.shape == (300, 300, 3)

    def test_load_from_stream(self, sample_png_bytes: bytes):
        stream = io.BytesIO(sample_png_bytes)
        img = load_image(stream)
        assert isinstance(img, np.ndarray)
        assert img.shape == (300, 300, 3)

    def test_load_from_pil_image(self, sample_rgb_array: np.ndarray):
        pil_img = Image.fromarray(sample_rgb_array)
        img = load_image(pil_img)
        assert isinstance(img, np.ndarray)
        assert np.array_equal(img, sample_rgb_array)

    def test_load_from_numpy_array(self, sample_rgb_array: np.ndarray):
        img = load_image(sample_rgb_array)
        assert isinstance(img, np.ndarray)
        assert np.array_equal(img, sample_rgb_array)

    def test_load_nonexistent_file(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.jpg"
        with pytest.raises(ImageReadError, match="does not exist"):
            load_image(missing)

    def test_load_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_bytes(b"")
        with pytest.raises(ImageReadError, match="empty"):
            load_image(empty_file)

    def test_load_empty_bytes(self):
        with pytest.raises(ImageReadError, match="empty"):
            load_image(b"")

    def test_reject_oversized_file(self, sample_jpeg_file: Path):
        with pytest.raises(ImageSizeError, match="exceeds maximum allowed"):
            load_image(sample_jpeg_file, max_file_size=100)

    def test_reject_oversized_dimensions(self, sample_rgb_array: np.ndarray):
        with pytest.raises(ImageSizeError, match="exceed maximum allowed dimension"):
            load_image(sample_rgb_array, max_dimension=100)


class TestImageResizing:
    """Tests for resizing images to model dimensions."""

    def test_direct_resize(self, sample_rgb_array: np.ndarray):
        target = (224, 224)
        resized = resize_image(sample_rgb_array, target_size=target, keep_aspect_ratio=False)
        assert resized.shape == (224, 224, 3)
        assert resized.dtype == np.uint8

    def test_custom_target_size(self, sample_rgb_array: np.ndarray):
        target = (128, 256)
        resized = resize_image(sample_rgb_array, target_size=target, keep_aspect_ratio=False)
        assert resized.shape == (128, 256, 3)

    def test_letterbox_padding(self, sample_rgb_array: np.ndarray):
        # Asymmetric target size with aspect ratio preserved
        target = (200, 400)
        resized = resize_image(sample_rgb_array, target_size=target, keep_aspect_ratio=True)
        assert resized.shape == (200, 400, 3)

    def test_same_dimensions_returns_copy(self, sample_rgb_array: np.ndarray):
        target = sample_rgb_array.shape[:2]
        resized = resize_image(sample_rgb_array, target_size=target)
        assert resized.shape == sample_rgb_array.shape
        assert np.array_equal(resized, sample_rgb_array)


class TestPixelNormalization:
    """Tests for model pixel normalization modes."""

    def test_rescale_mode(self, sample_rgb_array: np.ndarray):
        norm = normalize_pixels(sample_rgb_array, mode="rescale")
        assert norm.dtype == np.float32
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_centered_mode(self, sample_rgb_array: np.ndarray):
        norm = normalize_pixels(sample_rgb_array, mode="centered")
        assert norm.dtype == np.float32
        assert norm.min() >= -1.0
        assert norm.max() <= 1.0

    def test_standardize_mode(self, sample_rgb_array: np.ndarray):
        norm = normalize_pixels(sample_rgb_array, mode="standardize")
        assert norm.dtype == np.float32
        assert norm.shape == sample_rgb_array.shape

    def test_raw_mode(self, sample_rgb_array: np.ndarray):
        norm = normalize_pixels(sample_rgb_array, mode="raw")
        assert norm.dtype == np.float32
        assert norm.max() <= 255.0

    def test_invalid_mode_raises(self, sample_rgb_array: np.ndarray):
        with pytest.raises(ValueError, match="Unsupported normalization mode"):
            normalize_pixels(sample_rgb_array, mode="nonexistent_mode")  # type: ignore


class TestTensorPreparation:
    """Tests for tensor formatting and batch expansion."""

    def test_prepare_tensor_input_expands_batch(self, sample_rgb_array: np.ndarray):
        float_img = sample_rgb_array.astype(np.float32) / 255.0
        batch = prepare_tensor_input(float_img)
        assert batch.shape == (1, 300, 300, 3)
        assert batch.dtype == np.float32

    def test_prepare_tensor_existing_batch(self):
        arr = np.zeros((1, 224, 224, 3), dtype=np.float32)
        batch = prepare_tensor_input(arr)
        assert batch.shape == (1, 224, 224, 3)

    def test_prepare_tensor_invalid_dimensions(self):
        arr = np.zeros((224, 224), dtype=np.float32)
        with pytest.raises(ImageValidationError, match="Expected image with 3 or 4 dimensions"):
            prepare_tensor_input(arr)


class TestFullPreprocessingPipeline:
    """Tests for end-to-end preprocessing."""

    def test_preprocess_jpeg(self, sample_jpeg_file: Path):
        tensor, original = preprocess_image(sample_jpeg_file, target_size=(224, 224))
        assert tensor.shape == (1, 224, 224, 3)
        assert tensor.dtype == np.float32
        assert original.shape == (300, 300, 3)
        assert original.dtype == np.uint8

    def test_preprocess_bytes(self, sample_png_bytes: bytes):
        tensor, original = preprocess_image(sample_png_bytes, target_size=(224, 224))
        assert tensor.shape == (1, 224, 224, 3)
        assert tensor.dtype == np.float32
        assert original.shape == (300, 300, 3)
