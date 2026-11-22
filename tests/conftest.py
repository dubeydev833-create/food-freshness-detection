"""Pytest configuration and shared fixtures for FreshCheck AI tests."""

import io
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def sample_rgb_array() -> np.ndarray:
    """Return a synthetic 300x300 RGB test image."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Add varied color patches to simulate a food item
    img[:150, :150] = [220, 50, 50]    # Reddish patch
    img[:150, 150:] = [50, 180, 50]    # Greenish patch
    img[150:, :150] = [180, 140, 50]   # Yellow/brown patch
    img[150:, 150:] = [200, 200, 200]  # Light patch
    return img


@pytest.fixture
def sample_jpeg_file(tmp_path: Path, sample_rgb_array: np.ndarray) -> Path:
    """Create a temporary valid JPEG image file."""
    file_path = tmp_path / "test_fruit.jpg"
    bgr = cv2.cvtColor(sample_rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file_path), bgr)
    return file_path


@pytest.fixture
def sample_png_file(tmp_path: Path, sample_rgb_array: np.ndarray) -> Path:
    """Create a temporary valid PNG image file."""
    file_path = tmp_path / "test_veg.png"
    bgr = cv2.cvtColor(sample_rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file_path), bgr)
    return file_path


@pytest.fixture
def sample_jpeg_bytes(sample_rgb_array: np.ndarray) -> bytes:
    """Return encoded JPEG bytes."""
    pil_img = Image.fromarray(sample_rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes(sample_rgb_array: np.ndarray) -> bytes:
    """Return encoded PNG bytes."""
    pil_img = Image.fromarray(sample_rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def invalid_extension_file(tmp_path: Path) -> Path:
    """Create a file with an unsupported extension."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("not an image")
    return file_path


@pytest.fixture
def corrupted_image_file(tmp_path: Path) -> Path:
    """Create a corrupted JPG file."""
    file_path = tmp_path / "corrupted.jpg"
    file_path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted-junk-payload")
    return file_path


@pytest.fixture
def trained_keras_model_file(tmp_path: Path) -> Path:
    """Create a real minimal Keras .keras model file for end-to-end inference testing."""
    import keras
    from keras import layers

    # Build a tiny 3-class classifier matching FreshCheck AI requirements
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.GlobalAveragePooling2D()(inputs)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="test_freshcheck_model")

    model_path = tmp_path / "freshcheck_model.keras"
    model.save(str(model_path))
    return model_path
