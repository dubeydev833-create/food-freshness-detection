"""Unit tests for visual feature extraction and statistical image metrics."""

import numpy as np
import pytest

from image_processing.exceptions import FreshCheckError
from image_processing.feature_extraction import (
    FeatureExtractionError,
    analyze_brightness,
    analyze_color_distribution,
    analyze_contrast,
    analyze_texture,
    extract_visual_features,
)


class TestColorDistributionAnalysis:
    """Tests for color distribution and dominant colors extraction."""

    def test_color_distribution_output_structure(self, sample_rgb_array: np.ndarray):
        result = analyze_color_distribution(sample_rgb_array, num_dominant_colors=3)

        assert "channel_means" in result
        assert "channel_std" in result
        assert "hsv_means" in result
        assert "colorfulness_score" in result
        assert "dominant_colors" in result

        # Check channel values are positive floats
        assert result["channel_means"]["red"] > 0
        assert result["channel_means"]["green"] > 0
        assert result["channel_means"]["blue"] > 0

        # Dominant colors check
        assert len(result["dominant_colors"]) == 3
        first_dominant = result["dominant_colors"][0]
        assert "rgb" in first_dominant
        assert "hex" in first_dominant
        assert "proportion" in first_dominant
        assert first_dominant["hex"].startswith("#")

    def test_invalid_shape_raises(self):
        invalid = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(FeatureExtractionError, match="Expected 3-channel RGB image"):
            analyze_color_distribution(invalid)


class TestBrightnessAnalysis:
    """Tests for photometric luminance and optical exposure assessment."""

    def test_brightness_metrics(self, sample_rgb_array: np.ndarray):
        result = analyze_brightness(sample_rgb_array)

        assert "mean_luminance" in result
        assert "std_luminance" in result
        assert "exposure_assessment" in result
        assert result["exposure_assessment"] in ["underexposed", "optimal", "overexposed"]
        assert 0.0 <= result["mean_luminance"] <= 255.0

    def test_underexposed_image(self):
        dark_img = np.full((100, 100, 3), 10, dtype=np.uint8)
        result = analyze_brightness(dark_img)
        assert result["exposure_assessment"] == "underexposed"
        assert result["mean_luminance"] < 40.0

    def test_overexposed_image(self):
        bright_img = np.full((100, 100, 3), 245, dtype=np.uint8)
        result = analyze_brightness(bright_img)
        assert result["exposure_assessment"] == "overexposed"
        assert result["mean_luminance"] > 215.0


class TestContrastAnalysis:
    """Tests for RMS and Michelson visual contrast calculations."""

    def test_contrast_metrics(self, sample_rgb_array: np.ndarray):
        result = analyze_contrast(sample_rgb_array)

        assert "rms_contrast" in result
        assert "michelson_contrast" in result
        assert "intensity_range" in result
        assert "contrast_assessment" in result
        assert result["contrast_assessment"] in ["low", "medium", "high"]
        assert result["rms_contrast"] >= 0.0

    def test_low_contrast_uniform_image(self):
        flat_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = analyze_contrast(flat_img)
        assert result["rms_contrast"] == 0.0
        assert result["contrast_assessment"] == "low"


class TestTextureAnalysis:
    """Tests for OpenCV-based texture analysis."""

    def test_texture_metrics(self, sample_rgb_array: np.ndarray):
        result = analyze_texture(sample_rgb_array)

        assert "laplacian_variance" in result
        assert "mean_gradient_magnitude" in result
        assert "canny_edge_density" in result
        assert "texture_profile" in result
        assert result["laplacian_variance"] >= 0.0
        assert 0.0 <= result["canny_edge_density"] <= 1.0


class TestConsolidatedFeatureExtraction:
    """Tests for consolidated visual features helper and disclaimer."""

    def test_extract_visual_features_structure(self, sample_rgb_array: np.ndarray):
        features = extract_visual_features(sample_rgb_array)

        assert "color_distribution" in features
        assert "brightness" in features
        assert "contrast" in features
        assert "texture" in features
        assert "analysis_disclaimer" in features
        assert "do not independently confirm" in features["analysis_disclaimer"]
