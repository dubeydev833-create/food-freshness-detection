"""Unit and integration tests for FreshCheck AI inference engine and pipeline."""

from pathlib import Path
import numpy as np
import pytest

from image_processing.exceptions import (
    ImageValidationError,
    ModelInferenceError,
    ModelNotFoundError,
)
from image_processing.inference import (
    CLASS_NAMES,
    FoodFreshnessInferenceEngine,
    MockInferenceEngine,
    format_freshness_output,
)
from image_processing.pipeline import run_freshness_analysis


class TestStandardOutputContract:
    """Tests for Phase 6 standard output dictionary contract."""

    def test_format_freshness_output_exact_structure(self):
        probs = {"Fresh": 0.94, "Moderate": 0.04, "Spoiled": 0.02}
        result = format_freshness_output(
            predicted_class="Fresh",
            confidence=0.94,
            probabilities=probs,
            category="Fruits",
        )

        # Exact keys matching project contract
        assert set(result.keys()) == {
            "predicted_class",
            "confidence",
            "probabilities",
            "category",
        }
        assert result["predicted_class"] == "Fresh"
        assert result["confidence"] == 0.94
        assert result["category"] == "Fruits"

        # Check probabilities sub-dictionary and class ordering
        assert set(result["probabilities"].keys()) == set(CLASS_NAMES)
        assert result["probabilities"]["Fresh"] == 0.94
        assert result["probabilities"]["Moderate"] == 0.04
        assert result["probabilities"]["Spoiled"] == 0.02

    def test_class_names_exact_order(self):
        assert CLASS_NAMES == ("Fresh", "Moderate", "Spoiled")


class TestInferenceEngineModelHandling:
    """Tests for model loading, error handling, and caching."""

    def test_missing_model_file_raises_error(self, tmp_path: Path):
        missing_path = tmp_path / "nonexistent_model.keras"
        with pytest.raises(ModelNotFoundError, match="Trained model not found"):
            FoodFreshnessInferenceEngine(model_path=missing_path)

    def test_model_singleton_caching(self, trained_keras_model_file: Path):
        engine1 = FoodFreshnessInferenceEngine.get_instance(trained_keras_model_file)
        engine2 = FoodFreshnessInferenceEngine.get_instance(trained_keras_model_file)
        assert engine1 is engine2

    def test_real_keras_model_inference(
        self,
        trained_keras_model_file: Path,
        sample_jpeg_file: Path,
    ):
        engine = FoodFreshnessInferenceEngine(model_path=trained_keras_model_file)
        result = engine.predict(sample_jpeg_file, category="Vegetables")

        assert result["category"] == "Vegetables"
        assert result["predicted_class"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0

        probs = result["probabilities"]
        assert len(probs) == 3
        assert pytest.approx(sum(probs.values()), abs=1e-2) == 1.0

        # Confidence should match the max probability
        max_prob = max(probs.values())
        assert pytest.approx(result["confidence"], abs=1e-4) == max_prob

    def test_predict_tensor_invalid_shape_raises(self, trained_keras_model_file: Path):
        engine = FoodFreshnessInferenceEngine(model_path=trained_keras_model_file)
        invalid_tensor = np.zeros((224, 224, 3), dtype=np.float32)  # missing batch dim
        with pytest.raises(ImageValidationError, match="Expected tensor of shape"):
            engine.predict_tensor(invalid_tensor, category="Meat")


class TestMockInferenceEngine:
    """Tests for integration-test mock engine."""

    def test_mock_engine_returns_valid_contract(self):
        mock_engine = MockInferenceEngine(simulated_class="Fresh")
        tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        result = mock_engine.predict_tensor(tensor, category="Dairy")

        assert result["predicted_class"] == "Fresh"
        assert result["confidence"] == 0.94
        assert result["probabilities"]["Fresh"] == 0.94
        assert result["category"] == "Dairy"
        # Mock flag present to prevent confusing mock with real prediction
        assert result.get("_is_mock_test") is True


class TestEndToEndPipeline:
    """Tests for integrated analysis pipeline."""

    def test_run_freshness_analysis_full(
        self,
        trained_keras_model_file: Path,
        sample_jpeg_file: Path,
    ):
        engine = FoodFreshnessInferenceEngine(model_path=trained_keras_model_file)
        response = run_freshness_analysis(
            image_input=sample_jpeg_file,
            category="Meat",
            engine=engine,
            temperature_celsius=2.5,
            storage_condition="Refrigerated",
            storage_duration_days=2.0,
            include_visual_features=True,
        )

        assert "prediction" in response
        assert "metadata" in response
        assert "visual_features" in response

        # Prediction matches contract
        pred = response["prediction"]
        assert pred["predicted_class"] in CLASS_NAMES
        assert pred["category"] == "Meat"

        # Metadata is separated and unmodified
        meta = response["metadata"]
        assert meta["category"] == "Meat"
        assert meta["temperature_celsius"] == 2.5
        assert meta["storage_condition"] == "Refrigerated"

        # Visual features extracted
        vf = response["visual_features"]
        assert "color_distribution" in vf
        assert "brightness" in vf
        assert "contrast" in vf
        assert "texture" in vf
