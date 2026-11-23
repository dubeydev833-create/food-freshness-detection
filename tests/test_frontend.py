"""Unit and Integration Tests for FreshCheck AI Frontend (Member 4).

Validates application configuration, API client resilience, Plotly chart generation,
storage recommendations, metric calculations, and error handling.
"""

import io
import json
from unittest.mock import MagicMock, patch
from PIL import Image
import plotly.graph_objects as go
import pytest

from frontend.api_client import (
    BackendAPIError,
    BackendConnectionError,
    BackendTimeoutError,
    FreshCheckAPIClient,
    MockFreshCheckClient,
)
from frontend.components.charts import (
    create_category_distribution_chart,
    create_confidence_gauge,
    create_freshness_donut_chart,
    create_probability_bar_chart,
    create_timeline_chart,
)
from frontend.components.recommendations import get_storage_recommendations
from frontend.config import (
    CLASS_COLORS,
    FOOD_SAFETY_DISCLAIMER,
    FRESHNESS_CLASSES,
    SUPPORTED_CATEGORIES,
)


# ============================================================================
# 1. CONFIGURATION TESTS
# ============================================================================
def test_config_constants():
    """Verify supported categories and freshness classes match project contract."""
    assert "Fruits" in SUPPORTED_CATEGORIES
    assert "Vegetables" in SUPPORTED_CATEGORIES
    assert "Meat" in SUPPORTED_CATEGORIES
    assert "Dairy" in SUPPORTED_CATEGORIES
    assert len(SUPPORTED_CATEGORIES) == 4

    assert FRESHNESS_CLASSES == ("Fresh", "Moderate", "Spoiled")

    # Colors must be mapped for each class
    for cls in FRESHNESS_CLASSES:
        assert cls in CLASS_COLORS
        assert CLASS_COLORS[cls].startswith("#")

    # Safety disclaimer must be present and explicit
    assert "cannot confirm" in FOOD_SAFETY_DISCLAIMER.lower()
    assert "confidence" in FOOD_SAFETY_DISCLAIMER.lower()


# ============================================================================
# 2. API CLIENT TESTS
# ============================================================================
def test_api_client_initialization():
    """Verify API client initializes with default or custom URL."""
    client = FreshCheckAPIClient("http://localhost:8000/")
    assert client.base_url == "http://localhost:8000"

    client.set_base_url("http://192.168.1.100:5000/")
    assert client.base_url == "http://192.168.1.100:5000"


@patch("requests.get")
def test_api_client_health_check_success(mock_get):
    """Test successful health probe."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {"service": "FreshCheck AI Backend"}
    mock_get.return_value = mock_resp

    client = FreshCheckAPIClient("http://localhost:8000")
    health = client.check_health()

    assert health["status"] == "healthy"
    assert "latency_ms" in health
    assert health["url"] == "http://localhost:8000"


@patch("requests.get")
def test_api_client_connection_error(mock_get):
    """Verify BackendConnectionError is raised when backend is offline."""
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = FreshCheckAPIClient("http://localhost:8000")
    with pytest.raises(BackendConnectionError) as exc_info:
        client.check_health()
    assert "Could not connect to backend server" in str(exc_info.value)


@patch("requests.get")
def test_api_client_timeout_error(mock_get):
    """Verify BackendTimeoutError is raised when request times out."""
    import requests
    mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

    client = FreshCheckAPIClient("http://localhost:8000")
    with pytest.raises(BackendTimeoutError):
        client.check_health()


@patch("requests.post")
def test_api_client_predict_standard_contract(mock_post):
    """Test successful prediction request conforming to standard contract."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {
        "predicted_class": "Fresh",
        "confidence": 0.9412,
        "probabilities": {"Fresh": 0.9412, "Moderate": 0.0450, "Spoiled": 0.0138},
        "category": "Fruits",
        "metadata": {"temperature_celsius": 4.0},
    }
    mock_post.return_value = mock_resp

    client = FreshCheckAPIClient("http://localhost:8000")
    dummy_bytes = b"fake_jpeg_image_data"

    result = client.predict(
        image_bytes=dummy_bytes,
        filename="apple.jpg",
        category="Fruits",
        temperature_celsius=4.0,
    )

    assert result["predicted_class"] == "Fresh"
    assert result["confidence"] == 0.9412
    assert result["category"] == "Fruits"
    assert result["probabilities"]["Fresh"] == 0.9412


@patch("requests.post")
def test_api_client_predict_nested_backward_compatibility(mock_post):
    """Test backward compatibility when backend returns nested 'prediction' object."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {
        "prediction": {
            "predicted_class": "Moderate",
            "confidence": 0.82,
            "probabilities": {"Fresh": 0.10, "Moderate": 0.82, "Spoiled": 0.08},
            "category": "Vegetables",
        }
    }
    mock_post.return_value = mock_resp

    client = FreshCheckAPIClient("http://localhost:8000")
    result = client.predict(
        image_bytes=b"fake_bytes",
        filename="broccoli.png",
        category="Vegetables",
    )

    assert result["predicted_class"] == "Moderate"
    assert result["confidence"] == 0.82
    assert result["category"] == "Vegetables"


@patch("requests.post")
def test_api_client_predict_server_error(mock_post):
    """Test 422/500 API error propagation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": "Invalid image payload"}
    mock_post.return_value = mock_resp

    client = FreshCheckAPIClient("http://localhost:8000")
    with pytest.raises(BackendAPIError) as exc_info:
        client.predict(b"bad_bytes", "bad.jpg", "Fruits")

    assert exc_info.value.status_code == 422
    assert "Invalid image payload" in exc_info.value.detail


# ============================================================================
# 3. MOCK CLIENT TESTS (ISOLATED UI DEV MODE)
# ============================================================================
def test_mock_client_behavior():
    """Verify mock client provides valid contracts with explicit mock tag."""
    mock_client = MockFreshCheckClient()

    health = mock_client.check_health()
    assert health["status"] == "mock_mode"
    assert health["_is_mock"] is True

    history = mock_client.get_history()
    assert len(history) >= 5
    assert all(item.get("_is_mock") is True for item in history)

    prediction = mock_client.predict(
        image_bytes=b"dummy",
        filename="sample.jpg",
        category="Dairy",
        temperature_celsius=3.5,
    )
    assert prediction["predicted_class"] in FRESHNESS_CLASSES
    assert prediction["_is_mock"] is True
    assert prediction["category"] == "Dairy"

    # History should have received new prediction at index 0
    updated_history = mock_client.get_history()
    assert len(updated_history) == len(history) + 1
    assert updated_history[0]["category"] == "Dairy"

    # Clear history
    clear_res = mock_client.clear_history()
    assert clear_res["status"] == "success"
    assert len(mock_client.get_history()) == 0


# ============================================================================
# 4. PLOTLY VISUALIZATIONS TESTS
# ============================================================================
def test_create_confidence_gauge():
    """Test confidence gauge generation."""
    fig = create_confidence_gauge(0.942, "Fresh")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 94.2


def test_create_probability_bar_chart():
    """Test horizontal probability bar chart."""
    probs = {"Fresh": 0.90, "Moderate": 0.08, "Spoiled": 0.02}
    fig = create_probability_bar_chart(probs)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 3


def test_create_category_distribution_chart():
    """Test donut chart for categories."""
    # With data
    history = [
        {"category": "Fruits", "predicted_class": "Fresh"},
        {"category": "Vegetables", "predicted_class": "Moderate"},
        {"category": "Fruits", "predicted_class": "Fresh"},
    ]
    fig = create_category_distribution_chart(history)
    assert isinstance(fig, go.Figure)

    # Empty data
    empty_fig = create_category_distribution_chart([])
    assert isinstance(empty_fig, go.Figure)


def test_create_freshness_donut_chart():
    """Test donut chart for freshness classes."""
    history = [
        {"category": "Meat", "predicted_class": "Fresh"},
        {"category": "Meat", "predicted_class": "Spoiled"},
    ]
    fig = create_freshness_donut_chart(history)
    assert isinstance(fig, go.Figure)

    # Empty data
    empty_fig = create_freshness_donut_chart([])
    assert isinstance(empty_fig, go.Figure)


def test_create_timeline_chart():
    """Test timeline scatter chart."""
    history = [
        {"timestamp": "2026-11-23T10:00:00Z", "category": "Fruits", "predicted_class": "Fresh", "confidence": 0.95},
        {"timestamp": "2026-11-23T11:00:00Z", "category": "Dairy", "predicted_class": "Moderate", "confidence": 0.80},
    ]
    fig = create_timeline_chart(history)
    assert isinstance(fig, go.Figure)


# ============================================================================
# 5. RECOMMENDATIONS TESTS
# ============================================================================
def test_storage_recommendations():
    """Verify category-specific recommendations and temperature alerts."""
    # Fresh Fruit
    res_fruit = get_storage_recommendations("Fruits", "Fresh", temperature_celsius=3.0)
    assert res_fruit["category"] == "Fruits"
    assert len(res_fruit["action_items"]) > 0
    assert res_fruit["temperature_alert"] is None

    # Meat in Danger Zone temperature
    res_meat = get_storage_recommendations("Meat", "Fresh", temperature_celsius=15.0)
    assert res_meat["temperature_alert"] is not None
    assert "Danger Zone" in res_meat["temperature_alert"]

    # Spoiled item advice
    res_spoiled = get_storage_recommendations("Vegetables", "Spoiled")
    assert any("DO NOT CONSUME" in item for item in res_spoiled["action_items"])
