"""Comprehensive API integration test suite for FreshCheck AI Backend.

Uses pytest and FastAPI TestClient to validate:
- GET / (Application status)
- GET /health (Operational health & model availability)
- POST /predict (Prediction contract, validation, security, mock isolation)
- GET /history & DELETE /history (SQLite persistence & privacy compliance)
"""

from __future__ import annotations

import io
from pathlib import Path
import sqlite3
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import settings
from backend.database import get_db, init_db
from backend.main import app
from backend.schemas.prediction import PredictionResponse
from backend.services.inference_service import InferenceService, get_inference_service
from image_processing.inference import MockInferenceEngine


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary SQLite database path for isolated testing."""
    db_file = tmp_path / "test_freshcheck.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def mock_inference_service() -> InferenceService:
    """Return an InferenceService configured with a deterministic MockInferenceEngine."""
    mock_engine = MockInferenceEngine(simulated_class="Fresh")
    return InferenceService(engine=mock_engine)


@pytest.fixture
def client(test_db_path: Path, mock_inference_service: InferenceService):
    """FastAPI TestClient with isolated test database and mock inference engine."""

    def override_get_db():
        conn = sqlite3.connect(str(test_db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_inference_service] = lambda: mock_inference_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate valid in-memory JPEG image bytes."""
    img = Image.new("RGB", (120, 120), color=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate valid in-memory PNG image bytes."""
    img = Image.new("RGB", (120, 120), color=(40, 200, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSystemEndpoints:
    """Tests for GET / and GET /health endpoints (Phase 2)."""

    def test_root_status(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == settings.PROJECT_NAME
        assert data["version"] == settings.VERSION
        assert data["status"] == "online"
        assert data["docs_url"] == "/docs"
        assert data["health_url"] == "/health"

    def test_health_check_when_model_missing(self, client: TestClient):
        # Without real model weights on disk
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["backend_running"] is True
        assert data["database_connected"] is True
        assert "model_available" in data
        assert "model_path" in data


class TestPredictionEndpoint:
    """Tests for POST /predict endpoint (Phase 2, 3, & 4)."""

    def test_predict_success_standard_contract(self, client: TestClient, valid_jpeg_bytes: bytes):
        """Verify successful prediction matches Phase 3 JSON output contract."""
        response = client.post(
            "/predict",
            files={"file": ("apple.jpg", valid_jpeg_bytes, "image/jpeg")},
            data={
                "category": "Fruits",
                "temperature": "4.5",
                "storage_condition": "Refrigerated",
                "storage_duration": "2 days",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Validate with Pydantic model
        validated = PredictionResponse(**data)
        assert validated.success is True
        assert validated.category == "Fruits"
        assert validated.predicted_class == "Fresh"
        assert validated.confidence == 0.94
        assert validated.probabilities.Fresh == 0.94
        assert validated.probabilities.Moderate == 0.04
        assert validated.probabilities.Spoiled == 0.02
        assert validated.metadata.temperature_celsius == 4.5
        assert validated.metadata.storage_condition == "Refrigerated"
        assert validated.message == "Image-based freshness prediction completed."

    def test_predict_metadata_null_when_omitted(self, client: TestClient, valid_png_bytes: bytes):
        """Metadata fields must be null when omitted; never invent temperature."""
        response = client.post(
            "/predict",
            files={"file": ("spinach.png", valid_png_bytes, "image/png")},
            data={"category": "Vegetables"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["temperature_celsius"] is None
        assert data["metadata"]["storage_condition"] is None

    @pytest.mark.parametrize("category", ["fruits", "VEGETABLES", "Meat", " dairy "])
    def test_predict_category_case_insensitivity(
        self, client: TestClient, valid_jpeg_bytes: bytes, category: str
    ):
        response = client.post(
            "/predict",
            files={"file": ("food.jpg", valid_jpeg_bytes, "image/jpeg")},
            data={"category": category},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["category"] in {"Fruits", "Vegetables", "Meat", "Dairy"}

    def test_missing_model_error(self, test_db_path: Path, valid_jpeg_bytes: bytes):
        """When real model is missing, endpoint must return HTTP 503."""
        # Un-override inference service so it tries to load real missing model
        app.dependency_overrides.pop(get_inference_service, None)
        with TestClient(app) as unmocked_client:
            response = unmocked_client.post(
                "/predict",
                files={"file": ("food.jpg", valid_jpeg_bytes, "image/jpeg")},
                data={"category": "Fruits"},
            )
            assert response.status_code == 503
            assert "unavailable" in response.json()["detail"].lower()


class TestValidationAndSecurity:
    """Tests for Phase 4 validation and security requirements."""

    def test_reject_empty_file(self, client: TestClient):
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"category": "Fruits"},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_reject_corrupted_image(self, client: TestClient):
        corrupt_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00junk-payload-not-an-image"
        response = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
            data={"category": "Fruits"},
        )
        assert response.status_code == 400
        assert "corrupt" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()

    def test_reject_executable_upload(self, client: TestClient):
        """Strictly block executable binary files."""
        exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00fake_windows_pe_header"
        response = client.post(
            "/predict",
            files={"file": ("malware.exe", exe_bytes, "application/x-msdownload")},
            data={"category": "Fruits"},
        )
        assert response.status_code == 415
        assert "executable" in response.json()["detail"].lower() or "prohibited" in response.json()["detail"].lower()

    def test_reject_pdf_upload(self, client: TestClient):
        pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n"
        response = client.post(
            "/predict",
            files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
            data={"category": "Dairy"},
        )
        assert response.status_code == 415

    def test_reject_oversized_image(self, client: TestClient):
        """Reject files exceeding 10 MB limit with HTTP 413."""
        oversized = b"\xFF\xD8\xFF" + b"\x00" * (11 * 1024 * 1024)
        response = client.post(
            "/predict",
            files={"file": ("huge.jpg", oversized, "image/jpeg")},
            data={"category": "Fruits"},
        )
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()

    def test_reject_invalid_category(self, client: TestClient, valid_jpeg_bytes: bytes):
        response = client.post(
            "/predict",
            files={"file": ("food.jpg", valid_jpeg_bytes, "image/jpeg")},
            data={"category": "Beverages"},
        )
        assert response.status_code == 400
        assert "invalid category" in response.json()["detail"].lower()

    def test_reject_out_of_bounds_temperature(self, client: TestClient, valid_jpeg_bytes: bytes):
        response = client.post(
            "/predict",
            files={"file": ("food.jpg", valid_jpeg_bytes, "image/jpeg")},
            data={"category": "Meat", "temperature": "120.0"},
        )
        assert response.status_code == 400
        assert "outside realistic" in response.json()["detail"].lower()

    def test_reject_negative_duration(self, client: TestClient, valid_jpeg_bytes: bytes):
        response = client.post(
            "/predict",
            files={"file": ("food.jpg", valid_jpeg_bytes, "image/jpeg")},
            data={"category": "Meat", "storage_duration": "-3 days"},
        )
        assert response.status_code == 400
        assert "negative" in response.json()["detail"].lower()


class TestHistoryEndpoints:
    """Tests for GET /history and DELETE /history endpoints (Phase 5)."""

    def test_history_flow_and_privacy(self, client: TestClient, valid_jpeg_bytes: bytes):
        # 1. Clear initial history
        del_res = client.delete("/history")
        assert del_res.status_code == 200

        # 2. History is empty
        empty_res = client.get("/history")
        assert empty_res.status_code == 200
        assert empty_res.json()["total_count"] == 0

        # 3. Create prediction
        pred_res = client.post(
            "/predict",
            files={"file": ("steak.jpg", valid_jpeg_bytes, "image/jpeg")},
            data={"category": "Meat", "temperature": "-2.0", "storage_condition": "Frozen"},
        )
        assert pred_res.status_code == 200

        # 4. Verify history persisted correctly
        hist_res = client.get("/history")
        assert hist_res.status_code == 200
        data = hist_res.json()
        assert data["total_count"] == 1
        item = data["items"][0]
        assert item["category"] == "Meat"
        assert item["predicted_class"] == "Fresh"
        assert item["confidence"] == 0.94
        assert item["temperature_celsius"] == -2.0
        assert item["storage_condition"] == "Frozen"

        # PRIVACY VERIFICATION: Confirm database does not store image bytes
        assert "image" not in item
        assert "image_bytes" not in item
        assert "file_path" not in item

        # 5. Category filter
        veg_res = client.get("/history?category=Vegetables")
        assert veg_res.json()["total_count"] == 0

        meat_res = client.get("/history?category=Meat")
        assert meat_res.json()["total_count"] == 1

        # 6. Clear history
        clear_res = client.delete("/history")
        assert clear_res.status_code == 200
        assert clear_res.json()["deleted_count"] == 1

        # 7. Confirm cleared
        final_res = client.get("/history")
        assert final_res.json()["total_count"] == 0
