"""API Client for Member 3's FastAPI Backend.

Provides production-grade HTTP communication with the FreshCheck AI backend,
including health probing, prediction requests, history retrieval, and clear-history operations.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image

from frontend.config import (
    API_CLEAR_HISTORY_ENDPOINT,
    API_HEALTH_ENDPOINT,
    API_HISTORY_ENDPOINT,
    API_PREDICT_ENDPOINT,
    DEFAULT_BACKEND_URL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger("FreshCheckAI.FrontendAPI")


class BackendError(Exception):
    """Base exception for backend communication failures."""


class BackendConnectionError(BackendError):
    """Raised when the backend server is unreachable or offline."""


class BackendTimeoutError(BackendError):
    """Raised when backend fails to respond within timeout limit."""


class BackendAPIError(BackendError):
    """Raised when backend returns an HTTP 4xx or 5xx error."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Backend API Error ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


class FreshCheckAPIClient:
    """Production client for communicating with Member 3's FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or DEFAULT_BACKEND_URL).rstrip("/")

    def set_base_url(self, new_url: str) -> None:
        """Dynamically update backend URL."""
        self.base_url = new_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Probe backend health endpoint.

        Returns:
            Dict[str, Any]: Health details, e.g. {'status': 'healthy', 'latency_ms': 12.4}

        Raises:
            BackendConnectionError: If server cannot be reached.
            BackendTimeoutError: If request times out.
        """
        start_time = time.perf_counter()
        url = f"{self.base_url}{API_HEALTH_ENDPOINT}"
        try:
            resp = requests.get(url, timeout=3.0)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

            if resp.status_code == 200:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                data["status"] = "healthy"
                data["latency_ms"] = latency_ms
                data["url"] = self.base_url
                return data
            else:
                return {
                    "status": "degraded",
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "url": self.base_url,
                }
        except requests.exceptions.Timeout as exc:
            raise BackendTimeoutError(
                f"Health check timed out after 3.0s connecting to {self.base_url}"
            ) from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
            raise BackendConnectionError(
                f"Could not connect to backend server at {self.base_url}. Please ensure FastAPI is running."
            ) from exc

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        category: str,
        temperature_celsius: Optional[float] = None,
        storage_condition: Optional[str] = None,
        storage_duration_days: Optional[float] = None,
        storage_duration_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send food image and metadata to Member 3's POST /predict endpoint.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original image filename (e.g. 'apple.jpg').
            category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').
            temperature_celsius: Optional temperature in Celsius.
            storage_condition: Optional condition (e.g. 'Refrigerated').
            storage_duration_days: Optional duration in days.
            storage_duration_hours: Optional duration in hours.

        Returns:
            Dict[str, Any]: Parsed backend prediction response conforming to contract.

        Raises:
            BackendConnectionError: If server is offline.
            BackendTimeoutError: If inference takes too long.
            BackendAPIError: If validation or server error occurs.
        """
        url = f"{self.base_url}{API_PREDICT_ENDPOINT}"

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "image/jpeg"

        # Prepare multipart/form-data
        files = {
            "file": (filename, image_bytes, mime_type),
        }

        # Form data for metadata
        data: Dict[str, Any] = {
            "category": category,
        }
        if temperature_celsius is not None:
            data["temperature_celsius"] = str(temperature_celsius)
        if storage_condition:
            data["storage_condition"] = storage_condition
        if storage_duration_days is not None:
            data["storage_duration_days"] = str(storage_duration_days)
        if storage_duration_hours is not None:
            data["storage_duration_hours"] = str(storage_duration_hours)

        try:
            resp = requests.post(
                url,
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout as exc:
            raise BackendTimeoutError(
                f"Prediction request timed out after {REQUEST_TIMEOUT_SECONDS}s."
            ) from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
            raise BackendConnectionError(
                f"Failed to reach FastAPI backend at {url}. Ensure server is running."
            ) from exc

        if resp.status_code != 200:
            error_detail = "Unknown backend error"
            try:
                err_json = resp.json()
                error_detail = err_json.get("detail", err_json.get("message", resp.text))
            except Exception:
                error_detail = resp.text or f"HTTP {resp.status_code}"
            raise BackendAPIError(resp.status_code, str(error_detail))

        try:
            payload = resp.json()
        except ValueError as exc:
            raise BackendAPIError(200, "Backend returned invalid non-JSON response.") from exc

        # Standardize contract compatibility
        # If payload returned root-level contract
        predicted_class = payload.get("predicted_class")
        if not predicted_class and "prediction" in payload:
            # Backward-compatible nested response
            prediction = payload["prediction"]
            payload["predicted_class"] = prediction.get("predicted_class")
            payload["confidence"] = prediction.get("confidence")
            payload["probabilities"] = prediction.get("probabilities")
            payload["category"] = prediction.get("category", category)

        # Fallback category if missing in response
        if "category" not in payload:
            payload["category"] = category

        return payload

    def get_history(self) -> List[Dict[str, Any]]:
        """Fetch historical scans from Member 3's GET /history endpoint.

        Returns:
            List[Dict[str, Any]]: List of previous scan records.
        """
        url = f"{self.base_url}{API_HISTORY_ENDPOINT}"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout as exc:
            raise BackendTimeoutError("History request timed out.") from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
            raise BackendConnectionError(
                f"Could not connect to {url} to retrieve scan history."
            ) from exc

        if resp.status_code != 200:
            error_detail = "Failed to retrieve history"
            try:
                err_json = resp.json()
                error_detail = err_json.get("detail", resp.text)
            except Exception:
                pass
            raise BackendAPIError(resp.status_code, str(error_detail))

        try:
            data = resp.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "history" in data:
                return data["history"]
            return []
        except ValueError as exc:
            raise BackendAPIError(200, "Invalid JSON in history response.") from exc

    def clear_history(self) -> Dict[str, Any]:
        """Request backend to clear all scan history.

        Tries DELETE /history, and falls back to POST /history/clear if needed.
        """
        url = f"{self.base_url}{API_CLEAR_HISTORY_ENDPOINT}"
        try:
            # Try DELETE first
            resp = requests.delete(url, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code in (404, 405):
                # Fallback to POST /history/clear
                fallback_url = f"{self.base_url}{API_HISTORY_ENDPOINT}/clear"
                resp = requests.post(fallback_url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout as exc:
            raise BackendTimeoutError("Clear history request timed out.") from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
            raise BackendConnectionError(
                f"Could not connect to {url} to clear scan history."
            ) from exc

        if resp.status_code not in (200, 204):
            error_detail = "Failed to clear history"
            try:
                err_json = resp.json()
                error_detail = err_json.get("detail", resp.text)
            except Exception:
                pass
            raise BackendAPIError(resp.status_code, str(error_detail))

        try:
            return resp.json() if resp.status_code == 200 and resp.text else {"status": "success"}
        except Exception:
            return {"status": "success"}


# ============================================================================
# TEMPORARY MOCK CLIENT (FOR ISOLATED UI DEVELOPMENT ONLY)
# ============================================================================
class MockFreshCheckClient(FreshCheckAPIClient):
    """Temporary mock client used strictly when backend is not running.

    WARNING:
        This class is provided solely for isolated UI preview and testing
        when Member 3's backend is offline. Every record and prediction is
        explicitly tagged with '_is_mock': True and displays prominent alerts.
    """

    def __init__(self) -> None:
        super().__init__(base_url="http://mock-local-ui-dev")
        self._mock_history: List[Dict[str, Any]] = [
            {
                "id": "scan_mock_001",
                "timestamp": "2026-11-23T08:30:15Z",
                "category": "Fruits",
                "predicted_class": "Fresh",
                "confidence": 0.9412,
                "probabilities": {"Fresh": 0.9412, "Moderate": 0.0450, "Spoiled": 0.0138},
                "temperature_celsius": 4.0,
                "storage_condition": "Refrigerated",
                "storage_duration_days": 2.0,
                "_is_mock": True,
            },
            {
                "id": "scan_mock_002",
                "timestamp": "2026-11-23T09:15:42Z",
                "category": "Vegetables",
                "predicted_class": "Moderate",
                "confidence": 0.7850,
                "probabilities": {"Fresh": 0.1620, "Moderate": 0.7850, "Spoiled": 0.0530},
                "temperature_celsius": 18.0,
                "storage_condition": "Room Temperature",
                "storage_duration_days": 5.0,
                "_is_mock": True,
            },
            {
                "id": "scan_mock_003",
                "timestamp": "2026-11-23T10:05:11Z",
                "category": "Meat",
                "predicted_class": "Spoiled",
                "confidence": 0.9630,
                "probabilities": {"Fresh": 0.0110, "Moderate": 0.0260, "Spoiled": 0.9630},
                "temperature_celsius": 12.0,
                "storage_condition": "Cool & Dry",
                "storage_duration_days": 4.0,
                "_is_mock": True,
            },
            {
                "id": "scan_mock_004",
                "timestamp": "2026-11-23T10:45:00Z",
                "category": "Dairy",
                "predicted_class": "Fresh",
                "confidence": 0.9120,
                "probabilities": {"Fresh": 0.9120, "Moderate": 0.0710, "Spoiled": 0.0170},
                "temperature_celsius": 3.5,
                "storage_condition": "Refrigerated",
                "storage_duration_days": 1.0,
                "_is_mock": True,
            },
            {
                "id": "scan_mock_005",
                "timestamp": "2026-11-23T11:10:25Z",
                "category": "Fruits",
                "predicted_class": "Moderate",
                "confidence": 0.8240,
                "probabilities": {"Fresh": 0.1230, "Moderate": 0.8240, "Spoiled": 0.0530},
                "temperature_celsius": 21.0,
                "storage_condition": "Room Temperature",
                "storage_duration_days": 6.0,
                "_is_mock": True,
            },
        ]

    def check_health(self) -> Dict[str, Any]:
        return {
            "status": "mock_mode",
            "message": "Isolated UI Development Mode (Backend Offline)",
            "latency_ms": 1.0,
            "url": "mock://localhost",
            "_is_mock": True,
        }

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        category: str,
        temperature_celsius: Optional[float] = None,
        storage_condition: Optional[str] = None,
        storage_duration_days: Optional[float] = None,
        storage_duration_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Generate simulated classification based on category
        simulated_class = "Fresh"
        probs = {"Fresh": 0.9250, "Moderate": 0.0550, "Spoiled": 0.0200}
        confidence = 0.9250

        # Create record
        record = {
            "id": f"scan_mock_{len(self._mock_history) + 1:03d}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "predicted_class": simulated_class,
            "confidence": confidence,
            "probabilities": probs,
            "category": category,
            "metadata": {
                "category": category,
                "temperature_celsius": temperature_celsius,
                "storage_condition": storage_condition,
                "storage_duration_days": storage_duration_days,
                "storage_duration_hours": storage_duration_hours,
                "metadata_supplied": any(
                    v is not None
                    for v in (temperature_celsius, storage_condition, storage_duration_days, storage_duration_hours)
                ),
            },
            "_is_mock": True,
        }
        self._mock_history.insert(0, record)
        return record

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._mock_history)

    def clear_history(self) -> Dict[str, Any]:
        count = len(self._mock_history)
        self._mock_history.clear()
        return {"status": "success", "deleted_count": count, "_is_mock": True}
