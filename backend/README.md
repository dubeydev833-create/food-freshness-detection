# FreshCheck AI — Backend API Documentation (Member 3)

FastAPI REST backend for **FreshCheck AI**, an AI-powered food freshness assessment application.

The backend acts as the central integration layer connecting:
1. **Member 4's Streamlit frontend** (interactive web UI).
2. **Member 2's image processing & inference engine** (`image_processing/`).
3. **Member 1's trained Deep Learning model** (via Member 2's inference pipeline).

---

## 🎯 Supported Categories & Freshness Classes

### Supported Categories
- `Fruits`
- `Vegetables`
- `Meat`
- `Dairy`

*(Case-insensitive: `fruits`, `FRUITS`, `Fruits` are all accepted and normalized)*

### Freshness Classes (Strict Order)
1. `Fresh`
2. `Moderate`
3. `Spoiled`

---

## 🚀 Quickstart: Running the Backend

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```
*(Or `pip install -r requirements.txt` from repository root)*

### 2. Start the Uvicorn Server
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:8000/openapi.json`

### 3. Environment Variables (Optional)
- `FRESHCHECK_MODEL_PATH`: Custom path to Member 1's trained `.keras` model (default: `models/freshcheck_model.keras`).
- `FRESHCHECK_DB_PATH`: Custom path to SQLite database (default: `freshcheck.db`).
- `FRESHCHECK_DEBUG`: Set to `true` for detailed debug logs.

---

## 🌐 CORS Configuration

CORS is pre-configured for local frontend development. The following origins are allowed:
- `http://localhost:8501` (Default Streamlit port)
- `http://127.0.0.1:8501`
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:8000` / `http://127.0.0.1:8000`
- `*` (Wildcard allowed during local college development)

---

## 📡 API Endpoint Reference

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Application status & service information | `200 OK` |
| `GET` | `/health` | Health check & model availability status | `200 OK` |
| `POST` | `/predict` | Predict freshness from image & optional metadata | `200 OK` |
| `GET` | `/history` | Retrieve past prediction history (paginated) | `200 OK` |
| `DELETE` | `/history` | Clear prediction history database | `200 OK` |

---

### 1. `GET /` — Application Status

#### Response (200 OK)
```json
{
    "app": "FreshCheck AI Backend",
    "version": "1.0.0",
    "status": "online",
    "docs_url": "/docs",
    "health_url": "/health"
}
```

---

### 2. `GET /health` — Health & Model Check

#### Response (200 OK — Fully Healthy)
```json
{
    "status": "healthy",
    "backend_running": true,
    "model_available": true,
    "model_path": "models/freshcheck_model.keras",
    "database_connected": true,
    "message": "All FreshCheck AI backend systems and models are fully operational."
}
```

#### Response (200 OK — Degraded / Model Awaiting Placement)
```json
{
    "status": "degraded",
    "backend_running": true,
    "model_available": false,
    "model_path": "models/freshcheck_model.keras",
    "database_connected": true,
    "message": "Model not found at 'models/freshcheck_model.keras'. Please ensure Member 1's trained .keras model is in the models/ directory."
}
```

---

### 3. `POST /predict` — Freshness Assessment

Accepts `multipart/form-data`:

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `UploadFile` | **Yes** | Food image file (JPEG, PNG, WEBP, max 10MB) |
| `category` | `string` | **Yes** | One of `Fruits`, `Vegetables`, `Meat`, `Dairy` |
| `temperature` | `float` | No | Observed ambient/storage temp in Celsius (-40°C to 60°C) |
| `storage_condition` | `string` | No | E.g. `Refrigerated`, `Room Temperature`, `Frozen` |
| `storage_duration` | `string` | No | E.g. `2 days`, `12 hours`, `1 week` |

#### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@apple.jpg;type=image/jpeg" \
     -F "category=Fruits" \
     -F "temperature=5.0" \
     -F "storage_condition=Refrigerated" \
     -F "storage_duration=2 days"
```

#### Standard Response Structure (Phase 3 Contract)
```json
{
    "success": true,
    "category": "Fruits",
    "predicted_class": "Fresh",
    "confidence": 0.94,
    "probabilities": {
        "Fresh": 0.94,
        "Moderate": 0.04,
        "Spoiled": 0.02
    },
    "metadata": {
        "temperature_celsius": 5.0,
        "storage_condition": "Refrigerated"
    },
    "message": "Image-based freshness prediction completed."
}
```

> **Scientific Integrity & Privacy Notice**:
> - When `temperature` or `storage_condition` is omitted, `metadata.temperature_celsius` and `metadata.storage_condition` are returned as `null`. Temperature readings are never invented.
> - Uploaded images are processed strictly in memory and are **never saved permanently to disk or database**.

---

### 4. `GET /history` — Retrieve Prediction History

#### Query Parameters
- `limit` (int, default: 50, min: 1, max: 500): Max items to return.
- `offset` (int, default: 0): Items to skip for pagination.
- `category` (string, optional): Filter by category (e.g. `?category=Fruits`).

#### cURL Example
```bash
curl "http://127.0.0.1:8000/history?limit=10&category=Fruits"
```

#### Response (200 OK)
```json
{
    "success": true,
    "total_count": 1,
    "limit": 10,
    "offset": 0,
    "items": [
        {
            "id": 1,
            "prediction_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
            "timestamp": "2026-11-23T06:58:07.298579+00:00",
            "category": "Fruits",
            "predicted_class": "Fresh",
            "confidence": 0.94,
            "temperature_celsius": 5.0,
            "storage_condition": "Refrigerated",
            "storage_duration": "2 days",
            "message": "Image-based freshness prediction completed."
        }
    ]
}
```

---

### 5. `DELETE /history` — Clear Prediction History

#### cURL Example
```bash
curl -X DELETE "http://127.0.0.1:8000/history"
```

#### Response (200 OK)
```json
{
    "success": true,
    "message": "Prediction history cleared successfully.",
    "deleted_count": 1
}
```

---

## 🛡️ Security & Validation (Phase 4)

| Condition | HTTP Status | Response Example / Detail |
| :--- | :--- | :--- |
| **Empty File** (0 bytes) | `400 Bad Request` | `"Uploaded file is empty (0 bytes)."` |
| **Corrupted Image** | `400 Bad Request` | `"Corrupted or invalid image file: Failed to decode."` |
| **Invalid Category** | `400 Bad Request` | `"Invalid category 'Soda'. Supported: Fruits, Vegetables, Meat, Dairy."` |
| **Out-of-Bounds Temp** | `400 Bad Request` | `"Temperature 150°C is outside realistic bounds (-40°C to 60°C)."` |
| **Oversized Image** (>10MB) | `413 Content Too Large` | `"Image file exceeds maximum allowable size of 10 MB."` |
| **Dangerous Executable** (`.exe`, `.sh`, `.bat`) | `415 Unsupported Media Type` | `"Executable and arbitrary files with extension '.exe' are strictly prohibited."` |
| **Non-Image Format** (`.pdf`, `.txt`) | `415 Unsupported Media Type` | `"Invalid image format: File header does not match valid JPEG, PNG, or WEBP signatures."` |
| **Missing Model File** | `503 Service Unavailable` | `"Inference model is currently unavailable: Trained model not found."` |
| **Inference Failure** | `500 Internal Server Error` | `"Model inference execution failed: ..."` |

---

## 🤝 Member 4: Streamlit Frontend Integration Guide

Here is a ready-to-use Python snippet showing how Member 4 can integrate their Streamlit UI with the backend:

```python
import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="FreshCheck AI", layout="centered")
st.title("🍎 FreshCheck AI — Food Freshness Detection")

# Check backend health
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
    if health["status"] == "healthy":
        st.success("🟢 Backend & Model Online")
    else:
        st.warning(f"🟡 Backend Online (Model Status: {health.get('message')})")
except requests.exceptions.RequestException:
    st.error("🔴 Backend Offline. Please run `uvicorn backend.main:app --reload`.")
    st.stop()

# Input Form
with st.form("freshness_form"):
    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png", "webp"])
    category = st.selectbox("Food Category", ["Fruits", "Vegetables", "Meat", "Dairy"])
    temperature = st.number_value = st.number_input(
        "Observed Temperature (°C) [Optional]",
        value=None,
        min_value=-40.0,
        max_value=60.0,
        placeholder="Leave empty if unknown"
    )
    storage_condition = st.selectbox(
        "Storage Condition [Optional]",
        ["", "Room Temperature", "Refrigerated", "Frozen", "Cool & Dry", "Pantry"]
    )
    storage_duration = st.text_input("Storage Duration [Optional]", placeholder="e.g. 2 days")

    submit_button = st.form_submit_button("Assess Freshness")

if submit_button:
    if not uploaded_file:
        st.error("Please upload an image first!")
    else:
        with st.spinner("Analyzing freshness via Deep Learning model..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"category": category}
            if temperature is not None:
                data["temperature"] = float(temperature)
            if storage_condition:
                data["storage_condition"] = storage_condition
            if storage_duration:
                data["storage_duration"] = storage_duration

            response = requests.post(f"{BACKEND_URL}/predict", files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                st.subheader(f"Result: **{result['predicted_class']}** ({result['confidence']*100:.1f}%)")
                
                # Show probabilities bar chart
                probs = result["probabilities"]
                st.bar_chart(probs)

                # Show metadata
                meta = result["metadata"]
                if meta["temperature_celsius"] is not None or meta["storage_condition"] is not None:
                    st.info(f"Storage Info: {meta.get('storage_condition', 'N/A')} at {meta.get('temperature_celsius', 'N/A')}°C")
            else:
                st.error(f"Prediction failed ({response.status_code}): {response.json().get('detail')}")

# Historical predictions
st.markdown("---")
st.subheader("📜 Recent Prediction History")
hist_res = requests.get(f"{BACKEND_URL}/history?limit=10").json()
if hist_res.get("items"):
    st.table(hist_res["items"])
    if st.button("Clear History"):
        requests.delete(f"{BACKEND_URL}/history")
        st.experimental_rerun()
else:
    st.caption("No past predictions recorded yet.")
```

---

## 🧪 Running the Backend Test Suite

Run the full API test suite with pytest:
```bash
python -m pytest tests/test_backend_api.py -v
```

All 18 tests validate:
- Status and health checks.
- Exact Phase 3 standard contract response structure.
- Isolation between production model loading and test mock engines.
- Null handling for omitted metadata.
- Rejection of executables, corrupt images, unsupported formats, and oversized uploads.
- SQLite persistence and image privacy compliance.
