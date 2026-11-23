# 🥗 FreshCheck AI — Frontend & UI/UX (Member 4)

A modern, responsive, and intuitive web interface for **FreshCheck AI**, built with **Streamlit**, **Plotly**, **Pillow**, and **Requests**.

---

## 👥 College Project Role Division
- **Member 1**: Model Training, Weights & Deep Learning Architecture (`.keras`)
- **Member 2**: Computer Vision Preprocessing, Feature Extractors & Metadata Pipeline (`image_processing/`)
- **Member 3**: FastAPI Backend Service & History Database Persistence (`POST /predict`, `GET /history`, etc.)
- **Member 4 (Current Module)**: Complete Frontend, Visual Design System, UI/UX, Home Dashboard, Food Analyzer, Results View, Analytics, and Backend Integration.

---

## 🎨 Visual Design System & Palette

FreshCheck AI follows a clean, professional palette designed to elevate the project beyond a basic college assignment:
- **Primary Emerald Green** (`#10B981` / `#059669` / `#ECFDF5`): Symbolizes freshness, vitality, and safe produce.
- **Moderate Warning Amber** (`#F59E0B` / `#FFFBEB`): Highlights items nearing expiration or requiring prompt preparation.
- **Spoiled Critical Red** (`#EF4444` / `#FEF2F2`): Unambiguous danger indicator for items unsafe to consume.
- **Neutral Canvas** (`#F8FAFC` background, `#FFFFFF` cards, `#0F172A` headlines, `#E2E8F0` borders).
- **Typography & Components**: Plus Jakarta Sans, rounded cards (`14px`), subtle drop shadows, interactive Plotly visualizations, live connectivity badges, and actionable storage advice.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+ (Tested on Python 3.10 – 3.14)
- Virtual environment recommended

### 2. Install Dependencies
```bash
pip install -r frontend/requirements.txt
```

### 3. Launch Frontend Application
```bash
streamlit run frontend/app.py
```
The application will automatically open in your browser at `http://localhost:8501`.

### 4. Connect to Member 3's FastAPI Backend
By default, FreshCheck AI looks for Member 3's backend at `http://localhost:8000`.

To run with Member 3's FastAPI server:
```bash
# Terminal 1 (FastAPI Backend)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 (Streamlit Frontend)
streamlit run frontend/app.py
```

You can also specify a custom backend URL via environment variable:
```bash
# On Linux / macOS:
export BACKEND_URL="http://192.168.1.50:8000"

# On Windows PowerShell:
$env:BACKEND_URL = "http://localhost:8000"
```
Or directly adjust the **FastAPI Backend URL** input field in the application's left sidebar settings.

---

## 📁 Frontend Directory Architecture

```
frontend/
├── .streamlit/
│   └── config.toml           # Theme styling (emerald primary, font, upload limits)
├── assets/
│   └── style.css             # High-end CSS (cards, hero banner, badges, status pills)
├── components/
│   ├── __init__.py
│   ├── navbar.py             # Top hero banner and live connectivity indicator
│   ├── metrics_cards.py      # Summary KPI cards (Total, Fresh, Moderate, Spoiled)
│   ├── charts.py             # Plotly gauges, horizontal probability bars, donuts, timeline
│   ├── recommendations.py    # Science-backed food storage actions per category & status
│   └── modals.py             # Empty states, disclaimer alerts, offline guides
├── views/
│   ├── __init__.py
│   ├── dashboard.py          # Phase 3 Home Dashboard with real stats and recent scans
│   ├── analyzer.py           # Phase 4 Food Analyzer with drag-and-drop & camera input
│   ├── results.py            # Phase 5 Results display with probability graphs & disclaimer
│   └── history.py            # Phase 6 Scan History & Analytics with interactive filters
├── api_client.py             # HTTP client for Member 3 FastAPI with fallback mock mode
├── config.py                 # Application configuration, constants, and endpoints
├── requirements.txt          # Frontend dependencies
├── app.py                    # Main Streamlit application
└── README.md                 # Frontend documentation
```

---

## 🔌 API Contract with Member 3's FastAPI Backend

The frontend communicates with Member 3's backend via standard REST endpoints:

### 1. Health Probe
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy", "service": "FreshCheck AI Backend"}`

### 2. Neural Freshness Prediction
- **Endpoint**: `POST /predict`
- **Request Type**: `multipart/form-data`
  - `file`: Raw image bytes (`.jpg`, `.png`, `.webp`)
  - `category`: `"Fruits"` | `"Vegetables"` | `"Meat"` | `"Dairy"`
  - `temperature_celsius`: Optional float (e.g., `4.0`)
  - `storage_condition`: Optional string (`"Refrigerated"`, `"Room Temperature"`, etc.)
  - `storage_duration_days`: Optional float
- **Response**:
```json
{
  "predicted_class": "Fresh",
  "confidence": 0.9412,
  "probabilities": {
    "Fresh": 0.9412,
    "Moderate": 0.0450,
    "Spoiled": 0.0138
  },
  "category": "Fruits",
  "metadata": {
    "category": "Fruits",
    "temperature_celsius": 4.0,
    "storage_condition": "Refrigerated",
    "storage_duration_days": 2.0,
    "metadata_supplied": true
  }
}
```

### 3. Historical Scans
- **Endpoint**: `GET /history`
- **Response**: List of previous scan records with timestamps, categories, predicted classes, and confidence scores.

### 4. Clear History
- **Endpoint**: `DELETE /history` (or `POST /history/clear`)
- **Response**: `{"status": "success", "deleted_count": 12}`

---

## 🛡️ Scientific Integrity & Food Safety Notice

FreshCheck AI implements an explicit, persistent safety disclaimer across the analyzer, results, and dashboard pages:
> **Important Food Safety Disclaimer**: Predictions are generated strictly by image-based computer vision analysis and cannot confirm bacterial food safety, internal pathogens (such as *Salmonella*, *Listeria*, or *E. coli*), or chemical toxins. Confidence scores measure visual classification certainty, not a food safety guarantee. Always inspect smell, texture, storage history, and packaging integrity before consuming food.

---

## 🧪 Isolated UI Development Mode

When Member 3's FastAPI backend is offline during interface design or presentation dry-runs:
1. Open the left sidebar in the application.
2. Check **"Isolated UI Mock Mode"**.
3. FreshCheck AI will populate realistic test fixtures, with each record explicitly tagged `[TEST DATA]` and an amber banner stating mock mode is active.
4. Uncheck to immediately resume live communication with Member 3's real FastAPI server.
