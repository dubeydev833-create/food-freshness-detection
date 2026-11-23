# FreshCheck AI — Smart Food Freshness Detection System

FreshCheck AI is a four-member college mini-project that leverages deep learning and computer vision to evaluate the freshness of food items (Fruits, Vegetables, Meat, and Dairy) into three distinct quality levels: **Fresh**, **Moderate**, and **Spoiled**.

---

## 👥 Team Responsibilities
- **Member 1 (Machine Learning & Modeling)**: Deep learning model architecture, transfer learning with MobileNetV2/CNNs, model weights training, and evaluation (`.keras`).
- **Member 2 (Computer Vision & Preprocessing)**: Image format validation, resizing, normalization pipeline, visual feature extraction, and metadata verification (`image_processing/`).
- **Member 3 (Backend API & Database)**: FastAPI REST service, database storage for historical scans, model inference orchestration (`POST /predict`, `GET /history`).
- **Member 4 (Frontend & UI/UX)**: Modern Streamlit dashboard, visual design system, Food Analyzer, interactive Plotly visualizations, recommendations engine, and backend integration (`frontend/`).

---

## 🖥️ Running the Frontend Application (Member 4)

### 1. Installation
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r frontend/requirements.txt
```

### 2. Launch Streamlit Web UI
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.

### 3. Backend Integration
The frontend automatically connects to Member 3's FastAPI backend at `http://localhost:8000`.

To run the full stack:
```bash
# Terminal 1 — FastAPI Backend (Member 3)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit Frontend (Member 4)
streamlit run frontend/app.py
```

If the backend is not yet started, you can enable **"Isolated UI Mock Mode"** from the left sidebar to preview the UI layout and test workflows.

---

## 🔬 Freshness Classification Classes
1. 🟢 **Fresh**: Food shows optimal visual characteristics and firmness.
2. 🟡 **Moderate**: Early signs of aging, oxidation, or moisture loss; consume promptly.
3. 🔴 **Spoiled**: Visible decay, discoloration, mold, or spoilage; do not consume.

---

## ⚠️ Food Safety Disclaimer
FreshCheck AI provides computer-vision-based visual classification. An optical prediction **cannot confirm microbiological food safety or internal bacterial toxins**. Confidence scores reflect model pattern certainty, not food safety probability. Always inspect smell, texture, and expiration dates before consuming.
