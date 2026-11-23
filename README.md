# FreshCheck AI — Multi-Category Food Freshness Detection

FreshCheck AI is a computer vision and deep learning application that assesses the freshness of food items across four major categories:
- **Fruits**
- **Vegetables**
- **Meat**
- **Dairy**

Classification classes:
- **Fresh**
- **Moderate**
- **Spoiled**

---

## Team Responsibilities

- **Member 1**: Model architecture, training pipeline, and dataset preparation.
- **Member 2 (Image Processing & Inference)**: Input preprocessing, visual feature analysis, storage metadata validation, and inference engine.
- **Member 3 (FastAPI Backend)**: REST API development, request validation, deep-learning pipeline connection, prediction history SQLite persistence, and frontend integration.
- **Member 4**: Frontend web interface (Streamlit).

---

## Modules Developed (Member 2 — Image Processing & Inference)

Located in `image_processing/`:
- `preprocessing.py`: Multi-source image loading, format validation (JPG/JPEG/PNG/WEBP), corrupt image detection, dimensional safety, resizing, and pixel normalization.
- `feature_extraction.py`: Color distribution, luminance/exposure analysis, RMS contrast, and OpenCV spatial texture analysis.
- `metadata.py`: Environmental storage condition and temperature processing.
- `inference.py`: Cached deep learning inference engine loading Member 1's `.keras` model, returning predictions matching the project's standard contract.
- `pipeline.py`: Orchestration combining image processing, inference, metadata verification, and optional visual analytics.

---

## Modules Developed (Member 3 — FastAPI Backend)

Located in `backend/`:
- `main.py`: FastAPI application entry point, CORS middleware, lifespan events, and global exception handlers.
- `config.py`: Environment configuration, model path settings, database path, upload limits, and category lists.
- `database.py`: SQLite connection factory with WAL mode and schema initialization.
- `routes/predict.py`: `POST /predict` handling multipart uploads, format validation, inference pipeline execution, and SQLite history recording.
- `routes/health.py`: `GET /` (application status) and `GET /health` (operational health and model readiness).
- `routes/history.py`: `GET /history` (paginated past predictions) and `DELETE /history` (history purging).
- `schemas/`: Pydantic models for standard output contracts, metadata, health, and history records.
- `services/inference_service.py`: Encapsulates Member 2's inference engine and pipeline orchestration with mock isolation for testing.
- `services/validation_service.py`: Deep file validation, MIME and magic bytes inspection, executable rejection, and range checks.
- `services/history_service.py`: SQLite persistence layer for predictions and query management without storing user images.

### Starting the Backend
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```
API Documentation is available at `http://127.0.0.1:8000/docs`.

### Running Tests
```bash
# Run backend API tests
python -m pytest tests/test_backend_api.py -v

# Run all project tests
python -m pytest tests/ -v
```
