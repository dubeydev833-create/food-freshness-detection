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
- **Member 3**: FastAPI backend integration and API endpoints.
- **Member 4**: Frontend web interface.

---

## Modules Developed (Member 2)

Located in `image_processing/`:
- `preprocessing.py`: Multi-source image loading, format validation (JPG/JPEG/PNG), corrupt image detection, dimensional safety, resizing, and pixel normalization.
- `feature_extraction.py`: Color distribution, luminance/exposure analysis, RMS contrast, and OpenCV spatial texture analysis.
- `metadata.py`: Environmental storage condition and temperature processing.
- `inference.py`: Cached deep learning inference engine loading Member 1's `.keras` model, returning predictions matching the project's standard contract.
- `pipeline.py`: Orchestration combining image processing, inference, metadata verification, and optional visual analytics.

### Running Tests
```bash
python -m pytest tests/ -v
```
