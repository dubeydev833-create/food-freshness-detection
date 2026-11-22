# FreshCheck AI — Image Processing & Inference Module (Member 2)

This directory contains the computer vision preprocessing pipeline, statistical visual feature extractors, metadata validators, and the deep-learning inference engine for the **FreshCheck AI** food freshness detection application.

---

## 📂 Architecture Overview

```
image_processing/
├── __init__.py               # Package exports
├── exceptions.py             # Custom domain exceptions
├── preprocessing.py          # Format validation, resizing, normalization
├── feature_extraction.py     # Color, brightness, contrast, and texture analysis
├── metadata.py               # Auxiliary storage condition & temperature validation
├── inference.py              # Model caching, prediction, standard output contract
└── pipeline.py               # High-level pipeline orchestration for Member 3
```

---

## 🎯 Freshness Classes & Categories

### Freshness Classes (Strict Order)
1. `Fresh`
2. `Moderate`
3. `Spoiled`

### Supported Categories
- `Fruits`
- `Vegetables`
- `Meat`
- `Dairy`

---

## 📋 Standard Output Contract (Phase 6)

Every inference response returned to Member 3's FastAPI backend conforms to this exact dictionary structure:

```json
{
    "predicted_class": "Fresh",
    "confidence": 0.94,
    "probabilities": {
        "Fresh": 0.94,
        "Moderate": 0.04,
        "Spoiled": 0.02
    },
    "category": "Fruits"
}
```

---

## 🤝 Team Coordination & Interface Guidelines

### For Member 1 (Model Training & ML)
- **Model Format**: Save your trained model in modern `.keras` format (or legacy `.h5`).
- **Input Dimensions**: Default is `(224, 224, 3)`. You can specify custom dimensions via `target_size=(height, width)`.
- **Pixel Normalization**: Configurable via `normalization_mode`:
  - `"rescale"`: `[0.0, 1.0]` ($x / 255.0$) — *Current Project Default* (standard for models trained on normalized floats).
  - `"centered"`: `[-1.0, 1.0]` ($(x / 127.5) - 1.0$) — **REQUIRED** if you use a MobileNetV2 or Inception backbone trained with `tf.keras.applications.mobilenet_v2.preprocess_input`.
  - `"raw"`: `[0.0, 255.0]` `float32` — **REQUIRED** if you embedded an internal `keras.layers.Rescaling(1./255)` layer inside your Keras model architecture to prevent double normalization.
  - `"standardize"`: Applies ImageNet RGB channel mean `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]`.
- **Output Layer**: 3 softmax units corresponding strictly to `["Fresh", "Moderate", "Spoiled"]`.

### For Member 3 (Backend & API Integration)
1. **Initialize the engine once** in your FastAPI lifespan/startup event to avoid reloading:
   ```python
   from image_processing import FoodFreshnessInferenceEngine

   engine = FoodFreshnessInferenceEngine.get_instance("models/freshcheck_model.keras")
   ```
2. **Execute inference on uploaded file bytes**:
   ```python
   # Direct standard contract output (predicted_class, confidence, probabilities, category)
   result = engine.predict(file_bytes, category="Fruits")

   # Or execute the full orchestrated pipeline with metadata:
   from image_processing import run_freshness_analysis

   response = run_freshness_analysis(
       image_input=file_bytes,
       category="Fruits",
       engine=engine,
       temperature_celsius=4.0,
       storage_condition="Refrigerated",
       storage_duration_days=2.0,
       include_visual_features=True,
   )
   # response provides root-level contract fields:
   # response["predicted_class"], response["confidence"], response["probabilities"], response["category"]
   # along with decoupled telemetry: response["metadata"], response["visual_features"]
   # and backward-compatible nested dictionary: response["prediction"]
   ```
3. **Handle Custom Exceptions**:
   - `ImageFormatError`: When user uploads unsupported formats (e.g. GIF, PDF).
   - `ImageSizeError`: When image exceeds 20MB or 8192px limits.
   - `ImageReadError`: When image is corrupt or unreadable.
   - `ModelNotFoundError`: If model file is missing on disk.
   - `MetadataValidationError`: If temperature or category is invalid.

---

## 🔬 Scientific Integrity & Feature Analysis Notice

Visual features extracted by `feature_extraction.py` (color histograms, luminance, RMS contrast, Laplacian texture variance) provide descriptive optical telemetry. **They do not independently prove food spoilage**, as natural oxidation, cultivar pigmentation, and culinary browning can cause false positives. They are strictly decoupled from deep-learning predictions.

---

## 🧪 Testing

Run all unit and integration tests:
```bash
python -m pytest tests/ -v
```
All 64 tests validate image loading, format rejection, resizing, normalization modes, feature extraction, metadata sanitization, and end-to-end model inference.
