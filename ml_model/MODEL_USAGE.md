# FreshCheck AI — Model Usage Document
# Member 1: AI/ML Model Development
# Version: 1.0.0

---

## Overview

This document defines the **Model Contract** for the FreshCheck AI freshness
classifier. All team members (especially Member 2 — Inference Engine) must
follow this specification exactly to ensure correct predictions.

---

## Model File

| Property   | Value                                          |
|------------|------------------------------------------------|
| File       | `ml_model/model/freshness_model.keras`         |
| Format     | Keras v3 SavedModel (`.keras`)                 |
| Framework  | TensorFlow ≥ 2.13 / Keras ≥ 3.0               |
| Architecture | MobileNetV2 (ImageNet pre-trained backbone) + custom head |

---

## Input Specification

### Image Dimensions

```
Input shape: (1, 224, 224, 3)
             └─ batch size (always 1 for single inference)
                 └─ height (pixels)
                     └─ width (pixels)
                         └─ colour channels (RGB, NOT BGR)
```

> **Important**: The image must be in **RGB** channel order. If you are using
> OpenCV (`cv2`), which loads images as BGR, convert first:
> ```python
> rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
> ```

### Input Normalization — **CENTERED MODE**

The model was trained with **MobileNetV2 standard normalization**:

```
normalized_pixel = (raw_pixel / 127.5) - 1.0
```

- Input range: `[0, 255]` uint8
- Output range: `[-1.0, 1.0]` float32

**This is the `"centered"` normalization mode** in `image_processing/preprocessing.py`.

> **Do NOT use `rescale` mode** (÷ 255 → [0, 1]). That will produce wrong
> predictions because the backbone weights expect inputs in [-1, 1].

```python
# Correct preprocessing
image = image.astype(np.float32)
image = (image / 127.5) - 1.0
batch = np.expand_dims(image, axis=0)   # shape: (1, 224, 224, 3)
```

---

## Output Specification

### Class Order (FIXED — do not change)

```json
["Fresh", "Moderate", "Spoiled"]
  index 0    index 1     index 2
```

This order is documented in `ml_model/labels.json` and enforced by the
`flow_from_directory` class parameter during training.

### Output Shape

```
shape: (1, 3)   — one sample, three class probabilities
```

### Output Activation

Softmax — probabilities sum to 1.0:
```
output[0, 0] = P(Fresh)
output[0, 1] = P(Moderate)
output[0, 2] = P(Spoiled)
```

### Example Output

```python
array([[0.9241, 0.0548, 0.0211]])
# → Fresh with 92.4% confidence
```

---

## Loading the Model

```python
import numpy as np

# Method 1: Pure Keras (recommended)
import keras
model = keras.models.load_model("ml_model/model/freshness_model.keras", compile=False)

# Method 2: TensorFlow Keras (fallback)
import tensorflow as tf
model = tf.keras.models.load_model("ml_model/model/freshness_model.keras", compile=False)

# compile=False is intentional — we do not need training ops at inference time
```

---

## Running a Prediction

```python
import numpy as np
from PIL import Image

# 1. Load image (must be RGB)
img = Image.open("apple.jpg").convert("RGB")

# 2. Resize to model input dimensions
img = img.resize((224, 224), Image.BICUBIC)

# 3. Normalize — CENTERED MODE
arr = np.array(img, dtype=np.float32)
arr = (arr / 127.5) - 1.0

# 4. Add batch dimension
batch = np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)

# 5. Run inference
probs = model(batch, training=False).numpy().squeeze()  # shape: (3,)

# 6. Interpret output
CLASS_NAMES = ["Fresh", "Moderate", "Spoiled"]
predicted_class = CLASS_NAMES[np.argmax(probs)]
confidence = float(np.max(probs))

print(f"Class      : {predicted_class}")
print(f"Confidence : {confidence:.4f}")
print(f"All probs  : {dict(zip(CLASS_NAMES, probs.tolist()))}")
```

---

## Integration with Member 2 — Inference Engine

Member 2's `image_processing/inference.py` already implements the full
`FoodFreshnessInferenceEngine`. To use the real model, instantiate it with:

```python
from image_processing.inference import FoodFreshnessInferenceEngine

engine = FoodFreshnessInferenceEngine(
    model_path="ml_model/model/freshness_model.keras",
    target_size=(224, 224),
    normalization_mode="centered",   # ← MUST be "centered" for MobileNetV2
)

result = engine.predict("path/to/food_image.jpg", category="Fruits")
# Returns: {"predicted_class": "Fresh", "confidence": 0.9241, "probabilities": {...}}
```

---

## Reading `labels.json`

```python
import json

with open("ml_model/labels.json") as f:
    meta = json.load(f)

class_names = meta["class_names"]          # ["Fresh", "Moderate", "Spoiled"]
num_classes = meta["num_classes"]          # 3
normalization = meta["model_contract"]["normalization"]  # "centered"
input_shape = meta["model_contract"]["input_shape"]      # [224, 224, 3]
```

---

## Common Mistakes

| Mistake                             | Effect                        | Fix                          |
|-------------------------------------|-------------------------------|------------------------------|
| Using `rescale` normalization       | Wrong predictions             | Use `centered` mode          |
| Input in BGR channel order          | Wrong predictions             | Convert BGR→RGB before feed  |
| Not adding batch dimension          | Shape error `(224, 224, 3)`   | `np.expand_dims(arr, axis=0)` |
| Assuming Meat/Dairy work correctly  | Unreliable predictions        | Model trained on Fruits/Veg only |
| Trusting Moderate class fully       | May misclassify               | Class is synthetically generated |

---

## Model Limitations

1. **Food categories**: Only Fruits and Vegetables are in training data. Predictions
   on Meat or Dairy images are unreliable and should be clearly communicated to users.

2. **Moderate class**: Synthetically generated by augmenting Fresh images with
   brightness/noise perturbations. Real-world "borderline fresh" images may not
   match this distribution.

3. **Background sensitivity**: The model has seen both white-background studio
   images (Fruits-360) and real-world images. Performance may vary on images
   with complex backgrounds.

4. **Lighting**: Extreme lighting conditions may degrade performance.

---

## Installation Requirements

```bash
pip install tensorflow>=2.13 numpy pillow scikit-learn matplotlib seaborn
```

Tested on:
- Python 3.10, 3.11, 3.12
- TensorFlow 2.13, 2.15, 2.16
- Keras 3.x

---

## File Reference

```
ml_model/
├── train.py              — Training script
├── evaluate.py           — Evaluation and report generation
├── prepare_dataset.py    — Dataset cleaning and splitting
├── generate_reports.py   — Simulated report generation (integration testing)
├── labels.json           — Class names, indices, model contract
├── DATASET_SETUP.md      — Dataset download and preparation guide
├── MODEL_USAGE.md        — This file
├── model/
│   └── freshness_model.keras   — Trained model weights
└── reports/
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── training_accuracy.png
    └── training_loss.png
```
