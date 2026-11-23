# FreshCheck AI — Dataset Setup Guide
# Member 1: AI/ML Model Development

## Overview

This document explains how to download, verify and structure the training
datasets for FreshCheck AI before running `train.py`.

---

## Datasets Used

### 1. Fresh and Stale Classification (Primary Dataset)

| Field       | Detail                                               |
|-------------|------------------------------------------------------|
| Source      | Kaggle — `swoyam2609/fresh-and-stale-classification` |
| License     | CC0 1.0 Public Domain                                |
| Categories  | Fruits, Vegetables                                   |
| Labels      | `fresh`, `rotten` (binary)                           |
| Images      | ~10,000 total                                        |

**Label mapping applied:**
- `fresh` → **Fresh** class
- `rotten` → **Spoiled** class
- `Moderate` class: synthetically generated using borderline augmentation
  (brightness 0.6–0.8, gaussian noise, slight blur) on `fresh` images.
  This is a known limitation — see `reports/classification_report.txt`.

### 2. Fruits 360 (Supplementary — Fresh class only)

| Field       | Detail                                            |
|-------------|---------------------------------------------------|
| Source      | Kaggle — `moltean/fruits`                         |
| License     | CC BY-SA 4.0                                      |
| Categories  | Fruits only                                       |
| Labels      | Fresh only (white-background studio images)       |
| Images      | ~90,000 (subset of ~15,000 used)                  |

> **Warning**: Fruits-360 images are studio-shot (white background). They
> supplement the Fresh class but may reduce generalisation to real-world photos.
> Used at a 1:3 ratio with real-world fresh images.

---

## Categories NOT Covered

| Category | Reason                                                            |
|----------|-------------------------------------------------------------------|
| Meat     | No public dataset with reliable Fresh/Moderate/Spoiled labels found |
| Dairy    | No public dataset with reliable Fresh/Moderate/Spoiled labels found |

If your team obtains a labelled Meat or Dairy dataset, add it under
`ml_model/data/train/Fresh/`, `Moderate/`, `Spoiled/` and retrain.

---

## Download Instructions

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
# Place your Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d swoyam2609/fresh-and-stale-classification -p ml_model/data/raw/
kaggle datasets download -d moltean/fruits -p ml_model/data/raw/fruits360/
```

### Option B — Manual Download

1. Visit https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification
2. Click "Download" (requires free Kaggle account)
3. Extract to `ml_model/data/raw/`

---

## Running the Dataset Preparation Script

After downloading, run the data preparation script to split and organise:

```bash
python ml_model/prepare_dataset.py --raw-dir ml_model/data/raw --out-dir ml_model/data
```

This script:
1. Removes corrupted images
2. Deduplicates using perceptual hashing
3. Splits: 70% train / 15% val / 15% test (no data leakage)
4. Generates the Moderate class via augmentation
5. Reports final class distribution

---

## Expected Final Directory Structure

```
ml_model/data/
├── train/
│   ├── Fresh/       (~4,900 images)
│   ├── Moderate/    (~2,100 images — augmented)
│   └── Spoiled/     (~4,200 images)
├── val/
│   ├── Fresh/       (~1,050 images)
│   ├── Moderate/    (~450 images)
│   └── Spoiled/     (~900 images)
└── test/
    ├── Fresh/       (~1,050 images)
    ├── Moderate/    (~450 images)
    └── Spoiled/     (~900 images)
```

---

## Class Distribution Notes

| Class    | Approx. Images | Notes                              |
|----------|----------------|------------------------------------|
| Fresh    | 7,000          | Real + Fruits-360 studio images    |
| Moderate | 3,000          | Augmentation-generated borderline  |
| Spoiled  | 6,300          | Real rotten food images            |

**Class imbalance**: Moderate has ~50% fewer samples than Fresh/Spoiled.
The training script uses `class_weight` compensation automatically.

---

## License Compliance

| Dataset             | License        | Usage Permitted |
|---------------------|----------------|-----------------|
| Fresh-Stale Dataset | CC0 1.0        | Yes — no restrictions |
| Fruits 360          | CC BY-SA 4.0   | Yes — attribution required |

Attribution for Fruits 360:
> Oltean, Horea, and Radu-Emil Precup. "Fruits-360 dataset." Mendeley Data
> (2017). doi:10.17632/rp73yg93n8.1
