# ml_model/dataset — Dataset Registry

This directory documents every dataset considered, evaluated, and selected or
rejected for FreshCheck AI training. It does **not** contain image files.

## Directory Contents

| File | Purpose |
|------|---------|
| `README.md` | This file — overview and conventions |
| `dataset_sources.md` | Detailed per-dataset documentation |
| `dataset_audit.csv` | Machine-readable audit table |

## Naming Conventions

Canonical freshness class names used throughout this project:

| Canonical Name | Description |
|----------------|-------------|
| `Fresh` | Visually optimal — no signs of decay |
| `Moderate` | Early visible ageing; still consumable with caveats |
| `Spoiled` | Advanced decay; not safe or suitable for consumption |

> **Policy**: A dataset class is accepted only when the original authors
> explicitly label images by freshness state. Inferring freshness from time
> progression, brightness reduction, or noise injection is **not** accepted
> as a genuine label source.

## Download Root

After downloading, place raw datasets under:

```
ml_model/dataset/raw/
    fresh_and_stale/        ← swoyam2609/fresh-and-stale-classification
    meat_freshness/         ← (Kaggle or Roboflow — see dataset_sources.md)
```

Then run:

```bash
python ml_model/prepare_dataset.py \
    --raw-dir ml_model/dataset/raw \
    --out-dir ml_model/dataset/processed
```

## Supported Categories (verified)

| Category | Datasets | Classes Available | Notes |
|----------|----------|-----------------|-------|
| Fruits | Fresh & Stale Classification | Fresh, Spoiled | Binary only; Moderate **not** available |
| Vegetables | Fresh & Stale Classification | Fresh, Spoiled | Binary only; Moderate **not** available |
| Meat | Meat Freshness Image Dataset | Fresh, Half-Fresh (≈Moderate), Spoiled | 3-class; label mapping documented |
| Dairy | No verified freshness-labelled dataset found | — | See limitations |

## Known Limitations

1. **No genuine `Moderate` class** exists in any verified public dataset for
   Fruits and Vegetables. This class cannot be populated without real annotations.

2. **Dairy** has no suitable public dataset with Fresh/Moderate/Spoiled labels.
   Images of food type (cheese variety, yoghurt type) exist but freshness labels
   do not.

3. **Fruits-360** is a food-type recognition dataset, NOT a freshness dataset.
   It must NOT be treated as a source of freshness labels.

4. The Meat Freshness dataset uses `Half-Fresh` as the middle class. This is
   mapped to `Moderate` in our pipeline with documentation.
