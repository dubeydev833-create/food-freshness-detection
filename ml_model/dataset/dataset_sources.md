# FreshCheck AI — Dataset Sources
# Member 1: AI/ML Model Development
# Last updated: 2026-11-23

This document provides verified, source-cited information about every dataset
researched for FreshCheck AI. All entries include actual label verification,
licensing, and honest limitations.

---

## ACCEPTED DATASETS

---

### Dataset 1 — Fresh and Stale Classification

| Field | Value |
|-------|-------|
| **Name** | Fresh and Stale Classification |
| **Kaggle ID** | `swoyam2609/fresh-and-stale-classification` |
| **URL** | https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification |
| **License** | CC0 1.0 Public Domain (no restrictions) |
| **Food Categories** | Fruits, Vegetables |
| **Image Count** | ~10,000 total (approximate) |
| **Image Format** | JPEG |
| **Freshness Labels** | **BINARY ONLY**: `fresh_<item>` and `stale_<item>` |

**Actual Folder Labels:**
```
train/
    freshapple/     freshbanana/    freshorange/     freshtomato/
    freshpotato/    freshcapsicum/  freshcucumber/   freshokra/
    freshbittergourd/
    staleapple/     stalebanana/    staleorange/     staletomato/
    stalepotato/    stalecapsicum/  stalecucumber/   staleokra/
    stalebittergourd/
```

**Freshness Class Mapping:**
- `fresh_*` → **Fresh** ✓ (genuine label)
- `stale_*` → **Spoiled** ✓ (genuine label)
- `Moderate` → **NOT AVAILABLE** — this dataset does not provide a medium freshness class

**Academic Verification:** Used in multiple peer-reviewed papers on binary food
freshness classification. Labels are determined by visual inspection by the
dataset authors.

**Usage Decision:**
- Use `fresh_*` images for the **Fresh** class
- Use `stale_*` images for the **Spoiled** class
- Do NOT use this dataset to populate a Moderate class

**Download Command:**
```bash
pip install kaggle
kaggle datasets download -d swoyam2609/fresh-and-stale-classification \
    -p ml_model/dataset/raw/fresh_and_stale/
unzip ml_model/dataset/raw/fresh_and_stale/fresh-and-stale-classification.zip \
    -d ml_model/dataset/raw/fresh_and_stale/
```

**Estimated Size:** ~300 MB

---

### Dataset 2 — Meat Freshness Image Dataset

| Field | Value |
|-------|-------|
| **Name** | Meat Freshness Image Dataset |
| **Kaggle URL** | https://www.kaggle.com/datasets/vinayakshanawad/meat-quality-assessment |
| **Alternative** | Roboflow: search "Meat Freshness" at roboflow.com/universe |
| **License** | CC BY 4.0 (attribution required) |
| **Food Category** | Meat |
| **Image Count** | ~2,266 high-quality images |
| **Image Format** | JPEG |
| **Freshness Labels** | **THREE-CLASS**: `Fresh`, `Half_Fresh`, `Spoiled` |

**Actual Folder Labels:**
```
Fresh/          ← optimal state, vibrant colour, clear marbling
Half_Fresh/     ← transitional state; mild colour/texture change
Spoiled/        ← significant discoloration, compromised texture
```

**Freshness Class Mapping:**
- `Fresh` → **Fresh** ✓ (genuine label)
- `Half_Fresh` → **Moderate** ✓ (genuine label; mapped with documentation)
- `Spoiled` → **Spoiled** ✓ (genuine label)

> **Important**: `Half_Fresh` is mapped to `Moderate` in `prepare_dataset.py`
> via the `LABEL_MAP` dictionary. This mapping is documented here and in
> `labels.json`. The mapping is scientifically reasonable — "half-fresh"
> describes an intermediate freshness state equivalent to "moderate freshness".

**Label Mapping in Code:**
```python
"half_fresh": "Moderate",
"halffresh": "Moderate",
"half-fresh": "Moderate",
```

**Citation:**
> Vinayak Shanawad. "Meat Quality Assessment Dataset." Kaggle, 2022.

**Download Command:**
```bash
kaggle datasets download -d vinayakshanawad/meat-quality-assessment \
    -p ml_model/dataset/raw/meat_freshness/
```

**Estimated Size:** ~150 MB

---

## REJECTED / NOT USED DATASETS

---

### Rejected — Fruits-360

| Field | Value |
|-------|-------|
| **Name** | Fruits 360 |
| **Kaggle ID** | `moltean/fruits` |
| **License** | CC BY-SA 4.0 |
| **Rejection Reason** | Food-type classification dataset, NOT freshness dataset |

**Why Rejected:**
Fruits-360 labels are food type (Apple, Banana, etc.) and variety, NOT
freshness state. All images show fresh fruit against white backgrounds.
Using this dataset would populate only the Fresh class with studio-shot
images that do not generalise to real-world photos.

**Previous Use (Incorrect):** An earlier version of this project incorrectly
included Fruits-360 as a dataset source. This entry documents its rejection.

---

### Rejected — Generic "Food Freshness" Community Datasets

Several community-assembled Kaggle datasets claim three classes
(Fresh/Medium/Rotten) but upon inspection:

1. The "Medium" class is often created by photographing produce at different
   time points without controlled conditions.
2. No standardised annotation protocol is documented.
3. Images from different sources are merged without consistent labelling.

These datasets are **not used** due to unreliable ground truth labels.

---

### Not Available — Dairy Freshness

| Category | Finding |
|----------|---------|
| Cheese freshness | No public dataset with Fresh/Moderate/Spoiled labels found |
| Milk freshness | No public image dataset found (physical tests like pH/smell cannot be captured in photos) |
| Yoghurt freshness | No suitable dataset found |

**Research performed:**
- Kaggle search: "dairy freshness", "cheese spoiled", "milk fresh spoiled"
- Result: Datasets found are food-type datasets (cheese varieties by country/milk source) not freshness datasets
- Cheese mold datasets exist for quality-control in manufacturing but are not publicly available

**Conclusion:** Dairy is NOT supported in the current model. This is a genuine
limitation documented in `labels.json` and `classification_report.txt`.

---

## SUMMARY TABLE

| Category | Dataset | Genuine Labels Available | Used |
|----------|---------|------------------------|------|
| Fruits | Fresh & Stale Classification | Fresh, Spoiled | ✓ Yes |
| Vegetables | Fresh & Stale Classification | Fresh, Spoiled | ✓ Yes |
| Meat | Meat Freshness Image Dataset | Fresh, Moderate (Half_Fresh), Spoiled | ✓ Yes |
| Dairy | None found | None | ✗ No — limitation |

## Notes on the Moderate Class

For **Fruits and Vegetables**: No public dataset provides a genuine `Moderate`
freshness class. If you train a fruits/vegetables model, it will be **binary
(Fresh/Spoiled)** unless a genuine Moderate dataset is obtained.

For **Meat**: The `Half_Fresh` label from the Meat Freshness dataset provides a
genuine three-class setup. The combined model (Fruits + Vegetables + Meat) can
support a three-class output if the Moderate class is populated from meat data.

**This is a known limitation** — the Moderate class will be biased toward
meat images if the only three-class data comes from meat. All limitations are
documented in `classification_report.txt`.
