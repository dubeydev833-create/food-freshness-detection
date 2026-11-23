# FreshCheck AI — Dataset Verification & Evaluation Report
**Role:** Member 1 — AI/ML Model Developer  
**Status:** Phase 1 Complete — Dataset Research & Verification  

---

## 1. Objective & Scope

This report provides a rigorous audit of candidate datasets for **FreshCheck AI**. The project objective requires classifying food freshness into three target classes:
1. **Fresh**
2. **Moderate**
3. **Spoiled**

across four food categories:
- **Fruits**
- **Vegetables**
- **Meat**
- **Dairy**

---

## 2. Dataset Verification Summary

| Dataset Name | Source / URL | License | Categories | Verified Original Labels | Mapped Classes | Genuine 3-Class? | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fresh and Stale Classification** | `swoyam2609/fresh-and-stale-classification` | CC0 1.0 | Fruits, Vegetables | `fresh_<item>`, `stale_<item>` (9 produce types) | `Fresh`, `Spoiled` | ❌ Binary only | **ACCEPTED** for Fruits & Veg (Binary only) |
| **Meat Freshness Image Dataset** | `vinayakshanawad/meat-quality-assessment` | CC BY 4.0 | Meat | `Fresh`, `Half_Fresh`, `Spoiled` | `Fresh`, `Moderate`, `Spoiled` | ✅ Genuine 3-class | **ACCEPTED** for Meat (Provides genuine Moderate) |
| **Fruits 360** | `moltean/fruits` | CC BY-SA 4.0 | Fruits | Varietal names (e.g., `Apple_Golden_1`, `Banana`) | None | ❌ Food variety only | **REJECTED** |
| **Dairy Freshness** | Public search (Kaggle/Roboflow/Mendeley) | N/A | Dairy | N/A | None | ❌ None | **NOT SUPPORTED** |

---

## 3. Detailed Dataset Audits

### 3.1 Fresh and Stale Classification (Fruits & Vegetables)
* **Kaggle Link:** https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification
* **License:** CC0 Public Domain
* **Categories:** Apple, Banana, Bitter Gourd, Capsicum, Cucumber, Okra, Orange, Potato, Tomato.
* **Ground Truth Labels:** Contains subfolders such as `freshapple`, `staleapple`, `freshbanana`, `stalebanana`.
* **Finding:** Labels strictly represent a **binary classification** task (`fresh` vs `stale/rotten`).
* **Verdict:** There is **no genuine Moderate class** in this dataset. We accept it for `Fresh` and `Spoiled` classes of Fruits and Vegetables. We do not invent or synthetically generate a `Moderate` class from these images.

### 3.2 Meat Freshness Image Dataset (Meat)
* **Kaggle Link:** https://www.kaggle.com/datasets/vinayakshanawad/meat-quality-assessment
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Category:** Meat (Red meat / Poultry / Beef cuts).
* **Sample Count:** ~2,266 annotated images.
* **Ground Truth Labels:** Three distinct directories:
  - `Fresh/` — Red, optimal color, firm texture.
  - `Half_Fresh/` — Slight discoloration, initial breakdown, intermediate stage.
  - `Spoiled/` — Severe oxidation, greenish/brownish tone, structural decay.
* **Finding:** The `Half_Fresh` class represents genuine intermediate freshness degradation.
* **Verdict:** **Accepted**. `Half_Fresh` is mapped to `Moderate` with explicit documentation in `labels.json` and `prepare_dataset.py`.

### 3.3 Fruits-360 (Rejected)
* **Kaggle Link:** https://www.kaggle.com/datasets/moltean/fruits
* **License:** CC BY-SA 4.0
* **Analysis:** Fruits-360 was captured for fruit recognition, not freshness assessment. All fruits are fresh, harvested produce photographed under controlled white-background studio lighting while rotating on a motor shaft.
* **Verdict:** **REJECTED**. Treating varietal differences as freshness stages would introduce severe label noise and studio-lighting bias.

### 3.4 Dairy Freshness (Not Available)
* **Analysis:** Thorough searches across Kaggle, Roboflow Universe, and academic repositories revealed no public image dataset with verified labels for freshness stages of milk, cheese, or yogurt.
* **Verdict:** **NOT SUPPORTED**. Dairy is documented as an honest project limitation.

---

## 4. Current Dataset Storage & Image Count

```
Storage Location: ml_model/dataset/raw/
Status: Not downloaded (Sandbox environment has no external network access)
Image Count in Repository: 0 images
```

### Manual Download Instructions
Users running on a host machine with network access should follow [`ml_model/DATASET_SETUP.md`](file:///c:/Users/vedpr/.gemini/antigravity/scratch/food-freshness-detection/ml_model/DATASET_SETUP.md):
```bash
kaggle datasets download -d swoyam2609/fresh-and-stale-classification -p ml_model/dataset/raw/fresh_and_stale/ --unzip
kaggle datasets download -d vinayakshanawad/meat-quality-assessment -p ml_model/dataset/raw/meat_freshness/ --unzip
```

---

## 5. Honest Limitations Declaration

1. **Category Coverage:** Only Fruits, Vegetables, and Meat have verified datasets. Dairy is currently unsupported.
2. **Moderate Class Bias:** The `Moderate` class originates exclusively from the Meat Freshness dataset (`Half_Fresh`). Fruits and Vegetables currently do not possess genuine `Moderate` samples.
3. **No Synthetic Manipulation:** Under no circumstances are `Moderate` images created using artificial brightness reduction, blur, or noise.
