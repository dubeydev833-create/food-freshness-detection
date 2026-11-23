"""Generate synthetic report artifacts (pure numpy/stdlib — no matplotlib needed)."""
import numpy as np, json, os

SEED = 42
rng = np.random.default_rng(SEED)
CLASS_NAMES = ["Fresh", "Moderate", "Spoiled"]
REPORTS_DIR = "ml_model/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Simulate training history (20 epochs) ---
n = 20
train_acc = np.clip(np.linspace(0.62, 0.95, n) + rng.normal(0, 0.012, n), 0.5, 1.0)
val_acc   = np.clip(np.linspace(0.57, 0.91, n) + rng.normal(0, 0.018, n), 0.45, 1.0)
train_loss= np.clip(np.linspace(0.98, 0.18, n) + rng.normal(0, 0.025, n), 0.05, 1.5)
val_loss  = np.clip(np.linspace(1.05, 0.26, n) + rng.normal(0, 0.035, n), 0.05, 1.8)

history = {
    "accuracy": train_acc.tolist(),
    "val_accuracy": val_acc.tolist(),
    "loss": train_loss.tolist(),
    "val_loss": val_loss.tolist(),
}
with open(f"{REPORTS_DIR}/history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"Saved history.json — best val_acc: {max(val_acc):.4f}")

# --- Simulate predictions ---
n_samples = 1200
counts = [480, 300, 420]  # Fresh, Moderate, Spoiled
y_true = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
rng.shuffle(y_true)
confusion_probs = {0: [0.95, 0.03, 0.02], 1: [0.10, 0.83, 0.07], 2: [0.03, 0.04, 0.93]}
y_pred = np.array([rng.choice(3, p=confusion_probs[int(tl)]) for tl in y_true])

# Confusion matrix
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[int(t)][int(p)] += 1

# Per-class metrics
prec, rec, f1_scores = [], [], []
for i in range(3):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    prec.append(p); rec.append(r); f1_scores.append(f)

acc = float(np.mean(y_true == y_pred))
mac_p = float(np.mean(prec))
mac_r = float(np.mean(rec))
mac_f1 = float(np.mean(f1_scores))

print("Confusion matrix:")
print("           " + "  ".join(f"{c:>10}" for c in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:<11}" + "  ".join(f"{v:>10}" for v in row))
print(f"Accuracy: {acc:.4f}  Macro-Precision: {mac_p:.4f}  Macro-Recall: {mac_r:.4f}  Macro-F1: {mac_f1:.4f}")

# --- Write classification_report.txt ---
sep70 = "=" * 70
sep40 = "-" * 40
header_note = "[SIMULATED - for integration testing only. Replace with evaluate.py output.]"
col_header = f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
cm_header = "           " + "  ".join(f"{c:>10}" for c in CLASS_NAMES)

per_class_rows = []
for i, cls in enumerate(CLASS_NAMES):
    per_class_rows.append(
        f"  {cls:<12} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1_scores[i]:>10.4f} {counts[i]:>10}"
    )

cm_rows = []
for i, row in enumerate(cm):
    cm_rows.append(f"{CLASS_NAMES[i]:<11}" + "  ".join(f"{v:>10}" for v in row))

report_lines = [
    "FreshCheck AI - Classification Report",
    sep70,
    header_note,
    sep70,
    "Model     : ml_model/model/freshness_model.keras",
    "Norm Mode : centered  (pixel/127.5 - 1.0, range [-1.0, 1.0])",
    f"Classes   : {CLASS_NAMES}",
    f"Test Samples: {n_samples}",
    sep70,
    "",
    "OVERALL METRICS",
    sep40,
    f"Accuracy        : {acc:.4f} ({acc*100:.2f}%)",
    f"Macro Precision : {mac_p:.4f}",
    f"Macro Recall    : {mac_r:.4f}",
    f"Macro F1-Score  : {mac_f1:.4f}",
    "",
    "PER-CLASS METRICS",
    sep40,
    col_header,
    sep40,
    *per_class_rows,
    "",
    "CONFUSION MATRIX (rows=True label, columns=Predicted label)",
    sep40,
    cm_header,
    *cm_rows,
    "",
    "LIMITATIONS",
    sep40,
    "1. Training covers Fruits and Vegetables ONLY.",
    "   Meat and Dairy excluded — no suitable public freshness datasets found.",
    "",
    "2. Moderate class is synthetically generated via borderline augmentation.",
    "   Lower recall on Moderate is expected and documented.",
    "",
    "3. Studio-lit images (Fruits-360) may not generalise to real-world photos.",
    "",
    "4. Dataset bias toward Fruits; vegetable performance may be lower.",
]

with open(f"{REPORTS_DIR}/classification_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Saved classification_report.txt to {REPORTS_DIR}")
