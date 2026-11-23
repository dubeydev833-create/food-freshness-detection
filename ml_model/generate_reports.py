"""
FreshCheck AI — Synthetic Report Generator
Member 1 Responsibility: AI/ML Model Development

Generates realistic evaluation report artifacts (confusion matrix, accuracy
and loss curve plots, and text classification report) using synthetic but
training-realistic data.

This script is intended for:
1. Integration testing by Members 2, 3, 4 before the real model is trained.
2. CI pipeline smoke-tests.
3. Demonstrating the report format without requiring TensorFlow.

IMPORTANT: These are SIMULATED results. Actual evaluation requires running
evaluate.py against the trained model and real test data.
All synthetic files are tagged with a prominent header.

Usage:
    python ml_model/generate_reports.py

Requirements:
    pip install numpy matplotlib seaborn scikit-learn
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPORTS_DIR = Path("ml_model/reports")
CLASS_NAMES = ["Fresh", "Moderate", "Spoiled"]
SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Simulate training history
# ---------------------------------------------------------------------------
def simulate_history(n_epochs: int = 20) -> dict:
    """Generate a realistic training/validation accuracy and loss history."""
    # Accuracy: starts low, rises with noise, plateaus
    train_acc_base = np.linspace(0.62, 0.95, n_epochs)
    train_acc = np.clip(train_acc_base + rng.normal(0, 0.012, n_epochs), 0.5, 1.0)

    val_acc_base = np.linspace(0.57, 0.91, n_epochs)
    val_acc = np.clip(val_acc_base + rng.normal(0, 0.018, n_epochs), 0.45, 1.0)

    # Loss: starts high, decreases
    train_loss_base = np.linspace(0.98, 0.18, n_epochs)
    train_loss = np.clip(train_loss_base + rng.normal(0, 0.025, n_epochs), 0.05, 1.5)

    val_loss_base = np.linspace(1.05, 0.26, n_epochs)
    val_loss = np.clip(val_loss_base + rng.normal(0, 0.035, n_epochs), 0.05, 1.8)

    return {
        "accuracy": train_acc.tolist(),
        "val_accuracy": val_acc.tolist(),
        "loss": train_loss.tolist(),
        "val_loss": val_loss.tolist(),
    }


# ---------------------------------------------------------------------------
# Simulate predictions
# ---------------------------------------------------------------------------
def simulate_predictions(n_samples: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """Simulate realistic true labels and model predictions."""
    # Class distribution: 40% Fresh, 25% Moderate, 35% Spoiled
    counts = [int(n_samples * 0.40), int(n_samples * 0.25), int(n_samples * 0.35)]
    counts[2] += n_samples - sum(counts)  # adjust rounding
    y_true = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(y_true)

    # Confusion: model is ~92% accurate overall
    # - Fresh: 95% correct, 3% confused with Moderate, 2% with Spoiled
    # - Moderate: 83% correct (hardest class), 10% confused with Fresh, 7% with Spoiled
    # - Spoiled: 93% correct, 4% confused with Moderate, 3% with Fresh
    confusion_probs = {
        0: [0.95, 0.03, 0.02],  # True Fresh
        1: [0.10, 0.83, 0.07],  # True Moderate
        2: [0.03, 0.04, 0.93],  # True Spoiled
    }

    y_pred = np.array([
        rng.choice(3, p=confusion_probs[true_label])
        for true_label in y_true
    ])

    return y_true, y_pred


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def save_confusion_matrix(cm_arr: np.ndarray, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    # Manual coloured heatmap without seaborn
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, str(cm_arr[i, j]),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=13, fontweight="bold",
            )

    ax.set_title("FreshCheck AI — Confusion Matrix (Test Set)\n[SIMULATED — for integration testing]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    out_path = reports_dir / "confusion_matrix.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Reports] Confusion matrix saved: {out_path}")


def save_accuracy_plot(history: dict, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["accuracy"], "b-o", markersize=4, label="Training Accuracy")
    ax.plot(epochs, history["val_accuracy"], "r-s", markersize=4, label="Validation Accuracy")
    ax.set_title("FreshCheck AI — Training & Validation Accuracy\n[SIMULATED — for integration testing]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = reports_dir / "training_accuracy.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Reports] Accuracy curve saved: {out_path}")


def save_loss_plot(history: dict, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["loss"], "b-o", markersize=4, label="Training Loss")
    ax.plot(epochs, history["val_loss"], "r-s", markersize=4, label="Validation Loss")
    ax.set_title("FreshCheck AI — Training & Validation Loss\n[SIMULATED — for integration testing]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = reports_dir / "training_loss.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Reports] Loss curve saved: {out_path}")


# ---------------------------------------------------------------------------
# Classification report text
# ---------------------------------------------------------------------------
def build_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    acc = accuracy_score(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    cm_arr = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )

    text = (
        "FreshCheck AI — Classification Report\n"
        "=" * 70 + "\n"
        "⚠️  NOTICE: THIS IS A SIMULATED REPORT FOR INTEGRATION TESTING.\n"
        "    Real metrics require running evaluate.py on the trained model.\n"
        "=" * 70 + "\n"
        f"Model     : ml_model/model/freshness_model.keras\n"
        f"Norm Mode : centered  (pixel/127.5 - 1.0, range [-1, 1])\n"
        f"Classes   : {CLASS_NAMES}\n"
        f"Test Samples: {len(y_true)}\n"
        "=" * 70 + "\n\n"
        "OVERALL METRICS\n"
        "-" * 40 + "\n"
        f"Accuracy        : {acc:.4f} ({acc*100:.2f}%)\n"
        f"Macro Precision : {np.mean(precision):.4f}\n"
        f"Macro Recall    : {np.mean(recall):.4f}\n"
        f"Macro F1-Score  : {np.mean(f1):.4f}\n\n"
        "PER-CLASS METRICS\n"
        "-" * 40 + "\n"
        + report_str +
        "\n\nCONFUSION MATRIX (rows=True label, columns=Predicted label)\n"
        "-" * 40 + "\n"
    )

    header = "           " + "  ".join(f"{c:>10}" for c in CLASS_NAMES)
    text += header + "\n"
    for i, row in enumerate(cm_arr):
        text += f"{CLASS_NAMES[i]:<11}" + "  ".join(f"{v:>10}" for v in row) + "\n"

    text += (
        "\n\nLIMITATIONS\n"
        "-" * 40 + "\n"
        "1. Training data covers Fruits and Vegetables only.\n"
        "   Meat and Dairy categories are NOT included due to lack of\n"
        "   publicly available, freshness-labelled datasets.\n\n"
        "2. The 'Moderate' freshness class is synthetically generated via\n"
        "   borderline augmentation (brightness, noise). Lower recall on\n"
        "   Moderate is expected and documented.\n\n"
        "3. Studio-lit images (Fruits-360) may not generalise to real-world\n"
        "   photographs with varied lighting and backgrounds.\n\n"
        "4. The dataset bias toward Fruits may cause lower performance on\n"
        "   Vegetable images. Collecting more diverse images is recommended.\n"
    )

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    try:
        from sklearn.metrics import classification_report  # noqa
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    print("=" * 60)
    print("  FreshCheck AI — Generating Simulated Reports")
    print("  (For integration testing — NOT real model results)")
    print("=" * 60)

    # Simulate data
    history = simulate_history(n_epochs=20)
    y_true, y_pred = simulate_predictions(n_samples=1200)

    # Compute confusion matrix for image plot
    from sklearn.metrics import confusion_matrix
    cm_arr = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Save all reports
    save_confusion_matrix(cm_arr, REPORTS_DIR)
    save_accuracy_plot(history, REPORTS_DIR)
    save_loss_plot(history, REPORTS_DIR)

    # Save history JSON (used by evaluate.py --history-json)
    history_path = REPORTS_DIR / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Reports] Training history JSON saved: {history_path}")

    # Save classification report text
    report_text = build_classification_report_text(y_true, y_pred)
    report_path = REPORTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[Reports] Classification report saved: {report_path}")

    print(f"\n[Done] All reports saved to: {REPORTS_DIR.resolve()}")
    print("       Replace with real results by running: python ml_model/evaluate.py")


if __name__ == "__main__":
    main()
