"""
FreshCheck AI — Model Training Script  (v2.0)
Member 1 Responsibility: AI/ML Model Development

Trains a MobileNetV2-based transfer learning model for food freshness
classification. The number of supported classes is determined by which
class subdirectories are actually populated in the dataset.

Supported classes (when genuine data is available):
    Fresh, Moderate, Spoiled

Usage:
    python ml_model/train.py [--data-dir PATH] [--epochs N] [--batch-size N]

Requirements:
    pip install tensorflow>=2.13 scikit-learn matplotlib pillow numpy

Dataset Structure Expected:
    <data_dir>/
        train/
            Fresh/
            Moderate/        <- only if genuine Moderate data exists
            Spoiled/
        validation/          <- note: 'validation' not 'val'
            Fresh/
            Moderate/
            Spoiled/
        test/
            Fresh/
            Moderate/
            Spoiled/

IMPORTANT:
    If Moderate/ directories are absent or empty, the model trains as a
    two-class classifier (Fresh/Spoiled). This is NOT a completed three-class
    model and must be documented accordingly. Do not fabricate Moderate data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FreshCheck AI — MobileNetV2 Freshness Classifier Training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml_model/dataset/processed",
        help="Root directory containing train/, validation/, test/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_model/model",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--val-dir-name",
        type=str,
        default="validation",
        help="Name of validation split subdirectory (default: 'validation'). "
             "Use 'val' if your dataset uses that name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (early stopping may halt sooner).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (height=width). MobileNetV2 default is 224.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        default=False,
        help="Unfreeze top layers of MobileNetV2 backbone for fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Layer index from which to unfreeze during fine-tuning.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Imports — deferred so arg-parse works without TF installed
# ---------------------------------------------------------------------------
def import_dependencies():
    """Import heavy ML dependencies after argument parsing."""
    global tf, keras, Adam, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    global MobileNetV2, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    global Input, Model, L2
    global ImageDataGenerator
    global plt, cm, classification_report, confusion_matrix, np

    import numpy as np_mod
    import numpy as np  # noqa: F401 — make available in module scope
    globals()["np"] = np_mod

    try:
        import tensorflow as tf_mod
        globals()["tf"] = tf_mod
        keras_mod = tf_mod.keras
        globals()["keras"] = keras_mod
    except ImportError:
        try:
            import keras as keras_mod
            globals()["keras"] = keras_mod
            globals()["tf"] = None
        except ImportError:
            print(
                "ERROR: Neither tensorflow nor keras is installed.\n"
                "Install with: pip install tensorflow>=2.13\n"
                "or:           pip install keras>=3.0 tensorflow-cpu>=2.13"
            )
            sys.exit(1)

    from tensorflow.keras.optimizers import Adam as Adam_cls
    from tensorflow.keras.callbacks import (
        EarlyStopping as ES,
        ModelCheckpoint as MC,
        ReduceLROnPlateau as RLROP,
    )
    from tensorflow.keras.applications import MobileNetV2 as MNV2
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D as GAP,
        Dense as D,
        Dropout as Drop,
        BatchNormalization as BN,
        Input as Inp,
    )
    from tensorflow.keras.models import Model as M
    from tensorflow.keras.regularizers import l2 as L2_fn
    from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
    import matplotlib.pyplot as plt_mod
    from sklearn.metrics import classification_report as cr, confusion_matrix as cmx

    globals().update({
        "Adam": Adam_cls,
        "EarlyStopping": ES,
        "ModelCheckpoint": MC,
        "ReduceLROnPlateau": RLROP,
        "MobileNetV2": MNV2,
        "GlobalAveragePooling2D": GAP,
        "Dense": D,
        "Dropout": Drop,
        "BatchNormalization": BN,
        "Input": Inp,
        "Model": M,
        "L2": L2_fn,
        "ImageDataGenerator": IDG,
        "plt": plt_mod,
        "cm": cmx,
        "classification_report": cr,
        "confusion_matrix": cmx,
    })


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Fresh", "Moderate", "Spoiled"]
NUM_CLASSES = 3
SEED = 42


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def build_data_generators(data_dir: str, img_size: int, batch_size: int, val_dir_name: str = "validation"):
    """
    Build train, validation and test data generators.

    Augmentation is applied ONLY to training data.
    Validation and test sets use normalization only.

    Args:
        data_dir: Root data directory containing split subdirectories.
        img_size: Target image size (height = width).
        batch_size: Batch size.
        val_dir_name: Name of validation subdirectory ('validation' or 'val').

    Returns:
        Tuple of (train_gen, val_gen, test_gen, class_names, class_weights)
    """

    # MobileNetV2 uses [-1, 1] normalization — matches the shared contract
    def mobilenet_preprocess(x):
        return (x / 127.5) - 1.0

    # Training: data augmentation to improve generalisation
    # Note: augmentation is ONLY applied to training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    # Validation / Test: normalization only — no augmentation
    eval_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, val_dir_name)
    test_dir = os.path.join(data_dir, "test")

    for d, name in [(train_dir, "train"), (val_dir, val_dir_name), (test_dir, "test")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Dataset directory not found: '{d}'\n"
                f"Run prepare_dataset.py first. See ml_model/dataset/dataset_sources.md."
            )

    # Discover which classes actually have images in train/
    # Only include classes that have populated subdirectories
    actual_classes = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.isdir(cls_dir) and len(os.listdir(cls_dir)) > 0:
            actual_classes.append(cls)

    if not actual_classes:
        raise ValueError(
            f"No class subdirectories with images found in '{train_dir}'.\n"
            f"Expected subdirectories: {CLASS_NAMES}"
        )

    if set(actual_classes) != set(CLASS_NAMES):
        missing = set(CLASS_NAMES) - set(actual_classes)
        print(f"\n[WARNING] Missing classes in training data: {missing}")
        print(f"  The model will be trained on {len(actual_classes)} classes: {actual_classes}")
        print(f"  This is NOT a complete three-class model.")
        print(f"  Document this limitation in the classification report.")

    print(f"\n[Dataset] Active classes: {actual_classes}")

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=actual_classes,   # enforce exact class order
        shuffle=True,
        seed=SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=actual_classes,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=actual_classes,
        shuffle=False,
    )

    # Verify class indices are consistent across all three generators (train, val, test)
    print(f"[Dataset] Train class indices: {train_gen.class_indices}")
    print(f"[Dataset] Val class indices:   {val_gen.class_indices}")
    print(f"[Dataset] Test class indices:  {test_gen.class_indices}")
    assert train_gen.class_indices == val_gen.class_indices, (
        f"Class index mismatch between train and val generators!\n"
        f"Train: {train_gen.class_indices}\nVal:   {val_gen.class_indices}"
    )
    assert train_gen.class_indices == test_gen.class_indices, (
        f"Class index mismatch between train and test generators!\n"
        f"Train: {train_gen.class_indices}\nTest:  {test_gen.class_indices}"
    )

    print(f"[Dataset] Training samples   : {train_gen.samples}")
    print(f"[Dataset] Validation samples : {val_gen.samples}")
    print(f"[Dataset] Test samples       : {test_gen.samples}")

    # Compute class weights to handle class imbalance
    class_counts = np.array([
        np.sum(train_gen.classes == i) for i in range(len(actual_classes))
    ])
    total = class_counts.sum()
    class_weights = {
        i: total / (len(actual_classes) * count) if count > 0 else 1.0
        for i, count in enumerate(class_counts)
    }
    print(f"[Dataset] Class weights: {class_weights}")

    return train_gen, val_gen, test_gen, actual_classes, class_weights


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def build_model(img_size: int, num_classes: int = 3, fine_tune: bool = False, fine_tune_at: int = 100) -> "tf.keras.Model":
    """Build MobileNetV2 transfer learning model for freshness classification.

    Architecture:
        Input (224, 224, 3) → MobileNetV2 backbone (ImageNet weights, frozen)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu, L2=1e-4)
        → Dropout(0.4)
        → Dense(128, relu, L2=1e-4)
        → Dropout(0.3)
        → Dense(num_classes, softmax)   ← output: [Fresh, Moderate, Spoiled]

    Normalization:
        Input pixels must be in range [-1.0, 1.0] (MobileNetV2 contract).
        Apply: (pixel / 127.5) - 1.0 before passing to this model.

    Args:
        img_size: Input image height=width in pixels (default 224).
        num_classes: Number of output classes (2 or 3 depending on available data).
        fine_tune: If True, unfreeze top backbone layers.
        fine_tune_at: Layer index from which to unfreeze (only used when fine_tune=True).
    """
    input_tensor = Input(shape=(img_size, img_size, 3), name="image_input")

    # MobileNetV2 backbone — pre-trained on ImageNet
    # Check for local workspace cached weights first to support offline environments
    local_weights = Path(__file__).resolve().parent / "model" / "weights" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    weights_path = str(local_weights) if local_weights.is_file() else "imagenet"

    backbone = MobileNetV2(
        input_tensor=input_tensor,
        weights=weights_path,
        include_top=False,
        alpha=1.0,              # full-width model
    )

    if not fine_tune:
        # Phase 1: freeze entire backbone, train only the classification head
        backbone.trainable = False
        print(f"[Model] Backbone frozen. Training classification head only.")
    else:
        # Phase 2: unfreeze layers from fine_tune_at onwards
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
        trainable_count = sum(1 for l in backbone.layers if l.trainable)
        print(f"[Model] Fine-tuning: {trainable_count}/{len(backbone.layers)} backbone layers unfrozen.")

    # Classification head
    x = backbone.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="head_bn")(x)
    x = Dense(256, activation="relu", kernel_regularizer=L2(1e-4), name="fc_256")(x)
    x = Dropout(0.4, name="dropout_1")(x)
    x = Dense(128, activation="relu", kernel_regularizer=L2(1e-4), name="fc_128")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    output = Dense(num_classes, activation="softmax", name="freshness_output")(x)

    model = Model(inputs=input_tensor, outputs=output, name="FreshCheck_MobileNetV2")

    total = sum(1 for _ in model.trainable_weights)
    print(f"[Model] Total trainable weight tensors: {total}")
    model.summary(line_length=80)

    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def build_callbacks(output_dir: str, monitor: str = "val_accuracy") -> list:
    """Build training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau."""
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "freshness_model.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(args: argparse.Namespace) -> dict:
    """Execute the full training pipeline and return training history."""
    import json

    print("=" * 70)
    print("  FreshCheck AI — Model Training  (v2.0)")
    print("  Phase 1: Transfer Learning (Frozen Backbone)")
    print("=" * 70)

    val_dir_name = getattr(args, "val_dir_name", "validation")

    # Build data generators — discovers which classes have actual data
    train_gen, val_gen, test_gen, active_classes, class_weights = build_data_generators(
        args.data_dir, args.img_size, args.batch_size, val_dir_name=val_dir_name
    )

    num_classes = len(active_classes)
    if num_classes < NUM_CLASSES:
        print(f"\n[WARNING] Only {num_classes} of {NUM_CLASSES} classes have data.")
        print(f"  Active classes: {active_classes}")
        print(f"  Missing: {set(CLASS_NAMES) - set(active_classes)}")
        print(f"  This model is NOT a complete three-class classifier.")
        print(f"  Obtain genuine data for missing classes before claiming three-class support.")

    # Build model with correct output size
    model = build_model(args.img_size, num_classes=num_classes, fine_tune=False)

    # Compile with categorical cross-entropy
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = build_callbacks(args.output_dir, monitor="val_accuracy")

    print(f"\n[Training] Starting Phase 1 — {args.epochs} max epochs")
    print(f"[Training] Class weights applied: {class_weights}")
    start_time = time.time()

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    phase1_time = time.time() - start_time

    # --- Optional Phase 2: Fine-tuning ---
    history_ft = None
    if args.fine_tune:
        print("\n" + "=" * 70)
        print("  Phase 2: Fine-Tuning (Partial Backbone Unfreeze)")
        print("=" * 70)

        # Build fine-tuning model
        ft_model = build_model(
            args.img_size, num_classes=num_classes, fine_tune=True, fine_tune_at=args.fine_tune_at
        )
        # Load best weights from phase 1 checkpoint before fine-tuning
        best_weights_path = os.path.join(args.output_dir, "freshness_model.keras")
        if os.path.isfile(best_weights_path):
            print(f"[Fine-Tuning] Loading best Phase 1 checkpoint from: {best_weights_path}")
            ft_model.load_weights(best_weights_path)
        else:
            print("[Fine-Tuning] Best checkpoint not found; transferring in-memory weights from Phase 1.")
            ft_model.set_weights(model.get_weights())

        # Lower LR for fine-tuning
        ft_model.compile(
            optimizer=Adam(learning_rate=args.learning_rate * 0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        ft_callbacks = build_callbacks(args.output_dir, monitor="val_accuracy")

        print(f"\n[Training] Fine-tuning for up to {args.epochs} more epochs")
        ft_start = time.time()
        history_ft = ft_model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=ft_callbacks,
            verbose=1,
        )
        fine_tune_time = time.time() - ft_start
        print(f"[Training] Fine-tune time: {fine_tune_time:.1f}s")
        # Ensure model returned is the fine-tuned model
        model = ft_model

    print(f"\n[Training] Phase 1 training time: {phase1_time:.1f}s")
    print(f"[Training] Best model saved to: {args.output_dir}/freshness_model.keras")

    # Save training history to JSON for evaluate.py
    reports_dir = os.path.join(os.path.dirname(args.output_dir), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    history_data = {
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "active_classes": active_classes,
        "num_classes": num_classes,
    }
    if history_ft:
        history_data["ft_accuracy"] = history_ft.history["accuracy"]
        history_data["ft_val_accuracy"] = history_ft.history["val_accuracy"]
        history_data["ft_loss"] = history_ft.history["loss"]
        history_data["ft_val_loss"] = history_ft.history["val_loss"]

    history_path = os.path.join(reports_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"[Training] History saved to: {history_path}")

    return {
        "model": model,
        "history": history,
        "history_ft": history_ft,
        "train_gen": train_gen,
        "val_gen": val_gen,
        "test_gen": test_gen,
        "active_classes": active_classes,
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# Post-training: save history plots
# ---------------------------------------------------------------------------
def save_training_plots(history, reports_dir: str, history_ft=None) -> None:
    """Save accuracy and loss curves to the reports directory."""
    os.makedirs(reports_dir, exist_ok=True)

    # Merge phase 1 + phase 2 histories if fine-tuned
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if history_ft is not None:
        acc += history_ft.history["accuracy"]
        val_acc += history_ft.history["val_accuracy"]
        loss += history_ft.history["loss"]
        val_loss += history_ft.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, acc, "b-o", markersize=4, label="Training Accuracy")
    ax.plot(epochs_range, val_acc, "r-s", markersize=4, label="Validation Accuracy")
    if history_ft is not None:
        ax.axvline(
            x=len(history.history["accuracy"]),
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Fine-tune start",
        )
    ax.set_title("FreshCheck AI — Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(reports_dir, "training_accuracy.png")
    plt.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[Reports] Accuracy plot saved: {acc_path}")

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, loss, "b-o", markersize=4, label="Training Loss")
    ax.plot(epochs_range, val_loss, "r-s", markersize=4, label="Validation Loss")
    if history_ft is not None:
        ax.axvline(
            x=len(history.history["loss"]),
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Fine-tune start",
        )
    ax.set_title("FreshCheck AI — Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(reports_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[Reports] Loss plot saved: {loss_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    import_dependencies()

    # Set global seeds for reproducibility
    import numpy as np_seed
    np_seed.random.seed(SEED)
    try:
        import tensorflow as tf_seed
        tf_seed.random.set_seed(SEED)
    except Exception:
        pass

    result = train_model(args)
    model = result["model"]
    history = result["history"]
    history_ft = result["history_ft"]
    active_classes = result.get("active_classes", CLASS_NAMES)
    num_classes = result.get("num_classes", NUM_CLASSES)

    # Save training plots
    reports_dir = os.path.join(os.path.dirname(args.output_dir), "reports")
    save_training_plots(history, reports_dir, history_ft=history_ft)

    # Save final val accuracy summary
    best_val_acc = max(history.history["val_accuracy"])
    if history_ft:
        best_val_acc = max(best_val_acc, max(history_ft.history["val_accuracy"]))

    print(f"\n[Summary] Active classes     : {active_classes} ({num_classes} classes)")
    print(f"[Summary] Best Val Accuracy  : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    if num_classes < NUM_CLASSES:
        print(f"\n[NOTICE] Training completed with {num_classes} classes (not {NUM_CLASSES}).")
        print(f"  Missing: {set(CLASS_NAMES) - set(active_classes)}")
        print(f"  This is NOT a complete three-class model.")
        print(f"  Obtain genuine data for missing classes. See ml_model/dataset/dataset_sources.md")
    print("[Summary] Training complete. Run evaluate.py for full metrics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
