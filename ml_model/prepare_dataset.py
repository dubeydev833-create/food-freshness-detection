"""
FreshCheck AI — Dataset Preparation Script  (v2.0)
Member 1 Responsibility: AI/ML Model Development

Cleans, deduplicates and splits genuine freshness-labelled datasets into
train / validation / test directories with correct label structure.

IMPORTANT — WHAT THIS SCRIPT DOES AND DOES NOT DO:
  ✓ Loads genuinely labelled images from verified datasets
  ✓ Detects and reports corrupted images
  ✓ Detects exact duplicate images using MD5 hashing
  ✓ Reports near-duplicate candidates (perceptual hash, requires imagehash)
  ✓ Splits data with no image appearing in more than one split
  ✓ Performs stratified splitting to maintain class ratios
  ✗ Does NOT synthesise images to create missing classes
  ✗ Does NOT use brightness/noise/blur augmentation to fabricate Moderate labels
  ✗ Does NOT treat Fruits-360 as a freshness dataset

Usage:
    python ml_model/prepare_dataset.py \\
        --raw-dir ml_model/dataset/raw \\
        --out-dir ml_model/dataset/processed

Dataset structure expected under --raw-dir:
    raw/
        fresh_and_stale/        ← swoyam2609/fresh-and-stale-classification
            train/
                freshapple/ staleapple/ freshbanana/ ...
            test/
                freshapple/ staleapple/ ...
        meat_freshness/         ← vinayakshanawad/meat-quality-assessment
            Fresh/  Half_Fresh/  Spoiled/

Requirements:
    pip install pillow numpy scikit-learn tqdm
    pip install imagehash       # optional, for near-duplicate detection

Output structure:
    processed/
        train/     Fresh/   Moderate/   Spoiled/
        validation/ Fresh/  Moderate/   Spoiled/
        test/      Fresh/   Moderate/   Spoiled/
        dataset_summary.json
        dataset_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Label mapping — raw dataset folder names → canonical freshness class names
# Only genuine freshness labels are mapped. Food-type labels are excluded.
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    # --- Fresh & Stale Classification (swoyam2609) ---
    "freshapple": "Fresh",
    "freshbanana": "Fresh",
    "freshorange": "Fresh",
    "freshtomato": "Fresh",
    "freshpotato": "Fresh",
    "freshcapsicum": "Fresh",
    "freshcucumber": "Fresh",
    "freshokra": "Fresh",
    "freshbittergourd": "Fresh",
    "freshlongbeans": "Fresh",
    "freshmango": "Fresh",
    "freshlemon": "Fresh",
    "freshstrawberry": "Fresh",
    "freshkiwi": "Fresh",
    "freshgrape": "Fresh",
    "fresh": "Fresh",

    "staleapple": "Spoiled",
    "stalebanana": "Spoiled",
    "staleorange": "Spoiled",
    "staletomato": "Spoiled",
    "stalepotato": "Spoiled",
    "stalecapsicum": "Spoiled",
    "stalecucumber": "Spoiled",
    "staleokra": "Spoiled",
    "stalebittergourd": "Spoiled",
    "stalelong beans": "Spoiled",
    "stalelong_beans": "Spoiled",
    "stalekiwi": "Spoiled",
    "stalegrape": "Spoiled",
    "stalestrawberry": "Spoiled",
    "stalemango": "Spoiled",
    "stalelemon": "Spoiled",
    "stale": "Spoiled",

    # --- Meat Freshness Dataset (vinayakshanawad) ---
    # Half_Fresh is a genuine label — documented in dataset_sources.md
    "fresh": "Fresh",          # already in LABEL_MAP above
    "half_fresh": "Moderate",
    "halffresh": "Moderate",
    "half-fresh": "Moderate",
    "half fresh": "Moderate",
    "spoiled": "Spoiled",
    "rotten": "Spoiled",

    # --- Generic fallbacks (lowercase exact match) ---
    "moderate": "Moderate",
}

CLASS_NAMES = ["Fresh", "Moderate", "Spoiled"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
def is_valid_image(path: Path) -> bool:
    """Return True if the image file can be opened and is not corrupted."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def file_md5(path: Path) -> str:
    """Return MD5 hex digest of a file (for exact-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path) -> str | None:
    """Return perceptual hash string, or None if imagehash is not installed."""
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------
def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Scan raw_dir recursively and map canonical class names to image paths.

    Uses LABEL_MAP to convert folder names to class names.
    Skips any folder name not in LABEL_MAP (reports skipped folders).
    Does NOT infer freshness from time-sequence folder names or numbering.
    """
    class_to_paths: dict[str, list[Path]] = {c: [] for c in CLASS_NAMES}
    skipped_dirs: set[str] = set()

    for item in sorted(raw_dir.rglob("*")):
        if not item.is_file():
            continue
        if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        folder_raw = item.parent.name
        folder_key = folder_raw.lower().strip().replace(" ", "_")

        # Exact match first
        canonical = LABEL_MAP.get(folder_key) or LABEL_MAP.get(folder_raw.lower().strip())

        # Prefix matching as fallback (e.g. "freshapple_extra" → Fresh)
        if canonical is None:
            for key, val in LABEL_MAP.items():
                if folder_key.startswith(key) or folder_key.endswith(key):
                    canonical = val
                    break

        if canonical and canonical in class_to_paths:
            class_to_paths[canonical].append(item)
        else:
            if folder_raw not in skipped_dirs:
                print(f"  [Skip] Folder '{folder_raw}' — not in LABEL_MAP, skipping")
                skipped_dirs.add(folder_raw)

    return class_to_paths


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def remove_exact_duplicates(paths: list[Path]) -> tuple[list[Path], int]:
    """Remove files with identical MD5 hashes. Returns (unique_list, count_removed)."""
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    for p in paths:
        h = file_md5(p)
        if h not in seen:
            seen[h] = p
            unique.append(p)
    return unique, len(paths) - len(unique)


def find_near_duplicates(
    paths: list[Path],
    threshold: int = 8,
) -> list[tuple[Path, Path, int]]:
    """
    Find near-duplicate image pairs using perceptual hashing (pHash).

    Returns list of (path_a, path_b, hash_distance) for pairs with
    hamming distance <= threshold. Returns empty list if imagehash not installed.
    """
    hashes: list[tuple[Path, str]] = []
    for p in paths:
        ph = perceptual_hash(p)
        if ph is not None:
            hashes.append((p, ph))

    if not hashes:
        return []

    near_dups: list[tuple[Path, Path, int]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            # Simple Hamming distance on hex strings via imagehash objects
            try:
                import imagehash
                h1 = imagehash.hex_to_hash(hashes[i][1])
                h2 = imagehash.hex_to_hash(hashes[j][1])
                dist = h1 - h2
                if dist <= threshold:
                    near_dups.append((hashes[i][0], hashes[j][0], dist))
            except Exception:
                break
    return near_dups


# ---------------------------------------------------------------------------
# Stratified split — prevents any image appearing in multiple splits
# ---------------------------------------------------------------------------
def stratified_split(
    paths: list[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> dict[str, list[Path]]:
    """
    Split a list of image paths into train / validation / test with no overlap.

    Uses a fixed seed for reproducibility. All shuffling happens before split,
    so the same input always produces the same output.
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Copy and resize
# ---------------------------------------------------------------------------
def copy_images(
    split_to_paths: dict[str, list[Path]],
    cls: str,
    out_dir: Path,
    target_size: tuple[int, int] = (224, 224),
    resize: bool = True,
) -> dict[str, int]:
    """Copy (and optionally resize) images into the structured output directory."""
    from PIL import Image

    counts: dict[str, int] = {}
    for split, paths in split_to_paths.items():
        dest_dir = out_dir / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in paths:
            # Unique filename to avoid collisions across subdatasets
            dest_name = f"{src.parent.name}_{src.name}"
            dest = dest_dir / dest_name
            if dest.exists():
                stem, suffix = dest_name.rsplit(".", 1) if "." in dest_name else (dest_name, "jpg")
                dest = dest_dir / f"{stem}_{file_md5(src)[:6]}.{suffix}"
            try:
                if resize:
                    with Image.open(src) as img:
                        img_rgb = img.convert("RGB")
                        img_resized = img_rgb.resize(target_size, Image.BICUBIC)
                        img_resized.save(str(dest), "JPEG", quality=95)
                else:
                    shutil.copy2(src, dest)
                copied += 1
            except Exception as e:
                print(f"    [Warning] Failed to copy {src.name}: {e}")
        counts[split] = copied
    return counts


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def write_summary(
    stats: dict,
    corrupted_counts: dict,
    duplicate_counts: dict,
    out_dir: Path,
    near_dup_warnings: list,
) -> None:
    """Write dataset_summary.json and dataset_summary.txt."""
    summary = {
        "dataset_structure": out_dir.resolve().as_posix(),
        "class_names": CLASS_NAMES,
        "splits": {},
        "per_class_totals": {},
        "corrupted_removed": corrupted_counts,
        "exact_duplicates_removed": duplicate_counts,
        "near_duplicate_pairs_flagged": len(near_dup_warnings),
        "limitations": [
            "Moderate class images come exclusively from Meat data (Half_Fresh label).",
            "Fresh and Spoiled for Fruits/Vegetables; no genuine Moderate for these categories.",
            "Dairy category is not supported — no public freshness-labelled dataset found.",
            "No synthetic augmentation was used to create any class.",
        ],
    }
    for split in ["train", "validation", "test"]:
        summary["splits"][split] = {}
        for cls in CLASS_NAMES:
            count = stats.get(cls, {}).get(split, 0)
            summary["splits"][split][cls] = count

    for cls in CLASS_NAMES:
        total = sum(stats.get(cls, {}).get(s, 0) for s in ["train", "validation", "test"])
        summary["per_class_totals"][cls] = total

    json_path = out_dir / "dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable text summary
    lines = [
        "=" * 60,
        "  FreshCheck AI — Dataset Preparation Summary",
        "=" * 60,
        "",
        f"Output directory: {out_dir.resolve()}",
        "",
        "CLASS DISTRIBUTION",
        "-" * 55,
        f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}",
        "-" * 55,
    ]
    grand_total = 0
    split_totals = {s: 0 for s in ["train", "validation", "test"]}
    for cls in CLASS_NAMES:
        s = stats.get(cls, {"train": 0, "validation": 0, "test": 0})
        t = s.get("train", 0) + s.get("validation", 0) + s.get("test", 0)
        grand_total += t
        for sp in split_totals:
            split_totals[sp] += s.get(sp, 0)
        lines.append(
            f"{cls:<12} {s.get('train',0):>8} {s.get('validation',0):>8} "
            f"{s.get('test',0):>8} {t:>8}"
        )
    lines.append(
        f"{'TOTAL':<12} {split_totals['train']:>8} {split_totals['validation']:>8} "
        f"{split_totals['test']:>8} {grand_total:>8}"
    )
    lines += [
        "",
        "QUALITY CHECKS",
        "-" * 40,
    ]
    for cls in CLASS_NAMES:
        lines.append(f"  {cls:<12} Corrupted removed: {corrupted_counts.get(cls, 0):<4}  "
                     f"Exact duplicates removed: {duplicate_counts.get(cls, 0)}")
    if near_dup_warnings:
        lines.append(f"\n  Near-duplicate pairs flagged: {len(near_dup_warnings)}")
        lines.append("  (Review near_duplicates.csv for details)")

    lines += [
        "",
        "KNOWN LIMITATIONS",
        "-" * 40,
        "  - Moderate class: Meat only (Half_Fresh → Moderate). Genuine label.",
        "  - Fruits/Vegetables: Fresh and Spoiled only. No genuine Moderate.",
        "  - Dairy: Not supported. No public freshness-labelled dataset found.",
        "  - No synthetic augmentation was used to fabricate any class.",
        "",
        "Next step: python ml_model/train.py --data-dir ml_model/dataset/processed",
    ]

    txt_path = out_dir / "dataset_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[Summary] Written: {json_path}")
    print(f"[Summary] Written: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="FreshCheck AI — Dataset Preparation (v2.0)"
    )
    parser.add_argument(
        "--raw-dir", required=True,
        help="Root directory of raw downloaded datasets."
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for processed train/validation/test splits."
    )
    parser.add_argument(
        "--skip-resize", action="store_true",
        help="Copy images without resizing to 224×224 (not recommended for training)."
    )
    parser.add_argument(
        "--skip-near-dup", action="store_true",
        help="Skip perceptual hash near-duplicate detection (faster)."
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.is_dir():
        print(f"ERROR: Raw directory not found: '{raw_dir}'")
        print("Download datasets as described in ml_model/dataset/dataset_sources.md")
        return 1

    print("=" * 60)
    print("  FreshCheck AI — Dataset Preparation v2.0")
    print("=" * 60)
    print(f"\n[Step 1] Scanning: {raw_dir}")

    class_to_paths = collect_images(raw_dir)
    for cls, paths in class_to_paths.items():
        print(f"  {cls:<12}: {len(paths):>6} images found")

    total_found = sum(len(v) for v in class_to_paths.values())
    if total_found == 0:
        print("\nERROR: No images found. Check the raw directory structure.")
        print("Expected subdirectory names matching LABEL_MAP in prepare_dataset.py")
        return 1

    print(f"\n[Step 2] Corrupted image check...")
    corrupted_counts: dict[str, int] = {}
    for cls in CLASS_NAMES:
        before = len(class_to_paths[cls])
        valid: list[Path] = []
        bad = 0
        for p in class_to_paths[cls]:
            if is_valid_image(p):
                valid.append(p)
            else:
                bad += 1
                print(f"    [Corrupted] {p}")
        corrupted_counts[cls] = bad
        class_to_paths[cls] = valid
        print(f"  {cls:<12}: {bad} corrupted removed → {len(valid)} valid")

    print(f"\n[Step 3] Exact duplicate detection (MD5)...")
    duplicate_counts: dict[str, int] = {}
    for cls in CLASS_NAMES:
        unique, removed = remove_exact_duplicates(class_to_paths[cls])
        duplicate_counts[cls] = removed
        class_to_paths[cls] = unique
        print(f"  {cls:<12}: {removed} exact duplicates removed → {len(unique)} unique")

    near_dup_warnings: list[tuple] = []
    if not args.skip_near_dup:
        print(f"\n[Step 4] Near-duplicate detection (perceptual hash)...")
        try:
            import imagehash  # noqa: F401
            for cls in CLASS_NAMES:
                pairs = find_near_duplicates(class_to_paths[cls], threshold=8)
                near_dup_warnings.extend(pairs)
                if pairs:
                    print(f"  {cls:<12}: {len(pairs)} near-duplicate pairs flagged")
                else:
                    print(f"  {cls:<12}: no near-duplicates found")
            # Write near-duplicate report
            if near_dup_warnings:
                nd_path = out_dir / "near_duplicates.csv"
                nd_path.parent.mkdir(parents=True, exist_ok=True)
                with open(nd_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["path_a", "path_b", "hamming_distance"])
                    for a, b, d in near_dup_warnings:
                        w.writerow([str(a), str(b), d])
                print(f"  Near-duplicate pairs written to: {nd_path}")
        except ImportError:
            print("  [Skip] imagehash not installed — run: pip install imagehash")
    else:
        print(f"\n[Step 4] Near-duplicate detection: SKIPPED (--skip-near-dup)")

    # Verify we have data
    empty_classes = [cls for cls in CLASS_NAMES if len(class_to_paths[cls]) == 0]
    if empty_classes:
        print(f"\n[WARNING] The following classes have NO images: {empty_classes}")
        print("  These classes will be missing from the output splits.")
        print("  Do NOT proceed with training until this is resolved.")
        print("  See ml_model/dataset/dataset_sources.md for download instructions.")

    print(f"\n[Step 5] Stratified split (70/15/15)...")
    all_stats: dict[str, dict[str, int]] = {}
    all_split_paths: dict[str, dict[str, list[Path]]] = {}

    for cls in CLASS_NAMES:
        paths = class_to_paths[cls]
        if not paths:
            all_stats[cls] = {"train": 0, "validation": 0, "test": 0}
            all_split_paths[cls] = {"train": [], "validation": [], "test": []}
            continue

        split_paths = stratified_split(paths)
        all_split_paths[cls] = split_paths
        print(
            f"  {cls:<12}: train={len(split_paths['train'])}  "
            f"val={len(split_paths['validation'])}  "
            f"test={len(split_paths['test'])}"
        )

    print(f"\n[Step 6] Copying images to: {out_dir}")
    print(f"  Resize to 224×224: {'No' if args.skip_resize else 'Yes'}")

    for cls in CLASS_NAMES:
        split_paths = all_split_paths[cls]
        if not any(split_paths.values()):
            print(f"  {cls:<12}: Skipped (no images)")
            continue
        counts = copy_images(
            split_paths, cls, out_dir,
            target_size=(224, 224),
            resize=not args.skip_resize,
        )
        all_stats[cls] = counts
        print(f"  {cls:<12}: copied {sum(counts.values())} images")

    # Write summary
    write_summary(all_stats, corrupted_counts, duplicate_counts, out_dir, near_dup_warnings)

    # Final report
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\n{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 44)
    for cls in CLASS_NAMES:
        s = all_stats.get(cls, {"train": 0, "validation": 0, "test": 0})
        print(f"{cls:<12} {s.get('train',0):>8} {s.get('validation',0):>8} {s.get('test',0):>8}")

    print("\nLIMITATIONS:")
    print("  - Moderate class: Meat only (genuine 'Half_Fresh' label mapped to Moderate)")
    print("  - Fruits/Vegetables: Fresh + Spoiled only (no genuine Moderate class)")
    print("  - Dairy: Not supported")
    print("\nNext: python ml_model/train.py --data-dir", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
