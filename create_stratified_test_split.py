#!/usr/bin/env python3
"""
Stratified Test Split Generator for CVM Stage Classification.

This script creates a stratified test split based on:
1. Stage distribution defined line-by-line in stages.txt (ordered ordinally: 1.jpg, 2.jpg, ...).
2. Image and label data inside data/full.

Output layout:
  data/test/
    images/
      {id}.jpg
    labels/
      {id}.txt
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def natural_sort_key(s: Any) -> List[Any]:
    """Helper for natural/ordinal sorting (e.g., 1.jpg, 2.jpg, ..., 10.jpg)."""
    text = str(s)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def format_stage(val: str) -> str:
    """Normalizes stage strings like '1' or 'CS1' to 'CS1'."""
    val_clean = val.strip().upper()
    if val_clean.startswith("CS"):
        return val_clean
    return f"CS{val_clean}"


def render_distribution_bar(
    stage_counts: Dict[str, int],
    total: int,
    title: str = "CVM Stage Distribution:",
    stages_order: Optional[List[str]] = None,
) -> str:
    """Renders distribution text and ASCII block bars matching terminal output style."""
    if stages_order is None:
        stages_order = [f"CS{i}" for i in range(1, 7)]

    lines = [title]
    for stage_name in stages_order:
        count = stage_counts.get(stage_name, 0)
        pct = (count / total * 100.0) if total > 0 else 0.0
        bar = "█" * int(round(pct / 2.5))
        lines.append(f"  {stage_name}: {count:3d} ({pct:5.1f}%) {bar}")
    return "\n".join(lines)


def allocate_stratified_counts(
    class_counts: Dict[str, int],
    target_count: int,
) -> Dict[str, int]:
    """
    Allocates exact class sample sizes using the Largest Remainder Method (Hamilton's method)
    to guarantee the sum of allocated counts equals target_count while matching original proportions.
    """
    total_population = sum(class_counts.values())
    if target_count <= 0 or total_population == 0:
        return {k: 0 for k in class_counts}

    # Exact quota for each class
    quotas = {k: (cnt / total_population) * target_count for k, cnt in class_counts.items()}
    # Initial floor allocation (ensuring at least 1 if cnt > 0 and target_count >= len(class_counts))
    allocated = {k: math.floor(q) for k, q in quotas.items()}
    
    # Ensure every non-empty class gets at least 1 sample if possible
    if target_count >= sum(1 for cnt in class_counts.values() if cnt > 0):
        for k, cnt in class_counts.items():
            if cnt > 0 and allocated[k] == 0:
                allocated[k] = 1

    current_sum = sum(allocated.values())
    remainder = target_count - current_sum

    if remainder > 0:
        # Sort classes by fractional parts descending
        fractions = sorted(
            class_counts.keys(),
            key=lambda k: (quotas[k] - allocated[k]),
            reverse=True,
        )
        for i in range(remainder):
            cls = fractions[i % len(fractions)]
            allocated[cls] += 1
    elif remainder < 0:
        # In case we over-allocated due to minimum 1 enforcement, reduce from classes with largest allocations
        overage = -remainder
        reducible = sorted(
            [k for k in class_counts.keys() if allocated[k] > 1],
            key=lambda k: (quotas[k] - allocated[k]),
        )
        for i in range(overage):
            cls = reducible[i % len(reducible)]
            allocated[cls] -= 1

    return allocated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create stratified test split from stages.txt and data/full."
    )
    parser.add_argument(
        "--stages-file",
        type=str,
        default="stages.txt",
        help="Path to stages.txt file (default: stages.txt).",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/full",
        help="Source directory containing 'images' and 'labels' (default: data/full).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test",
        help="Output test directory path (default: data/test).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data to use for test split (e.g. 0.15 for 15%%) OR integer sample count (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        choices=["copy", "stages"],
        default="copy",
        help="How to generate test labels: 'copy' copies original files from source labels dir; "
        "'stages' writes stage directly from stages.txt (default: copy).",
    )
    parser.add_argument(
        "--save-ids",
        type=str,
        default="test_image_ids.json",
        help="Path to save JSON list of test image stem IDs (default: test_image_ids.json).",
    )
    parser.add_argument(
        "--save-manifest",
        type=str,
        default=None,
        help="Optional path to save full test split manifest JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the split and print statistics without writing/copying files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directory before copying new test split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stages_path = Path(args.stages_file)
    source_path = Path(args.source_dir)
    images_dir = source_path / "images"
    labels_dir = source_path / "labels"
    output_dir = Path(args.output_dir)

    print("=" * 65)
    print("      STRATIFIED CVM TEST SPLIT GENERATOR      ")
    print("=" * 65)
    print(f"Stages File:       {stages_path.resolve()}")
    print(f"Source Directory:  {source_path.resolve()}")
    print(f"Output Directory:  {output_dir.resolve()}")
    print(f"Test Ratio / Size: {args.test_size}")
    print(f"Random Seed:       {args.seed}")
    print(f"Label Source Mode: {args.label_source}")
    print("=" * 65)

    # 1. Validation of inputs
    if not stages_path.is_file():
        print(f"❌ Error: stages file not found at {stages_path}", file=sys.stderr)
        sys.exit(1)

    if not images_dir.is_dir():
        print(f"❌ Error: images directory not found at {images_dir}", file=sys.stderr)
        sys.exit(1)

    # 2. Read stages.txt
    with open(stages_path, "r", encoding="utf-8") as f:
        stage_lines = [line.strip() for line in f if line.strip()]

    # 3. Discover images and sort in ordinal / natural order
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = [p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_exts and not p.name.startswith(".")]
    image_paths.sort(key=lambda p: natural_sort_key(p.name))

    print(f"\n📂 Found {len(image_paths)} images in {images_dir}")
    print(f"📄 Found {len(stage_lines)} stages in {stages_path}")

    if len(image_paths) != len(stage_lines):
        print(
            f"❌ Error: Mismatch between number of images ({len(image_paths)}) and stages ({len(stage_lines)})!",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. Map images to stages ordinally (1:1 matching)
    dataset: List[Dict[str, Any]] = []
    full_counts = Counter()
    class_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, (img_p, stg_raw) in enumerate(zip(image_paths, stage_lines), start=1):
        stage_str = format_stage(stg_raw)
        stage_int = int(stage_str.replace("CS", ""))
        stem = img_p.stem
        rec = {
            "ordinal_index": idx,
            "filename": img_p.name,
            "stem": stem,
            "image_path": img_p,
            "stage_str": stage_str,
            "stage_int": stage_int,
        }
        dataset.append(rec)
        full_counts[stage_str] += 1
        class_groups[stage_str].append(rec)

    # Display full dataset distribution
    print("\n" + render_distribution_bar(full_counts, len(dataset), title="Original Full Dataset Distribution (from stages.txt):"))

    # 5. Determine test sample count
    total_samples = len(dataset)
    if args.test_size < 1.0:
        target_test_count = int(round(total_samples * args.test_size))
    else:
        target_test_count = int(args.test_size)

    if target_test_count <= 0 or target_test_count > total_samples:
        print(f"❌ Error: Invalid test target count {target_test_count} for dataset of {total_samples} items.", file=sys.stderr)
        sys.exit(1)

    # 6. Allocate sample size per class via stratified apportionment
    stages_order = [f"CS{i}" for i in range(1, 7)]
    allocated_counts = allocate_stratified_counts(full_counts, target_test_count)

    # 7. Sample items with seed
    import random
    rng = random.Random(args.seed)

    test_samples: List[Dict[str, Any]] = []
    train_samples: List[Dict[str, Any]] = []
    test_counts = Counter()

    for stage_name in stages_order:
        items = list(class_groups[stage_name])
        # Sort beforehand to ensure reproducible shuffle regardless of OS filesystem order
        items.sort(key=lambda x: natural_sort_key(x["filename"]))
        rng.shuffle(items)

        k = allocated_counts.get(stage_name, 0)
        selected_test = items[:k]
        remaining_train = items[k:]

        test_samples.extend(selected_test)
        train_samples.extend(remaining_train)
        test_counts[stage_name] = len(selected_test)

    # Sort test samples ordinally for clean presentation
    test_samples.sort(key=lambda x: natural_sort_key(x["filename"]))
    train_samples.sort(key=lambda x: natural_sort_key(x["filename"]))

    # 8. Print Test Distribution
    test_total = len(test_samples)
    pct_of_full = (test_total / total_samples) * 100.0
    print(f"\n" + render_distribution_bar(
        test_counts,
        test_total,
        title=f"Stratified Test Split Distribution ({test_total} samples, {pct_of_full:.1f}% of full):",
    ))

    # Print Comparative Stratification Table
    print("\n" + "=" * 65)
    print(f"{'Stage':<6} | {'Full Count':<11} | {'Full %':<8} | {'Test Count':<11} | {'Test %':<8} | {'Diff %':<8}")
    print("-" * 65)
    for stage_name in stages_order:
        fc = full_counts.get(stage_name, 0)
        fp = (fc / total_samples * 100.0) if total_samples else 0.0
        tc = test_counts.get(stage_name, 0)
        tp = (tc / test_total * 100.0) if test_total else 0.0
        diff = tp - fp
        print(f"{stage_name:<6} | {fc:11d} | {fp:7.2f}% | {tc:11d} | {tp:7.2f}% | {diff:+7.2f}%")
    print("-" * 65)
    print(f"{'Total':<6} | {total_samples:11d} | 100.00%  | {test_total:11d} | 100.00%  |")
    print("=" * 65)

    if args.dry_run:
        print("\n🔍 DRY-RUN MODE: No files were copied or created.")
        return

    # 9. Create Destination Directories
    test_images_dir = output_dir / "images"
    test_labels_dir = output_dir / "labels"

    if args.clean and output_dir.exists():
        print(f"🧹 Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚚 Copying {test_total} images and creating labels in {output_dir}...")

    copied_images = 0
    created_labels = 0

    for rec in test_samples:
        src_img = rec["image_path"]
        dst_img = test_images_dir / rec["filename"]
        shutil.copy2(src_img, dst_img)
        copied_images += 1

        dst_lbl = test_labels_dir / f"{rec['stem']}.txt"
        if args.label_source == "copy":
            src_lbl = labels_dir / f"{rec['stem']}.txt"
            if src_lbl.is_file():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # Fallback to writing stage integer if source label is absent
                dst_lbl.write_text(f"{rec['stage_int']}\n", encoding="utf-8")
        else:  # 'stages' mode
            dst_lbl.write_text(f"{rec['stage_int']}\n", encoding="utf-8")

        created_labels += 1

    print(f"✅ Successfully wrote {copied_images} images to {test_images_dir.resolve()}")
    print(f"✅ Successfully wrote {created_labels} labels to {test_labels_dir.resolve()}")

    # 10. Save test_image_ids.json
    if args.save_ids:
        save_ids_path = Path(args.save_ids)
        test_ids: List[Any] = []
        for s in test_samples:
            stem = s["stem"]
            test_ids.append(int(stem) if stem.isdigit() else stem)
        with open(save_ids_path, "w", encoding="utf-8") as f:
            json.dump(test_ids, f, indent=2)
        print(f"💾 Saved {len(test_ids)} test image IDs to: {save_ids_path.resolve()}")

    # 11. Optionally save manifest
    if args.save_manifest:
        manifest_path = Path(args.save_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "metadata": {
                "total_source_samples": total_samples,
                "total_test_samples": test_total,
                "test_ratio": args.test_size,
                "seed": args.seed,
                "label_source": args.label_source,
            },
            "class_distribution": {
                "full": {k: full_counts[k] for k in stages_order},
                "test": {k: test_counts[k] for k in stages_order},
            },
            "test_files": [
                {
                    "filename": s["filename"],
                    "stem": s["stem"],
                    "stage": s["stage_str"],
                }
                for s in test_samples
            ],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"📋 Saved manifest to: {manifest_path.resolve()}")

    print("\n🎉 Test split creation complete!")


if __name__ == "__main__":
    main()
