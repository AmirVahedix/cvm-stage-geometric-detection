#!/usr/bin/env python3
"""
Generate Calibrated CVM Test Set.

Workflow:
1. Loads ground-truth stages from distribution/runs/calibrated_gt.txt.
2. Maps each stage to its corresponding image in data/full/images using strict ordinal sort
   (matching runs_comparison.csv and existing distribution runs).
3. Randomly samples an equal number of images (default: 20) from each of the 6 CVM stages (CS1-CS6),
   yielding a balanced test split of 120 images.
4. Copies the selected images to data/test/images (or data/test if --flat is specified).
5. Generates 'test_calibrated_gt.txt' with one line per test sample in natural ordinal sort order,
   identically formatted to calibrated_gt.txt.
6. Generates individual label files in data/test/labels/{stem}.txt and a detailed mapping CSV
   data/test/test_calibrated_gt_mapping.csv for transparency and verification.
7. Also syncs a copy of test_calibrated_gt.txt to distribution/runs/test_calibrated_gt.txt.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ordinal_sort_key(filename: str) -> Tuple[int, int, List[Any]]:
    """
    Ordinal sort key for filenames (1.jpg, 2.jpg, ... 10.jpg, 100.jpg, 1101.jpg).
    Pure numeric stems are sorted by integer value; mixed stems use natural chunking.
    """
    stem = Path(filename).stem
    if stem.isdigit():
        return (0, int(stem), [])
    parts = re.split(r"(\d+)", stem)
    parsed = [int(p) if p.isdigit() else p.lower() for p in parts]
    return (1, 0, parsed)


def parse_args():
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate a balanced test set (20 images per CVM stage) based on calibrated_gt.txt."
    )
    parser.add_argument(
        "--calibrated-gt",
        type=str,
        default=str(repo_root / "distribution" / "runs" / "calibrated_gt.txt"),
        help="Path to calibrated_gt.txt (default: distribution/runs/calibrated_gt.txt).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default=str(repo_root / "distribution" / "runs_comparison.csv"),
        help="Path to runs_comparison.csv (default: distribution/runs_comparison.csv).",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(repo_root / "data" / "full"),
        help="Path to source directory containing full dataset images (default: data/full).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "data" / "test"),
        help="Target output directory for test set (default: data/test).",
    )
    parser.add_argument(
        "--num-per-stage",
        type=int,
        default=20,
        help="Number of images to randomly sample per CVM stage (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling; set to -1 for non-deterministic random (default: 42).",
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["standard", "flat"],
        default="standard",
        help="Output directory layout: 'standard' creates 'images/' and 'labels/' inside test folder; "
        "'flat' copies images directly to the test folder (default: standard).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(repo_root / "distribution" / "runs"),
        help="Path to runs directory to sync test_calibrated_gt.txt into (default: distribution/runs).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean/delete existing output directory before generating test set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview test set selection and distribution without copying files.",
    )
    return parser.parse_args()


def load_calibrated_gt(gt_path: Path) -> List[int]:
    """Reads calibrated_gt.txt and returns list of integer stages."""
    if not gt_path.is_file():
        raise FileNotFoundError(f"Calibrated GT file not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        stages = [int(line.strip()) for line in f if line.strip()]
    return stages


def resolve_source_images(source_dir: Path) -> Tuple[Path, List[Path]]:
    """Resolves image directory inside source_dir and returns sorted list of image paths."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    images_dir = source_dir / "images" if (source_dir / "images").is_dir() else source_dir
    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]
    images.sort(key=lambda p: ordinal_sort_key(p.name))
    return images_dir, images


def print_stage_distribution(counts: Dict[int, int], title: str):
    """Prints a styled ASCII distribution chart for CVM stages."""
    total = sum(counts.values())
    print("\n" + "=" * 65)
    print(f" {title:^63} ")
    print("=" * 65)
    for stage in range(1, 7):
        cnt = counts.get(stage, 0)
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        bar = "█" * int(round(pct / 2.5))
        print(f"  CS{stage} (Stage {stage}): {cnt:4d} images ({pct:5.1f}%)  {bar}")
    print("-" * 65)
    print(f"  Total:        {total:4d} images")
    print("=" * 65)


def main():
    args = parse_args()

    calibrated_gt_path = Path(args.calibrated_gt)
    comparison_csv_path = Path(args.comparison_csv)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None

    print("=" * 75)
    print("           CALIBRATED CVM TEST SET GENERATOR           ")
    print("=" * 75)
    print(f"📄 Calibrated GT Source:  {calibrated_gt_path}")
    print(f"📁 Source Directory:      {source_dir}")
    print(f"🎯 Target Output Dir:     {output_dir}")
    print(f"🎲 Random Seed:           {args.seed if args.seed >= 0 else 'None (Non-deterministic)'}")
    print(f"🔢 Samples Per Stage:     {args.num_per_stage} images/stage")
    print(f"📐 Folder Structure:      {args.structure}")
    print(f"🧹 Clean Existing Dir:    {args.clean}")
    print("=" * 75)

    # 1. Load Calibrated GT stages
    gt_stages = load_calibrated_gt(calibrated_gt_path)

    # 2. Discover and sort source images
    images_dir, source_images = resolve_source_images(source_dir)
    print(f"\n📂 Discovered {len(source_images)} images in {images_dir}")
    print(f"📋 Loaded {len(gt_stages)} ground-truth stage annotations from {calibrated_gt_path.name}")

    if len(source_images) != len(gt_stages):
        raise ValueError(
            f"Mismatch between number of source images ({len(source_images)}) "
            f"and GT stage annotations ({len(gt_stages)})!"
        )

    # Cross-verify with runs_comparison.csv if available
    if comparison_csv_path.is_file():
        with open(comparison_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            csv_rows = list(reader)
        if len(csv_rows) == len(source_images):
            for idx, (img_p, r, stg) in enumerate(zip(source_images, csv_rows, gt_stages)):
                csv_fn = r.get("filename", "")
                csv_stg = int(r.get("run5_gt_calibrated", -1))
                if csv_fn != img_p.name:
                    raise ValueError(f"Ordinal mismatch at #{idx}: Disk={img_p.name} vs CSV={csv_fn}")
                if csv_stg != stg:
                    raise ValueError(f"Stage mismatch at #{idx} ({img_p.name}): GT={stg} vs CSV={csv_stg}")
            print(f"✅ Strict 1-to-1 alignment verified against {comparison_csv_path.name}.")

    # 3. Group samples by stage
    samples_by_stage: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    full_counts: Dict[int, int] = defaultdict(int)

    for idx, (img_p, stg) in enumerate(zip(source_images, gt_stages), start=1):
        rec = {
            "full_ordinal_index": idx,
            "filename": img_p.name,
            "stem": img_p.stem,
            "path": img_p,
            "stage": stg,
        }
        samples_by_stage[stg].append(rec)
        full_counts[stg] += 1

    print_stage_distribution(full_counts, f"FULL DATASET DISTRIBUTION ({len(source_images)} IMAGES)")

    # 4. Check class availability
    num_per_stage = args.num_per_stage
    for stage in range(1, 7):
        avail = len(samples_by_stage[stage])
        if avail < num_per_stage:
            raise ValueError(
                f"Stage CS{stage} only has {avail} images available, "
                f"which is less than requested {num_per_stage} images!"
            )

    # 5. Randomly sample 20 images from each stage
    rng = random.Random(args.seed) if args.seed >= 0 else random.Random()
    selected_samples: List[Dict[str, Any]] = []
    test_counts: Dict[int, int] = defaultdict(int)

    for stage in range(1, 7):
        # Sort pool beforehand to ensure deterministic shuffle across systems
        pool = list(samples_by_stage[stage])
        pool.sort(key=lambda s: ordinal_sort_key(s["filename"]))
        selected = rng.sample(pool, num_per_stage)
        selected_samples.extend(selected)
        test_counts[stage] = len(selected)

    # 6. Sort selected test set strictly by ordinal order
    selected_samples.sort(key=lambda s: ordinal_sort_key(s["filename"]))

    # Add test ordinal index (1-based)
    for idx, sample in enumerate(selected_samples, start=1):
        sample["test_ordinal_index"] = idx

    print_stage_distribution(
        test_counts,
        f"BALANCED TEST SPLIT ({len(selected_samples)} IMAGES: {num_per_stage}/STAGE)",
    )

    # Print Comparative Table
    print("\n" + "=" * 70)
    print(f"{'Stage':<6} | {'Full Count':<11} | {'Full %':<8} | {'Test Count':<11} | {'Test %':<8}")
    print("-" * 70)
    total_full = len(source_images)
    total_test = len(selected_samples)
    for stage in range(1, 7):
        fc = full_counts.get(stage, 0)
        fp = (fc / total_full * 100.0) if total_full else 0.0
        tc = test_counts.get(stage, 0)
        tp = (tc / total_test * 100.0) if total_test else 0.0
        print(f"CS{stage:<4} | {fc:11d} | {fp:7.2f}% | {tc:11d} | {tp:7.2f}%")
    print("-" * 70)
    print(f"{'Total':<6} | {total_full:11d} | 100.00%  | {total_test:11d} | 100.00%  |")
    print("=" * 70)

    if args.dry_run:
        print("\n🔍 DRY-RUN MODE: No files were copied or created.")
        return

    # 7. Prepare output directory
    if args.clean and output_dir.exists():
        print(f"\n🧹 Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.structure == "standard":
        dst_images_dir = output_dir / "images"
        dst_labels_dir = output_dir / "labels"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)
    else:  # flat
        dst_images_dir = output_dir
        dst_labels_dir = output_dir / "labels"
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

    # 8. Copy selected images and write individual label files
    print(f"\n🚚 Copying {len(selected_samples)} test images to {dst_images_dir}...")
    for sample in selected_samples:
        src_path = sample["path"]
        dst_img_path = dst_images_dir / sample["filename"]
        shutil.copy2(src_path, dst_img_path)

        # Write individual label file (stage number)
        dst_lbl_path = dst_labels_dir / f"{sample['stem']}.txt"
        dst_lbl_path.write_text(f"{sample['stage']}\n", encoding="utf-8")

    print(f"✅ Successfully copied {len(selected_samples)} images.")
    print(f"✅ Successfully created {len(selected_samples)} label files in {dst_labels_dir}.")

    # 9. Generate test_calibrated_gt.txt (one line per sample, strictly ordinal sorted)
    test_calibrated_gt_content = "".join(f"{s['stage']}\n" for s in selected_samples)

    # Save inside data/test/
    test_calibrated_gt_path = output_dir / "test_calibrated_gt.txt"
    test_calibrated_gt_path.write_text(test_calibrated_gt_content, encoding="utf-8")
    print(f"💾 Generated primary GT file: {test_calibrated_gt_path} ({len(selected_samples)} lines)")

    # Also sync to distribution/runs/ if requested and directory exists
    if runs_dir and runs_dir.is_dir():
        runs_gt_path = runs_dir / "test_calibrated_gt.txt"
        runs_gt_path.write_text(test_calibrated_gt_content, encoding="utf-8")
        print(f"💾 Synced GT file to runs:    {runs_gt_path}")

    # 10. Generate detailed mapping CSV and JSON manifest for easy inspection
    csv_mapping_path = output_dir / "test_calibrated_gt_mapping.csv"
    with open(csv_mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["test_ordinal_index", "filename", "calibrated_stage", "full_ordinal_index"])
        for s in selected_samples:
            writer.writerow([s["test_ordinal_index"], s["filename"], s["stage"], s["full_ordinal_index"]])
    print(f"📊 Saved detailed mapping:   {csv_mapping_path}")

    manifest_path = output_dir / "test_manifest.json"
    manifest = {
        "metadata": {
            "source_gt_file": str(calibrated_gt_path),
            "num_per_stage": num_per_stage,
            "total_test_samples": len(selected_samples),
            "seed": args.seed,
            "structure": args.structure,
        },
        "class_distribution": {f"CS{k}": v for k, v in sorted(test_counts.items())},
        "samples": [
            {
                "test_ordinal_index": s["test_ordinal_index"],
                "filename": s["filename"],
                "stage": s["stage"],
                "full_ordinal_index": s["full_ordinal_index"],
            }
            for s in selected_samples
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Saved test manifest:      {manifest_path}")

    # 11. Print Preview of first 15 test items
    print("\nPreview of first 15 test items (strictly natural ordinal order):")
    print(f"{'Test Index':<12} | {'Filename':<15} | {'Calibrated Stage':<18} | {'Full Set Index'}")
    print("-" * 65)
    for s in selected_samples[:15]:
        print(
            f"{s['test_ordinal_index']:<12} | {s['filename']:<15} | "
            f"CS{s['stage']:<16} | #{s['full_ordinal_index']}"
        )
    print("=" * 65)
    print("🎉 Test set generation complete!")


if __name__ == "__main__":
    main()
