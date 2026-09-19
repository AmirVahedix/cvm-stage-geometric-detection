#!/usr/bin/env python3
"""
Create Balanced Test Splits for CVM Stage Classification.

This script partitions images from data/full into 3 test splits:
- data/test_split/1
- data/test_split/2
- data/test_split/3

For each class (CS1 to CS6), it extracts a balanced subset (default: 8 images per class
per split, resulting in 24 images per class across the 3 splits, or 8+8+9=25 if specified).

Standard layout created for each split:
  data/test_split/{split_id}/images/{id}.jpg
  data/test_split/{split_id}/labels/{id}.txt

Also supports --flat to place images directly into data/test_split/{split_id}/.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into balanced test folds by CVM class."
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
        default="data/test_split",
        help="Target output directory (default: data/test_split).",
    )
    parser.add_argument(
        "--split-sizes",
        type=int,
        nargs="+",
        default=[8, 8, 8],
        help="Number of images per class for each split folder (default: 8 8 8). "
        "Example for 25 total: --split-sizes 8 8 9",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["standard", "flat"],
        default="standard",
        help="Folder layout: 'standard' creates images/ & labels/ inside each split; 'flat' places files directly into each split folder.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip copying label .txt files (copies images only).",
    )
    return parser.parse_args()


def load_dataset_by_class(source_path: Path) -> Dict[str, List[Path]]:
    images_dir = source_path / "images"
    labels_dir = source_path / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{images_dir}' and '{labels_dir}' to exist in source directory '{source_path}'."
        )

    # Map image stem -> image file path
    image_files = {p.stem: p for p in images_dir.glob("*") if p.is_file() and not p.name.startswith(".")}
    label_files = {p.stem: p for p in labels_dir.glob("*.txt") if p.is_file() and not p.name.startswith(".")}

    class_to_images = defaultdict(list)
    common_stems = sorted(image_files.keys() & label_files.keys())

    for stem in common_stems:
        lbl_path = label_files[stem]
        class_id = lbl_path.read_text().strip()
        class_to_images[class_id].append(image_files[stem])

    return class_to_images


def main():
    args = parse_args()
    source_path = Path(args.source_dir)
    output_base = Path(args.output_dir)
    split_sizes = args.split_sizes
    num_splits = len(split_sizes)

    print("=" * 60)
    print("        BALANCED TEST SPLIT GENERATOR        ")
    print("=" * 60)
    print(f"Source Directory:  {source_path}")
    print(f"Output Directory:  {output_base}")
    print(f"Number of Splits:  {num_splits} ({', '.join(f'Split {i+1}: {s}/class' for i, s in enumerate(split_sizes))})")
    print(f"Total per Class:   {sum(split_sizes)} images")
    print(f"Random Seed:       {args.seed}")
    print(f"Folder Structure:  {args.structure}")
    print(f"Copy Labels:       {not args.no_labels}")
    print("=" * 60)

    # 1. Load data
    class_to_images = load_dataset_by_class(source_path)
    available_classes = sorted(class_to_images.keys())

    print(f"\nDiscovered {len(available_classes)} classes in {source_path}:")
    for cls in available_classes:
        print(f"  CS{cls}: {len(class_to_images[cls])} images available")

    # Check that each class has enough images
    total_required_per_class = sum(split_sizes)
    for cls in available_classes:
        avail = len(class_to_images[cls])
        if avail < total_required_per_class:
            raise ValueError(
                f"Class CS{cls} only has {avail} images, but {total_required_per_class} are required across all splits."
            )

    # 2. Shuffle and partition
    rng = random.Random(args.seed)
    # split_id -> class_id -> list of image Paths
    splits_data: Dict[int, Dict[str, List[Path]]] = {i + 1: defaultdict(list) for i in range(num_splits)}

    for cls in available_classes:
        shuffled = list(class_to_images[cls])
        rng.shuffle(shuffled)

        cursor = 0
        for split_idx, size in enumerate(split_sizes, start=1):
            splits_data[split_idx][cls] = shuffled[cursor : cursor + size]
            cursor += size

    # 3. Create folders and copy files
    labels_dir = source_path / "labels"
    manifest: Dict[str, Dict[str, List[str]]] = {}

    for split_idx in range(1, num_splits + 1):
        split_dir = output_base / str(split_idx)

        if args.structure == "standard":
            img_dest = split_dir / "images"
            lbl_dest = split_dir / "labels"
            img_dest.mkdir(parents=True, exist_ok=True)
            if not args.no_labels:
                lbl_dest.mkdir(parents=True, exist_ok=True)
        else:  # flat
            img_dest = split_dir
            lbl_dest = split_dir
            split_dir.mkdir(parents=True, exist_ok=True)

        manifest[f"split_{split_idx}"] = {}

        for cls in available_classes:
            items = splits_data[split_idx][cls]
            manifest[f"split_{split_idx}"][f"CS{cls}"] = [p.name for p in items]

            for img_path in items:
                # Copy image
                shutil.copy2(img_path, img_dest / img_path.name)

                # Copy label
                if not args.no_labels:
                    src_lbl = labels_dir / f"{img_path.stem}.txt"
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, lbl_dest / f"{img_path.stem}.txt")

    # Save manifest
    output_base.mkdir(parents=True, exist_ok=True)
    manifest_file = output_base / "splits_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4. Print Summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY TABLE")
    print("=" * 60)
    header = f"{'Split Folder':<20}" + "".join(f"{f'CS{c}':>8}" for c in available_classes) + f"{'Total':>10}"
    print(header)
    print("-" * len(header))

    for split_idx in range(1, num_splits + 1):
        folder_str = f"{output_base}/{split_idx}"
        counts = [len(splits_data[split_idx][c]) for c in available_classes]
        row_str = f"{folder_str:<20}" + "".join(f"{cnt:>8}" for cnt in counts) + f"{sum(counts):>10}"
        print(row_str)

    print("-" * len(header))
    totals_per_class = [sum(len(splits_data[s][c]) for s in range(1, num_splits + 1)) for c in available_classes]
    grand_total = sum(totals_per_class)
    print(f"{'Total Images':<20}" + "".join(f"{cnt:>8}" for cnt in totals_per_class) + f"{grand_total:>10}")
    print("=" * 60)
    print(f"Manifest saved to: {manifest_file}")
    print("All splits successfully generated!")


if __name__ == "__main__":
    main()
