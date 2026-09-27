#!/usr/bin/env python3
"""
Standalone Script: Run Calibrated CVM Calculator on Ground-Truth & Predicted Landmarks.

Workflow:
1. Loads ground-truth landmark annotations from data/export_cache.json.
2. Loads neural model predicted landmarks from distribution/predictions/predicted_landmarks.json.
3. Sorts all samples strictly by natural numeric / ordinal sort key (1.jpg, 2.jpg, ..., 10.jpg, ..., 1101.jpg),
   matching existing runs in distribution/runs/.
4. Runs the Calibrated CVM calculator (mode='calibrated') on:
   - Ground-truth landmarks -> outputs distribution/runs/calibrated_gt.txt
   - Predicted landmarks   -> outputs distribution/runs/calibrated_pred.txt
5. Updates distribution/runs_comparison.csv:
   - Preserves all original columns and data.
   - Adds new columns: run5_gt_calibrated and run6_pred_calibrated.
6. Updates distribution/runs_distribution.json:
   - Preserves all original metadata and previous runs (run1 - run4).
   - Adds run5_gt_calibrated and run6_pred_calibrated with stage counts and percentages.
   - Adds calibrated parameters to metadata and calibrated agreement metrics to agreement_analysis.
7. Prints a comprehensive terminal distribution and agreement report.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from src.cvm_calculator import (
    CVMInput,
    CVMThresholds,
    Point,
    VertebraC2,
    VertebraC3C4,
    classify_cvm_stage,
)

EXPECTED_LANDMARKS = [
    "C2_PI",
    "C2_IC",
    "C2_AI",
    "C3_PS",
    "C3_AS",
    "C3_PI",
    "C3_IC",
    "C3_AI",
    "C4_PS",
    "C4_AS",
    "C4_PI",
    "C4_IC",
    "C4_AI",
]


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


def build_task_indices(tasks: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Indexes Label Studio tasks by ID and by image filename from task data URL."""
    tasks_by_id: Dict[str, Dict[str, Any]] = {}
    tasks_by_filename: Dict[str, Dict[str, Any]] = {}

    for t in tasks:
        tid = str(t.get("id"))
        tasks_by_id[tid] = t

        img_url = t.get("data", {}).get("img") or t.get("file_upload") or ""
        if img_url:
            parsed = urlparse(img_url)
            qp = parse_qs(parsed.query)
            if "d" in qp:
                fn = os.path.basename(qp["d"][0])
            else:
                fn = os.path.basename(parsed.path)
            if fn:
                tasks_by_filename[fn] = t
                tasks_by_filename[Path(fn).stem] = t

    return tasks_by_id, tasks_by_filename


def extract_landmarks(task: Dict[str, Any]) -> Dict[str, Point]:
    """Extracts the 13 cervical vertebrae landmarks in image pixel coordinates from task annotations."""
    annotations = task.get("annotations", [])
    if not annotations:
        raise ValueError(f"Task {task.get('id')} has no annotations.")

    img_w, img_h = None, None
    points_dict: Dict[str, Tuple[float, float]] = {}

    ann = annotations[-1]
    results = ann.get("result", [])

    for res in results:
        if res.get("type") != "keypointlabels":
            continue
        val = res.get("value", {})
        labels = val.get("keypointlabels", [])
        if not labels:
            continue
        lbl_name = labels[0]
        if lbl_name in EXPECTED_LANDMARKS:
            orig_w = res.get("original_width")
            orig_h = res.get("original_height")
            if orig_w and orig_h:
                img_w, img_h = orig_w, orig_h

            norm_x = float(val["x"]) / 100.0
            norm_y = float(val["y"]) / 100.0
            points_dict[lbl_name] = (norm_x, norm_y)

    if img_w is None or img_h is None:
        raise ValueError(f"Could not resolve dimensions for task {task.get('id')}.")

    missing = set(EXPECTED_LANDMARKS) - set(points_dict.keys())
    if missing:
        raise ValueError(f"Task {task.get('id')} missing landmarks: {sorted(missing)}")

    return {
        k: Point(x=nx * img_w, y=ny * img_h)
        for k, (nx, ny) in points_dict.items()
    }


def compute_calibrated_stage(landmarks: Dict[str, Point], thresholds: CVMThresholds) -> int:
    """Computes the CVM stage (1-6) using the Calibrated CVM calculator."""
    c2 = VertebraC2(
        inferior_posterior=landmarks["C2_PI"],
        inferior_concavity=landmarks["C2_IC"],
        inferior_anterior=landmarks["C2_AI"],
    )
    c3 = VertebraC3C4(
        superior_posterior=landmarks["C3_PS"],
        superior_anterior=landmarks["C3_AS"],
        inferior_posterior=landmarks["C3_PI"],
        inferior_concavity=landmarks["C3_IC"],
        inferior_anterior=landmarks["C3_AI"],
    )
    c4 = VertebraC3C4(
        superior_posterior=landmarks["C4_PS"],
        superior_anterior=landmarks["C4_AS"],
        inferior_posterior=landmarks["C4_PI"],
        inferior_concavity=landmarks["C4_IC"],
        inferior_anterior=landmarks["C4_AI"],
    )

    cvm_input = CVMInput(c2=c2, c3=c3, c4=c4)
    result = classify_cvm_stage(cvm_input, thresholds=thresholds)
    stage_str = result["stage"]
    return int(stage_str.replace("CS", ""))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Calibrated CVM Calculator on ground-truth and predicted landmarks."
    )
    parser.add_argument(
        "--export-cache",
        type=str,
        default="data/export_cache.json",
        help="Path to Label Studio export JSON file (default: data/export_cache.json).",
    )
    parser.add_argument(
        "--predicted-landmarks",
        type=str,
        default="distribution/predictions/predicted_landmarks.json",
        help="Path to predicted landmarks JSON file (default: distribution/predictions/predicted_landmarks.json).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="distribution/runs",
        help="Output directory where run text files are stored (default: distribution/runs).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default="distribution/runs_comparison.csv",
        help="Path to runs comparison CSV file to update (default: distribution/runs_comparison.csv).",
    )
    parser.add_argument(
        "--distribution-json",
        type=str,
        default="distribution/runs_distribution.json",
        help="Path to runs distribution JSON file to update (default: distribution/runs_distribution.json).",
    )
    parser.add_argument(
        "--c4-concavity-depth-mm",
        type=float,
        default=1.20,
        help="C4 concavity depth threshold in mm for calibrated mode (default: 1.20 mm).",
    )
    parser.add_argument(
        "--c4-concavity-ratio",
        type=float,
        default=0.065,
        help="C4 concavity relative ratio threshold for calibrated mode (default: 0.065 = 6.5%%).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    export_cache_path = Path(args.export_cache)
    pred_landmarks_path = Path(args.predicted_landmarks)
    runs_dir = Path(args.runs_dir)
    comparison_csv_path = Path(args.comparison_csv)
    distribution_json_path = Path(args.distribution_json)

    print("=" * 75)
    print("      CALIBRATED CVM CALCULATOR EVALUATION ON GT & PREDICTED DATA      ")
    print("=" * 75)
    print(f"📦 GT Landmarks Export:     {export_cache_path}")
    print(f"🧠 Predicted Landmarks:     {pred_landmarks_path}")
    print(f"📁 Runs Output Dir:         {runs_dir}")
    print(f"📊 Runs Comparison CSV:     {comparison_csv_path}")
    print(f"📈 Runs Distribution JSON:  {distribution_json_path}")
    print(f"⚙️ Calibrated C4 Depth Th:  {args.c4_concavity_depth_mm} mm")
    print(f"⚙️ Calibrated C4 Ratio Th:  {args.c4_concavity_ratio * 100:.1f}%")
    print(f"🛡️ Concavity Order Guard:   Enabled (suppress C4 unless C2/C3 confirmed)")
    print("=" * 75)

    # 1. Verify and load input data
    if not export_cache_path.is_file():
        print(f"❌ Error: Export cache file not found: {export_cache_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_landmarks_path.is_file():
        print(f"❌ Error: Predicted landmarks file not found: {pred_landmarks_path}", file=sys.stderr)
        sys.exit(1)

    with open(export_cache_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    tasks_by_id, tasks_by_filename = build_task_indices(tasks)

    with open(pred_landmarks_path, "r", encoding="utf-8") as f:
        preds_dict: Dict[str, Dict[str, List[float]]] = json.load(f)

    # 2. Determine ordered sample list
    # If runs_comparison.csv exists, use its exact sample sequence to guarantee 100% order fidelity
    samples_order: List[Tuple[int, str, str]] = []  # (ordinal_index, filename, task_id)
    existing_csv_rows: List[Dict[str, str]] = []

    if comparison_csv_path.is_file():
        with open(comparison_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            existing_csv_rows = list(reader)
        for r in existing_csv_rows:
            samples_order.append((int(r["ordinal_index"]), r["filename"], r["task_id"]))
        print(f"📋 Loaded {len(samples_order)} samples from {comparison_csv_path.name} (strictly preserving order).")
    else:
        # Fallback to ordinal sorting of predicted landmark filenames
        sorted_filenames = sorted(preds_dict.keys(), key=ordinal_sort_key)
        for idx, fn in enumerate(sorted_filenames, start=1):
            stem = Path(fn).stem
            task = tasks_by_filename.get(fn) or tasks_by_filename.get(stem) or tasks_by_id.get(stem)
            tid = str(task.get("id")) if task else stem
            samples_order.append((idx, fn, tid))
        print(f"📋 Sorted {len(samples_order)} samples using ordinal filename sorting.")

    # 3. Configure Calibrated CVM Thresholds
    thresholds_calibrated = CVMThresholds(
        mode="calibrated",
        enable_calibrated=True,
        c4_concavity_depth_mm_threshold=args.c4_concavity_depth_mm,
        c4_concavity_ratio_threshold=args.c4_concavity_ratio,
        concavity_order_guard=True,
    )

    # 4. Compute Calibrated CVM Stages
    print("\n🚀 Computing Calibrated CVM stages...")
    calibrated_gt_stages: List[int] = []
    calibrated_pred_stages: List[int] = []

    for ordinal_idx, fn, tid in samples_order:
        stem = Path(fn).stem

        # Extract Ground-Truth landmarks
        task = tasks_by_id.get(tid) or tasks_by_filename.get(fn) or tasks_by_filename.get(stem)
        if not task:
            raise KeyError(f"Could not find Label Studio task for filename '{fn}' (task_id: {tid})")

        gt_points = extract_landmarks(task)
        gt_stage = compute_calibrated_stage(gt_points, thresholds_calibrated)
        calibrated_gt_stages.append(gt_stage)

        # Extract Predicted landmarks
        if fn not in preds_dict:
            raise KeyError(f"Could not find predicted landmarks for filename '{fn}'")

        pred_raw = preds_dict[fn]
        pred_points = {
            name: Point(x=coords[0], y=coords[1])
            for name, coords in pred_raw.items()
        }
        pred_stage = compute_calibrated_stage(pred_points, thresholds_calibrated)
        calibrated_pred_stages.append(pred_stage)

    total_samples = len(samples_order)
    print(f"✅ Successfully evaluated {total_samples} samples.")

    # 5. Write single .txt files to runs folder
    runs_dir.mkdir(parents=True, exist_ok=True)
    calibrated_gt_path = runs_dir / "calibrated_gt.txt"
    calibrated_pred_path = runs_dir / "calibrated_pred.txt"

    with open(calibrated_gt_path, "w", encoding="utf-8") as f:
        for s in calibrated_gt_stages:
            f.write(f"{s}\n")

    with open(calibrated_pred_path, "w", encoding="utf-8") as f:
        for s in calibrated_pred_stages:
            f.write(f"{s}\n")

    print(f"\n💾 Saved Calibrated run files in '{runs_dir}':")
    print(f"   - {calibrated_gt_path.name}  ({len(calibrated_gt_stages)} lines)")
    print(f"   - {calibrated_pred_path.name} ({len(calibrated_pred_stages)} lines)")

    # 6. Update runs_comparison.csv
    if comparison_csv_path.is_file() and existing_csv_rows:
        updated_csv_rows: List[Dict[str, str]] = []
        for r, s_gt, s_pred in zip(existing_csv_rows, calibrated_gt_stages, calibrated_pred_stages):
            updated_row = dict(r)
            updated_row["run5_gt_calibrated"] = str(s_gt)
            updated_row["run6_pred_calibrated"] = str(s_pred)
            updated_csv_rows.append(updated_row)

        fieldnames = list(existing_csv_rows[0].keys())
        if "run5_gt_calibrated" not in fieldnames:
            fieldnames.append("run5_gt_calibrated")
        if "run6_pred_calibrated" not in fieldnames:
            fieldnames.append("run6_pred_calibrated")

        with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(updated_csv_rows)

        print(f"\n📊 Updated '{comparison_csv_path}':")
        print(f"   - Added columns: 'run5_gt_calibrated', 'run6_pred_calibrated' ({len(updated_csv_rows)} rows)")

    # 7. Update runs_distribution.json
    gt_calib_counter = Counter(calibrated_gt_stages)
    pred_calib_counter = Counter(calibrated_pred_stages)

    if distribution_json_path.is_file():
        with open(distribution_json_path, "r", encoding="utf-8") as f:
            dist_data = json.load(f)

        # Update metadata parameters
        if "metadata" in dist_data and "parameters" in dist_data["metadata"]:
            dist_data["metadata"]["parameters"]["c4_concavity_depth_mm_threshold"] = args.c4_concavity_depth_mm
            dist_data["metadata"]["parameters"]["c4_concavity_ratio_threshold"] = args.c4_concavity_ratio
            dist_data["metadata"]["parameters"]["concavity_order_guard"] = True

        # Add Run 5 and Run 6
        if "runs" not in dist_data:
            dist_data["runs"] = {}

        dist_data["runs"]["run5_gt_calibrated"] = {
            "description": "Calibrated CVM calculator on ground-truth landmarks",
            "output_file": calibrated_gt_path.name,
            "landmarks": "ground_truth",
            "calibrated": True,
            "counts": {f"CS{k}": gt_calib_counter.get(k, 0) for k in range(1, 7)},
            "percentages": {
                f"CS{k}": round(gt_calib_counter.get(k, 0) / total_samples * 100, 2) if total_samples else 0.0
                for k in range(1, 7)
            },
        }

        dist_data["runs"]["run6_pred_calibrated"] = {
            "description": "Calibrated CVM calculator on predicted landmarks",
            "output_file": calibrated_pred_path.name,
            "landmarks": "predicted",
            "calibrated": True,
            "counts": {f"CS{k}": pred_calib_counter.get(k, 0) for k in range(1, 7)},
            "percentages": {
                f"CS{k}": round(pred_calib_counter.get(k, 0) / total_samples * 100, 2) if total_samples else 0.0
                for k in range(1, 7)
            },
        }

        # Update agreement analysis
        if "agreement_analysis" not in dist_data:
            dist_data["agreement_analysis"] = {}

        agr_pred_gt_calib = sum(1 for g, p in zip(calibrated_gt_stages, calibrated_pred_stages) if g == p)
        dist_data["agreement_analysis"]["pred_vs_gt_calibrated"] = {
            "matches": agr_pred_gt_calib,
            "total": total_samples,
            "accuracy_pct": round(agr_pred_gt_calib / total_samples * 100, 2) if total_samples else 0.0,
        }

        # Agreement vs standard on GT
        if existing_csv_rows and "run1_gt_standard" in existing_csv_rows[0]:
            run1_std = [int(r["run1_gt_standard"]) for r in existing_csv_rows]
            agr_calib_std_gt = sum(1 for c, s in zip(calibrated_gt_stages, run1_std) if c == s)
            dist_data["agreement_analysis"]["calibrated_vs_standard_on_gt"] = {
                "matches": agr_calib_std_gt,
                "total": total_samples,
                "agreement_pct": round(agr_calib_std_gt / total_samples * 100, 2) if total_samples else 0.0,
            }

        # Agreement vs standard on Pred
        if existing_csv_rows and "run2_pred_standard" in existing_csv_rows[0]:
            run2_std = [int(r["run2_pred_standard"]) for r in existing_csv_rows]
            agr_calib_std_pred = sum(1 for c, s in zip(calibrated_pred_stages, run2_std) if c == s)
            dist_data["agreement_analysis"]["calibrated_vs_standard_on_pred"] = {
                "matches": agr_calib_std_pred,
                "total": total_samples,
                "agreement_pct": round(agr_calib_std_pred / total_samples * 100, 2) if total_samples else 0.0,
            }

        with open(distribution_json_path, "w", encoding="utf-8") as f:
            json.dump(dist_data, f, indent=2)

        print(f"\n📈 Updated '{distribution_json_path}':")
        print("   - Added 'run5_gt_calibrated' & 'run6_pred_calibrated'")
        print("   - Updated 'metadata.parameters' with calibrated thresholds")
        print("   - Updated 'agreement_analysis' with calibrated metrics")

    # 8. Display Terminal Report
    print("\n" + "=" * 75)
    print("                      CALIBRATED STAGE DISTRIBUTIONS                    ")
    print("=" * 75)
    print(f"{'Stage':<8} | {'GT Calibrated':<18} | {'Pred Calibrated':<18}")
    print("-" * 75)
    for k in range(1, 7):
        c_gt = gt_calib_counter.get(k, 0)
        p_gt = (c_gt / total_samples * 100) if total_samples else 0.0
        c_pr = pred_calib_counter.get(k, 0)
        p_pr = (c_pr / total_samples * 100) if total_samples else 0.0
        print(f"CS{k:<6} | {c_gt:4d} ({p_gt:5.1f}%)         | {c_pr:4d} ({p_pr:5.1f}%)")
    print("-" * 75)
    print(f"Total    | {total_samples:<18d} | {total_samples:<18d}")
    print("=" * 75)

    matches_gt_pred = sum(1 for g, p in zip(calibrated_gt_stages, calibrated_pred_stages) if g == p)
    print(f"\n🎯 Agreement Metrics:")
    print(f"  * Pred Calibrated vs GT Calibrated: {matches_gt_pred}/{total_samples} ({matches_gt_pred/total_samples*100:.2f}%)")

    if existing_csv_rows and "run1_gt_standard" in existing_csv_rows[0]:
        run1_std = [int(r["run1_gt_standard"]) for r in existing_csv_rows]
        matches_pred_vs_gt_std = sum(1 for p, g in zip(calibrated_pred_stages, run1_std) if p == g)
        print(f"  * Pred Calibrated vs GT Standard:   {matches_pred_vs_gt_std}/{total_samples} ({matches_pred_vs_gt_std/total_samples*100:.2f}%)")


if __name__ == "__main__":
    main()
