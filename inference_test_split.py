#!/usr/bin/env python3
"""
CVM Stage Inference and Evaluation on Test Split.

Workflow:
1. Inspects the test dataset directory (default: data/test).
2. Discovers images and corresponding ground truth annotations.
3. Sorts all data items by number ordinal (e.g., 25, 32, ..., 101, ..., 1003).
4. Part 1 (Ground Truth):
   - Evaluates ground truth annotations for each sample.
   - Uses the CVM geometric calculator (src.cvm_calculator) on landmark annotations
     or reads the validated ground-truth stage.
   - Generates 'symbolic_on_gt.txt' with one line per sample containing an integer (1-6).
   - Prints class item counts for Ground Truth to the terminal.
5. Part 2 (Model Inference):
   - Loads the CephalometricSwinGCN neural network with trained weights (model/weights.pth).
   - Predicts 13 cervical vertebrae landmarks for each image.
   - Passes the predicted landmark coordinates to the CVM calculator to determine the stage (1-6).
   - Generates 'symbolic_on_inference.txt' with one line per sample containing an integer (1-6).
   - Prints class item counts for Model Inference to the terminal.
6. Displays a comparative summary table and agreement/accuracy metrics.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Automatically re-exec with local .venv python if torch is missing in current environment
try:
    import torch  # noqa: F401
except ImportError:
    script_dir = Path(__file__).resolve().parent
    venv_python = script_dir / ".venv" / "bin" / "python"
    venv_prefix = script_dir / ".venv"
    if venv_python.exists() and sys.prefix != str(venv_prefix):
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is unavailable
    def tqdm(iterable, desc="", unit="", **kwargs):
        if desc:
            print(f"--> {desc}...")
        return iterable

from src.cvm_calculator import (
    CVMInput,
    CVMThresholds,
    Point,
    VertebraC2,
    VertebraC3C4,
    classify_cvm_stage,
)
from src.inference import CVMPredictor

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CVM ground-truth calculation and model inference on a test split."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Path to test folder containing 'images' and 'labels' (default: data/test).",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="model/weights.pth",
        help="Path to trained model weights .pth file (default: model/weights.pth).",
    )
    parser.add_argument(
        "--gt-output",
        type=str,
        default="symbolic_on_gt.txt",
        help="Output path for ground-truth symbolic stages file (default: symbolic_on_gt.txt).",
    )
    parser.add_argument(
        "--inference-output",
        type=str,
        default="symbolic_on_inference.txt",
        help="Output path for inference symbolic stages file (default: symbolic_on_inference.txt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda", "auto"],
        help="Compute device for neural inference: 'cpu', 'mps', 'cuda', or 'auto' (default: cpu).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Model input image resolution (default: 640).",
    )
    parser.add_argument(
        "--export-cache",
        type=str,
        default="data/export_cache.json",
        help="Optional path to Label Studio export JSON for raw landmark recomputation (default: data/export_cache.json).",
    )
    parser.add_argument(
        "--recompute-gt-from-cache",
        action="store_true",
        help="If set, recomputes GT stages by running cvm_calculator on raw landmark annotations from export_cache.json.",
    )
    return parser.parse_args()


def natural_numeric_sort_key(path: Path) -> Tuple[int, int, List[Any]]:
    """
    Sort key that orders files by number ordinal (e.g., 1, 2, ..., 25, ..., 101, ..., 1003).
    Handles both purely numeric stems and alphanumeric filenames naturally.
    """
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), [])
    # Alphanumeric fallback with natural digit chunking
    parts = re.split(r"(\d+)", stem)
    parsed = [int(p) if p.isdigit() else p.lower() for p in parts]
    return (1, 0, parsed)


def build_cvm_input_from_landmarks(landmarks: Dict[str, Point]) -> CVMInput:
    """Constructs a CVMInput dataclass from a dictionary mapping landmark names to Point objects."""
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
    return CVMInput(c2=c2, c3=c3, c4=c4)


def extract_landmarks_from_task(
    task: Dict[str, Any], fallback_image_path: Optional[Path] = None
) -> Dict[str, Point]:
    """Extracts 13 vertebra landmarks in pixel coordinates from a Label Studio task."""
    annotations = task.get("annotations", [])
    if not annotations:
        raise ValueError(f"Task {task.get('id')} has no annotations.")

    img_w, img_h = None, None
    points_dict: Dict[str, Tuple[float, float]] = {}

    ann = annotations[-1]
    for res in ann.get("result", []):
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

    if (img_w is None or img_h is None) and fallback_image_path and fallback_image_path.exists():
        with Image.open(fallback_image_path) as im:
            img_w, img_h = im.size

    if img_w is None or img_h is None:
        raise ValueError(f"Cannot resolve image dimensions for task {task.get('id')}.")

    missing = set(EXPECTED_LANDMARKS) - set(points_dict.keys())
    if missing:
        raise ValueError(f"Task {task.get('id')} is missing landmarks: {sorted(missing)}")

    return {
        k: Point(x=nx * img_w, y=ny * img_h)
        for k, (nx, ny) in points_dict.items()
    }


def parse_label_content_as_landmarks(text: str, image_size: Optional[Tuple[int, int]] = None) -> Optional[CVMInput]:
    """
    Attempts to parse text as landmark coordinates (JSON or structured keypoint lines).
    Returns CVMInput if successful, otherwise None.
    """
    text_clean = text.strip()
    if not text_clean:
        return None

    # 1. Try JSON parsing
    if text_clean.startswith("{") and text_clean.endswith("}"):
        try:
            data = json.loads(text_clean)
            if "C2" in data and "C3" in data and "C4" in data:
                return CVMInput.from_dict(data)
            # Dictionary of landmark_name -> [x, y]
            if all(k in data for k in EXPECTED_LANDMARKS):
                pts = {k: Point.from_dict(data[k]) for k in EXPECTED_LANDMARKS}
                return build_cvm_input_from_landmarks(pts)
        except Exception:
            pass

    # 2. Try parsing 13 lines of coordinates (x y or x, y)
    lines = [ln.strip() for ln in text_clean.splitlines() if ln.strip()]
    if len(lines) == 13:
        try:
            pts_dict: Dict[str, Point] = {}
            for k, line in zip(EXPECTED_LANDMARKS, lines):
                parts = re.split(r"[\s,]+", line)
                if len(parts) >= 2:
                    pts_dict[k] = Point(x=float(parts[0]), y=float(parts[1]))
            if len(pts_dict) == 13:
                return build_cvm_input_from_landmarks(pts_dict)
        except Exception:
            pass

    return None


def get_ground_truth_stage(
    label_path: Optional[Path],
    image_path: Optional[Path],
    task: Optional[Dict[str, Any]],
    recompute_from_cache: bool,
    thresholds: CVMThresholds,
) -> int:
    """
    Resolves the ground truth stage for a given sample:
    1. If recompute_from_cache and task exists with landmarks, computes via cvm_calculator.
    2. If label file contains landmark coordinates, computes via cvm_calculator.
    3. If label file contains an integer (1-6) or stage string (CS1-CS6), returns it.
    4. If label file is missing/empty but task exists, computes via cvm_calculator.
    """
    # Option A: Explicit recomputation from export cache landmarks
    if recompute_from_cache and task:
        landmarks = extract_landmarks_from_task(task, fallback_image_path=image_path)
        cvm_input = build_cvm_input_from_landmarks(landmarks)
        res = classify_cvm_stage(cvm_input, thresholds=thresholds)
        return int(res["stage"].replace("CS", ""))

    # Option B: Read from label file
    if label_path and label_path.is_file():
        content = label_path.read_text(encoding="utf-8").strip()

        # Check if content is already a stage identifier (e.g., '1', '2', ..., '6' or 'CS1'...'CS6')
        stage_match = re.match(r"^(?:CS)?([1-6])$", content, re.IGNORECASE)
        if stage_match:
            return int(stage_match.group(1))

        # Check if content has landmark coordinate data
        cvm_input = parse_label_content_as_landmarks(content)
        if cvm_input is not None:
            res = classify_cvm_stage(cvm_input, thresholds=thresholds)
            return int(res["stage"].replace("CS", ""))

    # Option C: Fallback to task annotation in export cache
    if task:
        landmarks = extract_landmarks_from_task(task, fallback_image_path=image_path)
        cvm_input = build_cvm_input_from_landmarks(landmarks)
        res = classify_cvm_stage(cvm_input, thresholds=thresholds)
        return int(res["stage"].replace("CS", ""))

    raise ValueError(f"Could not determine ground truth stage for {image_path.name if image_path else label_path}")


def print_class_counts(counts: Counter, title: str):
    """Prints a formatted distribution table of CVM stages (CS1-CS6) to the terminal."""
    total = sum(counts.values())
    print("\n" + "=" * 55)
    print(f" {title:^53} ")
    print("=" * 55)
    for stage_num in range(1, 7):
        stage_name = f"CS{stage_num}"
        cnt = counts.get(stage_num, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        bar = "█" * int(round(pct / 2.5))
        print(f"  {stage_name} (Stage {stage_num}): {cnt:4d} items ({pct:5.1f}%)  {bar}")
    print("-" * 55)
    print(f"  Total:        {total:4d} items")
    print("=" * 55)


def main():
    args = parse_args()
    test_dir = Path(args.test_dir).resolve()

    if not test_dir.is_dir():
        print(f"❌ Error: Test directory does not exist: {test_dir}", file=sys.stderr)
        sys.exit(1)

    images_dir = test_dir / "images" if (test_dir / "images").is_dir() else test_dir
    labels_dir = test_dir / "labels" if (test_dir / "labels").is_dir() else test_dir

    print("=" * 65)
    print("        CVM STAGE INFERENCE AND EVALUATION (TEST SPLIT)        ")
    print("=" * 65)
    print(f"Test Directory:      {test_dir}")
    print(f"Images Source:       {images_dir}")
    print(f"Labels Source:       {labels_dir}")
    print(f"Model Weights:       {args.weights_path}")
    print(f"GT Output File:      {args.gt_output}")
    print(f"Inference Output:    {args.inference_output}")
    print(f"Inference Device:    {args.device}")
    print("=" * 65)

    # 1. Discover all test images
    all_image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]

    if not all_image_paths:
        print(f"❌ Error: No image files found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    # 2. Sort images by number ordinal (e.g. 25, 32, ..., 101, ..., 1003)
    all_image_paths.sort(key=natural_numeric_sort_key)
    print(f"\n📁 Discovered {len(all_image_paths)} images in test split.")
    print("🔢 Sorted in numeric ordinal order (e.g. 25.jpg, 32.jpg, ...)")

    # 3. Load export cache if available (for raw landmark mapping or fallback)
    tasks_by_stem: Dict[str, Dict[str, Any]] = {}
    export_cache_path = Path(args.export_cache)
    if export_cache_path.is_file():
        try:
            with open(export_cache_path, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)
            for t in tasks_data:
                tid = str(t.get("id"))
                tasks_by_stem[tid] = t
            print(f"📦 Loaded {len(tasks_by_stem)} tasks from export cache: {export_cache_path.name}")
        except Exception as e:
            print(f"⚠️ Could not load export cache ({e}), proceeding with files only.")

    thresholds = CVMThresholds()

    # -------------------------------------------------------------
    # PART 1: GROUND TRUTH EVALUATION & SYMBOLIC GENERATION
    # -------------------------------------------------------------
    print(f"\n[Part 1/2] Processing Ground Truth Annotations on {len(all_image_paths)} items...")
    gt_stages: List[int] = []
    gt_counts: Counter = Counter()
    gt_errors: List[Tuple[str, str]] = []

    for img_path in tqdm(all_image_paths, desc="Ground Truth", unit="item"):
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        task = tasks_by_stem.get(stem)

        try:
            stage_int = get_ground_truth_stage(
                label_path=lbl_path if lbl_path.is_file() else None,
                image_path=img_path,
                task=task,
                recompute_from_cache=args.recompute_gt_from_cache,
                thresholds=thresholds,
            )
            gt_stages.append(stage_int)
            gt_counts[stage_int] += 1
        except Exception as err:
            gt_errors.append((img_path.name, str(err)))
            gt_stages.append(-1)

    if gt_errors:
        print(f"\n⚠️ Encountered {len(gt_errors)} errors during ground-truth evaluation:")
        for fn, msg in gt_errors[:5]:
            print(f"   - {fn}: {msg}")

    # Write symbolic_on_gt.txt
    gt_output_path = Path(args.gt_output)
    gt_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_output_path, "w", encoding="utf-8") as f:
        for stage in gt_stages:
            f.write(f"{stage}\n")

    print(f"\n💾 Saved Ground Truth stages to: {gt_output_path.resolve()}")
    print_class_counts(gt_counts, "GROUND TRUTH CLASS ITEM COUNTS")

    # -------------------------------------------------------------
    # PART 2: MODEL INFERENCE & SYMBOLIC GENERATION
    # -------------------------------------------------------------
    print(f"\n[Part 2/2] Running Model Inference & CVM Calculation on {len(all_image_paths)} items...")
    print(f"🧠 Loading CephalometricSwinGCN from: {args.weights_path}")
    predictor = CVMPredictor(
        weights_path=args.weights_path,
        device=args.device,
        img_size=args.img_size,
    )
    print(f"🚀 Model initialized successfully on device: {predictor.device}")

    inf_stages: List[int] = []
    inf_counts: Counter = Counter()
    inf_errors: List[Tuple[str, str]] = []

    for img_path in tqdm(all_image_paths, desc="Model Inference", unit="item"):
        try:
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")

            # 1. Neural landmark prediction
            landmarks_dict, _, _ = predictor.predict_landmarks(rgb_img)

            # 2. Geometric CVM stage calculation
            cvm_input = CVMInput.from_dict(landmarks_dict)
            classification = classify_cvm_stage(cvm_input, thresholds=thresholds)

            stage_str = classification["stage"]  # "CS1" ... "CS6"
            stage_int = int(stage_str.replace("CS", ""))

            inf_stages.append(stage_int)
            inf_counts[stage_int] += 1
        except Exception as err:
            inf_errors.append((img_path.name, str(err)))
            inf_stages.append(-1)

    if inf_errors:
        print(f"\n⚠️ Encountered {len(inf_errors)} errors during model inference:")
        for fn, msg in inf_errors[:5]:
            print(f"   - {fn}: {msg}")

    # Write symbolic_on_inference.txt
    inf_output_path = Path(args.inference_output)
    inf_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inf_output_path, "w", encoding="utf-8") as f:
        for stage in inf_stages:
            f.write(f"{stage}\n")

    print(f"\n💾 Saved Model Inference stages to: {inf_output_path.resolve()}")
    print_class_counts(inf_counts, "MODEL INFERENCE CLASS ITEM COUNTS")

    # -------------------------------------------------------------
    # FINAL COMPARISON & ACCURACY SUMMARY
    # -------------------------------------------------------------
    valid_comparisons = [
        (g, p) for g, p in zip(gt_stages, inf_stages) if g != -1 and p != -1
    ]
    if valid_comparisons:
        matches = sum(1 for g, p in valid_comparisons if g == p)
        total_eval = len(valid_comparisons)
        accuracy = (matches / total_eval) * 100.0

        print("\n" + "=" * 65)
        print("                  EVALUATION COMPARISON SUMMARY                 ")
        print("=" * 65)
        print(f"Total Evaluated Samples:     {total_eval} / {len(all_image_paths)}")
        print(f"Exact Agreement (Matches):   {matches} / {total_eval} ({accuracy:.2f}%)")
        print(f"Discrepancies (Mismatches):  {total_eval - matches} / {total_eval} ({100.0 - accuracy:.2f}%)")
        print("-" * 65)
        print(f"{'Stage':<10} | {'Ground Truth':<15} | {'Inference':<15} | {'Delta':<10}")
        print("-" * 65)
        for stage_num in range(1, 7):
            gt_n = gt_counts.get(stage_num, 0)
            inf_n = inf_counts.get(stage_num, 0)
            delta = inf_n - gt_n
            delta_str = f"+{delta}" if delta > 0 else f"{delta}"
            print(f"CS{stage_num:<8} | {gt_n:<15} | {inf_n:<15} | {delta_str:<10}")
        print("=" * 65)

    print("\nPreview of first 10 items (Ordinal Order):")
    print(f"{'Index':<6} | {'Filename':<15} | {'GT Stage':<10} | {'Inference Stage':<15} | {'Match'}")
    print("-" * 65)
    for idx, (img_path, g, p) in enumerate(zip(all_image_paths[:10], gt_stages[:10], inf_stages[:10]), start=1):
        match_str = "✅" if g == p else "❌"
        print(f"{idx:<6} | {img_path.name:<15} | {g:<10} | {p:<15} | {match_str}")
    print("=" * 65)
    print("🎉 Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
