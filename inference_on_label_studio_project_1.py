#!/usr/bin/env python3
"""
Inference & Multi-Run CVM Classification Script on Label Studio Project 1 Dataset.

Workflow:
1. Reads full dataset images and Label Studio annotations (ground-truth landmarks).
2. Sorts images by number ordinal (1, 2, 3, ..., 10, ...).
3. Executes neural network inference (CephalometricSwinGCN) on the full dataset to produce predicted landmarks.
4. Executes the CVM geometric calculator across 4 runs:
   - Run 1: Standard CVM calculator on ground-truth landmarks.
   - Run 2: Standard CVM calculator on predicted landmarks.
   - Run 3: Fuzzy-hysteresis CVM calculator (--enable-fuzzy-hysteresis) on ground-truth landmarks.
   - Run 4: Fuzzy-hysteresis CVM calculator (--enable-fuzzy-hysteresis) on predicted landmarks.
5. Stores output data for all runs:
   - A single .txt file for each run containing the stage of each file in ordinal order.
   - Overall distribution of each of the 6 classes across all 4 runs (JSON & CSV summary).
   - Side-by-side comparative terminal report with agreement matrices.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Automatically re-exec with local .venv python if torch is missing in current environment
try:
    import torch  # noqa: F401
except ImportError:
    script_dir = Path(__file__).resolve().parent
    venv_python = script_dir / ".venv" / "bin" / "python"
    venv_prefix = script_dir / ".venv"
    if venv_python.exists() and sys.prefix != str(venv_prefix):
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit="", **kwargs):
        if desc:
            print(f"--> {desc}...")
        return iterable

# Import CVM geometric calculation engine & Model Predictor
from src.cvm_calculator import (
    CVMInput,
    CVMThresholds,
    Point,
    VertebraC2,
    VertebraC3C4,
    classify_cvm_stage,
)
from src.inference import CVMPredictor, LANDMARK_LABELS

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

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CVM neural landmark inference and 4-run CVM geometric calculations on Label Studio Project 1."
    )
    # Model & Device options
    parser.add_argument(
        "--weights-path",
        type=str,
        default=os.getenv("CVM_WEIGHTS_PATH", "model/weights.pth"),
        help="Path to trained model weights .pth file (default: model/weights.pth or $CVM_WEIGHTS_PATH).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for neural inference: 'auto', 'cuda', 'mps', or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image resolution for SwinGCN model (default: 640).",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="Delay in milliseconds between frames to allow thermal cooling on fanless devices (default: 0).",
    )

    # Dataset & Annotations paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/full",
        help="Root directory for full data containing 'images' and optional 'labels' (default: data/full).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Custom path to images directory (defaults to {data-dir}/images).",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Custom path to existing labels directory (defaults to {data-dir}/labels).",
    )
    parser.add_argument(
        "--export-cache",
        type=str,
        default="data/export_cache.json",
        help="Path to Label Studio export JSON file (default: data/export_cache.json).",
    )
    parser.add_argument(
        "--fetch-latest",
        action="store_true",
        help="Re-fetch export from Label Studio if API credentials are in .env.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio Project ID (defaults to LABEL_STUDIO_PROJECT_ID in .env or 1).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env configuration file (default: .env).",
    )

    # Sorting & Formatting options
    parser.add_argument(
        "--sort-order",
        type=str,
        choices=["ordinal", "natural", "alphabetical"],
        default="ordinal",
        help="Sorting order for files: 'ordinal' / 'natural' (1.jpg, 2.jpg, ... 10.jpg) or 'alphabetical' (default: ordinal).",
    )
    parser.add_argument(
        "--stage-format",
        type=str,
        choices=["int", "cs"],
        default="int",
        help="Format of stage per line in text files: 'int' (1-6) or 'cs' (CS1-CS6) (default: int).",
    )

    # Output paths for the 4 runs
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory where all run result files will be stored (default: current directory).",
    )
    parser.add_argument(
        "--run1-output",
        type=str,
        default="run1_gt_standard.txt",
        help="Filename for Run 1: Standard CVM on ground-truth landmarks (default: run1_gt_standard.txt).",
    )
    parser.add_argument(
        "--run2-output",
        type=str,
        default="run2_pred_standard.txt",
        help="Filename for Run 2: Standard CVM on predicted landmarks (default: run2_pred_standard.txt).",
    )
    parser.add_argument(
        "--run3-output",
        type=str,
        default="run3_gt_fuzzy.txt",
        help="Filename for Run 3: Fuzzy-hysteresis CVM on ground-truth landmarks (default: run3_gt_fuzzy.txt).",
    )
    parser.add_argument(
        "--run4-output",
        type=str,
        default="run4_pred_fuzzy.txt",
        help="Filename for Run 4: Fuzzy-hysteresis CVM on predicted landmarks (default: run4_pred_fuzzy.txt).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="runs_distribution.json",
        help="Filename for storing overall distribution of classes across all 4 runs (default: runs_distribution.json).",
    )
    parser.add_argument(
        "--save-mapping",
        type=str,
        default="runs_comparison.csv",
        help="Filename for saving detailed CSV table comparing all 4 runs per image (default: runs_comparison.csv).",
    )
    parser.add_argument(
        "--save-predicted-landmarks",
        type=str,
        default=None,
        help="Optional path to save all predicted landmarks to a JSON file.",
    )

    # CVM Calculator geometric parameters
    parser.add_argument(
        "--concavity-hysteresis-mm",
        type=float,
        default=0.15,
        help="Concavity hysteresis buffer (+/- mm) around threshold (default: 0.15).",
    )
    parser.add_argument(
        "--shape-fuzzy-margin",
        type=float,
        default=0.03,
        help="Shape ratio fuzzy transition margin (default: 0.03).",
    )
    parser.add_argument(
        "--strict-biological-hierarchy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce biological monotonicity for concavity notches (default: True).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to process (useful for quick testing/debugging).",
    )

    return parser.parse_args()


def ordinal_sort_key(path: Path) -> Tuple[int, int, List[Any]]:
    """
    Ordinal sort key (1, 2, 3, 4, ..., 10, ...).
    Pure numeric stems are sorted by integer value.
    Mixed alphanumeric stems are sorted naturally by split digit blocks.
    """
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), [])
    parts = re.split(r"(\d+)", stem)
    parsed = [int(p) if p.isdigit() else p.lower() for p in parts]
    return (1, 0, parsed)


def load_label_studio_export(
    export_path: str,
    fetch_latest: bool = False,
    env_file: str = ".env",
    project_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Loads export from local cache or fetches latest from Label Studio API."""
    if fetch_latest:
        from dotenv import load_dotenv
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        if os.path.exists(env_file):
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()

        ls_url = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
        api_token = os.getenv("LABEL_STUDIO_API_TOKEN") or ""
        proj_id = project_id or int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

        if ls_url and api_token:
            print(f"🌐 Fetching latest export for Project {proj_id} from {ls_url}...")
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5))
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update({"Authorization": f"Token {api_token}"})
            url = f"{ls_url}/api/projects/{proj_id}/export?exportType=JSON"
            resp = session.get(url, timeout=180)
            if resp.status_code == 200:
                tasks = resp.json()
                print(f"✅ Downloaded {len(tasks)} tasks from Label Studio.")
                os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, indent=2)
                return tasks
            else:
                print(f"⚠️ Failed to fetch from Label Studio ({resp.status_code}): {resp.text[:200]}")
                print(f"Falling back to local cache: {export_path}")

    if not os.path.exists(export_path):
        raise FileNotFoundError(f"Export cache file not found: {export_path}")

    print(f"📦 Loading Label Studio export from: {export_path}")
    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"✅ Loaded {len(tasks)} tasks from export cache.")
    return tasks


def build_task_indices(tasks: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Indexes tasks by ID and by image file name from task data URL."""
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


def extract_landmarks(
    task: Dict[str, Any],
    fallback_image_path: Optional[Path] = None,
) -> Dict[str, Point]:
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

    if (img_w is None or img_h is None) and fallback_image_path and fallback_image_path.exists():
        with Image.open(fallback_image_path) as im:
            img_w, img_h = im.size

    if img_w is None or img_h is None:
        raise ValueError(f"Could not resolve dimensions for task {task.get('id')}.")

    missing = set(EXPECTED_LANDMARKS) - set(points_dict.keys())
    if missing:
        raise ValueError(f"Task {task.get('id')} missing landmarks: {sorted(missing)}")

    pixel_points: Dict[str, Point] = {}
    for k, (nx, ny) in points_dict.items():
        pixel_points[k] = Point(x=nx * img_w, y=ny * img_h)

    return pixel_points


def compute_stage_from_landmarks(
    landmarks: Dict[str, Point],
    thresholds: Optional[CVMThresholds] = None,
) -> Tuple[int, str]:
    """Invokes geometric CVM calculator on the 13 landmarks and returns (stage_int, stage_str)."""
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
    if thresholds is None:
        thresholds = CVMThresholds()
    result = classify_cvm_stage(cvm_input, thresholds=thresholds)

    stage_str = result["stage"]  # "CS1" ... "CS6"
    stage_int = int(stage_str.replace("CS", ""))
    return stage_int, stage_str


def resolve_device(device_arg: str) -> str:
    """Selects best available device based on user preference and hardware."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def main():
    args = parse_args()

    # 1. Resolve image and label directory paths
    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir) if args.images_dir else data_dir / "images"
    labels_dir = Path(args.labels_dir) if args.labels_dir else data_dir / "labels"

    if not images_dir.is_dir():
        print(f"❌ Error: Images directory not found at: {images_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("      CVM MULTI-RUN EVALUATION ON FULL DATASET (LABEL STUDIO PROJECT 1)     ")
    print("=" * 80)
    print(f"📁 Images Source:     {images_dir}")
    if labels_dir.is_dir():
        print(f"📁 Reference Labels:  {labels_dir}")
    else:
        print(f"ℹ️ Reference labels dir not found at {labels_dir} (proceeding).")

    # 2. Gather image files
    all_image_paths = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]

    if not all_image_paths:
        print(f"❌ Error: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    # 3. Sort files according to ordinal sort order (1 2 3 4 ...)
    if args.sort_order in ["ordinal", "natural"]:
        all_image_paths.sort(key=ordinal_sort_key)
        sort_desc = "Ordinal numerical order (1.jpg, 2.jpg, ... 10.jpg, 100.jpg)"
    else:
        all_image_paths.sort(key=lambda p: p.name)
        sort_desc = "Alphabetical order (1.jpg, 10.jpg, 100.jpg, 2.jpg, ...)"

    if args.limit and args.limit > 0:
        all_image_paths = all_image_paths[:args.limit]
        print(f"⚠️ Limited run: processing first {len(all_image_paths)} images.")

    print(f"📋 Total dataset images: {len(all_image_paths)}")
    print(f"🔀 Sorting applied:      {sort_desc}")

    # 4. Load Label Studio Export (Ground-Truth Annotations)
    tasks = load_label_studio_export(
        export_path=args.export_cache,
        fetch_latest=args.fetch_latest,
        env_file=args.env_file,
        project_id=args.project_id,
    )
    tasks_by_id, tasks_by_filename = build_task_indices(tasks)

    # 5. Initialize Neural Network Predictor for Landmarks
    device = resolve_device(args.device)
    print(f"\n🧠 Initializing Neural Network (CephalometricSwinGCN)...")
    print(f"   Model Weights: {args.weights_path}")
    print(f"   Target Device: {device} (requested: {args.device})")
    if device == "cuda":
        print(f"   CUDA GPU:      {torch.cuda.get_device_name(0)}")

    predictor = CVMPredictor(
        weights_path=args.weights_path,
        device=device,
        img_size=args.img_size,
    )

    # Prepare CVM Threshold Configurations for Standard & Fuzzy-Hysteresis
    thresholds_standard = CVMThresholds(
        enable_fuzzy_hysteresis=False,
        concavity_hysteresis_mm=args.concavity_hysteresis_mm,
        shape_fuzzy_margin=args.shape_fuzzy_margin,
        strict_biological_hierarchy=args.strict_biological_hierarchy,
    )
    thresholds_fuzzy = CVMThresholds(
        enable_fuzzy_hysteresis=True,
        concavity_hysteresis_mm=args.concavity_hysteresis_mm,
        shape_fuzzy_margin=args.shape_fuzzy_margin,
        strict_biological_hierarchy=args.strict_biological_hierarchy,
    )

    # 6. Step 1: Run Model Landmark Inference & Ground-Truth Extraction
    print(f"\n🚀 Phase 1: Running Model Inference & Extracting Ground-Truth Landmarks...")
    samples_data: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []
    predicted_landmarks_export: Dict[str, Any] = {}

    for idx, img_path in enumerate(tqdm(all_image_paths, desc="Inference & Extraction", unit="img"), start=1):
        fn = img_path.name
        stem = img_path.stem

        # Match task from export cache
        task = tasks_by_id.get(stem) or tasks_by_filename.get(fn) or tasks_by_filename.get(stem)
        if not task:
            errors.append((fn, "No matching task in export cache"))
            continue

        # Ground-truth reference label file (if exists in data/full/labels)
        gt_file_label: Optional[int] = None
        if labels_dir.is_dir():
            lbl_file = labels_dir / f"{stem}.txt"
            if lbl_file.is_file():
                try:
                    txt = lbl_file.read_text(encoding="utf-8").strip()
                    if txt.isdigit():
                        gt_file_label = int(txt)
                except Exception:
                    pass

        try:
            # 1. Extract ground-truth landmarks from annotations
            gt_landmarks = extract_landmarks(task, fallback_image_path=img_path)

            # 2. Predict landmarks using CephalometricSwinGCN neural model
            with Image.open(img_path) as pil_img:
                _, landmarks_px, _ = predictor.predict_landmarks(pil_img)

            # Convert predicted landmarks list to Dict[str, Point]
            pred_landmarks: Dict[str, Point] = {
                name: Point(x=x, y=y)
                for name, (x, y) in zip(LANDMARK_LABELS, landmarks_px)
            }

            if args.save_predicted_landmarks:
                predicted_landmarks_export[fn] = {
                    name: [p.x, p.y] for name, p in pred_landmarks.items()
                }

            samples_data.append({
                "ordinal_index": idx,
                "filename": fn,
                "stem": stem,
                "task_id": task.get("id"),
                "gt_landmarks": gt_landmarks,
                "pred_landmarks": pred_landmarks,
                "gt_file_label": gt_file_label,
            })

        except Exception as err:
            errors.append((fn, str(err)))

        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)

    if errors:
        print(f"⚠️ Encountered {len(errors)} issues during Phase 1:")
        for fn_err, msg in errors[:5]:
            print(f"   - {fn_err}: {msg}")

    print(f"✅ Successfully processed {len(samples_data)} images for CVM stage calculation.")

    # 7. Step 2: Execute CVM Geometric Calculator Across 4 Runs
    print(f"\n⚙️ Phase 2: Running CVM Geometric Calculator Across 4 Runs...")
    print("   Run 1: Standard CVM calculator on ground-truth landmarks")
    print("   Run 2: Standard CVM calculator on predicted landmarks")
    print("   Run 3: Fuzzy-hysteresis CVM calculator (--enable-fuzzy-hysteresis) on ground-truth landmarks")
    print("   Run 4: Fuzzy-hysteresis CVM calculator (--enable-fuzzy-hysteresis) on predicted landmarks")

    run1_stages: List[int] = []
    run2_stages: List[int] = []
    run3_stages: List[int] = []
    run4_stages: List[int] = []

    records: List[Dict[str, Any]] = []

    for item in samples_data:
        gt_pts = item["gt_landmarks"]
        pred_pts = item["pred_landmarks"]

        # Run 1: Standard on Ground-Truth
        r1_int, r1_str = compute_stage_from_landmarks(gt_pts, thresholds=thresholds_standard)
        # Run 2: Standard on Predicted
        r2_int, r2_str = compute_stage_from_landmarks(pred_pts, thresholds=thresholds_standard)
        # Run 3: Fuzzy-Hysteresis on Ground-Truth
        r3_int, r3_str = compute_stage_from_landmarks(gt_pts, thresholds=thresholds_fuzzy)
        # Run 4: Fuzzy-Hysteresis on Predicted
        r4_int, r4_str = compute_stage_from_landmarks(pred_pts, thresholds=thresholds_fuzzy)

        run1_stages.append(r1_int)
        run2_stages.append(r2_int)
        run3_stages.append(r3_int)
        run4_stages.append(r4_int)

        records.append({
            "ordinal_index": item["ordinal_index"],
            "filename": item["filename"],
            "stem": item["stem"],
            "task_id": item["task_id"],
            "run1_gt_standard": r1_int,
            "run2_pred_standard": r2_int,
            "run3_gt_fuzzy": r3_int,
            "run4_pred_fuzzy": r4_int,
            "gt_file_label": item["gt_file_label"],
        })

    # 8. Store Single .txt Files for Each Run (strictly ordered by ordinal sort)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_files_info = [
        ("Run 1 (GT Standard)", output_dir / args.run1_output, run1_stages),
        ("Run 2 (Pred Standard)", output_dir / args.run2_output, run2_stages),
        ("Run 3 (GT Fuzzy)", output_dir / args.run3_output, run3_stages),
        ("Run 4 (Pred Fuzzy)", output_dir / args.run4_pred_fuzzy if hasattr(args, "run4_pred_fuzzy") else output_dir / args.run4_output, run4_stages),
    ]

    print(f"\n💾 Saving single .txt files for each run (Format: '{args.stage_format}', Ordinal Sort)...")
    for name, path, stages in run_files_info:
        with open(path, "w", encoding="utf-8") as f:
            for s in stages:
                if args.stage_format == "cs":
                    f.write(f"CS{s}\n")
                else:
                    f.write(f"{s}\n")
        print(f"   ✅ {name:22s} -> {path.resolve()} ({len(stages)} lines)")

    # 9. Compute Overall Distributions for each of the 6 classes across all 4 runs
    total_count = len(records)
    runs_counters = {
        "run1_gt_standard": Counter(run1_stages),
        "run2_pred_standard": Counter(run2_stages),
        "run3_gt_fuzzy": Counter(run3_stages),
        "run4_pred_fuzzy": Counter(run4_stages),
    }

    distribution_data: Dict[str, Any] = {
        "metadata": {
            "total_images": total_count,
            "sort_order": args.sort_order,
            "stage_format": args.stage_format,
            "device": device,
            "weights_path": str(Path(args.weights_path).resolve()),
            "parameters": {
                "concavity_hysteresis_mm": args.concavity_hysteresis_mm,
                "shape_fuzzy_margin": args.shape_fuzzy_margin,
                "strict_biological_hierarchy": args.strict_biological_hierarchy,
            },
        },
        "runs": {
            "run1_gt_standard": {
                "description": "Standard CVM calculator on ground-truth landmarks",
                "output_file": str(Path(args.run1_output).name),
                "landmarks": "ground_truth",
                "fuzzy_hysteresis": False,
                "counts": {f"CS{k}": runs_counters["run1_gt_standard"].get(k, 0) for k in range(1, 7)},
                "percentages": {
                    f"CS{k}": round(runs_counters["run1_gt_standard"].get(k, 0) / total_count * 100, 2)
                    if total_count else 0.0
                    for k in range(1, 7)
                },
            },
            "run2_pred_standard": {
                "description": "Standard CVM calculator on predicted landmarks",
                "output_file": str(Path(args.run2_output).name),
                "landmarks": "predicted",
                "fuzzy_hysteresis": False,
                "counts": {f"CS{k}": runs_counters["run2_pred_standard"].get(k, 0) for k in range(1, 7)},
                "percentages": {
                    f"CS{k}": round(runs_counters["run2_pred_standard"].get(k, 0) / total_count * 100, 2)
                    if total_count else 0.0
                    for k in range(1, 7)
                },
            },
            "run3_gt_fuzzy": {
                "description": "Fuzzy-hysteresis CVM calculator on ground-truth landmarks",
                "output_file": str(Path(args.run3_output).name),
                "landmarks": "ground_truth",
                "fuzzy_hysteresis": True,
                "counts": {f"CS{k}": runs_counters["run3_gt_fuzzy"].get(k, 0) for k in range(1, 7)},
                "percentages": {
                    f"CS{k}": round(runs_counters["run3_gt_fuzzy"].get(k, 0) / total_count * 100, 2)
                    if total_count else 0.0
                    for k in range(1, 7)
                },
            },
            "run4_pred_fuzzy": {
                "description": "Fuzzy-hysteresis CVM calculator on predicted landmarks",
                "output_file": str(Path(args.run4_output).name),
                "landmarks": "predicted",
                "fuzzy_hysteresis": True,
                "counts": {f"CS{k}": runs_counters["run4_pred_fuzzy"].get(k, 0) for k in range(1, 7)},
                "percentages": {
                    f"CS{k}": round(runs_counters["run4_pred_fuzzy"].get(k, 0) / total_count * 100, 2)
                    if total_count else 0.0
                    for k in range(1, 7)
                },
            },
        },
    }

    # Calculate agreement metrics between runs
    if total_count > 0:
        agr_2_vs_1 = sum(1 for r in records if r["run2_pred_standard"] == r["run1_gt_standard"])
        agr_4_vs_3 = sum(1 for r in records if r["run4_pred_fuzzy"] == r["run3_gt_fuzzy"])
        agr_3_vs_1 = sum(1 for r in records if r["run3_gt_fuzzy"] == r["run1_gt_standard"])
        agr_4_vs_2 = sum(1 for r in records if r["run4_pred_fuzzy"] == r["run2_pred_standard"])

        distribution_data["agreement_analysis"] = {
            "pred_vs_gt_standard": {
                "matches": agr_2_vs_1,
                "total": total_count,
                "accuracy_pct": round(agr_2_vs_1 / total_count * 100, 2),
            },
            "pred_vs_gt_fuzzy": {
                "matches": agr_4_vs_3,
                "total": total_count,
                "accuracy_pct": round(agr_4_vs_3 / total_count * 100, 2),
            },
            "fuzzy_vs_standard_on_gt": {
                "matches": agr_3_vs_1,
                "total": total_count,
                "agreement_pct": round(agr_3_vs_1 / total_count * 100, 2),
            },
            "fuzzy_vs_standard_on_pred": {
                "matches": agr_4_vs_2,
                "total": total_count,
                "agreement_pct": round(agr_4_vs_2 / total_count * 100, 2),
            },
        }

    # Save summary JSON file
    summary_path = output_dir / args.summary_json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(distribution_data, f, indent=2)
    print(f"📊 Overall distribution JSON saved to: {summary_path.resolve()}")

    # Save detailed CSV comparison file
    if args.save_mapping:
        mapping_path = output_dir / args.save_mapping
        with open(mapping_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ordinal_index",
                "filename",
                "task_id",
                "run1_gt_standard",
                "run2_pred_standard",
                "run3_gt_fuzzy",
                "run4_pred_fuzzy",
                "reference_label",
            ])
            for r in records:
                writer.writerow([
                    r["ordinal_index"],
                    r["filename"],
                    r["task_id"],
                    r["run1_gt_standard"],
                    r["run2_pred_standard"],
                    r["run3_gt_fuzzy"],
                    r["run4_pred_fuzzy"],
                    r["gt_file_label"] if r["gt_file_label"] is not None else "",
                ])
        print(f"📊 Detailed run comparison CSV saved to: {mapping_path.resolve()}")

    # Optional: Save predicted landmarks JSON
    if args.save_predicted_landmarks:
        pl_path = output_dir / args.save_predicted_landmarks
        with open(pl_path, "w", encoding="utf-8") as f:
            json.dump(predicted_landmarks_export, f, indent=2)
        print(f"💾 Predicted landmarks saved to: {pl_path.resolve()}")

    # 10. Display Formatted Distribution Summary Table in Terminal
    print("\n" + "=" * 90)
    print("                    OVERALL CVM STAGE DISTRIBUTION SUMMARY (6 CLASSES)                   ")
    print("=" * 90)
    print(
        f"{'Stage':<10} | "
        f"{'Run 1: GT (Std)':<18} | "
        f"{'Run 2: Pred (Std)':<18} | "
        f"{'Run 3: GT (Fuzzy)':<18} | "
        f"{'Run 4: Pred (Fuzzy)':<18}"
    )
    print("-" * 90)

    for k in range(1, 7):
        stage_lbl = f"CS{k} ({k})"
        c1 = runs_counters["run1_gt_standard"].get(k, 0)
        p1 = (c1 / total_count * 100) if total_count else 0.0

        c2 = runs_counters["run2_pred_standard"].get(k, 0)
        p2 = (c2 / total_count * 100) if total_count else 0.0

        c3 = runs_counters["run3_gt_fuzzy"].get(k, 0)
        p3 = (c3 / total_count * 100) if total_count else 0.0

        c4 = runs_counters["run4_pred_fuzzy"].get(k, 0)
        p4 = (c4 / total_count * 100) if total_count else 0.0

        print(
            f"{stage_lbl:<10} | "
            f"{c1:4d} ({p1:5.1f}%)       | "
            f"{c2:4d} ({p2:5.1f}%)       | "
            f"{c3:4d} ({p3:5.1f}%)       | "
            f"{c4:4d} ({p4:5.1f}%)"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<10} | "
        f"{total_count:4d} (100.0%)      | "
        f"{total_count:4d} (100.0%)      | "
        f"{total_count:4d} (100.0%)      | "
        f"{total_count:4d} (100.0%)"
    )
    print("=" * 90)

    if total_count > 0 and "agreement_analysis" in distribution_data:
        aa = distribution_data["agreement_analysis"]
        print("\n📈 Agreement & Concordance Metrics:")
        print(
            f"   • Model Accuracy (Standard)  [Run 2 vs Run 1]: {aa['pred_vs_gt_standard']['matches']}/{total_count} "
            f"({aa['pred_vs_gt_standard']['accuracy_pct']}%)"
        )
        print(
            f"   • Model Accuracy (Fuzzy)     [Run 4 vs Run 3]: {aa['pred_vs_gt_fuzzy']['matches']}/{total_count} "
            f"({aa['pred_vs_gt_fuzzy']['accuracy_pct']}%)"
        )
        print(
            f"   • Fuzzy vs Standard on GT    [Run 3 vs Run 1]: {aa['fuzzy_vs_standard_on_gt']['matches']}/{total_count} "
            f"({aa['fuzzy_vs_standard_on_gt']['agreement_pct']}%)"
        )
        print(
            f"   • Fuzzy vs Standard on Pred  [Run 4 vs Run 2]: {aa['fuzzy_vs_standard_on_pred']['matches']}/{total_count} "
            f"({aa['fuzzy_vs_standard_on_pred']['agreement_pct']}%)"
        )

    print("\nFirst 5 Samples Preview (Ordinal Order):")
    for r in records[:5]:
        print(
            f"   [{r['ordinal_index']:3d}] {r['filename']:<12s} -> "
            f"Run 1: {r['run1_gt_standard']} | "
            f"Run 2: {r['run2_pred_standard']} | "
            f"Run 3: {r['run3_gt_fuzzy']} | "
            f"Run 4: {r['run4_pred_fuzzy']}"
        )
    print("=" * 90)
    print("🎉 All 4 runs successfully computed, distributions analyzed, and files saved.")


if __name__ == "__main__":
    main()
