#!/usr/bin/env python3
"""
Inference Script for Label Studio Project 1.

Workflow:
1. Reads downloaded images and existing label data from data/full (or specified --data-dir).
2. Mode A (Default - Annotated Landmarks):
   - Reads project export from data/export_cache.json (or --export-cache).
   - Extracts annotated cervical vertebrae landmarks (13 keypoints) for each image.
   - Runs the geometric CVM calculator (src.cvm_calculator) to determine the CVM stage (CS1-CS6).
   - Default output file: stages.txt
3. Mode B (--inferece / --inference - Model Landmark Prediction):
   - Loads the deep learning model (CephalometricSwinGCN) with trained weights (--weights-path).
   - Predicts the 13 cervical vertebrae landmark coordinates directly from image pixels.
   - Runs the geometric CVM calculator on the predicted coordinates to determine the stage.
   - Default output file: stages_inference.txt
4. Sorts the results according to the names of the downloaded files (default: alphabetical string sort;
   natural/numeric sort also supported via --sort-order natural).
5. Stores the stages line-by-line in a single output file.
6. Optionally saves a detailed mapping (CSV/JSON) and verifies against existing labels in data/full/labels.
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

from PIL import Image
from tqdm import tqdm

# Automatically re-exec with local .venv python if --inferece is requested and torch is missing
if any(arg in sys.argv for arg in ["--inferece", "--inference"]):
    try:
        import torch  # noqa: F401
    except ImportError:
        script_dir = Path(__file__).resolve().parent
        venv_python = script_dir / ".venv" / "bin" / "python"
        venv_prefix = script_dir / ".venv"
        if venv_python.exists() and sys.prefix != str(venv_prefix):
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Import CVM geometric calculation engine
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

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CVM stage calculation on images from Label Studio Project 1 (via annotations or model inference)."
    )
    parser.add_argument(
        "--inferece",
        "--inference",
        dest="inference",
        action="store_true",
        help="Use deep learning model (CephalometricSwinGCN) to predict 13 landmarks from image pixels instead of using Label Studio annotations.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=os.getenv("CVM_WEIGHTS_PATH", "model/weights.pth"),
        help="Path to trained model weights .pth file (default: model/weights.pth or $CVM_WEIGHTS_PATH).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device for model: 'cpu', 'mps', or 'cuda' (default: cpu to prevent MacBook Air overheating).",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="Delay in milliseconds between frames to allow thermal cooling on fanless devices (default: 0).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image resolution for SwinGCN model (default: 640).",
    )
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
        "--output-file",
        "-o",
        type=str,
        default=None,
        help="Destination path for single output file containing stages line by line "
        "(default: 'stages_inference.txt' with --inferece, 'stages.txt' with annotations).",
    )
    parser.add_argument(
        "--sort-order",
        type=str,
        choices=["alphabetical", "natural", "alpha", "numeric"],
        default="alphabetical",
        help="Order to sort the downloaded filenames: 'alphabetical' (e.g. 1.jpg, 10.jpg, 100.jpg...) "
        "or 'natural' (e.g. 1.jpg, 2.jpg, 3.jpg...) (default: alphabetical).",
    )
    parser.add_argument(
        "--stage-format",
        type=str,
        choices=["int", "cs"],
        default="int",
        help="Format of stage per line: 'int' (1-6) or 'cs' (CS1-CS6) (default: int).",
    )
    parser.add_argument(
        "--save-mapping",
        type=str,
        default=None,
        help="Optional path to save CSV mapping file with filename, task_id, calculated_stage, and existing_label.",
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
    args = parser.parse_args()

    # Automatically set output filename if not explicitly provided
    if args.output_file is None:
        args.output_file = "stages_inference.txt" if args.inference else "stages.txt"

    return args


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


def compute_stage_from_landmarks(landmarks: Dict[str, Point]) -> Tuple[int, str]:
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
    thresholds = CVMThresholds()
    result = classify_cvm_stage(cvm_input, thresholds=thresholds)

    stage_str = result["stage"]  # "CS1" ... "CS6"
    stage_int = int(stage_str.replace("CS", ""))
    return stage_int, stage_str


def natural_sort_key(path: Path) -> List[Any]:
    """Key for natural human sorting (e.g. 1.jpg, 2.jpg, ..., 10.jpg)."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def main():
    args = parse_args()

    # 1. Resolve image and label directory paths
    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir) if args.images_dir else data_dir / "images"
    labels_dir = Path(args.labels_dir) if args.labels_dir else data_dir / "labels"

    if not images_dir.is_dir():
        print(f"❌ Error: Images directory not found at: {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"📁 Reading images from: {images_dir}")
    if labels_dir.is_dir():
        print(f"📁 Reading existing labels from: {labels_dir}")
    else:
        print(f"ℹ️ Labels directory not found at {labels_dir} (proceeding without ground-truth comparison).")

    # 2. Gather image files
    all_image_paths = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]

    if not all_image_paths:
        print(f"❌ Error: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    # 3. Sort files according to specified sort order
    is_natural = args.sort_order in ["natural", "numeric"]
    if is_natural:
        all_image_paths.sort(key=natural_sort_key)
        sort_desc = "natural numeric order (e.g., 1.jpg, 2.jpg, ... 10.jpg)"
    else:
        all_image_paths.sort(key=lambda p: p.name)
        sort_desc = "alphabetical string order (e.g., 1.jpg, 10.jpg, 100.jpg, ...)"

    print(f"📋 Found {len(all_image_paths)} downloaded image files.")
    print(f"🔀 Sorting applied: {sort_desc}")

    # 4. Initialize Inference Pipeline / Load Export Cache
    predictor = None
    tasks_by_id = {}
    tasks_by_filename = {}

    if args.inference:
        print("\n🧠 Mode: Model Inference (--inferece active)")
        print(f"   Model Weights: {args.weights_path}")
        from src.inference import CVMPredictor

        predictor = CVMPredictor(
            weights_path=args.weights_path,
            device=args.device,
            img_size=args.img_size,
        )
        print(f"   Inference Device: {predictor.device}")

        # If export cache exists, load for optional task ID matching
        if os.path.exists(args.export_cache):
            try:
                tasks = load_label_studio_export(
                    export_path=args.export_cache,
                    fetch_latest=False,
                )
                tasks_by_id, tasks_by_filename = build_task_indices(tasks)
            except Exception:
                pass
    else:
        print("\n📐 Mode: Geometric Calculation from Label Studio Annotations")
        tasks = load_label_studio_export(
            export_path=args.export_cache,
            fetch_latest=args.fetch_latest,
            env_file=args.env_file,
            project_id=args.project_id,
        )
        tasks_by_id, tasks_by_filename = build_task_indices(tasks)

    # 5. Process each image in sorted filename order
    mode_label = "Model Inference (SwinGCN + Geometry)" if args.inference else "Geometric Calculation (Annotations)"
    print(f"\n🚀 Running {mode_label} on {len(all_image_paths)} images...")
    processed_records = []
    errors = []
    label_matches = 0
    label_mismatches = 0
    stage_counts = Counter()

    for img_path in tqdm(all_image_paths, desc="Processing Images", unit="img"):
        fn = img_path.name
        stem = img_path.stem

        # Match image to task if available
        task = tasks_by_id.get(stem) or tasks_by_filename.get(fn) or tasks_by_filename.get(stem)
        tid = task.get("id") if task else stem

        # Read existing ground-truth label if present in labels_dir
        gt_label: Optional[str] = None
        if labels_dir.is_dir():
            lbl_file = labels_dir / f"{stem}.txt"
            if lbl_file.is_file():
                try:
                    gt_label = lbl_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        try:
            if args.inference:
                # Mode B: Model Landmark Prediction + Geometric Calculation
                assert predictor is not None
                pred_result = predictor.predict(img_path, annotate=False)
                stage_str = pred_result.stage
                stage_int = int(stage_str.replace("CS", ""))
            else:
                # Mode A: Label Studio Human Annotations + Geometric Calculation
                if not task:
                    errors.append((fn, "No matching task found in export cache"))
                    continue
                landmarks = extract_landmarks(task, fallback_image_path=img_path)
                stage_int, stage_str = compute_stage_from_landmarks(landmarks)

            # Check agreement with existing label
            if gt_label is not None and gt_label.isdigit():
                if int(gt_label) == stage_int:
                    label_matches += 1
                else:
                    label_mismatches += 1

            stage_counts[f"CS{stage_int}"] += 1
            processed_records.append({
                "filename": fn,
                "stem": stem,
                "task_id": tid,
                "stage_int": stage_int,
                "stage_str": stage_str,
                "gt_label": gt_label,
            })
        except Exception as err:
            errors.append((fn, str(err)))

        if args.delay_ms > 0:
            import time
            time.sleep(args.delay_ms / 1000.0)

    # 6. Store stages line by line in the single output file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in processed_records:
            if args.stage_format == "cs":
                f.write(f"{rec['stage_str']}\n")
            else:
                f.write(f"{rec['stage_int']}\n")

    print(f"\n💾 Saved {len(processed_records)} stages line by line to: {output_path.resolve()}")

    # 7. Optionally save mapping CSV file
    if args.save_mapping:
        mapping_path = Path(args.save_mapping)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["line_number", "filename", "task_id", "stage", "gt_label"])
            for idx, rec in enumerate(processed_records, start=1):
                writer.writerow([idx, rec["filename"], rec["task_id"], rec["stage_int"], rec["gt_label"] or ""])
        print(f"📊 Saved detailed mapping file to: {mapping_path.resolve()}")

    # 8. Summary Report
    print("\n" + "=" * 60)
    print("        INFERENCE ON LABEL STUDIO PROJECT 1 SUMMARY         ")
    print("=" * 60)
    print(f"Execution Mode:          {'Deep Learning Model Inference' if args.inference else 'Geometric Calculation (Human Annotations)'}")
    print(f"Total Images Found:      {len(all_image_paths)}")
    print(f"Successfully Evaluated:  {len(processed_records)}")
    if errors:
        print(f"Errors Encountered:      {len(errors)}")
        for err_fn, msg in errors[:5]:
            print(f"  - {err_fn}: {msg}")

    if label_matches + label_mismatches > 0:
        total_eval = label_matches + label_mismatches
        acc = (label_matches / total_eval) * 100.0
        print(f"Agreement with Labels:   {label_matches}/{total_eval} ({acc:.2f}%)")

    print(f"Output File:             {output_path.resolve()}")
    print(f"Line Order:              {sort_desc}")
    print(f"Format:                  {'CS1-CS6' if args.stage_format == 'cs' else '1-6'}")

    print("\nCVM Stage Distribution:")
    for stage_num in range(1, 7):
        stage_name = f"CS{stage_num}"
        count = stage_counts.get(stage_name, 0)
        pct = (count / len(processed_records) * 100) if processed_records else 0.0
        bar = "█" * int(round(pct / 2.5))
        print(f"  {stage_name}: {count:3d} ({pct:5.1f}%) {bar}")

    print("\nFirst 5 Lines Preview:")
    for idx, rec in enumerate(processed_records[:5], start=1):
        val = rec["stage_str"] if args.stage_format == "cs" else rec["stage_int"]
        print(f"  Line {idx:3d}: {val}  (from {rec['filename']})")
    print("=" * 60)
    print("🎉 Done! All stages successfully computed and saved.")


if __name__ == "__main__":
    main()
