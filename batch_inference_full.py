#!/usr/bin/env python3
"""
Batch Inference Script for Full Dataset (All Images from Label Studio).

Workflow:
1. Authenticates with Label Studio using credentials from .env.
2. Retrieves project export (Project ID 1) for the entire dataset.
3. Concurrently downloads all images to data/full/images/{id}.jpg.
4. Extracts the 13 cervical vertebrae keypoint coordinates for each annotated task.
5. Executes the geometric CVM calculator (src.cvm_calculator) to classify CVM stage (CS1-CS6).
6. Stores the determined integer stage (1-6) in data/full/labels/{id}.txt.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch download all images from Label Studio and determine CVM stages using geometric rules."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/full",
        help="Target output directory for images and labels (default: data/full).",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio Project ID (defaults to LABEL_STUDIO_PROJECT_ID from .env, or 1).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env configuration file (default: .env).",
    )
    parser.add_argument(
        "--export-cache",
        type=str,
        default="data/export_cache.json",
        help="Path to cache Label Studio project export JSON to speed up subsequent runs.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Force re-download of project export from Label Studio even if cache exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading image if it already exists in the destination folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total number of images to process (useful for quick testing).",
    )
    return parser.parse_args()


def get_authenticated_session(
    api_token: str,
    workers: int = 8,
) -> requests.Session:
    """Creates a requests.Session with connection pooling and retries."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        pool_connections=workers * 2,
        pool_maxsize=workers * 2,
        max_retries=retries,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Authorization": f"Token {api_token}"})
    return session


def fetch_project_export(
    session: requests.Session,
    ls_url: str,
    project_id: int,
    cache_path: Optional[str] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieves project export JSON, utilizing local cache if available."""
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        print(f"📦 Loading cached Label Studio export from: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    export_url = f"{ls_url}/api/projects/{project_id}/export?exportType=JSON"
    print(f"🌐 Requesting project {project_id} export from Label Studio ({export_url})...")
    t0 = time.time()
    resp = session.get(export_url, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch export from Label Studio (Status {resp.status_code}): {resp.text[:300]}"
        )
    tasks = resp.json()
    elapsed = time.time() - t0
    print(f"✅ Successfully retrieved {len(tasks)} tasks in {elapsed:.1f}s.")

    if cache_path:
        cache_p = Path(cache_path)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_p, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        print(f"💾 Saved export cache to: {cache_path}")

    return tasks


def get_image_url_and_ext(task: Dict[str, Any], ls_url: str) -> Tuple[str, str]:
    """Resolves full image download URL and file extension from task data."""
    img_path = task.get("data", {}).get("img") or task.get("file_upload") or ""
    if not img_path:
        raise ValueError(f"Task {task.get('id')} has no image URL.")

    full_url = img_path if img_path.startswith("http") else f"{ls_url.rstrip('/')}/{img_path.lstrip('/')}"

    # Determine extension
    parsed = urlparse(full_url)
    query_params = parse_qs(parsed.query)
    if "d" in query_params:
        filename = os.path.basename(query_params["d"][0])
    else:
        filename = os.path.basename(parsed.path)

    ext = Path(filename).suffix.lower()
    if not ext or ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        ext = ".jpg"

    return full_url, ext


def download_single_image(
    session: requests.Session,
    url: str,
    target_path: Path,
    skip_existing: bool = False,
) -> bool:
    """Downloads an image and writes it directly to target_path."""
    if skip_existing and target_path.exists() and target_path.stat().st_size > 0:
        return True

    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed for {url} (Status {resp.status_code})")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(resp.content)
    return True


def extract_landmarks(
    task: Dict[str, Any],
    fallback_image_path: Optional[Path] = None,
) -> Dict[str, Point]:
    """Extracts the 13 cervical vertebrae landmarks in image pixel coordinates."""
    annotations = task.get("annotations", [])
    if not annotations:
        raise ValueError(f"Task {task.get('id')} has no annotations.")

    # Image dimensions for percentage conversion
    img_w, img_h = None, None

    points_dict: Dict[str, Point] = {}

    # Prefer the most recent / completed annotation
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

    # If dimensions were not embedded in annotation result, inspect image file
    if (img_w is None or img_h is None) and fallback_image_path and fallback_image_path.exists():
        with Image.open(fallback_image_path) as im:
            img_w, img_h = im.size

    if img_w is None or img_h is None:
        raise ValueError(f"Could not resolve dimensions for task {task.get('id')}.")

    missing = set(EXPECTED_LANDMARKS) - set(points_dict.keys())
    if missing:
        raise ValueError(f"Task {task.get('id')} is missing landmarks: {sorted(missing)}")

    pixel_points: Dict[str, Point] = {}
    for k, (nx, ny) in points_dict.items():
        pixel_points[k] = Point(x=nx * img_w, y=ny * img_h)

    return pixel_points


def compute_stage_from_landmarks(landmarks: Dict[str, Point]) -> int:
    """Invokes geometric CVM calculator on the 13 landmarks and returns integer stage (1-6)."""
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

    stage_str = result["stage"]  # e.g. "CS1" ... "CS6"
    return int(stage_str.replace("CS", ""))


def main():
    args = parse_args()

    # 1. Load environment and credentials
    if os.path.exists(args.env_file):
        load_dotenv(dotenv_path=args.env_file)
    else:
        load_dotenv()

    ls_url = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
    api_token = os.getenv("LABEL_STUDIO_API_TOKEN") or ""
    project_id = args.project_id or int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

    if not ls_url:
        print("❌ Error: LABEL_STUDIO_URL is missing from environment or .env.", file=sys.stderr)
        sys.exit(1)
    if not api_token:
        print("❌ Error: LABEL_STUDIO_API_TOKEN is missing from environment or .env.", file=sys.stderr)
        sys.exit(1)

    # 2. Setup output directories
    base_out = Path(args.output_dir)
    images_dir = base_out / "images"
    labels_dir = base_out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 3. Initialize session and fetch project export
    session = get_authenticated_session(api_token=api_token, workers=args.workers)
    tasks = fetch_project_export(
        session=session,
        ls_url=ls_url,
        project_id=project_id,
        cache_path=args.export_cache,
        force_refresh=args.force_export,
    )

    # Filter tasks that have an image URL
    tasks_with_images = []
    for t in tasks:
        img_url = t.get("data", {}).get("img") or t.get("file_upload")
        if img_url:
            tasks_with_images.append(t)

    if args.limit and args.limit > 0:
        tasks_with_images = tasks_with_images[: args.limit]
        print(f"⚡ Applying limit: processing first {len(tasks_with_images)} tasks.")

    print(f"🎯 Total tasks with images to process: {len(tasks_with_images)}")

    # 4. Concurrently download images
    print(f"\n⬇️ Downloading {len(tasks_with_images)} images to {images_dir}...")
    task_image_paths: Dict[Any, Path] = {}

    def download_worker(task):
        tid = task["id"]
        img_url, ext = get_image_url_and_ext(task, ls_url)
        target_path = images_dir / f"{tid}{ext}"
        download_single_image(session, img_url, target_path, skip_existing=args.skip_existing)
        return tid, target_path

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_worker, t): t["id"] for t in tasks_with_images}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading Images", unit="img"):
            try:
                tid, p = fut.result()
                task_image_paths[tid] = p
            except Exception as e:
                tid = futures[fut]
                print(f"\n❌ Error downloading image for task {tid}: {e}")

    # 5. Process landmarks and calculate CVM stages
    print(f"\n📐 Computing CVM stages using geometric formulas for {len(tasks_with_images)} images...")
    stage_counts = Counter()
    successful = 0
    skipped_unannotated = 0
    errors = []

    for task in tqdm(tasks_with_images, desc="Evaluating CVM Stages", unit="sample"):
        tid = task["id"]
        img_path = task_image_paths.get(tid)

        # Check if task has annotations
        if not task.get("annotations"):
            skipped_unannotated += 1
            continue

        try:
            landmarks = extract_landmarks(task, fallback_image_path=img_path)
            cvm_stage_int = compute_stage_from_landmarks(landmarks)

            # Write single integer label file: data/full/labels/{tid}.txt
            lbl_path = labels_dir / f"{tid}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(f"{cvm_stage_int}\n")

            stage_counts[f"CS{cvm_stage_int}"] += 1
            successful += 1
        except Exception as err:
            errors.append((tid, str(err)))

    # 6. Summary Report
    print("\n" + "=" * 50)
    print("        FULL DATASET BATCH PROCESSING SUMMARY        ")
    print("=" * 50)
    print(f"Total Tasks in Export:  {len(tasks)}")
    print(f"Images Processed:       {len(task_image_paths)} (stored in {images_dir})")
    print(f"Labels Generated:       {successful} (stored in {labels_dir})")
    if skipped_unannotated:
        print(f"Skipped (Unannotated):  {skipped_unannotated}")
    if errors:
        print(f"Errors Encountered:     {len(errors)}")
        for err_id, msg in errors[:5]:
            print(f"  - Task {err_id}: {msg}")

    print("\nDetermined CVM Stage Distribution:")
    for stage_num in range(1, 7):
        stage_name = f"CS{stage_num}"
        count = stage_counts.get(stage_name, 0)
        pct = (count / successful * 100) if successful > 0 else 0.0
        bar = "█" * int(round(pct / 3))
        print(f"  {stage_name}: {count:3d} ({pct:5.1f}%) {bar}")
    print("=" * 50)
    print("🎉 Done! All full dataset images and CVM label files successfully generated.")


if __name__ == "__main__":
    main()
