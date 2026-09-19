import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from src.cvm_calculator import (
    CVMInput,
    CVMThresholds,
    classify_cvm_stage,
)
from src.model import CephalometricSwinGCN

LANDMARK_LABELS = [
    "C2_PI",  # 0: Inferior-Posterior
    "C2_IC",  # 1: Inferior-Concavity
    "C2_AI",  # 2: Inferior-Anterior
    "C3_PS",  # 3: Superior-Posterior
    "C3_AS",  # 4: Superior-Anterior
    "C3_PI",  # 5: Inferior-Posterior
    "C3_IC",  # 6: Inferior-Concavity
    "C3_AI",  # 7: Inferior-Anterior
    "C4_PS",  # 8: Superior-Posterior
    "C4_AS",  # 9: Superior-Anterior
    "C4_PI",  # 10: Inferior-Posterior
    "C4_IC",  # 11: Inferior-Concavity
    "C4_AI",  # 12: Inferior-Anterior
]

STAGE_DESCRIPTIONS = {
    "CS1": "Lower borders of C2, C3, and C4 are flat. Bodies of C3 and C4 are trapezoidal (Peak growth in ~2 years).",
    "CS2": "Lower border of C2 is concave. Bodies of C3 and C4 are trapezoidal (Peak growth in ~1 year).",
    "CS3": "Lower borders of C2 and C3 are concave. Bodies are trapezoidal/horizontal (Peak growth begins).",
    "CS4": "Lower borders of C2, C3, and C4 are concave. Bodies are rectangular horizontal (Peak growth completed).",
    "CS5": "Lower borders of C2, C3, and C4 are concave. At least one vertebra is square (Peak growth passed).",
    "CS6": "Lower borders of C2, C3, and C4 are concave. At least one vertebra is rectangular vertical (Maturation complete).",
}


@dataclass
class PredictionResult:
    """Encapsulates the complete inference and geometric classification results."""
    stage: str
    details: Dict[str, Any]
    landmarks_dict: Dict[str, Dict[str, Tuple[float, float]]]
    landmarks_array: List[Tuple[float, float]]
    normalized_landmarks: List[Tuple[float, float]]
    image_size: Tuple[int, int]  # (W, H)
    annotated_image: Optional[Image.Image] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts results into a JSON-serializable dictionary."""
        return {
            "stage": self.stage,
            "clinical_summary": STAGE_DESCRIPTIONS.get(self.stage, ""),
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "landmarks": {
                name: list(coords)
                for name, coords in zip(LANDMARK_LABELS, self.landmarks_array)
            },
            "landmarks_by_vertebra": {
                vert: {k: list(v) for k, v in pts.items()}
                for vert, pts in self.landmarks_dict.items()
            },
            "details": self.details,
        }


def draw_landmarks_on_image(
    image: Image.Image,
    landmarks: Dict[str, Dict[str, Tuple[float, float]]],
    draw_labels: bool = False,
) -> Image.Image:
    """
    Renders cervical vertebra landmark points, connecting anatomical outlines,
    and optional text labels on the PIL image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    W, H = image.size

    line_w = max(2, int(min(W, H) * 0.005))
    dot_r = max(4, int(min(W, H) * 0.008))

    c2_color = (255, 127, 80)   # Coral
    c3_color = (32, 178, 170)   # Teal
    c4_color = (65, 105, 225)   # Royal Blue

    c2 = landmarks["C2"]
    c3 = landmarks["C3"]
    c4 = landmarks["C4"]

    # C2 inferior border
    draw.line(
        [c2["inferior-posterior"], c2["inferior-concavity"], c2["inferior-anterior"]],
        fill=c2_color,
        width=line_w,
    )

    # C3 closed loop outline
    c3_outline = [
        c3["superior-posterior"],
        c3["superior-anterior"],
        c3["inferior-anterior"],
        c3["inferior-concavity"],
        c3["inferior-posterior"],
        c3["superior-posterior"],
    ]
    draw.line(c3_outline, fill=c3_color, width=line_w)

    # C4 closed loop outline
    c4_outline = [
        c4["superior-posterior"],
        c4["superior-anterior"],
        c4["inferior-anterior"],
        c4["inferior-concavity"],
        c4["inferior-posterior"],
        c4["superior-posterior"],
    ]
    draw.line(c4_outline, fill=c4_color, width=line_w)

    def draw_dot(pt: Tuple[float, float], color: Tuple[int, int, int]):
        draw.ellipse(
            [pt[0] - dot_r, pt[1] - dot_r, pt[0] + dot_r, pt[1] + dot_r],
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )

    for pt in c2.values():
        draw_dot(pt, c2_color)
    for pt in c3.values():
        draw_dot(pt, c3_color)
    for pt in c4.values():
        draw_dot(pt, c4_color)

    if draw_labels:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for vert, pts in [("C2", c2), ("C3", c3), ("C4", c4)]:
            for k, pt in pts.items():
                short_name = f"{vert}_{''.join([w[0].upper() for w in k.split('-')])}"
                draw.text(
                    (pt[0] + dot_r + 2, pt[1] - dot_r - 2),
                    short_name,
                    fill=(255, 255, 255),
                    font=font,
                )

    return annotated


def visualize_with_matplotlib(
    image: Image.Image,
    result: PredictionResult,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Renders a publication-grade two-panel visualization using Matplotlib:
      - Left: Clean Full Lateral Cephalometric X-ray with landmark overlay & contours.
      - Right: Zoomed cervical spine detail with non-intrusive, semi-transparent labels positioned outside the bone contours.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c2_color = "#FF7F50"   # Coral
    c3_color = "#20B2AA"   # Teal
    c4_color = "#4169E1"   # Royal Blue

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), facecolor="#0F172A")
    fig.canvas.manager.set_window_title(f"CVM Prediction: {result.stage}")

    # Title header
    stage_text = f"CVM Stage: {result.stage}"
    desc_text = STAGE_DESCRIPTIONS.get(result.stage, "")
    fig.suptitle(
        f"{stage_text}\n{desc_text}",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.97,
    )

    c2 = result.landmarks_dict["C2"]
    c3 = result.landmarks_dict["C3"]
    c4 = result.landmarks_dict["C4"]

    c2_pts = [c2["inferior-posterior"], c2["inferior-concavity"], c2["inferior-anterior"]]
    c3_pts = [
        c3["superior-posterior"],
        c3["superior-anterior"],
        c3["inferior-anterior"],
        c3["inferior-concavity"],
        c3["inferior-posterior"],
        c3["superior-posterior"],
    ]
    c4_pts = [
        c4["superior-posterior"],
        c4["superior-anterior"],
        c4["inferior-anterior"],
        c4["inferior-concavity"],
        c4["inferior-posterior"],
        c4["superior-posterior"],
    ]

    # Calculate ROI bounding box around the 13 landmarks for zooming the right view
    xs = [p[0] for p in result.landmarks_array]
    ys = [p[1] for p in result.landmarks_array]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 30)
    span_y = max(max_y - min_y, 30)

    # Generous horizontal padding so labels on the left/right fit cleanly
    pad_x = span_x * 0.75
    pad_y = span_y * 0.35

    roi_x0 = max(0, min_x - pad_x)
    roi_y0 = max(0, min_y - pad_y)
    roi_x1 = min(result.image_size[0], max_x + pad_x)
    roi_y1 = min(result.image_size[1], max_y + pad_y)

    for ax in axes:
        ax.imshow(image, cmap="gray" if image.mode == "L" else None)
        ax.set_facecolor("#0F172A")

        # C2 line & points
        ax.plot([p[0] for p in c2_pts], [p[1] for p in c2_pts], color=c2_color, linewidth=2.0, zorder=3)
        for p in c2_pts:
            ax.scatter(p[0], p[1], color=c2_color, s=36, edgecolors="white", linewidths=1.0, zorder=5)

        # C3 outline & points
        ax.plot([p[0] for p in c3_pts], [p[1] for p in c3_pts], color=c3_color, linewidth=2.0, zorder=3)
        for p in c3_pts:
            ax.scatter(p[0], p[1], color=c3_color, s=36, edgecolors="white", linewidths=1.0, zorder=5)

        # C4 outline & points
        ax.plot([p[0] for p in c4_pts], [p[1] for p in c4_pts], color=c4_color, linewidth=2.0, zorder=3)
        for p in c4_pts:
            ax.scatter(p[0], p[1], color=c4_color, s=36, edgecolors="white", linewidths=1.0, zorder=5)

    # --- Left Panel: Full Cephalogram (Clean, no ROI rectangle) ---
    axes[0].set_title("Full Cephalometric X-ray", color="#94A3B8", fontsize=13, fontweight="600", pad=10)
    axes[0].axis("off")

    # --- Right Panel: Zoomed Cervical Spine Detail ---
    axes[1].set_title("Zoomed Vertebrae (C2, C3, C4) & 13 Landmarks", color="#94A3B8", fontsize=13, fontweight="600", pad=10)
    axes[1].set_xlim(roi_x0, roi_x1)
    axes[1].set_ylim(roi_y1, roi_y0)  # Inverted for image coordinate space
    axes[1].axis("off")

    # Smart label placement offsets:
    # Posterior landmarks placed to the left; Anterior landmarks placed to the right;
    # Concavity landmarks placed into open disc space.
    label_offsets = {
        # Posterior landmarks (left side) -> offset left
        "C2_PI": (-38, 0, "right", "center"),
        "C3_PS": (-38, -6, "right", "center"),
        "C3_PI": (-38, 0, "right", "center"),
        "C4_PS": (-38, -6, "right", "center"),
        "C4_PI": (-38, 0, "right", "center"),
        # Anterior landmarks (right side) -> offset right
        "C2_AI": (38, 0, "left", "center"),
        "C3_AS": (38, -6, "left", "center"),
        "C3_AI": (38, 0, "left", "center"),
        "C4_AS": (38, -6, "left", "center"),
        "C4_AI": (38, 0, "left", "center"),
        # Concavity landmarks -> placed into adjacent open spaces
        "C2_IC": (0, -16, "center", "bottom"),
        "C3_IC": (0, 16, "center", "top"),
        "C4_IC": (0, 16, "center", "top"),
    }

    # Add non-intrusive landmark annotations with thin leader lines and semi-transparent badges
    for name, (px, py) in zip(LANDMARK_LABELS, result.landmarks_array):
        dx, dy, ha, va = label_offsets.get(name, (30, 0, "left", "center"))
        axes[1].annotate(
            name,
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7,
            fontweight="bold",
            color="#F1F5F9",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="#0F172A",
                ec="#64748B",
                alpha=0.45,  # Semi-transparent to reveal underlying structures
                linewidth=0.5,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#94A3B8",
                lw=0.6,
                alpha=0.55,
                shrinkA=2,
                shrinkB=3,
            ),
            zorder=6,
        )

    # Clean Legend at bottom (C2, C3, C4 only)
    legend_elements = [
        Patch(facecolor=c2_color, edgecolor="white", label="C2 (Axis: PI, IC, AI)"),
        Patch(facecolor=c3_color, edgecolor="white", label="C3 (PS, AS, PI, IC, AI)"),
        Patch(facecolor=c4_color, edgecolor="white", label="C4 (PS, AS, PI, IC, AI)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#1E293B",
        edgecolor="#334155",
        fontsize=10.5,
        labelcolor="white",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Saved] Matplotlib plot saved to: {save_path}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[Warning] Could not display matplotlib interactive window: {e}")


class CVMPredictor:
    """
    End-to-end inference engine for Lateral Cephalometric X-ray images.
    Loads CephalometricSwinGCN model weights and computes CVM maturation stages.
    """

    def __init__(
        self,
        weights_path: str = "model/weights.pth",
        device: Optional[str] = None,
        img_size: int = 640,
    ):
        self.img_size = img_size
        self.weights_path = weights_path

        # Resolve device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = CephalometricSwinGCN(
            num_landmarks=13,
            pretrained=False,
            img_size=img_size,
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Normalization constants (ImageNet defaults used by Swin backbone)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepares a PIL Image for model forward pass.
        Returns:
            - tensor: [1, 3, img_size, img_size]
            - original_size: (W, H)
        """
        rgb_image = image.convert("RGB")
        original_size = rgb_image.size  # (W, H)

        resized = rgb_image.resize(
            (self.img_size, self.img_size), resample=Image.Resampling.BILINEAR
        )
        tensor = TF.to_tensor(resized)  # [3, H, W] in [0.0, 1.0]
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor, original_size

    def predict_landmarks(
        self, image: Image.Image
    ) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Runs neural model inference to extract 13 landmark coordinates.
        Returns:
            - landmarks_dict: vertebra-grouped coordinates in original image pixel space
            - landmarks_px: list of 13 coordinates (x, y) in original image pixel space
            - landmarks_norm: list of 13 coordinates (x, y) in [0, 1] normalized space
        """
        tensor, (W, H) = self.preprocess(image)

        with torch.no_grad():
            _, coords_norm_tensor = self.model(tensor)

        coords_norm = coords_norm_tensor.squeeze(0).cpu().numpy()  # [13, 2]

        landmarks_px: List[Tuple[float, float]] = []
        landmarks_norm: List[Tuple[float, float]] = []

        for i in range(13):
            nx, ny = float(coords_norm[i, 0]), float(coords_norm[i, 1])
            px, py = nx * W, ny * H
            landmarks_norm.append((nx, ny))
            landmarks_px.append((px, py))

        # Map 13 landmarks to C2, C3, C4 structure
        landmarks_dict = {
            "C2": {
                "inferior-posterior": landmarks_px[0],
                "inferior-concavity": landmarks_px[1],
                "inferior-anterior": landmarks_px[2],
            },
            "C3": {
                "superior-posterior": landmarks_px[3],
                "superior-anterior": landmarks_px[4],
                "inferior-posterior": landmarks_px[5],
                "inferior-concavity": landmarks_px[6],
                "inferior-anterior": landmarks_px[7],
            },
            "C4": {
                "superior-posterior": landmarks_px[8],
                "superior-anterior": landmarks_px[9],
                "inferior-posterior": landmarks_px[10],
                "inferior-concavity": landmarks_px[11],
                "inferior-anterior": landmarks_px[12],
            },
        }

        return landmarks_dict, landmarks_px, landmarks_norm

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        thresholds: Optional[CVMThresholds] = None,
        annotate: bool = True,
        draw_labels: bool = False,
    ) -> PredictionResult:
        """
        Complete pipeline: Landmark detection -> Geometric analysis -> CVM stage classification.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        W, H = pil_img.size

        # 1. Run model inference for landmark detection
        landmarks_dict, landmarks_px, landmarks_norm = self.predict_landmarks(pil_img)

        # 2. Geometric CVM stage calculation
        cvm_input = CVMInput.from_dict(landmarks_dict)
        classification = classify_cvm_stage(cvm_input, thresholds=thresholds)

        # 3. Landmark annotation
        annotated_img = None
        if annotate:
            annotated_img = draw_landmarks_on_image(
                pil_img, landmarks_dict, draw_labels=draw_labels
            )

        return PredictionResult(
            stage=classification["stage"],
            details=classification["details"],
            landmarks_dict=landmarks_dict,
            landmarks_array=landmarks_px,
            normalized_landmarks=landmarks_norm,
            image_size=(W, H),
            annotated_image=annotated_img,
        )


def print_cli_report(result: PredictionResult, image_path: Optional[str] = None):
    """Prints a detailed terminal report of CVM prediction and geometric metrics."""
    print("\n" + "=" * 70)
    print("  CERVICAL VERTEBRAL MATURATION (CVM) PREDICTION REPORT")
    print("=" * 70)
    if image_path:
        print(f"Input Image : {image_path} (Size: {result.image_size[0]}x{result.image_size[1]})")
    print(f"Final Stage : \033[1;32m{result.stage}\033[0m")
    print(f"Clinical    : {STAGE_DESCRIPTIONS.get(result.stage, '')}")
    if result.details.get("table_2_exact_match"):
        print("Rule Match  : \033[1;34mExact match with Table 2\033[0m")
    s = result.details.get("spatial_calibration_mm_per_px", 0.375)
    th_mm = result.details.get("concavity_threshold_mm", 1.0)
    print(f"Calibration : S = {s:.3f} mm/pixel (Concavity Presence Threshold: {th_mm:.1f} mm)")
    print("-" * 70)

    # Vertebra C2
    c2 = result.details["C2"]
    c2_concave_str = "CONCAVE" if c2["is_concave"] else "FLAT"
    print(f"\n[Vertebra C2 (Axis)]")
    print(f"  Concavity Depth : {c2['concavity_depth_px']:.2f} px ({c2['concavity_depth_mm']:.2f} mm)")
    print(f"  Concavity Ratio : {c2['concavity_ratio']:.3f} -> Status: {c2_concave_str} (Notch: {c2.get('notch', 0)})")

    # Vertebra C3 & C4
    for vert_id in ["C3", "C4"]:
        v = result.details[vert_id]
        sh = v["shape_metrics"]
        v_concave_str = "CONCAVE" if v["is_concave"] else "FLAT"
        print(f"\n[Vertebra {vert_id}]")
        print(f"  Concavity Depth : {v['concavity_depth_px']:.2f} px ({v['concavity_depth_mm']:.2f} mm) -> Status: {v_concave_str} (Notch: {v.get('notch', 0)})")
        print(f"  Shape Index (SI): {sh['shape_index']:.3f} (Ha={sh['h_anterior']:.1f}px, Hp={sh['h_posterior']:.1f}px, Ws={sh['w_superior']:.1f}px, Wi={sh['w_inferior']:.1f}px)")
        print(f"  Taper Ratio (TR): {sh['taper_ratio']:.3f} (Ha / Hp)")
        print(f"  Classified Shape: \033[1;34m{sh['shape']}\033[0m")

    print("\n[Predicted 13 Landmarks (Image Pixel Coordinates)]")
    print(f"  {'#':<3} {'Landmark':<10} {'X (px)':<10} {'Y (px)':<10} {'X (norm)':<10} {'Y (norm)':<10}")
    print("  " + "-" * 55)
    for idx, (name, (px, py), (nx, ny)) in enumerate(
        zip(LANDMARK_LABELS, result.landmarks_array, result.normalized_landmarks)
    ):
        print(f"  {idx:<3} {name:<10} {px:<10.1f} {py:<10.1f} {nx:<10.4f} {ny:<10.4f}")
    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lateral Cephalometric X-ray CVM Stage Inference CLI"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to cephalometric X-ray image (PNG, JPG, TIFF, etc.)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="model/weights.pth",
        help="Path to model weights checkpoint (default: model/weights.pth)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device to use (default: auto)",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display interactive Matplotlib visualization window (default: --show, use --no-show to disable)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save Matplotlib publication-grade figure to path (e.g. plot.png)",
    )
    parser.add_argument(
        "--output-image",
        "-o",
        type=str,
        default=None,
        help="Save raw image with visualized landmarks overlay",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        type=str,
        default=None,
        help="Save inference results and metrics as JSON",
    )
    # Section 2.5 parameters
    parser.add_argument(
        "--calibration-factor",
        "--pixel-to-mm",
        dest="pixel_to_mm",
        type=float,
        default=0.375,
        help="Spatial calibration factor S in mm/pixel (default: 0.375 mm/px)",
    )
    parser.add_argument(
        "--concavity-threshold-mm",
        type=float,
        default=1.0,
        help="Physical concavity depth threshold in mm (default: 1.0 mm)",
    )
    parser.add_argument(
        "--concavity-threshold",
        type=float,
        default=0.05,
        help="Fallback relative concavity ratio threshold if absolute depth disabled (default: 0.05)",
    )
    parser.add_argument(
        "--use-absolute-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use physical mm concavity depth threshold instead of relative ratio (default: True)",
    )
    parser.add_argument(
        "--taper-threshold",
        type=float,
        default=1.15,
        help="Trapezoid tapering ratio threshold TR >= threshold (default: 1.15)",
    )
    parser.add_argument(
        "--trapezoid-si-threshold",
        type=float,
        default=0.75,
        help="Trapezoid shape index threshold SI <= threshold (default: 0.75)",
    )
    parser.add_argument(
        "--rect-horizontal-threshold",
        type=float,
        default=0.85,
        help="Rectangular horizontal shape index upper threshold (default: 0.85)",
    )
    parser.add_argument(
        "--rect-vertical-threshold",
        type=float,
        default=1.15,
        help="Rectangular vertical shape index threshold SI >= threshold (default: 1.15)",
    )
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw text labels beside landmark points on output image",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed terminal output and only output the stage",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    thresholds = CVMThresholds(
        use_absolute_depth=args.use_absolute_depth,
        pixel_to_mm=args.pixel_to_mm,
        concavity_depth_mm_threshold=args.concavity_threshold_mm,
        concavity_ratio_threshold=args.concavity_threshold,
        trapezoid_taper_threshold=args.taper_threshold,
        trapezoid_si_threshold=args.trapezoid_si_threshold,
        rect_horizontal_si_threshold=args.rect_horizontal_threshold,
        rect_vertical_si_threshold=args.rect_vertical_threshold,
    )

    predictor = CVMPredictor(
        weights_path=args.weights,
        device=args.device,
    )

    pil_img = Image.open(args.image).convert("RGB")

    result = predictor.predict(
        image_input=pil_img,
        thresholds=thresholds,
        annotate=bool(args.output_image) or not args.quiet,
        draw_labels=args.draw_labels,
    )

    if args.quiet:
        print(result.stage)
    else:
        print_cli_report(result, image_path=args.image)

    if args.output_image and result.annotated_image:
        result.annotated_image.save(args.output_image)
        print(f"[Saved] Visualized landmarks saved to: {args.output_image}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[Saved] JSON metrics saved to: {args.output_json}")

    # Display / save Matplotlib visualization
    if args.show or args.save_plot:
        visualize_with_matplotlib(
            image=pil_img,
            result=result,
            save_path=args.save_plot,
            show=args.show,
        )


if __name__ == "__main__":
    main()
