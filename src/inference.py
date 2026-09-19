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
    and optional text labels on the image.
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
    stage_desc = {
        "CS1": "Lower borders of C2, C3, and C4 are flat. Bodies of C3 and C4 are trapezoidal (Peak growth in ~2 years).",
        "CS2": "Lower border of C2 is concave. Bodies of C3 and C4 are trapezoidal (Peak growth in ~1 year).",
        "CS3": "Lower borders of C2 and C3 are concave. Bodies are trapezoidal/horizontal (Peak growth begins).",
        "CS4": "Lower borders of C2, C3, and C4 are concave. Bodies are rectangular horizontal (Peak growth completed).",
        "CS5": "Lower borders of C2, C3, and C4 are concave. At least one vertebra is square (Peak growth passed).",
        "CS6": "Lower borders of C2, C3, and C4 are concave. At least one vertebra is rectangular vertical (Maturation complete).",
    }

    print("\n" + "=" * 70)
    print("  CERVICAL VERTEBRAL MATURATION (CVM) PREDICTION REPORT")
    print("=" * 70)
    if image_path:
        print(f"Input Image : {image_path} (Size: {result.image_size[0]}x{result.image_size[1]})")
    print(f"Final Stage : \033[1;32m{result.stage}\033[0m")
    print(f"Clinical    : {stage_desc.get(result.stage, '')}")
    print("-" * 70)

    # Vertebra C2
    c2 = result.details["C2"]
    c2_concave_str = "CONCAVE" if c2["is_concave"] else "FLAT"
    print(f"\n[Vertebra C2 (Axis)]")
    print(f"  Concavity Depth : {c2['concavity_depth']:.2f} px")
    print(f"  Concavity Ratio : {c2['concavity_ratio']:.3f} -> Status: {c2_concave_str}")

    # Vertebra C3 & C4
    for vert_id in ["C3", "C4"]:
        v = result.details[vert_id]
        sh = v["shape_metrics"]
        v_concave_str = "CONCAVE" if v["is_concave"] else "FLAT"
        print(f"\n[Vertebra {vert_id}]")
        print(f"  Concavity Ratio : {v['concavity_ratio']:.3f} -> Status: {v_concave_str}")
        print(f"  Width/Height    : {sh['wh_ratio']:.3f} (Avg W: {sh['w_average']:.1f}px, Avg H: {sh['h_average']:.1f}px)")
        print(f"  Tapering (Ha/Hp): {sh['taper_ratio']:.3f}")
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
        "--output-image",
        "-o",
        type=str,
        default=None,
        help="Optional path to save image with visualized landmarks overlay",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        type=str,
        default=None,
        help="Optional path to save inference results and metrics as JSON",
    )
    parser.add_argument(
        "--concavity-threshold",
        type=float,
        default=0.05,
        help="Concavity ratio threshold (default: 0.05)",
    )
    parser.add_argument(
        "--taper-threshold",
        type=float,
        default=0.90,
        help="Trapezoid tapering ratio threshold (default: 0.90)",
    )
    parser.add_argument(
        "--rect-horizontal-threshold",
        type=float,
        default=1.20,
        help="Rectangular horizontal ratio threshold (default: 1.20)",
    )
    parser.add_argument(
        "--rect-vertical-threshold",
        type=float,
        default=0.85,
        help="Rectangular vertical ratio threshold (default: 0.85)",
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
        concavity_ratio_threshold=args.concavity_threshold,
        trapezoid_height_ratio_threshold=args.taper_threshold,
        rect_horizontal_threshold=args.rect_horizontal_threshold,
        rect_vertical_threshold=args.rect_vertical_threshold,
    )

    predictor = CVMPredictor(
        weights_path=args.weights,
        device=args.device,
    )

    result = predictor.predict(
        image_input=args.image,
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


if __name__ == "__main__":
    main()
