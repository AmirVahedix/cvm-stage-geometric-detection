import os
import sys
from pathlib import Path

from src.inference import (
    CVMPredictor,
    build_arg_parser,
    print_cli_report,
)
from src.cvm_calculator import CVMThresholds


def main():
    parser = build_arg_parser()
    # Make image optional in main.py so running `python main.py` defaults to demo sample_xray.png
    for action in parser._actions:
        if action.dest == "image":
            action.required = False
            action.default = "sample_xray.png"

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    thresholds = CVMThresholds(
        concavity_ratio_threshold=args.concavity_threshold,
        trapezoid_height_ratio_threshold=args.taper_threshold,
        rect_horizontal_threshold=args.rect_horizontal_threshold,
        rect_vertical_threshold=args.rect_vertical_threshold,
    )

    print(f"[CVM Detection] Loading model and running inference on: {args.image}...")
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
        import json
        with open(args.output_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[Saved] JSON metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
