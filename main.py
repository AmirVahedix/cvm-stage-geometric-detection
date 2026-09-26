import os
import sys
from PIL import Image

from src.inference import (
    CVMPredictor,
    build_arg_parser,
    print_cli_report,
    visualize_with_matplotlib,
)
from src.cvm_calculator import CVMThresholds


def main():
    parser = build_arg_parser()
    for action in parser._actions:
        if action.dest == "image":
            action.required = False
            action.default = "sample_xray.png"

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    thresholds = CVMThresholds(
        use_absolute_depth=args.use_absolute_depth,
        pixel_to_mm=args.pixel_to_mm,
        concavity_depth_mm_threshold=args.concavity_threshold_mm,
        concavity_ratio_threshold=args.concavity_threshold,
        trapezoid_taper_threshold=args.taper_threshold,
        trapezoid_si_threshold=args.trapezoid_si_threshold,
        rect_horizontal_si_threshold=args.rect_horizontal_threshold,
        rect_vertical_si_threshold=args.rect_vertical_threshold,
        enable_fuzzy_hysteresis=args.enable_fuzzy_hysteresis,
        concavity_hysteresis_mm=args.concavity_hysteresis_mm,
        shape_fuzzy_margin=args.shape_fuzzy_margin,
        strict_biological_hierarchy=args.strict_biological_hierarchy,
    )

    print(f"[CVM Detection] Running inference on: {args.image}...")
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
        import json
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
