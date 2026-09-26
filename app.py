import os
from PIL import Image
import gradio as gr

from src.cvm_calculator import CVMThresholds
from src.inference import CVMPredictor, draw_landmarks_on_image

# Global predictor instance (loaded once)
PREDICTOR: CVMPredictor | None = None


def get_predictor() -> CVMPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        weights_path = os.getenv("CVM_WEIGHTS_PATH", "model/weights.pth")
        PREDICTOR = CVMPredictor(weights_path=weights_path)
    return PREDICTOR


def format_results_to_html(result: dict) -> str:
    """
    Renders a premium HTML dashboard breaking down CVM staging details.
    """
    stage = result["stage"]
    details = result["details"]

    stage_desc = {
        "CS1": "The lower borders of all the three vertebrae (C2, C3, and C4) are flat. The bodies of both C3 and C4 are trapezoidal in shape. Peak mandibular growth is expected to occur 2 years after this stage.",
        "CS2": "The lower border of C2 is concave. The bodies of both C3 and C4 are trapezoidal. Peak mandibular growth is expected to occur 1 year after this stage.",
        "CS3": "The lower borders of both C2 and C3 are concave. The bodies of both C3 and C4 are trapezoidal or rectangular horizontal. Peak mandibular growth begins during this stage.",
        "CS4": "The lower borders of C2, C3, and C4 are all concave. The bodies of both C3 and C4 are rectangular horizontal. Peak mandibular growth is completed at or before this stage.",
        "CS5": "The lower borders of C2, C3, and C4 are all concave. At least one of C3 or C4 is square. Peak mandibular growth has passed.",
        "CS6": "The lower borders of C2, C3, and C4 are all concave. At least one of C3 or C4 is rectangular vertical. Late/post-pubertal growth is complete.",
    }

    desc = stage_desc.get(stage, "Unknown CVM Stage")

    html = f"""
    <div style="font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; padding: 5px;">
        <div style="background: linear-gradient(135deg, #4A00E0, #8E2DE2); color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.8; font-weight: 600;">Prediction Result</div>
            <h2 style="margin: 5px 0 0 0; font-size: 32px; font-weight: 800; letter-spacing: 0.5px;">{stage}</h2>
            <p style="margin: 12px 0 0 0; font-size: 14px; opacity: 0.95; line-height: 1.6; font-weight: 500;">{desc}</p>
        </div>
        
        <h3 style="color: #1A202C; margin-bottom: 12px; font-size: 18px; border-bottom: 2px solid #E2E8F0; padding-bottom: 6px; font-weight: 700;">Vertebrae Analysis Breakdown</h3>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 16px;">
    """

    # C2 Vertebra Card
    c2 = details["C2"]
    c2_concave_status = (
        "<span style='color: #DD6B20; font-weight: 700;'>Concave (Notch)</span>"
        if c2["is_concave"]
        else "Flat"
    )
    s_calib = details.get("spatial_calibration_mm_per_px", 0.375)
    th_depth_mm = details.get("concavity_threshold_mm", 1.0)
    exact_match_badge = (
        "<div style='margin-top: 6px; display: inline-block; background-color: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;'>Table 2 Exact Rule Match</div>"
        if details.get("table_2_exact_match")
        else ""
    )

    html += f"""
            <div style="background-color: #F8FAFC; border-left: 5px solid #FF7F50; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-top: 1px solid #EDF2F7; border-right: 1px solid #EDF2F7; border-bottom: 1px solid #EDF2F7;">
                <div style="color: #FF7F50; font-weight: 700; font-size: 15px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Vertebra C2 (Axis)</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 13.5px; color: #4A5568;">
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600; width: 65%;">Inferior Concavity Depth:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{c2['concavity_depth_px']:.2f} px ({c2['concavity_depth_mm']:.2f} mm)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Concavity Threshold:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{th_depth_mm:.1f} mm (S = {s_calib:.3f} mm/px)</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; font-weight: 600;">Classification Status:</td>
                        <td style="padding: 6px 0; text-align: right;">{c2_concave_status}</td>
                    </tr>
                </table>
            </div>
    """

    # C3 and C4 Vertebrae Cards
    for vert_id, vert_color in [("C3", "#20B2AA"), ("C4", "#4169E1")]:
        v_data = details[vert_id]
        sh = v_data["shape_metrics"]
        v_concave_status = (
            f"<span style='color: {vert_color}; font-weight: 700;'>Concave (Notch)</span>"
            if v_data["is_concave"]
            else "Flat"
        )

        html += f"""
            <div style="background-color: #F8FAFC; border-left: 5px solid {vert_color}; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-top: 1px solid #EDF2F7; border-right: 1px solid #EDF2F7; border-bottom: 1px solid #EDF2F7;">
                <div style="color: {vert_color}; font-weight: 700; font-size: 15px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Vertebra {vert_id}</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 13.5px; color: #4A5568;">
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600; width: 65%;">Inferior Concavity Depth:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{v_data['concavity_depth_px']:.2f} px ({v_data['concavity_depth_mm']:.2f} mm)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Notch Status:</td>
                        <td style="padding: 6px 0; text-align: right;">{v_concave_status}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Shape Index SI (Ha+Hp)/(Ws+Wi):</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{sh['shape_index']:.3f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Taper Ratio TR (Ha/Hp):</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{sh['taper_ratio']:.3f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; font-weight: 700; color: #2D3748; font-size: 14.5px;">Classified Shape:</td>
                        <td style="padding: 6px 0; text-align: right; font-weight: 800; color: #2D3748; font-size: 14.5px;">{sh['shape']}</td>
                    </tr>
                </table>
            </div>
        """

    html += """
        </div>
    </div>
    """
    return html


def predict_cvm(
    image: Image.Image,
    calibration_s: float,
    threshold_concavity_mm: float,
    threshold_taper: float,
    threshold_trapezoid_si: float,
    threshold_horizontal_si: float,
    threshold_vertical_si: float,
    enable_fuzzy_hysteresis: bool = False,
):
    if image is None:
        return (
            None,
            "<div style='color: red; padding: 10px; font-weight: bold;'>Error: Please upload an image first.</div>",
        )

    # 1. Prepare thresholds from Section 2.5
    thresholds = CVMThresholds(
        use_absolute_depth=True,
        pixel_to_mm=calibration_s,
        concavity_depth_mm_threshold=threshold_concavity_mm,
        trapezoid_taper_threshold=threshold_taper,
        trapezoid_si_threshold=threshold_trapezoid_si,
        rect_horizontal_si_threshold=threshold_horizontal_si,
        rect_vertical_si_threshold=threshold_vertical_si,
        enable_fuzzy_hysteresis=enable_fuzzy_hysteresis,
    )

    # 2. Run real model inference and CVM classification
    predictor = get_predictor()
    result = predictor.predict(
        image_input=image,
        thresholds=thresholds,
        annotate=True,
    )

    # 3. Render HTML report
    html_report = format_results_to_html(
        {"stage": result.stage, "details": result.details}
    )

    return result.annotated_image, html_report


# --- Build Gradio Interface ---

custom_css = """
footer {visibility: hidden}
.gradio-container {
    background-color: #0F172A !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    border: none !important;
    color: white !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #4F46E5, #4338CA) !important;
}
"""

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 25px; padding-top: 15px; color: white;">
            <h1 style="font-size: 34px; font-weight: 800; margin: 0; background: linear-gradient(to right, #818CF8, #C084FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Cervical Vertebral Maturation (CVM) Stage Classifier
            </h1>
            <p style="font-size: 16px; color: #94A3B8; margin-top: 8px; max-width: 650px; margin-left: auto; margin-right: auto;">
                Upload a lateral cephalometric X-ray image to automatically detect the 13 cervical vertebrae landmarks with Cephalometric Swin-GCN and predict the skeletal maturation stage.
            </p>
        </div>
        """
    )

    with gr.Row():
        # Left column: Image Upload and Threshold Controls
        with gr.Column(scale=5):
            input_image = gr.Image(type="pil", label="Lateral Cephalometric X-ray Image")

            with gr.Accordion(label="CVM Classification Threshold Settings (Section 2.5)", open=False):
                th_calibration = gr.Slider(
                    minimum=0.10,
                    maximum=0.80,
                    step=0.005,
                    value=0.375,
                    label="Spatial Calibration Factor S (mm/pixel)",
                    info="Section 2.5.1 calibration factor (default: 0.375 mm/px)",
                )
                th_concavity = gr.Slider(
                    minimum=0.20,
                    maximum=2.50,
                    step=0.05,
                    value=1.0,
                    label="Concavity Depth Threshold (mm)",
                    info="Physical concavity threshold for notch presence (default: 1.0 mm)",
                )
                th_taper = gr.Slider(
                    minimum=1.00,
                    maximum=1.50,
                    step=0.01,
                    value=1.15,
                    label="Trapezoid Taper Ratio Threshold (TR >= threshold)",
                    info="Section 2.5.2: TR = Ha/Hp >= 1.15 for Trapezoidal (default: 1.15)",
                )
                th_trapezoid_si = gr.Slider(
                    minimum=0.50,
                    maximum=0.85,
                    step=0.01,
                    value=0.75,
                    label="Trapezoid Shape Index Threshold (SI <= threshold)",
                    info="Section 2.5.2: SI <= 0.75 for Trapezoidal (default: 0.75)",
                )
                th_horizontal = gr.Slider(
                    minimum=0.75,
                    maximum=0.95,
                    step=0.01,
                    value=0.85,
                    label="Rectangular Horizontal SI Threshold (SI <= threshold)",
                    info="Section 2.5.2: 0.75 < SI <= 0.85 for Rectangular Horizontal (default: 0.85)",
                )
                th_vertical = gr.Slider(
                    minimum=1.05,
                    maximum=1.35,
                    step=0.01,
                    value=1.15,
                    label="Rectangular Vertical SI Threshold (SI >= threshold)",
                    info="Section 2.5.2: SI >= 1.15 for Rectangular Vertical (default: 1.15)",
                )
                enable_fuzzy_hysteresis = gr.Checkbox(
                    value=False,
                    label="Enable Hysteresis Buffer & Fuzzy Transition Zone",
                    info="Apply concavity hysteresis buffer and fuzzy transition margin on vertebral shapes (default: False)",
                )

            predict_btn = gr.Button("Predict CVM Stage", variant="primary")


        # Right column: Visualization and Detailed Breakdown Report
        with gr.Column(scale=6):
            output_image = gr.Image(
                type="pil", label="Predicted Landmarks Overlay", interactive=False
            )
            output_report = gr.HTML(label="Detailed Analysis Report")

    # Wire up the prediction event
    predict_btn.click(
        fn=predict_cvm,
        inputs=[
            input_image,
            th_calibration,
            th_concavity,
            th_taper,
            th_trapezoid_si,
            th_horizontal,
            th_vertical,
            enable_fuzzy_hysteresis,
        ],
        outputs=[
            output_image,
            output_report,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"),
        css=custom_css,
    )
