import os
import math
from PIL import Image, ImageDraw
import gradio as gr

from src.cvm_calculator import (
    CVMInput,
    CVMThresholds,
    classify_cvm_stage,
    Point
)

def generate_mock_landmarks(W: float, H: float, stage: str) -> dict:
    """
    Generates mock coordinates for 13 landmarks scaled to the uploaded image size.
    The points are generated to mathematically result in the chosen CVM stage.
    """
    cx = W * 0.5
    w = W * 0.18  # vertebra width
    
    # Configure variables for the target CVM stage
    # C2 Concavity: flat (0.01) or concave (0.08)
    # C3 Concavity: flat (0.01) or concave (0.08)
    # C4 Concavity: flat (0.01) or concave (0.08)
    # C3 Shape / C4 Shape: "Trapezoidal", "Rectangular Horizontal", "Square", "Rectangular Vertical"
    if stage == "CS1":
        c2_concave = 0.01
        c3_concave = 0.01
        c4_concave = 0.01
        c3_shape = "Trapezoidal"
        c4_shape = "Trapezoidal"
    elif stage == "CS2":
        c2_concave = 0.08
        c3_concave = 0.01
        c4_concave = 0.01
        c3_shape = "Trapezoidal"
        c4_shape = "Trapezoidal"
    elif stage == "CS3":
        c2_concave = 0.08
        c3_concave = 0.08
        c4_concave = 0.01
        c3_shape = "Rectangular Horizontal"
        c4_shape = "Rectangular Horizontal"
    elif stage == "CS4":
        c2_concave = 0.08
        c3_concave = 0.08
        c4_concave = 0.08
        c3_shape = "Rectangular Horizontal"
        c4_shape = "Rectangular Horizontal"
    elif stage == "CS5":
        c2_concave = 0.08
        c3_concave = 0.08
        c4_concave = 0.08
        c3_shape = "Square"
        c4_shape = "Rectangular Horizontal"
    elif stage == "CS6":
        c2_concave = 0.08
        c3_concave = 0.08
        c4_concave = 0.08
        c3_shape = "Rectangular Vertical"
        c4_shape = "Square"
    else:  # Fallback to CS3
        c2_concave = 0.08
        c3_concave = 0.08
        c4_concave = 0.01
        c3_shape = "Rectangular Horizontal"
        c4_shape = "Rectangular Horizontal"

    # Define vertical centers
    y2_bot = H * 0.30
    y3_center = H * 0.52
    y4_center = H * 0.74

    def get_vertebra_points(y_center, shape, concavity_ratio):
        # Calculate heights based on shape class
        if shape == "Trapezoidal":
            # posterior height = w/1.3, anterior height = 0.75 * posterior height
            h_post = w / 1.3
            h_ant = 0.75 * h_post
        elif shape == "Rectangular Horizontal":
            h_post = h_ant = w / 1.5
        elif shape == "Square":
            h_post = h_ant = w / 1.0
        elif shape == "Rectangular Vertical":
            h_post = h_ant = w / 0.7
        else:
            h_post = h_ant = w / 1.5

        y_top_post = y_center - h_post / 2
        y_bot_post = y_center + h_post / 2
        y_top_ant = y_center - h_ant / 2
        y_bot_ant = y_center + h_ant / 2

        sp = (cx - w/2, y_top_post)
        sa = (cx + w/2, y_top_ant)
        ip = (cx - w/2, y_bot_post)
        ia = (cx + w/2, y_bot_ant)

        # concavity point calculation using normal vector
        dx = ia[0] - ip[0]
        dy = ia[1] - ip[1]
        base_len = math.sqrt(dx**2 + dy**2)
        depth = concavity_ratio * base_len

        mid_x = (ip[0] + ia[0]) / 2
        mid_y = (ip[1] + ia[1]) / 2

        # normal vector pointing upwards (y starts at top, so upward is -y)
        # to guarantee upward bending, we adjust sign using dx
        ic_x = mid_x + depth * (dy / base_len)
        ic_y = mid_y - depth * (dx / base_len)
        ic = (ic_x, ic_y)

        return {
            "superior-posterior": sp,
            "superior-anterior": sa,
            "inferior-posterior": ip,
            "inferior-anterior": ia,
            "inferior-concavity": ic
        }

    # C2 only has inferior landmarks
    c2_ip = (cx - w/2, y2_bot)
    c2_ia = (cx + w/2, y2_bot)
    c2_ic = (cx, y2_bot - c2_concave * w)

    c3_pts = get_vertebra_points(y3_center, c3_shape, c3_concave)
    c4_pts = get_vertebra_points(y4_center, c4_shape, c4_concave)

    return {
        "C2": {
            "inferior-posterior": c2_ip,
            "inferior-concavity": c2_ic,
            "inferior-anterior": c2_ia
        },
        "C3": c3_pts,
        "C4": c4_pts
    }

def draw_landmarks(image: Image.Image, landmarks: dict) -> Image.Image:
    """
    Plots the cervical vertebra landmarks and outlines on the image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    W, H = image.size

    # Responsive scaling for lines and dots
    line_w = max(2, int(min(W, H) * 0.006))
    dot_r = max(4, int(min(W, H) * 0.009))

    # Distinct colors matching the dashboard border
    c2_color = (255, 127, 80)   # Coral
    c3_color = (32, 178, 170)   # Teal / Light Sea Green
    c4_color = (65, 105, 225)   # Royal Blue

    c2 = landmarks["C2"]
    c3 = landmarks["C3"]
    c4 = landmarks["C4"]

    # Draw C2 inferior border outline
    draw.line([c2["inferior-posterior"], c2["inferior-concavity"], c2["inferior-anterior"]], fill=c2_color, width=line_w)

    # Draw C3 closed loop outline
    c3_outline = [
        c3["superior-posterior"],
        c3["superior-anterior"],
        c3["inferior-anterior"],
        c3["inferior-concavity"],
        c3["inferior-posterior"],
        c3["superior-posterior"]
    ]
    draw.line(c3_outline, fill=c3_color, width=line_w)

    # Draw C4 closed loop outline
    c4_outline = [
        c4["superior-posterior"],
        c4["superior-anterior"],
        c4["inferior-anterior"],
        c4["inferior-concavity"],
        c4["inferior-posterior"],
        c4["superior-posterior"]
    ]
    draw.line(c4_outline, fill=c4_color, width=line_w)

    # Helper function to draw circles
    def draw_dot(pt, color):
        draw.ellipse([pt[0] - dot_r, pt[1] - dot_r, pt[0] + dot_r, pt[1] + dot_r], fill=color, outline=(255, 255, 255), width=1)

    # Draw landmarks
    for pt in c2.values():
        draw_dot(pt, c2_color)
    for pt in c3.values():
        draw_dot(pt, c3_color)
    for pt in c4.values():
        draw_dot(pt, c4_color)

    return annotated

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
        "CS6": "The lower borders of C2, C3, and C4 are all concave. At least one of C3 or C4 is rectangular vertical. Late/post-pubertal growth is complete."
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
    c2_concave_status = "<span style='color: #DD6B20; font-weight: 700;'>Concave</span>" if details["C2"]["is_concave"] else "Flat"
    html += f"""
            <div style="background-color: #F8FAFC; border-left: 5px solid #FF7F50; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-top: 1px solid #EDF2F7; border-right: 1px solid #EDF2F7; border-bottom: 1px solid #EDF2F7;">
                <div style="color: #FF7F50; font-weight: 700; font-size: 15px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Vertebra C2 (Axis)</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 13.5px; color: #4A5568;">
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600; width: 65%;">Inferior Concavity Depth:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{details["C2"]["concavity_depth"]:.2f} px</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Inferior Concavity Ratio:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{details["C2"]["concavity_ratio"]:.3f}</td>
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
        v_concave_status = f"<span style='color: {vert_color}; font-weight: 700;'>Concave</span>" if v_data["is_concave"] else "Flat"

        html += f"""
            <div style="background-color: #F8FAFC; border-left: 5px solid {vert_color}; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-top: 1px solid #EDF2F7; border-right: 1px solid #EDF2F7; border-bottom: 1px solid #EDF2F7;">
                <div style="color: {vert_color}; font-weight: 700; font-size: 15px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Vertebra {vert_id}</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 13.5px; color: #4A5568;">
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600; width: 65%;">Inferior Concavity Ratio:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{v_data["concavity_ratio"]:.3f} ({v_concave_status})</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Width-to-Height Ratio:</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{sh["wh_ratio"]:.3f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 6px 0; font-weight: 600;">Tapering Ratio (Ha/Hp):</td>
                        <td style="padding: 6px 0; text-align: right; font-family: monospace;">{sh["taper_ratio"]:.3f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; font-weight: 700; color: #2D3748; font-size: 14.5px;">Classified Shape:</td>
                        <td style="padding: 6px 0; text-align: right; font-weight: 800; color: #2D3748; font-size: 14.5px;">{sh["shape"]}</td>
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
    target_stage: str,
    threshold_concavity: float,
    threshold_taper: float,
    threshold_horizontal: float,
    threshold_vertical: float
):
    if image is None:
        return None, "<div style='color: red; padding: 10px; font-weight: bold;'>Error: Please upload an image first.</div>"
    
    # 1. Fetch width and height
    W, H = image.size
    
    # 2. Run mock model to generate 13 landmarks scaled to the image size
    landmarks = generate_mock_landmarks(W, H, target_stage)
    
    # 3. Create CVM thresholds configuration
    thresholds = CVMThresholds(
        use_absolute_depth=False,
        concavity_ratio_threshold=threshold_concavity,
        trapezoid_height_ratio_threshold=threshold_taper,
        rect_horizontal_threshold=threshold_horizontal,
        rect_vertical_threshold=threshold_vertical
    )
    
    # 4. Parse landmarks into CVMInput
    cvm_input = CVMInput.from_dict(landmarks)
    
    # 5. Run classification rules
    result = classify_cvm_stage(cvm_input, thresholds=thresholds)
    
    # 6. Plot landmarks on image
    annotated_image = draw_landmarks(image, landmarks)
    
    # 7. Render report details
    html_report = format_results_to_html(result)
    
    return annotated_image, html_report


# --- Build Gradio Interface ---

# We can create a default mock cephalometric image using PIL to make the demo self-contained
def create_sample_cephalometric():
    # Create a nice dark gray background lateral head/neck contour mock
    img = Image.new("RGB", (600, 800), color=(20, 24, 33))
    draw = ImageDraw.Draw(img)
    
    # Draw a soft white skeleton-like contour to simulate X-ray
    # Head contour
    draw.arc([50, 50, 450, 450], start=180, end=360, fill=(45, 55, 72), width=3)
    # Nose profile
    draw.line([(50, 250), (30, 280), (70, 310)], fill=(45, 55, 72), width=3)
    # Jaw outline
    draw.line([(70, 310), (100, 380), (250, 420), (350, 350)], fill=(45, 55, 72), width=3)
    
    # Cervical Vertebrae soft background guides
    draw.rectangle([250, 200, 360, 240], outline=(35, 40, 50), width=2) # C2
    draw.rectangle([240, 350, 360, 430], outline=(35, 40, 50), width=2) # C3
    draw.rectangle([230, 500, 360, 600], outline=(35, 40, 50), width=2) # C4
    
    return img

sample_img = create_sample_cephalometric()
sample_img_path = "sample_xray.png"
sample_img.save(sample_img_path)

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
                Upload a lateral cephalometric X-ray image to automatically predict the skeletal maturation stage using cervical vertebrae landmarks.
            </p>
        </div>
        """
    )
    
    with gr.Row():
        # Left column: Controls and Input
        with gr.Column(scale=5):
            input_image = gr.Image(type="pil", label="Lateral Cephalometric X-ray Image")
            
            target_stage = gr.Dropdown(
                choices=["CS1", "CS2", "CS3", "CS4", "CS5", "CS6"],
                value="CS3",
                label="Simulated Model Checkpoint Stage"
            )
            
            with gr.Accordion(label="CVM Classification Threshold Settings", open=False):
                th_concavity = gr.Slider(
                    minimum=0.01, maximum=0.20, step=0.01, value=0.05,
                    label="Concavity Ratio Threshold",
                    info="Threshold for inferior border concavity (default 5%)"
                )
                th_taper = gr.Slider(
                    minimum=0.50, maximum=0.99, step=0.01, value=0.90,
                    label="Trapezoid Taper Ratio Threshold",
                    info="Maximum taper ratio (Ha/Hp) to be classified as Trapezoidal (default 0.90)"
                )
                th_horizontal = gr.Slider(
                    minimum=1.00, maximum=1.50, step=0.05, value=1.20,
                    label="Rectangular Horizontal Threshold",
                    info="Minimum width-to-height ratio to be Rectangular Horizontal (default 1.20)"
                )
                th_vertical = gr.Slider(
                    minimum=0.60, maximum=0.95, step=0.05, value=0.85,
                    label="Rectangular Vertical Threshold",
                    info="Maximum width-to-height ratio to be Rectangular Vertical (default 0.85)"
                )
                
            predict_btn = gr.Button("Predict CVM Stage", variant="primary")
            
            # Example panel
            gr.Examples(
                examples=[[sample_img_path, "CS3", 0.05, 0.90, 1.20, 0.85]],
                inputs=[input_image, target_stage, th_concavity, th_taper, th_horizontal, th_vertical],
                label="Quick Demo Sample"
            )
            
        # Right column: Visualization and Detailed Breakdown Report
        with gr.Column(scale=6):
            output_image = gr.Image(type="pil", label="Predicted Landmarks Overlay", interactive=False)
            output_report = gr.HTML(label="Detailed Analysis Report")

    # Wire up the prediction event
    predict_btn.click(
        fn=predict_cvm,
        inputs=[
            input_image,
            target_stage,
            th_concavity,
            th_taper,
            th_horizontal,
            th_vertical
        ],
        outputs=[
            output_image,
            output_report
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"),
        css=custom_css
    )
