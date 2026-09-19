"""
Demonstration of CVM Stage Classification comparing:
1. Proposed Symbolic Clinical Geometric Rules
2. Model 4: Ablation Variant (iii) (-Symbolic, Replaced Rules with MLP)
"""

from src.cvm import (
    CVMInput,
    classify_cvm_stage,
    classify_cvm_stage_mlp,
)


def print_vertebra_details(name: str, data: dict):
    print(f"\n--- {name} Details ---")
    print(f"  Inferior Concavity depth: {data['concavity_depth']:.3f} px")
    print(f"  Inferior Concavity ratio: {data['concavity_ratio']:.3f} (threshold: 0.05)")
    print(f"  Is Concave?             : {data['is_concave']}")
    if "shape_metrics" in data:
        sh = data["shape_metrics"]
        print(f"  Posterior Height        : {sh['h_posterior']:.3f} px")
        print(f"  Anterior Height         : {sh['h_anterior']:.3f} px")
        print(f"  Average Height          : {sh['h_average']:.3f} px")
        print(f"  Average Width           : {sh['w_average']:.3f} px")
        print(f"  Tapering ratio (Ha/Hp)  : {sh['taper_ratio']:.3f} (threshold for trapezoid: <=0.90)")
        print(f"  Width-to-Height ratio   : {sh['wh_ratio']:.3f} (rect horizontal: >=1.20, vertical: <0.85)")
        print(f"  Vertebral Shape         : {sh['shape']}")


def main():
    print("=" * 75)
    print(" CVM STAGE CLASSIFICATION DEMONSTRATION & ABLATION STUDY")
    print(" Comparing: Clinical Geometric Rules vs. Ablation Variant (iii) (MLP)")
    print("=" * 75)

    # 1. Simulating landmark predictions from Model 1 for CVM Stage CS3:
    # 13 points (26 coordinate numbers)
    mock_landmarks_cs3 = {
        "C2": {
            "inferior-posterior": (100.0, 200.0),
            "inferior-concavity": (105.0, 199.2),  # depth = 0.8, ratio = 0.08 (concave)
            "inferior-anterior": (110.0, 200.0),
        },
        "C3": {
            "inferior-posterior": (98.0, 250.0),
            "inferior-concavity": (103.0, 249.2),  # depth = 0.8, ratio = 0.08 (concave)
            "inferior-anterior": (108.0, 250.0),
            "superior-posterior": (98.0, 244.0),
            "superior-anterior": (108.0, 244.0),   # width = 10, height = 6 (horizontal)
        },
        "C4": {
            "inferior-posterior": (96.0, 300.0),
            "inferior-concavity": (101.0, 299.9),  # depth = 0.1, ratio = 0.01 (flat)
            "inferior-anterior": (106.0, 300.0),
            "superior-posterior": (96.0, 294.0),
            "superior-anterior": (106.0, 294.0),   # width = 10, height = 6 (horizontal)
        },
    }

    print("\n[Input] Parsing Model 1 landmark coordinates (13 points, 26 numbers)...")
    cvm_input = CVMInput.from_dict(mock_landmarks_cs3)
    flat_coords = cvm_input.to_flat_coords()
    print(f"  -> Extracted 26 coordinates: {flat_coords[:6]} ... (total {len(flat_coords)})")

    # -------------------------------------------------------------
    # Method 1: Proposed Framework (Clinical Geometric Rules)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(" [1] Running Symbolic Clinical Geometric Rules...")
    result_rules = classify_cvm_stage(cvm_input, method="rules")
    print(f"  -> Method           : {result_rules.get('method', 'rules')}")
    print(f"  -> Predicted Stage  : {result_rules['stage']}")

    details = result_rules["details"]
    print_vertebra_details("C2", details["C2"])
    print_vertebra_details("C3", details["C3"])
    print_vertebra_details("C4", details["C4"])

    # -------------------------------------------------------------
    # Method 2: Ablation Variant (iii) (Redirecting coords to MLP)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(" [2] Running Ablation Variant (iii) (Redirecting output to 2-layer MLP)...")
    result_mlp = classify_cvm_stage(cvm_input, method="mlp")
    print(f"  -> Method           : {result_mlp['method']} ({result_mlp['details']['model_architecture']})")
    print(f"  -> Predicted Stage  : {result_mlp['stage']}")
    print(f"  -> Confidence       : {result_mlp['confidence'] * 100.0:.2f}%")
    print("  -> Class Probability Distribution:")
    for stg, prob in result_mlp["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"     {stg}: {prob * 100.0:5.1f}% | {bar}")

    print("\n" + "=" * 75)
    print(f" SUMMARY COMPARISON:")
    print(f"  - Hard Geometric Rules  : {result_rules['stage']}")
    print(f"  - MLP Classifier        : {result_mlp['stage']} (Confidence: {result_mlp['confidence']*100:.1f}%)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
