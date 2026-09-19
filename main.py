from src import CVMInput, classify_cvm_stage


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
    print("=" * 60)
    print(" CVM STAGE GEOMETRIC DETECTION DEMONSTRATION")
    print("=" * 60)

    # 1. Simulating landmark predictions for CVM Stage CS3:
    # - C2 has a concave inferior border
    # - C3 has a concave inferior border
    # - C4 has a flat inferior border
    # - C3 & C4 are rectangular horizontal
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
        }
    }

    print("\n[Input] Parsing mock landmark coordinates (CS3 simulated data)...")
    cvm_input = CVMInput.from_dict(mock_landmarks_cs3)

    # 2. Perform the classification
    result = classify_cvm_stage(cvm_input)

    # 3. Print the results
    print(f"\n[Result] Predicted maturation stage: {result['stage']}")
    
    details = result["details"]
    print_vertebra_details("C2", details["C2"])
    print_vertebra_details("C3", details["C3"])
    print_vertebra_details("C4", details["C4"])
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
