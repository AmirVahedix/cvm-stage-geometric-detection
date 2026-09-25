from .cvm_calculator import (
    Point,
    VertebraC2,
    VertebraC3C4,
    CVMInput,
    CVMThresholds,
    classify_cvm_stage,
    euclidean_distance,
    perpendicular_distance,
)
def __getattr__(name: str):
    if name == "CephalometricSwinGCN":
        from .model import CephalometricSwinGCN
        return CephalometricSwinGCN
    if name in (
        "CVMPredictor",
        "PredictionResult",
        "LANDMARK_LABELS",
        "draw_landmarks_on_image",
        "visualize_with_matplotlib",
    ):
        from . import inference
        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Point",
    "VertebraC2",
    "VertebraC3C4",
    "CVMInput",
    "CVMThresholds",
    "classify_cvm_stage",
    "euclidean_distance",
    "perpendicular_distance",
    "CephalometricSwinGCN",
    "CVMPredictor",
    "PredictionResult",
    "LANDMARK_LABELS",
    "draw_landmarks_on_image",
    "visualize_with_matplotlib",
]
