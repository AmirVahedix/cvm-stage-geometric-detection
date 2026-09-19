import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union, List


@dataclass
class Point:
    """Represents a 2D landmark point with x and y coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_dict(cls, data: Union[Dict[str, float], Tuple[float, float], list]) -> 'Point':
        """Utility constructor from various data formats."""
        if isinstance(data, dict):
            return cls(x=float(data['x']), y=float(data['y']))
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            return cls(x=float(data[0]), y=float(data[1]))
        raise ValueError(f"Cannot parse Point from data: {data}")


@dataclass
class VertebraC2:
    """Represents C2 vertebra landmarks (3 points)."""
    inferior_posterior: Point
    inferior_concavity: Point
    inferior_anterior: Point

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VertebraC2':
        return cls(
            inferior_posterior=Point.from_dict(data['inferior-posterior']),
            inferior_concavity=Point.from_dict(data['inferior-concavity']),
            inferior_anterior=Point.from_dict(data['inferior-anterior']),
        )


@dataclass
class VertebraC3C4:
    """Represents C3 or C4 vertebra landmarks (5 points)."""
    inferior_posterior: Point
    inferior_concavity: Point
    inferior_anterior: Point
    superior_posterior: Point
    superior_anterior: Point

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VertebraC3C4':
        return cls(
            inferior_posterior=Point.from_dict(data['inferior-posterior']),
            inferior_concavity=Point.from_dict(data['inferior-concavity']),
            inferior_anterior=Point.from_dict(data['inferior-anterior']),
            superior_posterior=Point.from_dict(data['superior-posterior']),
            superior_anterior=Point.from_dict(data['superior-anterior']),
        )


@dataclass
class CVMInput:
    """Represents the complete set of landmarks for C2, C3, and C4."""
    c2: VertebraC2
    c3: VertebraC3C4
    c4: VertebraC3C4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CVMInput':
        return cls(
            c2=VertebraC2.from_dict(data['C2']),
            c3=VertebraC3C4.from_dict(data['C3']),
            c4=VertebraC3C4.from_dict(data['C4']),
        )

    def to_list(self) -> List[Tuple[float, float]]:
        """Returns the 13 landmarks in standard clinical order."""
        return [
            self.c2.inferior_posterior.to_tuple(),
            self.c2.inferior_concavity.to_tuple(),
            self.c2.inferior_anterior.to_tuple(),
            self.c3.superior_posterior.to_tuple(),
            self.c3.superior_anterior.to_tuple(),
            self.c3.inferior_posterior.to_tuple(),
            self.c3.inferior_concavity.to_tuple(),
            self.c3.inferior_anterior.to_tuple(),
            self.c4.superior_posterior.to_tuple(),
            self.c4.superior_anterior.to_tuple(),
            self.c4.inferior_posterior.to_tuple(),
            self.c4.inferior_concavity.to_tuple(),
            self.c4.inferior_anterior.to_tuple(),
        ]

    def to_flat_coords(self) -> List[float]:
        """Returns the 26 coordinate numbers (x, y) flattened."""
        return [coord for pt in self.to_list() for coord in pt]


@dataclass
class CVMThresholds:
    """Configuration thresholds for shape and concavity classification."""
    # Concavity thresholds
    # If absolute_depth is used, concavity_depth_mm > concavity_depth_mm_threshold
    # Otherwise, relative concavity ratio (depth / base_length) > concavity_ratio_threshold
    use_absolute_depth: bool = False
    concavity_ratio_threshold: float = 0.05  # 5% of base length
    concavity_depth_mm_threshold: float = 1.0  # 1.0 mm (requires pixel_to_mm)
    pixel_to_mm: Optional[float] = None

    # Shape ratios thresholds (C3 and C4)
    # Trapezoid: if anterior height / posterior height < trapezoid_height_ratio_threshold
    trapezoid_height_ratio_threshold: float = 0.90
    # Rectangular Horizontal: if width / height >= rect_horizontal_threshold
    rect_horizontal_threshold: float = 1.20
    # Square: if square_lower_threshold <= width / height < rect_horizontal_threshold
    # Rectangular Vertical: if width / height < rect_vertical_threshold
    rect_vertical_threshold: float = 0.85


# --- Geometric Helper Functions ---

def euclidean_distance(p1: Point, p2: Point) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Calculates the perpendicular distance from a point to the line segment connecting line_start and line_end."""
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    base_len = math.sqrt(dx ** 2 + dy ** 2)
    if base_len == 0:
        return euclidean_distance(point, line_start)
    
    # Perpendicular distance to line formula: |dx*(y1-y0) - dy*(x1-x0)| / base_len
    numerator = abs(dx * (line_start.y - point.y) - dy * (line_start.x - point.x))
    return numerator / base_len


# --- Feature Classification Functions ---

def calculate_concavity(
    ip: Point, ic: Point, ia: Point, thresholds: CVMThresholds
) -> Tuple[float, float, bool]:
    """
    Calculates the concavity depth and ratio of a vertebra.
    Returns: (depth, ratio, is_concave)
    """
    depth = perpendicular_distance(ic, ip, ia)
    base_len = euclidean_distance(ip, ia)
    ratio = depth / base_len if base_len > 0 else 0.0

    if thresholds.use_absolute_depth:
        if thresholds.pixel_to_mm is None:
            raise ValueError("pixel_to_mm must be set if use_absolute_depth is True.")
        depth_mm = depth * thresholds.pixel_to_mm
        is_concave = depth_mm >= thresholds.concavity_depth_mm_threshold
    else:
        is_concave = ratio >= thresholds.concavity_ratio_threshold

    return depth, ratio, is_concave


def calculate_shape(
    sp: Point, sa: Point, ip: Point, ia: Point, thresholds: CVMThresholds
) -> Dict[str, Any]:
    """
    Calculates shape metrics and classifies the shape of a C3/C4 vertebra.
    Returns a dictionary of metrics and the resulting shape class.
    """
    h_posterior = euclidean_distance(sp, ip)
    h_anterior = euclidean_distance(sa, ia)
    h_average = (h_posterior + h_anterior) / 2.0

    w_superior = euclidean_distance(sp, sa)
    w_inferior = euclidean_distance(ip, ia)
    w_average = (w_superior + w_inferior) / 2.0

    # Trapezoid tapering ratio: anterior height / posterior height
    taper_ratio = h_anterior / h_posterior if h_posterior > 0 else 1.0
    
    # Width-to-height ratio
    wh_ratio = w_average / h_average if h_average > 0 else 1.0

    # Classification logic
    if taper_ratio <= thresholds.trapezoid_height_ratio_threshold:
        shape_class = "Trapezoidal"
    elif wh_ratio >= thresholds.rect_horizontal_threshold:
        shape_class = "Rectangular Horizontal"
    elif wh_ratio < thresholds.rect_vertical_threshold:
        shape_class = "Rectangular Vertical"
    else:
        shape_class = "Square"

    return {
        "h_posterior": h_posterior,
        "h_anterior": h_anterior,
        "h_average": h_average,
        "w_superior": w_superior,
        "w_inferior": w_inferior,
        "w_average": w_average,
        "taper_ratio": taper_ratio,
        "wh_ratio": wh_ratio,
        "shape": shape_class,
    }


def classify_cvm_stage(
    input_data: Union[CVMInput, Dict[str, Any], Any],
    thresholds: Optional[CVMThresholds] = None,
    method: str = "rules",
    mlp_model: Optional[Any] = None,
    mlp_weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classifies the Cervical Vertebral Maturation (CVM) stage (CS1-CS6) based on 13 landmarks.
    
    Args:
        input_data: CVMInput containing the landmark points for C2, C3, and C4 (or raw landmark dictionary / array).
        thresholds: Configuration parameters for CVM classification when using geometric rules.
        method: Staging engine method. 'rules' (default) uses clinical geometric rules;
                'mlp' redirects model coordinates to the lightweight 2-layer MLP classifier
                (Ablation Variant iii: -Symbolic, Replaced Rules with MLP).
        mlp_model: Optional pre-loaded CVMStageMLP instance when method='mlp'.
        mlp_weights_path: Optional path to MLP checkpoint weights when method='mlp'.
        
    Returns:
        A dictionary containing:
            - 'stage': The predicted CVM stage (string, e.g., 'CS1' to 'CS6').
            - 'method': 'rules' or 'mlp'.
            - 'details': Detailed metrics and classifications for C2, C3, and C4 (or MLP details).
    """
    if method.lower() == "mlp":
        from src.cvm.mlp import classify_cvm_stage_mlp
        return classify_cvm_stage_mlp(input_data, model=mlp_model, weights_path=mlp_weights_path)

    if thresholds is None:
        thresholds = CVMThresholds()

    # 1. Evaluate C2 Concavity
    c2_depth, c2_ratio, c2_concave = calculate_concavity(
        input_data.c2.inferior_posterior,
        input_data.c2.inferior_concavity,
        input_data.c2.inferior_anterior,
        thresholds
    )

    # 2. Evaluate C3 Concavity & Shape
    c3_depth, c3_ratio, c3_concave = calculate_concavity(
        input_data.c3.inferior_posterior,
        input_data.c3.inferior_concavity,
        input_data.c3.inferior_anterior,
        thresholds
    )
    c3_shape_metrics = calculate_shape(
        input_data.c3.superior_posterior,
        input_data.c3.superior_anterior,
        input_data.c3.inferior_posterior,
        input_data.c3.inferior_anterior,
        thresholds
    )

    # 3. Evaluate C4 Concavity & Shape
    c4_depth, c4_ratio, c4_concave = calculate_concavity(
        input_data.c4.inferior_posterior,
        input_data.c4.inferior_concavity,
        input_data.c4.inferior_anterior,
        thresholds
    )
    c4_shape_metrics = calculate_shape(
        input_data.c4.superior_posterior,
        input_data.c4.superior_anterior,
        input_data.c4.inferior_posterior,
        input_data.c4.inferior_anterior,
        thresholds
    )

    # --- Rule Engine for CVM Stage Staging ---
    # We compile the features to determine the stage:
    shape3 = c3_shape_metrics["shape"]
    shape4 = c4_shape_metrics["shape"]

    # 1. Determine CVM Stage by concavities first, then refine with shape.
    # CS1: Flat C2, C3, C4. C3 & C4 are Trapezoidal.
    # CS2: C2 concave. C3 & C4 are Trapezoidal.
    # CS3: C2 & C3 concave. C3 & C4 are Trapezoidal or Rectangular Horizontal.
    # CS4: C2, C3, C4 all concave. C3 & C4 are Rectangular Horizontal.
    # CS5: C2, C3, C4 all concave. At least one is Square.
    # CS6: C2, C3, C4 all concave. At least one is Rectangular Vertical.
    
    # In case of discrepancies between concavity development and shape development:
    # We prioritize the concavities to narrow down the growth phase, but also check the shapes.
    if c2_concave and c3_concave and c4_concave:
        # High maturity stages: CS4, CS5, CS6
        if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
            stage = "CS6"
        elif shape3 == "Square" or shape4 == "Square":
            stage = "CS5"
        else:
            # Both horizontal or trapezoidal
            stage = "CS4"
    elif c2_concave and c3_concave:
        # CS3 stage: C2 & C3 are concave, C4 is flat.
        if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
            stage = "CS6"
        elif shape3 == "Square" or shape4 == "Square":
            stage = "CS5"
        else:
            stage = "CS3"
    elif c2_concave:
        # CS2 stage: C2 is concave, C3 & C4 are flat.
        if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
            stage = "CS6"
        elif shape3 == "Square" or shape4 == "Square":
            stage = "CS5"
        elif shape3 == "Rectangular Horizontal" and shape4 == "Rectangular Horizontal":
            stage = "CS4"
        else:
            stage = "CS2"
    else:
        # CS1 stage: C2 is flat.
        if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
            stage = "CS6"
        elif shape3 == "Square" or shape4 == "Square":
            stage = "CS5"
        elif c3_concave and c4_concave:
            stage = "CS4"
        elif c3_concave:
            stage = "CS3"
        else:
            stage = "CS1"

    details = {
        "C2": {
            "concavity_depth": c2_depth,
            "concavity_ratio": c2_ratio,
            "is_concave": c2_concave,
        },
        "C3": {
            "concavity_depth": c3_depth,
            "concavity_ratio": c3_ratio,
            "is_concave": c3_concave,
            "shape_metrics": c3_shape_metrics,
        },
        "C4": {
            "concavity_depth": c4_depth,
            "concavity_ratio": c4_ratio,
            "is_concave": c4_concave,
            "shape_metrics": c4_shape_metrics,
        },
    }

    return {
        "stage": stage,
        "method": "rules",
        "details": details,
    }
