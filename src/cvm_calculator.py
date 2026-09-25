import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union


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


@dataclass
class CVMThresholds:
    """Configuration thresholds for shape and concavity classification based on manuscript Section 2.5."""
    # Concavity thresholds (Section 2.5.1)
    # Using spatial calibration factor S = 0.375 mm/pixel
    # Presence: d_{c, k}^{mm} >= 1.0 mm
    use_absolute_depth: bool = True
    pixel_to_mm: float = 0.375  # S = 0.375 mm/pixel
    concavity_depth_mm_threshold: float = 1.0  # 1.0 mm
    concavity_ratio_threshold: float = 0.05  # Fallback relative ratio if use_absolute_depth=False

    # Shape ratio thresholds (C3 and C4, Section 2.5.2)
    # Trapezoidal: TR >= 1.15 or SI <= 0.75
    trapezoid_taper_threshold: float = 1.15
    trapezoid_si_threshold: float = 0.75
    # Rectangular Horizontal: 0.75 < SI <= 0.85 and TR < 1.15
    rect_horizontal_si_threshold: float = 0.85
    # Square: 0.90 <= SI <= 1.10
    square_si_lower: float = 0.90
    square_si_upper: float = 1.10
    # Rectangular Vertical: SI >= 1.15
    rect_vertical_si_threshold: float = 1.15

    # --- Hysteresis Buffer & Fuzzy Transition Parameters ---
    enable_fuzzy_hysteresis: bool = True
    concavity_hysteresis_mm: float = 0.15  # Hysteresis buffer +/- around concavity threshold (e.g. [0.85, 1.15] mm)
    concavity_ratio_hysteresis: float = 0.015  # Fallback relative buffer for concavity ratio
    shape_fuzzy_margin: float = 0.03  # Buffer margin for borderline shape ratios (e.g. trapezoid preservation)
    strict_biological_hierarchy: bool = True  # Monotonic progression (C2 notch precedes C3, C3 precedes C4)

    # Backward compatibility aliases
    @property
    def trapezoid_height_ratio_threshold(self) -> float:
        return self.trapezoid_taper_threshold

    @property
    def rect_horizontal_threshold(self) -> float:
        return self.rect_horizontal_si_threshold

    @property
    def rect_vertical_threshold(self) -> float:
        return self.rect_vertical_si_threshold


# --- Table 2 Ground-Truth Diagnostic Rule Table ---
TABLE_2_RULES = [
    {
        "stage": "CS1",
        "c2_notch": 0, "c3_notch": 0, "c4_notch": 0,
        "c3_shape": "Trapezoidal", "c4_shape": "Trapezoidal",
    },
    {
        "stage": "CS2",
        "c2_notch": 1, "c3_notch": 0, "c4_notch": 0,
        "c3_shape": "Trapezoidal", "c4_shape": "Trapezoidal",
    },
    {
        "stage": "CS3",
        "c2_notch": 1, "c3_notch": 1, "c4_notch": 0,
        "c3_shape": "Rectangular Horizontal", "c4_shape": "Trapezoidal",
    },
    {
        "stage": "CS4",
        "c2_notch": 1, "c3_notch": 1, "c4_notch": 1,
        "c3_shape": "Rectangular Horizontal", "c4_shape": "Rectangular Horizontal",
    },
    {
        "stage": "CS5",
        "c2_notch": 1, "c3_notch": 1, "c4_notch": 1,
        "c3_shape": "Square", "c4_shape": "Square",
    },
    {
        "stage": "CS6",
        "c2_notch": 1, "c3_notch": 1, "c4_notch": 1,
        "c3_shape": "Rectangular Vertical", "c4_shape": "Rectangular Vertical",
    },
]


# --- Geometric Helper Functions ---

def euclidean_distance(p1: Point, p2: Point) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """
    Calculates the perpendicular Euclidean distance from point (p_IC) to the
    inferior cortical baseline connecting line_start (p_PI) and line_end (p_AI):
    d_{c, k} = |(x_AI - x_PI)(y_PI - y_IC) - (x_PI - x_IC)(y_AI - y_PI)| / sqrt((x_AI - x_PI)^2 + (y_AI - y_PI)^2)
    """
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    base_len = math.sqrt(dx ** 2 + dy ** 2)
    if base_len == 0:
        return euclidean_distance(point, line_start)
    
    # Perpendicular distance to line formula (Section 2.5.1)
    numerator = abs(dx * (line_start.y - point.y) - (line_start.x - point.x) * dy)
    return numerator / base_len


# --- Feature Classification Functions ---

def compute_fuzzy_concavity(
    depth_val: float,
    threshold: float,
    margin: float,
) -> Dict[str, Any]:
    """
    Computes fuzzy membership degree and hysteresis transition state for concavity.
    Transition zone: [threshold - margin, threshold + margin].
    Returns:
        - degree: float in [0.0, 1.0] (0.0 = definitely flat, 1.0 = definitely concave)
        - state: 'definitely_flat', 'transition_zone', or 'definitely_concave'
        - is_definite_concave: bool (>= threshold + margin)
        - is_definite_flat: bool (< threshold - margin)
        - is_transition: bool (within buffer zone)
    """
    low = max(0.0, threshold - margin)
    high = threshold + margin
    span = high - low if high > low else 1e-6

    if depth_val <= low:
        deg = 0.0
        state = "definitely_flat"
    elif depth_val >= high:
        deg = 1.0
        state = "definitely_concave"
    else:
        deg = (depth_val - low) / span
        state = "transition_zone"

    return {
        "degree": round(float(deg), 4),
        "state": state,
        "is_definite_concave": depth_val >= high,
        "is_definite_flat": depth_val < low,
        "is_transition": low <= depth_val < high,
        "transition_lower": round(float(low), 3),
        "transition_upper": round(float(high), 3),
    }


def compute_fuzzy_shape(
    shape_index: float,
    taper_ratio: float,
    thresholds: CVMThresholds,
) -> Dict[str, Any]:
    """
    Computes continuous fuzzy membership degrees across the 4 CVM shape categories.
    """
    margin = thresholds.shape_fuzzy_margin if thresholds.enable_fuzzy_hysteresis else 0.0
    tr_th = thresholds.trapezoid_taper_threshold
    si_trap = thresholds.trapezoid_si_threshold

    # Trapezoidal affinity (tapering or small height-to-width)
    denom = 2 * margin if margin > 0 else 1e-6
    mu_tr = max(0.0, min(1.0, (taper_ratio - (tr_th - margin)) / denom)) if margin > 0 else (1.0 if taper_ratio >= tr_th else 0.0)
    mu_si = max(0.0, min(1.0, ((si_trap + margin) - shape_index) / denom)) if margin > 0 else (1.0 if shape_index <= si_trap else 0.0)
    mu_trap = max(mu_tr, mu_si)

    # Rectangular horizontal affinity (0.75 < SI <= 0.85)
    if shape_index <= si_trap - margin or shape_index >= thresholds.square_si_lower:
        mu_horiz = 0.0
    else:
        mu_horiz = max(0.0, min(1.0, 1.0 - abs(shape_index - 0.80) / 0.12))

    # Square affinity (0.90 <= SI <= 1.10)
    if shape_index <= thresholds.square_si_lower - margin or shape_index >= thresholds.square_si_upper + margin:
        mu_square = 0.0
    else:
        mu_square = max(0.0, min(1.0, 1.0 - abs(shape_index - 1.00) / 0.15))

    # Rectangular vertical affinity (SI >= 1.15)
    vert_th = thresholds.rect_vertical_si_threshold
    if shape_index <= vert_th - margin:
        mu_vert = 0.0
    else:
        mu_vert = max(0.0, min(1.0, (shape_index - (vert_th - margin)) / denom)) if margin > 0 else (1.0 if shape_index >= vert_th else 0.0)

    return {
        "membership_trapezoidal": round(float(mu_trap), 3),
        "membership_rect_horizontal": round(float(mu_horiz), 3),
        "membership_square": round(float(mu_square), 3),
        "membership_rect_vertical": round(float(mu_vert), 3),
    }


def calculate_concavity(
    ip: Point, ic: Point, ia: Point, thresholds: CVMThresholds
) -> Tuple[float, float, bool]:
    """
    Calculates the concavity depth and ratio of a vertebra following Section 2.5.1.
    Using spatial calibration factor S = 0.375 mm/pixel:
        d_{c, k}^{mm} = d_{c, k} * S
        C_k = 1 if d_{c, k}^{mm} >= 1.0 mm else 0
    Returns: (depth_px, ratio, is_concave)
    """
    depth_px = perpendicular_distance(ic, ip, ia)
    base_len = euclidean_distance(ip, ia)
    ratio = depth_px / base_len if base_len > 0 else 0.0

    s = thresholds.pixel_to_mm if thresholds.pixel_to_mm is not None else 0.375
    depth_mm = depth_px * s

    if thresholds.use_absolute_depth:
        is_concave = depth_mm >= thresholds.concavity_depth_mm_threshold
    else:
        is_concave = ratio >= thresholds.concavity_ratio_threshold

    return depth_px, ratio, is_concave


def calculate_shape(
    sp: Point, sa: Point, ip: Point, ia: Point, thresholds: CVMThresholds
) -> Dict[str, Any]:
    """
    Calculates morphometric dimensions and classifies the shape of a C3/C4 vertebra
    strictly following Section 2.5.2 of the manuscript, with optional fuzzy buffer transitions.

    Dimensions:
        H_a = ||p_AS - p_AI||_2 (anterior height)
        H_p = ||p_PS - p_PI||_2 (posterior height)
        W_s = ||p_PS - p_AS||_2 (superior width)
        W_i = ||p_PI - p_AI||_2 (inferior width)

    Indices:
        SI = (H_a + H_p) / (W_s + W_i)  [Shape Index]
        TR = H_a / H_p                  [Taper Ratio]

    Classification Rules (Section 2.5.2):
        - Trapezoidal (S_trap): TR >= 1.15 or SI <= 0.75
        - Rectangular Horizontal (S_horiz): 0.75 < SI <= 0.85 and TR < 1.15
        - Square (S_sq): 0.90 <= SI <= 1.10
        - Rectangular Vertical (S_vert): SI >= 1.15

    Buffer intervals (assigned to nearest categorical threshold):
        - (0.85, 0.90): midpoint 0.875 -> SI <= 0.875 maps to 0.85 (Rect. Horizontal), > 0.875 maps to 0.90 (Square)
        - (1.10, 1.15): midpoint 1.125 -> SI <= 1.125 maps to 1.10 (Square), > 1.125 maps to 1.15 (Rect. Vertical)
    """
    h_posterior = euclidean_distance(sp, ip)  # H_p
    h_anterior = euclidean_distance(sa, ia)   # H_a
    w_superior = euclidean_distance(sp, sa)   # W_s
    w_inferior = euclidean_distance(ip, ia)   # W_i

    width_sum = w_superior + w_inferior
    shape_index = (h_anterior + h_posterior) / width_sum if width_sum > 0 else 1.0
    taper_ratio = h_anterior / h_posterior if h_posterior > 0 else 1.0

    margin = thresholds.shape_fuzzy_margin if thresholds.enable_fuzzy_hysteresis else 0.0

    # Buffer midpoints computed dynamically from configured thresholds
    buffer_horiz_sq = (thresholds.rect_horizontal_si_threshold + thresholds.square_si_lower) / 2.0  # 0.875
    buffer_sq_vert = (thresholds.square_si_upper + thresholds.rect_vertical_si_threshold) / 2.0    # 1.125

    if (taper_ratio >= thresholds.trapezoid_taper_threshold - margin) or (shape_index <= thresholds.trapezoid_si_threshold + margin):
        shape_class = "Trapezoidal"
    elif shape_index <= buffer_horiz_sq:
        shape_class = "Rectangular Horizontal"
    elif shape_index <= buffer_sq_vert:
        shape_class = "Square"
    else:
        shape_class = "Rectangular Vertical"

    fuzzy_shape = compute_fuzzy_shape(shape_index, taper_ratio, thresholds)

    return {
        "h_posterior": h_posterior,
        "h_anterior": h_anterior,
        "w_superior": w_superior,
        "w_inferior": w_inferior,
        "h_average": (h_posterior + h_anterior) / 2.0,
        "w_average": (w_superior + w_inferior) / 2.0,
        "shape_index": shape_index,
        "taper_ratio": taper_ratio,
        "si": shape_index,
        "tr": taper_ratio,
        "wh_ratio": shape_index,  # Shape index is height/width as defined in Section 2.5.2
        "shape": shape_class,
        "fuzzy_memberships": fuzzy_shape,
    }


def classify_cvm_stage(
    input_data: CVMInput, thresholds: Optional[CVMThresholds] = None
) -> Dict[str, Any]:
    """
    Classifies the Cervical Vertebral Maturation (CVM) stage (CS1-CS6) based on 13 landmarks
    following Section 2.5 and Table 2 of the manuscript.
    
    Args:
        input_data: CVMInput containing the landmark points for C2, C3, and C4.
        thresholds: Configuration parameters for CVM classification. If None, default thresholds are used.
        
    Returns:
        A dictionary containing:
            - 'stage': The predicted CVM stage (string, e.g., 'CS1' to 'CS6').
            - 'details': Detailed metrics and classifications for C2, C3, and C4.
    """
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

    s = thresholds.pixel_to_mm if thresholds.pixel_to_mm is not None else 0.375
    c2_depth_mm = c2_depth * s
    c3_depth_mm = c3_depth * s
    c4_depth_mm = c4_depth * s

    th_depth = thresholds.concavity_depth_mm_threshold
    hyst_mm = thresholds.concavity_hysteresis_mm if thresholds.enable_fuzzy_hysteresis else 0.0

    c2_fuzzy = compute_fuzzy_concavity(c2_depth_mm, th_depth, hyst_mm)
    c3_fuzzy = compute_fuzzy_concavity(c3_depth_mm, th_depth, hyst_mm)
    c4_fuzzy = compute_fuzzy_concavity(c4_depth_mm, th_depth, hyst_mm)

    shape3 = c3_shape_metrics["shape"]
    shape4 = c4_shape_metrics["shape"]

    # --- Notch Resolution (with Hysteresis & Biological Hierarchy) ---
    if thresholds.enable_fuzzy_hysteresis:
        # 1. C2 notch resolution:
        if c2_depth_mm >= th_depth + hyst_mm:
            n2 = 1
        elif c2_depth_mm < th_depth - hyst_mm:
            n2 = 0
        else:
            n2 = 1 if c2_depth_mm >= th_depth else 0

        # 2. C3 notch resolution: requires prior C2 maturity (monotonicity)
        if thresholds.strict_biological_hierarchy and n2 == 0:
            n3 = 0
        elif c3_depth_mm >= th_depth + hyst_mm:
            n3 = 1
        elif c3_depth_mm < th_depth - hyst_mm:
            n3 = 0
        else:
            # Transition zone [th - margin, th + margin]
            # Ambiguous concavity requires shape maturation to confirm notch presence
            if shape3 != "Trapezoidal" and c3_depth_mm >= th_depth:
                n3 = 1
            else:
                n3 = 0

        # 3. C4 notch resolution: requires prior C3 maturity (monotonicity)
        if thresholds.strict_biological_hierarchy and n3 == 0:
            n4 = 0
        elif c4_depth_mm >= th_depth + hyst_mm:
            n4 = 1
        elif c4_depth_mm < th_depth - hyst_mm:
            n4 = 0
        else:
            # Transition zone [th - margin, th + margin]
            if shape4 != "Trapezoidal" and c4_depth_mm >= th_depth:
                n4 = 1
            else:
                n4 = 0
    else:
        n2 = 1 if c2_concave else 0
        n3 = 1 if c3_concave else 0
        n4 = 1 if c4_concave else 0

    # 1. Check exact match with Table 2
    exact_stage = None
    for rule in TABLE_2_RULES:
        if (
            rule["c2_notch"] == n2
            and rule["c3_notch"] == n3
            and rule["c4_notch"] == n4
            and rule["c3_shape"] == shape3
            and rule["c4_shape"] == shape4
        ):
            exact_stage = rule["stage"]
            break

    if exact_stage is not None:
        stage = exact_stage
    elif not thresholds.enable_fuzzy_hysteresis:
        # Legacy fallback logic for exact backward compatibility
        if c2_concave and c3_concave and c4_concave:
            if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
                stage = "CS6"
            elif shape3 == "Square" or shape4 == "Square":
                stage = "CS5"
            else:
                stage = "CS4"
        elif c2_concave and c3_concave:
            if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
                stage = "CS6"
            elif shape3 == "Square" or shape4 == "Square":
                stage = "CS5"
            else:
                stage = "CS3"
        elif c2_concave:
            if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
                stage = "CS6"
            elif shape3 == "Square" or shape4 == "Square":
                stage = "CS5"
            elif shape3 == "Rectangular Horizontal" and shape4 == "Rectangular Horizontal":
                stage = "CS4"
            elif shape3 == "Rectangular Horizontal":
                stage = "CS3"
            else:
                stage = "CS2"
        else:
            if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
                stage = "CS6"
            elif shape3 == "Square" or shape4 == "Square":
                stage = "CS5"
            elif c3_concave and c4_concave:
                stage = "CS4"
            elif c3_concave:
                stage = "CS3"
            elif shape3 == "Rectangular Horizontal":
                stage = "CS3"
            else:
                stage = "CS1"
    else:
        # Robust clinical hierarchical fallback with hysteresis & biological hierarchy
        # Mature adult vertebral shapes (CS5/CS6)
        if shape3 == "Rectangular Vertical" or shape4 == "Rectangular Vertical":
            stage = "CS6"
        elif shape3 == "Square" or shape4 == "Square":
            stage = "CS5"
        elif n2 == 1 and n3 == 1 and n4 == 1:
            stage = "CS4"
        elif n2 == 1 and n3 == 1:
            stage = "CS3"
        elif n2 == 1:
            stage = "CS2"
        else:
            stage = "CS1"

    details = {
        "spatial_calibration_mm_per_px": s,
        "concavity_threshold_mm": thresholds.concavity_depth_mm_threshold,
        "fuzzy_hysteresis_enabled": thresholds.enable_fuzzy_hysteresis,
        "concavity_hysteresis_mm": thresholds.concavity_hysteresis_mm if thresholds.enable_fuzzy_hysteresis else 0.0,
        "effective_notches": {"C2": n2, "C3": n3, "C4": n4},
        "C2": {
            "concavity_depth": c2_depth,
            "concavity_depth_px": c2_depth,
            "concavity_depth_mm": c2_depth_mm,
            "concavity_ratio": c2_ratio,
            "is_concave": bool(n2 == 1 if thresholds.enable_fuzzy_hysteresis else c2_concave),
            "raw_is_concave": c2_concave,
            "notch": n2,
            "fuzzy": c2_fuzzy,
        },
        "C3": {
            "concavity_depth": c3_depth,
            "concavity_depth_px": c3_depth,
            "concavity_depth_mm": c3_depth_mm,
            "concavity_ratio": c3_ratio,
            "is_concave": bool(n3 == 1 if thresholds.enable_fuzzy_hysteresis else c3_concave),
            "raw_is_concave": c3_concave,
            "notch": n3,
            "shape_metrics": c3_shape_metrics,
            "fuzzy": c3_fuzzy,
        },
        "C4": {
            "concavity_depth": c4_depth,
            "concavity_depth_px": c4_depth,
            "concavity_depth_mm": c4_depth_mm,
            "concavity_ratio": c4_ratio,
            "is_concave": bool(n4 == 1 if thresholds.enable_fuzzy_hysteresis else c4_concave),
            "raw_is_concave": c4_concave,
            "notch": n4,
            "shape_metrics": c4_shape_metrics,
            "fuzzy": c4_fuzzy,
        },
        "table_2_exact_match": exact_stage is not None,
    }

    return {
        "stage": stage,
        "details": details,
    }
