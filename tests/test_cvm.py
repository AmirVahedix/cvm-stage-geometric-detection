import unittest
from src.cvm_calculator import (
    Point,
    VertebraC2,
    VertebraC3C4,
    CVMInput,
    CVMThresholds,
    euclidean_distance,
    perpendicular_distance,
    calculate_concavity,
    calculate_shape,
    classify_cvm_stage,
)


class TestCVMGeometricFunctions(unittest.TestCase):
    def test_euclidean_distance(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        self.assertAlmostEqual(euclidean_distance(p1, p2), 5.0)
        
        # Test same point
        self.assertEqual(euclidean_distance(p1, p1), 0.0)

    def test_perpendicular_distance(self):
        # Line from (0, 0) to (10, 0)
        line_start = Point(0, 0)
        line_end = Point(10, 0)
        
        # Point (5, 3) should have perpendicular distance 3
        p = Point(5, 3)
        self.assertAlmostEqual(perpendicular_distance(p, line_start, line_end), 3.0)

        # Point (2, -4) should have perpendicular distance 4
        p2 = Point(2, -4)
        self.assertAlmostEqual(perpendicular_distance(p2, line_start, line_end), 4.0)

        # Point on the line should be 0
        p3 = Point(7, 0)
        self.assertAlmostEqual(perpendicular_distance(p3, line_start, line_end), 0.0)


class TestCVMFeatureClassification(unittest.TestCase):
    def setUp(self):
        # Default thresholds from Section 2.5: S = 0.375 mm/px, depth >= 1.0 mm
        self.thresholds = CVMThresholds(
            use_absolute_depth=True,
            pixel_to_mm=0.375,
            concavity_depth_mm_threshold=1.0,
        )

    def test_calculate_concavity_flat(self):
        # Base length = 10, depth = 1.0 px -> 1.0 * 0.375 = 0.375 mm < 1.0 mm -> Flat
        ip = Point(0, 0)
        ic = Point(5, 1.0)
        ia = Point(10, 0)
        
        depth, ratio, is_concave = calculate_concavity(ip, ic, ia, self.thresholds)
        self.assertAlmostEqual(depth, 1.0)
        self.assertAlmostEqual(ratio, 0.10)
        self.assertFalse(is_concave)

    def test_calculate_concavity_concave(self):
        # Base length = 10, depth = 3.0 px -> 3.0 * 0.375 = 1.125 mm >= 1.0 mm -> Concave
        ip = Point(0, 0)
        ic = Point(5, 3.0)
        ia = Point(10, 0)
        
        depth, ratio, is_concave = calculate_concavity(ip, ic, ia, self.thresholds)
        self.assertAlmostEqual(depth, 3.0)
        self.assertAlmostEqual(ratio, 0.30)
        self.assertTrue(is_concave)

    def test_calculate_shape_trapezoidal_by_si(self):
        # Ha = 5, Hp = 5, Ws = 10, Wi = 10 -> SI = 10 / 20 = 0.50 <= 0.75 -> Trapezoidal
        sp = Point(0, 5)
        sa = Point(10, 5)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Trapezoidal")
        self.assertAlmostEqual(metrics["shape_index"], 0.50)
        self.assertAlmostEqual(metrics["taper_ratio"], 1.0)

    def test_calculate_shape_trapezoidal_by_tr(self):
        # Ha = 12, Hp = 10, Ws = 10, Wi = 10 -> TR = 1.20 >= 1.15 -> Trapezoidal
        sp = Point(0, 10)
        sa = Point(10, 12)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Trapezoidal")
        self.assertAlmostEqual(metrics["taper_ratio"], 1.20)

    def test_calculate_shape_rectangular_horizontal(self):
        # Ha = 8, Hp = 8, Ws = 10, Wi = 10 -> SI = 16 / 20 = 0.80 (0.75 < SI <= 0.85, TR < 1.15)
        sp = Point(0, 8)
        sa = Point(10, 8)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Rectangular Horizontal")
        self.assertAlmostEqual(metrics["shape_index"], 0.80)
        self.assertAlmostEqual(metrics["taper_ratio"], 1.0)

    def test_calculate_shape_square(self):
        # Ha = 10, Hp = 10, Ws = 10, Wi = 10 -> SI = 20 / 20 = 1.00 (0.90 <= SI <= 1.10)
        sp = Point(0, 10)
        sa = Point(10, 10)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Square")
        self.assertAlmostEqual(metrics["shape_index"], 1.00)

    def test_calculate_shape_rectangular_vertical(self):
        # Ha = 12, Hp = 12, Ws = 10, Wi = 10 -> SI = 24 / 20 = 1.20 >= 1.15
        sp = Point(0, 12)
        sa = Point(10, 12)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Rectangular Vertical")
        self.assertAlmostEqual(metrics["shape_index"], 1.20)

    def test_buffer_intervals(self):
        # Buffer 1: (0.85, 0.90), midpoint = 0.875
        # SI = 17.4 / 20 = 0.87 <= 0.875 -> nearest is 0.85 -> Rectangular Horizontal
        sp1 = Point(0, 8.7)
        sa1 = Point(10, 8.7)
        ip = Point(0, 0)
        ia = Point(10, 0)
        m1 = calculate_shape(sp1, sa1, ip, ia, self.thresholds)
        self.assertEqual(m1["shape"], "Rectangular Horizontal")

        # SI = 17.6 / 20 = 0.88 > 0.875 -> nearest is 0.90 -> Square
        sp2 = Point(0, 8.8)
        sa2 = Point(10, 8.8)
        m2 = calculate_shape(sp2, sa2, ip, ia, self.thresholds)
        self.assertEqual(m2["shape"], "Square")

        # Buffer 2: (1.10, 1.15), midpoint = 1.125
        # SI = 22.2 / 20 = 1.11 <= 1.125 -> nearest is 1.10 -> Square
        sp3 = Point(0, 11.1)
        sa3 = Point(10, 11.1)
        m3 = calculate_shape(sp3, sa3, ip, ia, self.thresholds)
        self.assertEqual(m3["shape"], "Square")

        # SI = 22.8 / 20 = 1.14 > 1.125 -> nearest is 1.15 -> Rectangular Vertical
        sp4 = Point(0, 11.4)
        sa4 = Point(10, 11.4)
        m4 = calculate_shape(sp4, sa4, ip, ia, self.thresholds)
        self.assertEqual(m4["shape"], "Rectangular Vertical")


class TestCVMStaging(unittest.TestCase):
    def helper_make_c2(self, concave: bool) -> VertebraC2:
        ip = Point(0, 0)
        ia = Point(10, 0)
        # S = 0.375 mm/px: 3.0 px = 1.125 mm (concave), 0.5 px = 0.1875 mm (flat)
        ic = Point(5, 3.0 if concave else 0.5)
        return VertebraC2(ip, ic, ia)

    def helper_make_c3c4(self, concave: bool, shape: str) -> VertebraC3C4:
        ip = Point(0, 0)
        ia = Point(10, 0)
        ic = Point(5, 3.0 if concave else 0.5)
        
        if shape == "Trapezoidal":
            # SI = 10 / 20 = 0.50 <= 0.75
            sp = Point(0, 5)
            sa = Point(10, 5)
        elif shape == "Rectangular Horizontal":
            # SI = 16 / 20 = 0.80 (0.75 < SI <= 0.85)
            sp = Point(0, 8)
            sa = Point(10, 8)
        elif shape == "Square":
            # SI = 20 / 20 = 1.00 (0.90 <= SI <= 1.10)
            sp = Point(0, 10)
            sa = Point(10, 10)
        elif shape == "Rectangular Vertical":
            # SI = 24 / 20 = 1.20 >= 1.15
            sp = Point(0, 12)
            sa = Point(10, 12)
        else:
            raise ValueError(f"Unknown shape {shape}")
            
        return VertebraC3C4(ip, ic, ia, sp, sa)

    def test_cs1_stage(self):
        # Table 2: CS1 -> C2=0, C3=0, C4=0, C3=Trapezoidal, C4=Trapezoidal
        c2 = self.helper_make_c2(concave=False)
        c3 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        c4 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS1")
        self.assertTrue(result["details"]["table_2_exact_match"])
        self.assertFalse(result["details"]["C2"]["is_concave"])
        self.assertFalse(result["details"]["C3"]["is_concave"])
        self.assertFalse(result["details"]["C4"]["is_concave"])

    def test_cs2_stage(self):
        # Table 2: CS2 -> C2=1, C3=0, C4=0, C3=Trapezoidal, C4=Trapezoidal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        c4 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS2")
        self.assertTrue(result["details"]["table_2_exact_match"])
        self.assertTrue(result["details"]["C2"]["is_concave"])
        self.assertFalse(result["details"]["C3"]["is_concave"])
        self.assertFalse(result["details"]["C4"]["is_concave"])

    def test_cs3_stage(self):
        # Table 2: CS3 -> C2=1, C3=1, C4=0, C3=Rect. Horizontal, C4=Trapezoidal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        c4 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS3")
        self.assertTrue(result["details"]["table_2_exact_match"])
        self.assertTrue(result["details"]["C2"]["is_concave"])
        self.assertTrue(result["details"]["C3"]["is_concave"])
        self.assertFalse(result["details"]["C4"]["is_concave"])

    def test_cs4_stage(self):
        # Table 2: CS4 -> C2=1, C3=1, C4=1, C3=Rect. Horizontal, C4=Rect. Horizontal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        c4 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS4")
        self.assertTrue(result["details"]["table_2_exact_match"])
        self.assertTrue(result["details"]["C2"]["is_concave"])
        self.assertTrue(result["details"]["C3"]["is_concave"])
        self.assertTrue(result["details"]["C4"]["is_concave"])

    def test_cs5_stage(self):
        # Table 2: CS5 -> C2=1, C3=1, C4=1, C3=Square, C4=Square
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Square")
        c4 = self.helper_make_c3c4(concave=True, shape="Square")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS5")
        self.assertTrue(result["details"]["table_2_exact_match"])

    def test_cs5_stage_transitional(self):
        # Transitional CS5: C2=1, C3=1, C4=1, C3=Square, C4=Rectangular Horizontal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Square")
        c4 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS5")
        self.assertFalse(result["details"]["table_2_exact_match"])

    def test_cs6_stage(self):
        # Table 2: CS6 -> C2=1, C3=1, C4=1, C3=Rect. Vertical, C4=Rect. Vertical
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Vertical")
        c4 = self.helper_make_c3c4(concave=True, shape="Rectangular Vertical")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS6")
        self.assertTrue(result["details"]["table_2_exact_match"])

    def test_cs6_stage_transitional(self):
        # Transitional CS6: C2=1, C3=1, C4=1, C3=Rect. Vertical, C4=Square
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Vertical")
        c4 = self.helper_make_c3c4(concave=True, shape="Square")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS6")
        self.assertFalse(result["details"]["table_2_exact_match"])


if __name__ == "__main__":
    unittest.main()
