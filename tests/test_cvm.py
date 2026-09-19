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
        self.thresholds = CVMThresholds()

    def test_calculate_concavity_flat(self):
        ip = Point(0, 0)
        ic = Point(5, 0.1)  # tiny concavity depth (0.1) relative to base length (10)
        ia = Point(10, 0)
        
        depth, ratio, is_concave = calculate_concavity(ip, ic, ia, self.thresholds)
        self.assertAlmostEqual(depth, 0.1)
        self.assertAlmostEqual(ratio, 0.01)
        self.assertFalse(is_concave)

    def test_calculate_concavity_concave(self):
        ip = Point(0, 0)
        ic = Point(5, 0.8)  # concavity depth 0.8 relative to base length 10 (ratio 0.08 >= 0.05)
        ia = Point(10, 0)
        
        depth, ratio, is_concave = calculate_concavity(ip, ic, ia, self.thresholds)
        self.assertAlmostEqual(depth, 0.8)
        self.assertAlmostEqual(ratio, 0.08)
        self.assertTrue(is_concave)

    def test_calculate_shape_trapezoidal(self):
        # Posterior height = 10, Anterior height = 7 (taper ratio = 0.7 <= 0.90)
        sp = Point(0, 10)
        sa = Point(10, 7)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Trapezoidal")
        self.assertAlmostEqual(metrics["taper_ratio"], 0.7)

    def test_calculate_shape_rectangular_horizontal(self):
        # Average height = 6, average width = 10 (wh_ratio = 10/6 = 1.67 >= 1.20)
        sp = Point(0, 6)
        sa = Point(10, 6)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Rectangular Horizontal")
        self.assertAlmostEqual(metrics["wh_ratio"], 1.67, places=2)

    def test_calculate_shape_square(self):
        # Average height = 10, average width = 10 (wh_ratio = 1.0)
        sp = Point(0, 10)
        sa = Point(10, 10)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Square")
        self.assertAlmostEqual(metrics["wh_ratio"], 1.0)

    def test_calculate_shape_rectangular_vertical(self):
        # Average height = 15, average width = 10 (wh_ratio = 10/15 = 0.67 < 0.85)
        sp = Point(0, 15)
        sa = Point(10, 15)
        ip = Point(0, 0)
        ia = Point(10, 0)
        
        metrics = calculate_shape(sp, sa, ip, ia, self.thresholds)
        self.assertEqual(metrics["shape"], "Rectangular Vertical")
        self.assertAlmostEqual(metrics["wh_ratio"], 0.67, places=2)


class TestCVMStaging(unittest.TestCase):
    def helper_make_c2(self, concave: bool) -> VertebraC2:
        ip = Point(0, 0)
        ia = Point(10, 0)
        ic = Point(5, 0.8 if concave else 0.1)
        return VertebraC2(ip, ic, ia)

    def helper_make_c3c4(self, concave: bool, shape: str) -> VertebraC3C4:
        ip = Point(0, 0)
        ia = Point(10, 0)
        ic = Point(5, 0.8 if concave else 0.1)
        
        if shape == "Trapezoidal":
            # posterior height = 10, anterior height = 7 (taper ratio = 0.7)
            sp = Point(0, 10)
            sa = Point(10, 7)
        elif shape == "Rectangular Horizontal":
            # average height = 6 (wh_ratio = 1.67)
            sp = Point(0, 6)
            sa = Point(10, 6)
        elif shape == "Square":
            # average height = 10 (wh_ratio = 1.0)
            sp = Point(0, 10)
            sa = Point(10, 10)
        elif shape == "Rectangular Vertical":
            # average height = 15 (wh_ratio = 0.67)
            sp = Point(0, 15)
            sa = Point(10, 15)
        else:
            raise ValueError(f"Unknown shape {shape}")
            
        return VertebraC3C4(ip, ic, ia, sp, sa)

    def test_cs1_stage(self):
        # CS1: All flat, C3 & C4 are trapezoidal
        c2 = self.helper_make_c2(concave=False)
        c3 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        c4 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS1")
        self.assertFalse(result["details"]["C2"]["is_concave"])
        self.assertFalse(result["details"]["C3"]["is_concave"])
        self.assertFalse(result["details"]["C4"]["is_concave"])

    def test_cs2_stage(self):
        # CS2: C2 concave, C3 & C4 flat. C3 & C4 are trapezoidal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        c4 = self.helper_make_c3c4(concave=False, shape="Trapezoidal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS2")
        self.assertTrue(result["details"]["C2"]["is_concave"])
        self.assertFalse(result["details"]["C3"]["is_concave"])
        self.assertFalse(result["details"]["C4"]["is_concave"])

    def test_cs3_stage(self):
        # CS3: C2 & C3 concave, C4 flat. C3 & C4 are rectangular horizontal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        c4 = self.helper_make_c3c4(concave=False, shape="Rectangular Horizontal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS3")
        self.assertTrue(result["details"]["C2"]["is_concave"])
        self.assertTrue(result["details"]["C3"]["is_concave"])
        self.assertFalse(result["details"]["C4"]["is_concave"])

    def test_cs4_stage(self):
        # CS4: C2, C3, C4 all concave. C3 & C4 are rectangular horizontal
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        c4 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS4")
        self.assertTrue(result["details"]["C2"]["is_concave"])
        self.assertTrue(result["details"]["C3"]["is_concave"])
        self.assertTrue(result["details"]["C4"]["is_concave"])
        self.assertEqual(result["details"]["C3"]["shape_metrics"]["shape"], "Rectangular Horizontal")
        self.assertEqual(result["details"]["C4"]["shape_metrics"]["shape"], "Rectangular Horizontal")

    def test_cs5_stage(self):
        # CS5: C2, C3, C4 all concave. At least one of C3 & C4 is square
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Square")
        c4 = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS5")
        
        # Test other vertebra being square
        c3_alt = self.helper_make_c3c4(concave=True, shape="Rectangular Horizontal")
        c4_alt = self.helper_make_c3c4(concave=True, shape="Square")
        cvm_input_alt = CVMInput(c2, c3_alt, c4_alt)
        result_alt = classify_cvm_stage(cvm_input_alt)
        self.assertEqual(result_alt["stage"], "CS5")

    def test_cs6_stage(self):
        # CS6: C2, C3, C4 all concave. At least one of C3 & C4 is rectangular vertical
        c2 = self.helper_make_c2(concave=True)
        c3 = self.helper_make_c3c4(concave=True, shape="Rectangular Vertical")
        c4 = self.helper_make_c3c4(concave=True, shape="Square")
        
        cvm_input = CVMInput(c2, c3, c4)
        result = classify_cvm_stage(cvm_input)
        self.assertEqual(result["stage"], "CS6")


if __name__ == "__main__":
    unittest.main()
