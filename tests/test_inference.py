import unittest
import os
from PIL import Image
from src.inference import CVMPredictor, LANDMARK_LABELS, draw_landmarks_on_image
from src.cvm_calculator import CVMThresholds


class TestCVMInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_path = "model/weights.pth"
        cls.test_image_path = "sample_xray.png"
        assert os.path.exists(cls.weights_path), "Model weights file not found"
        assert os.path.exists(cls.test_image_path), "Sample image not found"
        cls.predictor = CVMPredictor(weights_path=cls.weights_path, device="cpu")

    def test_landmark_prediction_structure(self):
        img = Image.open(self.test_image_path)
        landmarks_dict, landmarks_px, landmarks_norm = self.predictor.predict_landmarks(img)

        # Check count of landmarks
        self.assertEqual(len(landmarks_px), 13)
        self.assertEqual(len(landmarks_norm), 13)
        self.assertEqual(len(LANDMARK_LABELS), 13)

        # Check vertebra dictionary keys
        self.assertIn("C2", landmarks_dict)
        self.assertIn("C3", landmarks_dict)
        self.assertIn("C4", landmarks_dict)

        self.assertEqual(len(landmarks_dict["C2"]), 3)
        self.assertEqual(len(landmarks_dict["C3"]), 5)
        self.assertEqual(len(landmarks_dict["C4"]), 5)

        # Check normalized range
        for nx, ny in landmarks_norm:
            self.assertGreaterEqual(nx, 0.0)
            self.assertLessEqual(nx, 1.0)
            self.assertGreaterEqual(ny, 0.0)
            self.assertLessEqual(ny, 1.0)

        # Check pixel coordinates range
        W, H = img.size
        for px, py in landmarks_px:
            self.assertGreaterEqual(px, 0.0)
            self.assertLessEqual(px, W)
            self.assertGreaterEqual(py, 0.0)
            self.assertLessEqual(py, H)

    def test_end_to_end_prediction(self):
        result = self.predictor.predict(
            image_input=self.test_image_path,
            thresholds=CVMThresholds(),
            annotate=True,
            draw_labels=True,
        )

        # Valid stage
        self.assertIn(result.stage, ["CS1", "CS2", "CS3", "CS4", "CS5", "CS6"])

        # Details verified
        self.assertIn("C2", result.details)
        self.assertIn("C3", result.details)
        self.assertIn("C4", result.details)

        # Annotated image returned
        self.assertIsNotNone(result.annotated_image)
        self.assertEqual(result.annotated_image.size, result.image_size)

        # Dictionary serialization
        res_dict = result.to_dict()
        self.assertIn("stage", res_dict)
        self.assertIn("landmarks", res_dict)
        self.assertIn("details", res_dict)


if __name__ == "__main__":
    unittest.main()
