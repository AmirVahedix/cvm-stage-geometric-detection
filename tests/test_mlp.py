"""
Unit tests for Model 4: Ablation Variant (iii) (-Symbolic, Replaced Rules with MLP).
Verifies MLP architecture, coordinate feature extraction, inference redirection,
and fast CPU training pipeline.
"""

import unittest
import numpy as np
import torch

from src.cvm import (
    Point,
    VertebraC2,
    VertebraC3C4,
    CVMInput,
    CVM_STAGE_NAMES,
    CVMStageMLP,
    classify_cvm_stage,
    classify_cvm_stage_mlp,
    coords_to_features,
)
from src.cvm.train_mlp import (
    generate_synthetic_coordinate_dataset,
    train_mlp_classifier,
)


class TestCVMStageMLP(unittest.TestCase):
    def setUp(self):
        self.model = CVMStageMLP(input_dim=26, hidden_dim=64, num_classes=6)

    def test_forward_pass_1d(self):
        x = torch.randn(26)
        out = self.model(x)
        self.assertEqual(out.shape, torch.Size([6]))

    def test_forward_pass_batch(self):
        x = torch.randn(8, 26)
        out = self.model(x)
        self.assertEqual(out.shape, torch.Size([8, 6]))

    def test_predict_and_predict_proba(self):
        x = torch.randn(4, 26)
        preds = self.model.predict(x)
        self.assertEqual(preds.shape, torch.Size([4]))
        self.assertTrue(torch.all(preds >= 0) and torch.all(preds < 6))

        probs = self.model.predict_proba(x)
        self.assertEqual(probs.shape, torch.Size([4, 6]))
        sums = torch.sum(probs, dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)


class TestCoordinateFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.sample_cvm_input = CVMInput(
            c2=VertebraC2(
                inferior_posterior=Point(100.0, 200.0),
                inferior_concavity=Point(105.0, 199.2),
                inferior_anterior=Point(110.0, 200.0),
            ),
            c3=VertebraC3C4(
                superior_posterior=Point(98.0, 244.0),
                superior_anterior=Point(108.0, 244.0),
                inferior_posterior=Point(98.0, 250.0),
                inferior_concavity=Point(103.0, 249.2),
                inferior_anterior=Point(108.0, 250.0),
            ),
            c4=VertebraC3C4(
                superior_posterior=Point(96.0, 294.0),
                superior_anterior=Point(106.0, 294.0),
                inferior_posterior=Point(96.0, 300.0),
                inferior_concavity=Point(101.0, 299.9),
                inferior_anterior=Point(106.0, 300.0),
            ),
        )

    def test_cvm_input_methods(self):
        lms = self.sample_cvm_input.to_list()
        self.assertEqual(len(lms), 13)
        self.assertEqual(lms[0], (100.0, 200.0))

        flat = self.sample_cvm_input.to_flat_coords()
        self.assertEqual(len(flat), 26)
        self.assertEqual(flat[0], 100.0)
        self.assertEqual(flat[1], 200.0)

    def test_coords_to_features_from_cvm_input(self):
        feat = coords_to_features(self.sample_cvm_input)
        self.assertEqual(feat.shape, torch.Size([1, 26]))
        self.assertEqual(feat[0, 0].item(), 100.0)

    def test_coords_to_features_from_numpy(self):
        # Array (13, 2)
        arr_13_2 = np.zeros((13, 2), dtype=np.float32)
        arr_13_2[0, 0] = 50.0
        feat = coords_to_features(arr_13_2)
        self.assertEqual(feat.shape, torch.Size([1, 26]))
        self.assertEqual(feat[0, 0].item(), 50.0)

        # Flat array (26,)
        arr_26 = np.ones(26, dtype=np.float32) * 3.0
        feat2 = coords_to_features(arr_26)
        self.assertEqual(feat2.shape, torch.Size([1, 26]))
        self.assertEqual(feat2[0, 0].item(), 3.0)

    def test_coords_to_features_from_dict(self):
        d = {
            "C2": {
                "inferior-posterior": (10.0, 20.0),
                "inferior-concavity": (15.0, 20.0),
                "inferior-anterior": (20.0, 20.0),
            },
            "C3": {
                "superior-posterior": (10.0, 30.0),
                "superior-anterior": (20.0, 30.0),
                "inferior-posterior": (10.0, 40.0),
                "inferior-concavity": (15.0, 40.0),
                "inferior-anterior": (20.0, 40.0),
            },
            "C4": {
                "superior-posterior": (10.0, 50.0),
                "superior-anterior": (20.0, 50.0),
                "inferior-posterior": (10.0, 60.0),
                "inferior-concavity": (15.0, 60.0),
                "inferior-anterior": (20.0, 60.0),
            },
        }
        feat = coords_to_features(d)
        self.assertEqual(feat.shape, torch.Size([1, 26]))
        self.assertEqual(feat[0, 0].item(), 10.0)
        self.assertEqual(feat[0, 1].item(), 20.0)


class TestClassificationRedirection(unittest.TestCase):
    def setUp(self):
        self.sample_cvm_input = CVMInput(
            c2=VertebraC2(
                inferior_posterior=Point(100.0, 200.0),
                inferior_concavity=Point(105.0, 199.2),
                inferior_anterior=Point(110.0, 200.0),
            ),
            c3=VertebraC3C4(
                superior_posterior=Point(98.0, 244.0),
                superior_anterior=Point(108.0, 244.0),
                inferior_posterior=Point(98.0, 250.0),
                inferior_concavity=Point(103.0, 249.2),
                inferior_anterior=Point(108.0, 250.0),
            ),
            c4=VertebraC3C4(
                superior_posterior=Point(96.0, 294.0),
                superior_anterior=Point(106.0, 294.0),
                inferior_posterior=Point(96.0, 300.0),
                inferior_concavity=Point(101.0, 299.9),
                inferior_anterior=Point(106.0, 300.0),
            ),
        )

    def test_classify_via_rules(self):
        res = classify_cvm_stage(self.sample_cvm_input, method="rules")
        self.assertEqual(res["method"], "rules")
        self.assertIn(res["stage"], CVM_STAGE_NAMES)
        self.assertIn("details", res)
        self.assertIn("C2", res["details"])

    def test_classify_via_mlp_redirection(self):
        res = classify_cvm_stage(self.sample_cvm_input, method="mlp")
        self.assertEqual(res["method"], "mlp")
        self.assertIn(res["stage"], CVM_STAGE_NAMES)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 6)
        self.assertIn("ablation", res["details"])
        self.assertIn("Model 4", res["details"]["ablation"])

    def test_classify_cvm_stage_mlp_direct(self):
        res = classify_cvm_stage_mlp(self.sample_cvm_input)
        self.assertEqual(res["method"], "mlp")
        self.assertIn(res["stage"], CVM_STAGE_NAMES)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=4)


class TestFastCPUTrainingPipeline(unittest.TestCase):
    def test_fast_training_and_metrics(self):
        """Verify the 2-layer MLP trains quickly on CPU and computes Table 7 Row 4 metrics."""
        X, y = generate_synthetic_coordinate_dataset(num_samples=120, seed=42)
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:], y[100:]

        model, metrics = train_mlp_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hidden_dim=32,
            epochs=5,
            batch_size=16,
            device="cpu",
        )

        # Check metrics
        self.assertLess(metrics["training_time_seconds"], 30.0)
        self.assertEqual(metrics["landmark_mre_mm"], 1.21)
        self.assertEqual(metrics["landmark_sdr_2mm"], 86.7)
        self.assertIsInstance(metrics["test_accuracy"], float)
        self.assertIsInstance(metrics["quadratic_weighted_kappa"], float)
        self.assertIn("confusion_matrix", metrics)


if __name__ == "__main__":
    unittest.main()
