"""Tests for configuration settings and directory resolution."""

import unittest
from pathlib import Path

from config.settings import (
    ALGORITHM_KEYS,
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAVED_MODELS_DIR,
    SRC_DIR,
    SUPPORTED_ALGORITHMS,
    settings,
)


class TestConfig(unittest.TestCase):
    """Test suite for system configuration."""

    def test_base_directory_structure(self) -> None:
        """Verify all defined paths resolve to Path objects within the workspace."""
        for path_name, path_obj in [
            ("BASE_DIR", BASE_DIR),
            ("DATA_DIR", DATA_DIR),
            ("RAW_DATA_DIR", RAW_DATA_DIR),
            ("PROCESSED_DATA_DIR", PROCESSED_DATA_DIR),
            ("MODELS_DIR", MODELS_DIR),
            ("SAVED_MODELS_DIR", SAVED_MODELS_DIR),
            ("LOGS_DIR", LOGS_DIR),
            ("SRC_DIR", SRC_DIR),
            ("REPORTS_DIR", REPORTS_DIR),
        ]:
            self.assertIsInstance(path_obj, Path, f"{path_name} must be a Path instance")
            self.assertTrue(path_obj.is_absolute(), f"{path_name} must be an absolute path")

    def test_supported_algorithms_strictly_enforced(self) -> None:
        """Verify only the three authorized algorithms are registered."""
        expected_algorithms = [
            "Support Vector Machine (SVM)",
            "Multilayer Perceptron (MLP)",
            "Decision Tree",
        ]
        self.assertEqual(list(SUPPORTED_ALGORITHMS), expected_algorithms)
        self.assertEqual(list(ALGORITHM_KEYS), ["svm", "mlp", "decision_tree"])

    def test_hyperparameter_presences(self) -> None:
        """Verify default hyperparameters exist for each approved algorithm."""
        self.assertIn("kernel", settings.models.svm_params)
        self.assertIn("C", settings.models.svm_params)
        self.assertIn("hidden_layer_sizes", settings.models.mlp_params)
        self.assertIn("activation", settings.models.mlp_params)
        self.assertIn("criterion", settings.models.decision_tree_params)
        self.assertIn("max_depth", settings.models.decision_tree_params)

    def test_dataset_specification(self) -> None:
        """Verify CIC-IDS-2017 dataset configurations."""
        self.assertEqual(settings.dataset.name, "CIC-IDS-2017")
        self.assertGreater(len(settings.dataset.known_attack_classes), 0)
        self.assertIn("BENIGN", settings.dataset.known_attack_classes)
        self.assertIn("DoS Hulk", settings.dataset.known_attack_classes)
        self.assertEqual(settings.dataset.random_state, 42)

    def test_reporting_specification(self) -> None:
        """Verify CP-11 reporting configurations and directory."""
        self.assertEqual(settings.reporting.reports_dir, REPORTS_DIR)
        self.assertGreater(settings.reporting.max_bundle_size_mb, 0)
        self.assertTrue(settings.reporting.default_redact_ips)
        self.assertTrue(settings.reporting.default_include_analyst_notes)


if __name__ == "__main__":
    unittest.main()
