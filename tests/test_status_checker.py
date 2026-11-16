"""Unit tests for status_checker module."""

import tempfile
import unittest
from pathlib import Path

from dashboard.status_checker import (
    check_dataset_status,
    check_latest_analysis_status,
    check_model_status,
    check_traffic_analysis_status,
    get_detection_pipeline_stages,
    get_project_milestones,
)


class TestStatusChecker(unittest.TestCase):
    """Test suite for verifiable status derivation."""

    def test_dataset_status_empty_dir(self) -> None:
        """Verify dataset status when raw directory contains no CSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            status = check_dataset_status(raw_dir=Path(tmp_dir))
            self.assertFalse(status.is_loaded)
            self.assertEqual(status.file_count, 0)
            self.assertEqual(status.summary_text, "Dataset Not Loaded")
            self.assertEqual(status.available_files, [])

    def test_dataset_status_with_csv_files(self) -> None:
        """Verify dataset status dynamically identifies available CSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "Monday-WorkingHours.pcap_ISCX.csv"
            csv_path.write_text("Flow ID,Source IP", encoding="utf-8")

            status = check_dataset_status(raw_dir=Path(tmp_dir))
            self.assertTrue(status.is_loaded)
            self.assertEqual(status.file_count, 1)
            self.assertEqual(status.summary_text, "1 File Available")
            self.assertIn("Monday-WorkingHours.pcap_ISCX.csv", status.available_files)

    def test_model_status_no_models(self) -> None:
        """Verify model status when no model artifacts exist in the registry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            status = check_model_status(models_dir=Path(tmp_dir))
            self.assertEqual(status.trained_count, 0)
            self.assertEqual(status.total_models, 3)
            self.assertEqual(status.summary_text, "No Trained Models")

            # Check individual algorithm keys
            self.assertIn("svm", status.models)
            self.assertIn("mlp", status.models)
            self.assertIn("decision_tree", status.models)

            for key in ["svm", "mlp", "decision_tree"]:
                self.assertFalse(status.models[key].is_trained)
                self.assertEqual(status.models[key].status_label, "No Trained Models")

    def test_model_status_with_artifact(self) -> None:
        """Verify model status when a model artifact is placed in registry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "svm.joblib"
            model_file.write_bytes(b"dummy_serialized_model")

            status = check_model_status(models_dir=Path(tmp_dir))
            self.assertEqual(status.trained_count, 1)
            self.assertEqual(status.summary_text, "1/3 Models Available")
            self.assertTrue(status.models["svm"].is_trained)
            self.assertEqual(status.models["svm"].status_label, "Trained & Ready")
            self.assertFalse(status.models["mlp"].is_trained)
            self.assertFalse(status.models["decision_tree"].is_trained)

    def test_traffic_analysis_status_non_fabricated(self) -> None:
        """Verify traffic analysis status does not fabricate statistics."""
        status = check_traffic_analysis_status()
        self.assertFalse(status.is_available)
        self.assertEqual(status.summary_text, "No Analysis Yet")

    def test_latest_analysis_status_non_fabricated(self) -> None:
        """Verify latest analysis status displays empty state."""
        status = check_latest_analysis_status()
        self.assertFalse(status.has_run)
        self.assertEqual(status.summary_text, "No Analysis Yet")
        self.assertIsNone(status.timestamp)

    def test_detection_pipeline_stages_structure(self) -> None:
        """Verify detection pipeline stages count and pending statuses."""
        stages = get_detection_pipeline_stages()
        self.assertEqual(len(stages), 6)
        expected_ids = [
            "traffic_input",
            "preprocessing",
            "feature_engineering",
            "ml_classification",
            "prediction_results",
            "reports_export",
        ]
        self.assertEqual([s.stage_id for s in stages], expected_ids)
        for s in stages:
            self.assertEqual(s.status, "Pending")

    def test_project_milestones_count_and_current_cp03(self) -> None:
        """Verify exactly 12 milestones with CP-01/02 Completed, CP-03 Active, rest Planned."""
        milestones = get_project_milestones()
        self.assertEqual(len(milestones), 12)
        milestone_dict = {m.code: m for m in milestones}

        self.assertEqual(milestone_dict["CP-01"].status, "Completed")
        self.assertEqual(milestone_dict["CP-02"].status, "Completed")
        self.assertEqual(milestone_dict["CP-03"].status, "Active")

        for i in range(4, 13):
            code = f"CP-{i:02d}"
            self.assertEqual(milestone_dict[code].status, "Planned")


if __name__ == "__main__":
    unittest.main()
