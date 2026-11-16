"""Integration and unit tests for CP-03 Application Dashboard."""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

from dashboard.navigation import ROUTES
from dashboard.status_checker import (
    check_dataset_status,
    check_latest_analysis_status,
    check_model_status,
    check_traffic_analysis_status,
    get_detection_pipeline_stages,
    get_project_milestones,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestDashboardOverview(unittest.TestCase):
    """Test suite for CP-03 Application Dashboard functionality and state verification."""

    def test_overview_renders_without_exceptions(self) -> None:
        """Verify the Overview Dashboard loads and renders cleanly in Streamlit."""
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0, f"Overview page raised exceptions: {app.exception}")
        self.assertEqual(app.session_state["nids_active_route"], "overview")

    def test_executive_summary_cards_verifiable_state(self) -> None:
        """Verify all four summary cards derive strictly from verifiable project state."""
        dataset_status = check_dataset_status()
        model_status = check_model_status()
        traffic_status = check_traffic_analysis_status()
        latest_status = check_latest_analysis_status()

        # In unpopulated state, assert expected non-fabricated text
        self.assertEqual(dataset_status.summary_text, "Dataset Not Loaded")
        self.assertEqual(model_status.summary_text, "No Trained Models")
        self.assertEqual(traffic_status.summary_text, "No Analysis Yet")
        self.assertEqual(latest_status.summary_text, "No Analysis Yet")

    def test_three_authorized_model_cards(self) -> None:
        """Verify exactly three authorized algorithms appear with untrained status."""
        model_status = check_model_status()
        self.assertEqual(model_status.total_models, 3)
        self.assertEqual(set(model_status.models.keys()), {"svm", "mlp", "decision_tree"})

        for key, model in model_status.models.items():
            self.assertFalse(model.is_trained)
            self.assertEqual(model.status_label, "No Trained Models")
            self.assertIsNone(model.artifact_path)

    def test_detection_pipeline_stages(self) -> None:
        """Verify detection pipeline stages count, titles, and non-operational pending status."""
        stages = get_detection_pipeline_stages()
        self.assertEqual(len(stages), 6)
        expected_checkpoints = ["CP-04", "CP-05", "CP-05", "CP-06", "CP-08 & CP-09", "CP-10"]

        for idx, stage in enumerate(stages):
            self.assertEqual(stage.status, "Pending")
            self.assertEqual(stage.target_checkpoint, expected_checkpoints[idx])

    def test_milestone_progression_consistency(self) -> None:
        """Verify all 12 milestones correctly reflect project progression."""
        milestones = get_project_milestones()
        self.assertEqual(len(milestones), 12)

        status_by_code = {m.code: m.status for m in milestones}
        self.assertEqual(status_by_code["CP-01"], "Completed")
        self.assertEqual(status_by_code["CP-02"], "Completed")
        self.assertEqual(status_by_code["CP-03"], "Active")

        for cp_num in range(4, 13):
            self.assertEqual(status_by_code[f"CP-{cp_num:02d}"], "Planned")


if __name__ == "__main__":
    unittest.main()
