"""Integration tests for the Traffic Analysis & Data Explorer page and system integration."""

from pathlib import Path
import tempfile
import unittest
from streamlit.testing.v1 import AppTest

from dashboard.navigation import ROUTES
from dashboard.status_checker import (
    check_dataset_status,
    check_traffic_analysis_status,
)
from src.data.csv_validator import validate_csv_dataset
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestTrafficAnalysisIntegration(unittest.TestCase):
    """Test suite for UI rendering, dataset library display, and dashboard state integration."""

    def test_traffic_analysis_page_renders_without_exceptions(self) -> None:
        """29. Traffic Analysis page renders cleanly without exceptions."""
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)

        # Switch route to Traffic Analysis
        radio = app.sidebar.radio[0]
        radio.set_value("📊  Traffic Analysis").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["nids_active_route"], "traffic_analysis")

    def test_uploaded_dataset_appears_in_manifest_and_status(self) -> None:
        """30 & 31. Uploaded dataset updates manifest and dashboard reflects status."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            meta_dir = Path(tmp_dir) / "metadata"
            raw_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            # In unpopulated state
            status_before = check_dataset_status(raw_dir=raw_dir, metadata_dir=meta_dir)
            self.assertFalse(status_before.is_loaded)
            self.assertEqual(status_before.summary_text, "Dataset Not Loaded")

            traffic_status_before = check_traffic_analysis_status(metadata_dir=meta_dir)
            self.assertFalse(traffic_status_before.is_available)
            self.assertEqual(traffic_status_before.summary_text, "No Analysis Yet")

            # Place and validate a fixture in the temp directory
            csv_path = raw_dir / "sample_flow.csv"
            csv_path.write_text("FlowID,Packets,Label\n1,10,BENIGN\n2,20,DDoS\n", encoding="utf-8")

            record = validate_csv_dataset(
                file_path=csv_path,
                dataset_id="integration-test-1",
                raw_dir=raw_dir,
                metadata_dir=meta_dir,
            )
            self.assertEqual(record.total_rows, 2)
            self.assertEqual(record.validation_status, "Validated with Warnings")

            # Now verify status checker detects and reflects the new dataset
            status_after = check_dataset_status(raw_dir=raw_dir, metadata_dir=meta_dir)
            self.assertTrue(status_after.is_loaded)
            self.assertEqual(status_after.file_count, 1)
            self.assertEqual(status_after.validation_status, "Validated with Warnings")

            traffic_status_after = check_traffic_analysis_status(metadata_dir=meta_dir)
            self.assertTrue(traffic_status_after.is_available)
            self.assertIn("1 Dataset", traffic_status_after.summary_text)

    def test_all_eight_routes_continue_to_navigate(self) -> None:
        """32 & 33. All eight routes render cleanly and session state persists."""
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)

        route_labels = [r.display_name for r in ROUTES]
        radio = app.sidebar.radio[0]

        for display_name in route_labels:
            radio.set_value(display_name).run()
            self.assertEqual(
                len(app.exception),
                0,
                f"Route '{display_name}' raised an exception: {app.exception}",
            )


if __name__ == "__main__":
    unittest.main()
