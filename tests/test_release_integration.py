"""Release Integration & End-to-End Governance Test Suite for v1.0.0 (CP-12).

Validates the complete integrated product lifecycle, release version declarations,
three-algorithm invariant enforcement, cross-checkpoint data contracts,
CSV formula injection neutralization, and path-traversal safeguards.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List
import unittest
import zipfile

import numpy as np
import pandas as pd
import pypdf

from config.settings import ALGORITHM_KEYS, SUPPORTED_ALGORITHMS, settings
from dashboard.navigation import ROUTE_MAP, ROUTES
from src import __version__
from src.explorer.review_models import AnalystDisposition
from src.explorer.review_store import ReviewStore
from src.reporting.evidence_bundle import EvidenceBundlePackager
from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.report_data_aggregator import (
    ReportDataAggregator,
    redact_ip_address,
    sanitize_csv_cell,
    verify_and_load_detection_run,
)
from src.reporting.report_models import (
    RedactionConfig,
    ReportClassification,
    ReportDataPackage,
    ReportMetadata,
    ReportStatus,
    ReportType,
)
from src.reporting.report_registry import ReportRegistry
from src.reporting.report_service import ReportService
from src.reporting.structured_exporter import StructuredExporter
from src.threat_intelligence.mapping_registry import MappingRegistry


class TestReleaseIntegration(unittest.TestCase):
    """v1.0.0 Final Release Integration & Governance Verification."""

    def setUp(self) -> None:
        """Create isolated sandbox test workspace."""
        self.temp_dir = tempfile.mkdtemp(prefix="nids_release_test_")
        self.base_path = Path(self.temp_dir)

        self.analyses_dir = self.base_path / "analyses"
        self.reviews_dir = self.base_path / "reviews"
        self.mappings_dir = self.base_path / "mappings"
        self.reports_dir = self.base_path / "reports"

        for p in (self.analyses_dir, self.reviews_dir, self.mappings_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)

        # Build synthetic production run
        self.prod_run_id = "detect_run_release_v1_001"
        self._create_synthetic_detection_run(self.prod_run_id, row_count=60)

    def tearDown(self) -> None:
        """Clean up isolated sandbox test workspace."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_detection_run(self, run_id: str, row_count: int = 50) -> Path:
        """Helper to create a valid CP-08 run directory."""
        run_dir = self.analyses_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        classes = ["BENIGN", "PortScan", "DoS Hulk"]
        for i in range(row_count):
            lbl = classes[i % len(classes)]
            rows.append({
                "source_row_index": i,
                "source_ip": f"192.168.1.{100 + (i % 10)}",
                "destination_ip": "10.0.0.1",
                "source_port": 40000 + i,
                "destination_port": 80,
                "protocol": 6,
                "consensus_prediction": lbl,
                "prediction_svm": lbl,
                "prediction_mlp": lbl,
                "prediction_decision_tree": lbl,
            })

        df = pd.DataFrame(rows)
        pred_path = run_dir / "predictions.csv"
        df.to_csv(pred_path, index=False)

        with open(pred_path, "rb") as f:
            pred_sha = hashlib.sha256(f.read()).hexdigest()

        comp_rows = [
            {"Aspect": "Predicted Attacks", "SVM": "40", "MLP": "40", "DecisionTree": "40"},
            {"Aspect": "Predicted Benign", "SVM": "20", "MLP": "20", "DecisionTree": "20"},
        ]
        comp_df = pd.DataFrame(comp_rows)
        comp_path = run_dir / "model_comparison.csv"
        comp_df.to_csv(comp_path, index=False)

        with open(comp_path, "rb") as f:
            comp_sha = hashlib.sha256(f.read()).hexdigest()

        manifest = {
            "detection_run_id": run_id,
            "status": "Completed",
            "source_dataset_id": "dataset_cicids2017",
            "source_filename": "traffic_flow_test.csv",
            "engineering_run_id": "eng_run_001",
            "is_test_fixture": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_algorithms": ["SVM", "MLP", "Decision Tree"],
            "row_accounting": {
                "input_rows": row_count,
                "successfully_predicted_rows": row_count,
                "rejected_rows": 0,
            },
            "output_artifacts": {
                "predictions.csv": pred_sha,
                "model_comparison.csv": comp_sha,
            },
        }

        with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return run_dir

    # ==========================================================================
    # 1. Release Versioning Declarations
    # ==========================================================================

    def test_release_version_declarations(self) -> None:
        """Verify unified version 1.0.0 declaration across package and configuration."""
        self.assertEqual(__version__, "1.0.0")
        self.assertEqual(settings.version, "1.0.0")
        self.assertEqual(settings.ui.app_version, "1.0.0")

    # ==========================================================================
    # 2. Invariant: Strictly Three ML Algorithms
    # ==========================================================================

    def test_strictly_three_authorized_algorithms_enforced(self) -> None:
        """Verify only the three authorized algorithms (SVM, MLP, Decision Tree) exist."""
        expected = [
            "Support Vector Machine (SVM)",
            "Multilayer Perceptron (MLP)",
            "Decision Tree",
        ]
        self.assertEqual(list(SUPPORTED_ALGORITHMS), expected)
        self.assertEqual(list(ALGORITHM_KEYS), ["svm", "mlp", "decision_tree"])
        self.assertEqual(len(SUPPORTED_ALGORITHMS), 3)

    # ==========================================================================
    # 3. Complete End-to-End Lifecycle: Detection -> Reviews -> ATT&CK -> Reports
    # ==========================================================================

    def test_e2e_detection_to_evidence_package(self) -> None:
        """Verify full pipeline: CP-08 run -> CP-09 review -> CP-10 mapping -> CP-11 report & ZIP."""
        # Add analyst review
        r_store = ReviewStore(reviews_dir=self.reviews_dir)
        r_store.add_review_event(
            detection_run_id=self.prod_run_id,
            source_dataset_id="ds_test",
            source_row_index=1,
            disposition=AnalystDisposition.POTENTIALLY_MALICIOUS,
            analyst_notes="=cmd|'/C calc'!A0 verified attack flow from 192.168.1.101",
            reviewer_label="Lead Analyst",
        )

        svc = ReportService(
            reports_dir=self.reports_dir,
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
        )

        meta = svc.generate_report(
            detection_run_id=self.prod_run_id,
            report_type=ReportType.CONSOLIDATED,
            title="Final v1.0.0 Consolidated Audit",
            classification=ReportClassification.TLP_CLEAR,
            redaction_config=RedactionConfig(redact_ips=True, include_analyst_notes=True),
            allow_test_fixtures=False,
            create_bundle=True,
        )

        self.assertEqual(meta.status, ReportStatus.COMPLETED)
        r_dir = self.reports_dir / meta.report_id
        self.assertTrue(r_dir.exists())

        # 1. Verify PDF validity
        pdf_path = r_dir / "report.pdf"
        self.assertTrue(pdf_path.exists())
        reader = pypdf.PdfReader(str(pdf_path))
        self.assertGreaterEqual(len(reader.pages), 1)

        # 2. Verify structured report_data.json
        json_path = r_dir / "report_data.json"
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as jf:
            payload = json.load(jf)
        self.assertEqual(payload["metadata"]["report_id"], meta.report_id)
        self.assertEqual(payload["metadata"]["total_records_analyzed"], 60)

        # 3. Verify evidence bundle ZIP integrity
        bundle_path = r_dir / f"evidence_bundle_{meta.report_id}.zip"
        self.assertTrue(bundle_path.exists())
        with zipfile.ZipFile(bundle_path, "r") as zf:
            self.assertIsNone(zf.testzip())
            namelist = zf.namelist()
            self.assertIn("report.pdf", namelist)
            self.assertIn("report_data.json", namelist)
            self.assertIn("evidence_manifest.json", namelist)
            self.assertIn("exports/prediction_summary.csv", namelist)
            self.assertIn("exports/analyst_review_summary.csv", namelist)

        # 4. Verify CSV formula injection neutralization in exported CSVs
        csv_path = r_dir / "exports" / "analyst_review_summary.csv"
        self.assertTrue(csv_path.exists())
        csv_content = csv_path.read_text(encoding="utf-8")
        # Ensure raw formula trigger is neutralized
        self.assertNotIn("=cmd|'/C calc'", csv_content)

        # 5. Verify registry historical discovery and cryptographic integrity
        registry = ReportRegistry(reports_dir=self.reports_dir)
        reports = registry.list_reports(include_test_fixtures=False)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["report_id"], meta.report_id)
        ver_res = registry.verify_report_integrity(meta.report_id)
        self.assertTrue(ver_res["verified"])

    # ==========================================================================
    # 4. Security & Path Traversal Safeguards
    # ==========================================================================

    def test_path_traversal_rejection_across_subsystems(self) -> None:
        """Verify path traversal is rejected across detection, review, and report loaders."""
        with self.assertRaises(PermissionError):
            verify_and_load_detection_run("../escape", analyses_dir=self.analyses_dir)

        r_store = ReviewStore(reviews_dir=self.reviews_dir)
        with self.assertRaises(PermissionError):
            r_store._get_run_review_dir("../escape")

        registry = ReportRegistry(reports_dir=self.reports_dir)
        with self.assertRaises(PermissionError):
            registry._get_report_dir("../escape")

        packager = EvidenceBundlePackager(report_dir=self.reports_dir)
        with self.assertRaises(PermissionError):
            packager._sanitize_arcname("../../escape.txt")

    # ==========================================================================
    # 5. UI Route Integrity: All 8 Routes Available
    # ==========================================================================

    def test_all_eight_ui_routes_configured(self) -> None:
        """Verify all 8 authorized navigation routes are registered with callable renderers."""
        self.assertEqual(len(ROUTES), 8)
        expected_routes = {
            "overview",
            "traffic_analysis",
            "detection_center",
            "model_laboratory",
            "threat_intelligence",
            "prediction_explorer",
            "reports",
            "about",
        }
        self.assertEqual(set(ROUTE_MAP.keys()), expected_routes)
        for route in ROUTES:
            self.assertTrue(callable(route.render_fn))


if __name__ == "__main__":
    unittest.main()
