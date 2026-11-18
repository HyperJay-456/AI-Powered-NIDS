"""Comprehensive Test Suite for Reporting & Evidence Export (CP-11).

Covers Categories A through I:
  - Category A: Source Verification & Fixture Isolation
  - Category B: Deterministic Report Aggregation & Metric Calculation
  - Category C: Report Generation Lifecycle & Archetypes
  - Category D: Publication-Grade PDF Correctness & Readability
  - Category E: Structured Exports & CSV Formula Injection Protection
  - Category F: Evidence Bundle ZIP Creation & Path Traversal Safeguards
  - Category G: Report Registry Discovery & Cryptographic Verification
  - Category H: Privacy Controls & IP Redaction
  - Category I: UI Rendering & Empty State Handling
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import unittest
import zipfile

import pandas as pd
import pypdf

from src.explorer.review_models import AnalystDisposition, ReviewEvent, ReviewStatus
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
    EvidenceItem,
    EvidenceManifest,
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
from src.threat_intelligence.attack_models import MappingStatus, MappingStrength
from src.threat_intelligence.mapping_registry import MappingRegistry


class TestReportingSubsystem(unittest.TestCase):
    """Unit and integration tests for CP-11 Reporting & Evidence Export subsystem."""

    def setUp(self) -> None:
        """Create isolated sandbox test directories."""
        self.temp_dir = tempfile.mkdtemp(prefix="nids_test_rep_")
        self.base_path = Path(self.temp_dir)

        self.analyses_dir = self.base_path / "analyses"
        self.reviews_dir = self.base_path / "reviews"
        self.mappings_dir = self.base_path / "mappings"
        self.reports_dir = self.base_path / "reports"
        self.evaluations_dir = self.base_path / "evaluations"

        for p in (self.analyses_dir, self.reviews_dir, self.mappings_dir, self.reports_dir, self.evaluations_dir):
            p.mkdir(parents=True, exist_ok=True)

        # Setup standard synthetic CP-08 production run
        self.prod_run_id = "detect_run_20261118_prod_001"
        self._create_synthetic_detection_run(
            run_id=self.prod_run_id,
            is_fixture=False,
            row_count=100,
            has_corrupt_sha=False,
        )

        # Setup synthetic test fixture run
        self.fixture_run_id = "detect_run_20261118_fixture_002"
        self._create_synthetic_detection_run(
            run_id=self.fixture_run_id,
            is_fixture=True,
            row_count=20,
            has_corrupt_sha=False,
        )

        # Setup synthetic CP-09 reviews
        self.review_store = ReviewStore(reviews_dir=self.reviews_dir)
        self.review_store.add_review_event(
            detection_run_id=self.prod_run_id,
            source_dataset_id="ds_test",
            source_row_index=5,
            disposition=AnalystDisposition.POTENTIALLY_MALICIOUS,
            analyst_notes="Suspicious SYN flood from 192.168.1.105 targeting port 80",
            reviewer_label="Senior Analyst Alice",
        )
        self.review_store.add_review_event(
            detection_run_id=self.prod_run_id,
            source_dataset_id="ds_test",
            source_row_index=12,
            disposition=AnalystDisposition.FALSE_POSITIVE,
            analyst_notes="=cmd|'/C calc'!A0 legitimate internal vulnerability scanner",
            reviewer_label="Analyst Bob",
        )

    def tearDown(self) -> None:
        """Clean up isolated sandbox test directories."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_detection_run(
        self,
        run_id: str,
        is_fixture: bool = False,
        row_count: int = 50,
        has_corrupt_sha: bool = False,
    ) -> Path:
        """Helper to create a fully consistent CP-08 run directory."""
        run_dir = self.analyses_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        classes = ["BENIGN", "PortScan", "DoS Hulk", "FTP-Patator"]
        for i in range(row_count):
            lbl = classes[i % len(classes)]
            rows.append({
                "source_row_index": i,
                "source_ip": f"192.168.1.{100 + (i % 20)}",
                "destination_ip": "10.0.0.1",
                "source_port": 40000 + i,
                "destination_port": 80 if i % 2 == 0 else 21,
                "protocol": 6,
                "consensus_prediction": lbl,
                "prediction_svm": lbl,
                "prediction_mlp": lbl if i % 5 != 0 else "BENIGN",
                "prediction_decision_tree": lbl,
            })

        df = pd.DataFrame(rows)
        pred_path = run_dir / "predictions.csv"
        df.to_csv(pred_path, index=False)

        with open(pred_path, "rb") as f:
            pred_sha = hashlib.sha256(f.read()).hexdigest()

        if has_corrupt_sha:
            pred_sha = "0000000000000000000000000000000000000000000000000000000000000000"

        # Model comparison CSV
        comp_rows = [
            {"Aspect": "Predicted Attacks", "SVM": "25", "MLP": "20", "DecisionTree": "25"},
            {"Aspect": "Predicted Benign", "SVM": "25", "MLP": "30", "DecisionTree": "25"},
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
            "source_filename": "traffic_capture.csv",
            "engineering_run_id": "eng_run_001",
            "is_test_fixture": is_fixture,
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
    # Category A: Source Verification & Fixture Isolation
    # ==========================================================================

    def test_verify_and_load_detection_run_valid(self) -> None:
        """Verify loading genuine CP-08 detection artifacts succeeds."""
        manifest, df_pred, df_comp = verify_and_load_detection_run(
            detection_run_id=self.prod_run_id,
            analyses_dir=self.analyses_dir,
            allow_test_fixtures=False,
        )
        self.assertEqual(manifest["detection_run_id"], self.prod_run_id)
        self.assertEqual(len(df_pred), 100)
        self.assertIsNotNone(df_comp)

    def test_fixture_isolation_prevents_unauthorized_production_report(self) -> None:
        """Verify test fixture run is rejected when allow_test_fixtures is False."""
        with self.assertRaises(ValueError) as ctx:
            verify_and_load_detection_run(
                detection_run_id=self.fixture_run_id,
                analyses_dir=self.analyses_dir,
                allow_test_fixtures=False,
            )
        self.assertIn("test fixture", str(ctx.exception).lower())

    def test_fixture_allowed_when_explicitly_flagged(self) -> None:
        """Verify test fixture run is permitted when allow_test_fixtures is True."""
        manifest, df_pred, _ = verify_and_load_detection_run(
            detection_run_id=self.fixture_run_id,
            analyses_dir=self.analyses_dir,
            allow_test_fixtures=True,
        )
        self.assertEqual(manifest["detection_run_id"], self.fixture_run_id)
        self.assertEqual(len(df_pred), 20)

    def test_checksum_corruption_detection(self) -> None:
        """Verify SHA-256 mismatch in predictions.csv raises ValueError."""
        corrupt_id = "detect_run_corrupt_001"
        self._create_synthetic_detection_run(
            run_id=corrupt_id,
            is_fixture=False,
            row_count=10,
            has_corrupt_sha=True,
        )
        with self.assertRaises(ValueError) as ctx:
            verify_and_load_detection_run(
                detection_run_id=corrupt_id,
                analyses_dir=self.analyses_dir,
            )
        self.assertIn("checksum mismatch", str(ctx.exception).lower())

    def test_path_traversal_prevention(self) -> None:
        """Verify directory traversal in detection_run_id is blocked."""
        with self.assertRaises(PermissionError):
            verify_and_load_detection_run(
                detection_run_id="../etc/passwd",
                analyses_dir=self.analyses_dir,
            )

    # ==========================================================================
    # Category B: Deterministic Report Aggregation
    # ==========================================================================

    def test_report_data_aggregator_metrics(self) -> None:
        """Verify accurate calculation of counts, percentages, and dispositions."""
        aggregator = ReportDataAggregator(
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
            evaluations_dir=self.evaluations_dir,
        )
        package = aggregator.aggregate(
            detection_run_id=self.prod_run_id,
            report_type=ReportType.EXECUTIVE,
            title="Executive Threat Summary",
            classification=ReportClassification.TLP_CLEAR,
        )

        meta = package.metadata
        self.assertEqual(meta.total_records_analyzed, 100)
        self.assertEqual(meta.predicted_attack_records + meta.predicted_benign_records, 100)
        self.assertEqual(meta.analyst_reviewed_count, 2)

        # Check prediction distribution
        dist = package.prediction_distribution
        self.assertGreater(len(dist), 0)
        total_dist_count = sum(d["count"] for d in dist)
        self.assertEqual(total_dist_count, 100)

        # Check analyst summary
        analyst = package.analyst_summary
        self.assertEqual(analyst["reviewed_records"], 2)
        self.assertEqual(analyst["unreviewed_records"], 98)
        self.assertEqual(analyst["disposition_counts"].get(AnalystDisposition.POTENTIALLY_MALICIOUS), 1)
        self.assertEqual(analyst["disposition_counts"].get(AnalystDisposition.FALSE_POSITIVE), 1)

    # ==========================================================================
    # Category C & D: Report Generation & PDF Correctness
    # ==========================================================================

    def test_all_report_archetypes(self) -> None:
        """Verify report generation for all 4 authorized report archetypes."""
        svc = ReportService(
            reports_dir=self.reports_dir,
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
        )
        for rtype in ReportType.ALL_TYPES:
            meta = svc.generate_report(
                detection_run_id=self.prod_run_id,
                report_type=rtype,
                title=f"Archetype Test - {rtype}",
                create_bundle=False,
            )
            self.assertEqual(meta.status, ReportStatus.COMPLETED)
            r_dir = self.reports_dir / meta.report_id
            pdf_path = r_dir / "report.pdf"
            self.assertTrue(pdf_path.exists())
            reader = pypdf.PdfReader(str(pdf_path))
            self.assertGreaterEqual(len(reader.pages), 1)

    def test_missing_manifest_raises_error(self) -> None:
        """Verify error when run_manifest.json is missing."""
        bad_run = self.analyses_dir / "detect_run_no_manifest"
        bad_run.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            verify_and_load_detection_run("detect_run_no_manifest", analyses_dir=self.analyses_dir)

    def test_missing_predictions_raises_error(self) -> None:
        """Verify error when predictions.csv is missing."""
        bad_run = self.analyses_dir / "detect_run_no_pred"
        bad_run.mkdir(parents=True, exist_ok=True)
        with open(bad_run / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"detection_run_id": "detect_run_no_pred", "status": "Completed"}, f)
        with self.assertRaises(FileNotFoundError):
            verify_and_load_detection_run("detect_run_no_pred", analyses_dir=self.analyses_dir)

    def test_corrupted_zip_integrity_verification(self) -> None:
        """Verify registry flags corrupted ZIP bundle."""
        svc = ReportService(
            reports_dir=self.reports_dir,
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
        )
        meta = svc.generate_report(detection_run_id=self.prod_run_id, create_bundle=True)
        bundle_path = self.reports_dir / meta.report_id / f"evidence_bundle_{meta.report_id}.zip"

        # Tamper with the zip file content
        with open(bundle_path, "wb") as f:
            f.write(b"CORRUPTED_BYTES")

        registry = ReportRegistry(reports_dir=self.reports_dir)
        res = registry.verify_report_integrity(meta.report_id)
        self.assertFalse(res["verified"])
        self.assertFalse(res["checks"].get("bundle_checksum", True))

    def test_unicode_and_long_notes_pdf_generation(self) -> None:
        """Verify PDF generation handles Unicode text and long notes without crashing."""
        aggregator = ReportDataAggregator(
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
        )
        long_notes = "A" * 1500 + " 🛡️ Chinese: 网络入侵检测 Spanish: Detección de intrusos"
        self.review_store.add_review_event(
            detection_run_id=self.prod_run_id,
            source_dataset_id="ds_test",
            source_row_index=20,
            disposition=AnalystDisposition.NEEDS_FURTHER_INVESTIGATION,
            analyst_notes=long_notes[:1800],
            reviewer_label="Reviewer λ",
        )
        pkg = aggregator.aggregate(
            detection_run_id=self.prod_run_id,
            report_type=ReportType.ANALYST_REVIEW,
            title="Unicode & Long Notes Test 🛡️",
        )
        gen = PDFReportGenerator()
        out_path, sha, size, pages = gen.generate(
            data_package=pkg,
            output_path=self.reports_dir / "unicode_test.pdf",
        )
        self.assertTrue(out_path.exists())
        self.assertGreater(size, 0)
        self.assertGreaterEqual(pages, 1)

    def test_pdf_generation_creates_valid_readable_document(self) -> None:
        """Verify PDF is generated, has valid structure, >= 1 pages, and expected text."""
        aggregator = ReportDataAggregator(
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
        )
        pkg = aggregator.aggregate(
            detection_run_id=self.prod_run_id,
            report_type=ReportType.TECHNICAL,
            title="Technical Threat Report",
        )

        gen = PDFReportGenerator()
        pdf_path = self.reports_dir / "test_report.pdf"
        out_path, sha, size, pages = gen.generate(data_package=pkg, output_path=pdf_path)

        self.assertTrue(out_path.exists())
        self.assertGreater(size, 0)
        self.assertGreaterEqual(pages, 1)

        # Inspect using pypdf
        reader = pypdf.PdfReader(str(out_path))
        self.assertEqual(len(reader.pages), pages)
        first_page_text = reader.pages[0].extract_text()

        self.assertIn("Technical Threat Report", first_page_text)
        self.assertIn("ACADEMIC / DEFENSIVE USE ONLY", first_page_text)

    def test_report_service_full_lifecycle(self) -> None:
        """Verify ReportService produces PDF, JSON, CSVs, and evidence ZIP bundle."""
        svc = ReportService(
            reports_dir=self.reports_dir,
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
        )

        meta = svc.generate_report(
            detection_run_id=self.prod_run_id,
            report_type=ReportType.CONSOLIDATED,
            title="Full Consolidated Audit",
            classification=ReportClassification.ACADEMIC,
            allow_test_fixtures=False,
            create_bundle=True,
        )

        self.assertEqual(meta.status, ReportStatus.COMPLETED)
        r_dir = self.reports_dir / meta.report_id
        self.assertTrue(r_dir.exists())

        # Verify files on disk
        self.assertTrue((r_dir / "report.pdf").exists())
        self.assertTrue((r_dir / "report_data.json").exists())
        self.assertTrue((r_dir / "evidence_manifest.json").exists())
        self.assertTrue((r_dir / f"evidence_bundle_{meta.report_id}.zip").exists())

        # Verify structured exports
        exports_dir = r_dir / "exports"
        self.assertTrue((exports_dir / "prediction_summary.csv").exists())
        self.assertTrue((exports_dir / "model_comparison.csv").exists())
        self.assertTrue((exports_dir / "analyst_review_summary.csv").exists())
        self.assertTrue((exports_dir / "attck_mapping_summary.csv").exists())

    # ==========================================================================
    # Category E: Structured Exports & Formula Injection Protection
    # ==========================================================================

    def test_csv_formula_injection_sanitization(self) -> None:
        """Verify formula-triggering characters (=, +, -, @) are sanitized with prefix single quote."""
        self.assertEqual(sanitize_csv_cell("=SUM(A1:A10)"), "'=SUM(A1:A10)")
        self.assertEqual(sanitize_csv_cell("+cmd|'/C calc'!A0"), "'+cmd|'/C calc'!A0")
        self.assertEqual(sanitize_csv_cell("-123"), "'-123")
        self.assertEqual(sanitize_csv_cell("@lookup"), "'@lookup")
        self.assertEqual(sanitize_csv_cell("BENIGN"), "BENIGN")
        self.assertEqual(sanitize_csv_cell(1234), 1234)

    # ==========================================================================
    # Category F: Evidence Bundle ZIP & Path Traversal
    # ==========================================================================

    def test_evidence_bundle_zip_integrity_and_contents(self) -> None:
        """Verify evidence bundle ZIP is valid, testzip() passes, and items match manifest."""
        svc = ReportService(
            reports_dir=self.reports_dir,
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
            mappings_dir=self.mappings_dir,
        )
        meta = svc.generate_report(
            detection_run_id=self.prod_run_id,
            report_type=ReportType.EXECUTIVE,
            create_bundle=True,
        )
        bundle_path = self.reports_dir / meta.report_id / f"evidence_bundle_{meta.report_id}.zip"
        self.assertTrue(bundle_path.exists())

        with zipfile.ZipFile(bundle_path, "r") as zf:
            self.assertIsNone(zf.testzip())
            namelist = zf.namelist()
            self.assertIn("report.pdf", namelist)
            self.assertIn("report_data.json", namelist)
            self.assertIn("evidence_manifest.json", namelist)
            self.assertIn("exports/prediction_summary.csv", namelist)

    def test_evidence_bundle_path_traversal_protection(self) -> None:
        """Verify packager rejects path traversal archive member names."""
        packager = EvidenceBundlePackager(report_dir=self.reports_dir)
        with self.assertRaises(PermissionError):
            packager._sanitize_arcname("../secret.txt")

    # ==========================================================================
    # Category G: Report Registry Discovery & Verification
    # ==========================================================================

    def test_registry_discovery_and_verification(self) -> None:
        """Verify ReportRegistry discovers historical reports and checks integrity."""
        svc = ReportService(
            reports_dir=self.reports_dir,
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
        )
        meta = svc.generate_report(detection_run_id=self.prod_run_id, report_type=ReportType.EXECUTIVE)

        registry = ReportRegistry(reports_dir=self.reports_dir)
        reports_list = registry.list_reports(include_test_fixtures=True, verify_checksums=True)
        self.assertEqual(len(reports_list), 1)
        self.assertEqual(reports_list[0]["report_id"], meta.report_id)
        self.assertTrue(reports_list[0]["pdf_valid"])

        # Check integrity verification method
        res = registry.verify_report_integrity(meta.report_id)
        self.assertTrue(res["verified"])
        self.assertTrue(res["checks"]["pdf_checksum"])
        self.assertTrue(res["checks"]["bundle_checksum"])

    # ==========================================================================
    # Category H: Privacy Controls & IP Redaction
    # ==========================================================================

    def test_ip_redaction_utility(self) -> None:
        """Verify IPv4 host portion is masked properly."""
        self.assertEqual(redact_ip_address("192.168.1.100"), "192.168.1.xxx")
        self.assertEqual(redact_ip_address("10.0.0.1"), "10.0.0.xxx")
        self.assertEqual(redact_ip_address(""), "")
        self.assertEqual(redact_ip_address("NonIPString"), "NonIPString")

    def test_analyst_notes_exclusion_policy(self) -> None:
        """Verify analyst notes are stripped when include_analyst_notes is False."""
        aggregator = ReportDataAggregator(
            analyses_dir=self.analyses_dir,
            reviews_dir=self.reviews_dir,
        )
        redact_cfg = RedactionConfig(include_analyst_notes=False)
        pkg = aggregator.aggregate(
            detection_run_id=self.prod_run_id,
            redaction_config=redact_cfg,
        )
        events = pkg.analyst_summary.get("recent_events", [])
        self.assertGreater(len(events), 0)
        for ev in events:
            self.assertIn("EXCLUDED", ev["notes"])

    # ==========================================================================
    # Category I: UI Rendering Smoke Test
    # ==========================================================================

    def test_reports_page_render_smoke(self) -> None:
        """Verify dashboard/pages/reports.py render function executes cleanly."""
        from dashboard.pages.reports import render
        try:
            render()
        except Exception as e:
            self.fail(f"dashboard/pages/reports.py render() raised unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
