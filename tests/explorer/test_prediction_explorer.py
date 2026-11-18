"""Comprehensive Verification & Regression Suite for Prediction Explorer (CP-09).

Tests all 8 required functional verification categories:
  Category A: Detection Run Discovery & Eligibility Gating
  Category B: Prediction Reading & CP-08 Schema Compatibility
  Category C: Multi-Attribute Filtering, Sorting & Pagination
  Category D: Individual Record Inspection & Network Telemetry
  Category E: Model Consensus & Disagreement Exploration
  Category F: Human Analyst Review Workflow
  Category G: Audit Trail, Cryptographic Hash Chain & Zero-Mutation Invariant
  Category H: Dashboard Integration & UI Rendering
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from config.settings import AppConfig
from src.detection.detection_manifest import (
    DetectionRunManifest,
    ModelInferenceMetadata,
    RowAccountingRecord,
)
from src.explorer.disagreement_service import DisagreementService
from src.explorer.explorer_registry import EligibleRunRecord, ExplorerRegistry
from src.explorer.prediction_query import PredictionFilterParams, PredictionQueryEngine
from src.explorer.prediction_reader import PredictionReader
from src.explorer.record_inspector import RecordInspector
from src.explorer.review_models import (
    AnalystDisposition,
    ReviewEvent,
    ReviewStatus,
    ReviewSummary,
)
from src.explorer.review_service import ReviewService
from src.explorer.review_store import ReviewStore


def _create_mock_detection_run(
    analyses_dir: Path,
    run_id: str,
    status: str = "Completed",
    is_test_fixture: bool = False,
    is_multi_model: bool = True,
    has_ground_truth: bool = True,
    corrupt_predictions_sha: bool = False,
    omit_predictions_file: bool = False,
    omit_manifest: bool = False,
    omit_source_row_index: bool = False,
    warnings: list = None,
) -> Path:
    """Helper creating an isolated mock detection run directory."""
    run_dir = analyses_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if omit_manifest:
        return run_dir

    # Build predictions dataframe
    if is_multi_model:
        pred_data = {
            "source_row_index": [0, 1, 2, 3, 4],
            "svm_prediction": ["BENIGN", "PortScan", "DoS Hulk", "BENIGN", "DDoS"],
            "svm_score": [1.45, 2.12, 0.88, 1.95, 3.20],
            "mlp_prediction": ["BENIGN", "PortScan", "BENIGN", "BENIGN", "DDoS"],
            "mlp_score": [0.99, 0.87, 0.52, 0.96, 0.91],
            "decision_tree_prediction": ["BENIGN", "PortScan", "DoS Hulk", "SSH-Patator", "DDoS"],
            "decision_tree_score": [1.0, 0.95, 0.80, 0.70, 1.0],
            "agreement_status": ["Agreement", "Agreement", "Disagreement", "Disagreement", "Agreement"],
            "differing_models": ["None", "None", "MLP", "Decision Tree", "None"],
        }
    else:
        pred_data = {
            "source_row_index": [0, 1, 2, 3, 4],
            "predicted_class": ["BENIGN", "PortScan", "DoS Hulk", "BENIGN", "DDoS"],
            "model_score": [1.45, 2.12, 0.88, 1.95, 3.20],
            "score_type": ["decision_margin"] * 5,
        }

    if has_ground_truth:
        pred_data["ground_truth_label"] = ["BENIGN", "PortScan", "DoS Hulk", "BENIGN", "DDoS"]

    if omit_source_row_index:
        pred_data.pop("source_row_index", None)

    pred_df = pd.DataFrame(pred_data)
    pred_path = run_dir / "predictions.csv"

    output_artifacts = {}
    if not omit_predictions_file:
        pred_df.to_csv(pred_path, index=False)
        real_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
        output_artifacts["predictions.csv"] = (
            "0000000000000000000000000000000000000000000000000000000000000000"
            if corrupt_predictions_sha
            else real_sha
        )

    # Build manifest
    algos = ["svm", "mlp", "decision_tree"] if is_multi_model else ["svm"]
    models_meta = {
        algo: ModelInferenceMetadata(
            algorithm=algo,
            algorithm_display_name=algo.upper(),
            training_run_id=f"train_{algo}",
            model_checksum="dummy_checksum",
            engineering_run_id="eng_run_123",
            status="Completed",
            predicted_rows=5,
            score_type="decision_margin" if algo == "svm" else "uncalibrated_probabilities",
        )
        for algo in algos
    }

    manifest = DetectionRunManifest(
        detection_run_id=run_id,
        created_at_utc="2026-11-18T10:00:00Z",
        completed_at_utc="2026-11-18T10:01:00Z",
        status=status,
        source_dataset_id="cicids_sample",
        source_filename="sample_flows.csv",
        source_sha256="sample_sha256",
        source_validation_status="Valid",
        source_overlaps_training=False,
        source_overlap_details=None,
        has_ground_truth_target=has_ground_truth,
        ground_truth_column_name="Label" if has_ground_truth else None,
        selected_algorithms=algos,
        models=models_meta,
        engineering_run_id="eng_run_123",
        feature_schema_fingerprint="schema_fp",
        batch_size=10000,
        row_accounting=RowAccountingRecord(
            input_rows=5,
            successfully_predicted_rows=5,
            rejected_rows=0,
            skipped_rows=0,
        ),
        class_distribution={algo: {"BENIGN": 3, "PortScan": 1, "DoS Hulk": 1} for algo in algos},
        output_artifacts=output_artifacts,
        warnings=warnings or [],
        run_type="test_fixture" if is_test_fixture else "production",
        is_test_fixture=is_test_fixture,
    )

    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return run_dir


class TestPredictionExplorerSuite(unittest.TestCase):
    """Exhaustive test suite for CP-09 Prediction Explorer & Analyst Review."""

    def setUp(self) -> None:
        """Create isolated temporary workspaces for each test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.tmp_dir.name)
        self.analyses_dir = self.root_path / "data" / "analyses"
        self.reviews_dir = self.root_path / "data" / "analyst_reviews"
        self.raw_dir = self.root_path / "data" / "raw"

        self.analyses_dir.mkdir(parents=True, exist_ok=True)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.registry = ExplorerRegistry(analyses_dir=self.analyses_dir)
        self.review_store = ReviewStore(reviews_dir=self.reviews_dir)
        self.review_service = ReviewService(review_store=self.review_store)
        self.reader = PredictionReader(analyses_dir=self.analyses_dir, review_store=self.review_store)
        self.inspector = RecordInspector(
            prediction_reader=self.reader,
            review_store=self.review_store,
            raw_data_dir=self.raw_dir,
        )

    def tearDown(self) -> None:
        """Clean up temporary workspaces."""
        self.tmp_dir.cleanup()

    # --------------------------------------------------------------------------
    # CATEGORY A: Detection Run Discovery & Eligibility Gating
    # --------------------------------------------------------------------------

    def test_valid_completed_production_run(self) -> None:
        """Verify that a genuine completed production run passes 9-point verification."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_prod_01")
        runs = self.registry.list_eligible_runs()
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertTrue(r.is_eligible)
        self.assertEqual(r.detection_run_id, "detect_run_prod_01")
        self.assertEqual(r.status, "Completed")
        self.assertEqual(r.predicted_rows, 5)

    def test_completed_with_warnings_run(self) -> None:
        """Verify that a run marked Completed with Warnings is eligible."""
        _create_mock_detection_run(
            self.analyses_dir,
            "detect_run_warn_01",
            status="Completed with Warnings",
            warnings=["Training data overlap detected"],
        )
        runs = self.registry.list_eligible_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "Completed with Warnings")
        self.assertIn("Training data overlap detected", runs[0].warnings)

    def test_failed_and_incomplete_run_exclusion(self) -> None:
        """Verify that Failed runs and runs missing predictions.csv are excluded."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_failed", status="Failed")
        _create_mock_detection_run(self.analyses_dir, "detect_run_missing_pred", omit_predictions_file=True)

        eligible = self.registry.list_eligible_runs()
        self.assertEqual(len(eligible), 0)

        all_runs = self.registry.list_all_runs(include_test_fixtures=True)
        status_map = {r.detection_run_id: r.status for r in all_runs}
        self.assertEqual(status_map.get("detect_run_failed"), "Failed")
        self.assertEqual(status_map.get("detect_run_missing_pred"), "Incomplete")

    def test_fixture_isolation(self) -> None:
        """Verify that test fixtures are isolated from production run queries."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_fixture_01", is_test_fixture=True)
        self.assertEqual(len(self.registry.list_eligible_runs(include_test_fixtures=False)), 0)
        self.assertEqual(len(self.registry.list_eligible_runs(include_test_fixtures=True)), 1)

    def test_corrupted_artifact_checksum_rejection(self) -> None:
        """Verify that a SHA-256 checksum mismatch flags the run as Corrupted and ineligible."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_corrupted", corrupt_predictions_sha=True)
        eligible = self.registry.list_eligible_runs()
        self.assertEqual(len(eligible), 0)
        all_runs = self.registry.list_all_runs()
        self.assertEqual(all_runs[0].status, "Corrupted")

    def test_unsupported_schema_missing_source_row_index(self) -> None:
        """Verify that artifacts missing source_row_index are rejected."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_bad_schema", omit_source_row_index=True)
        eligible = self.registry.list_eligible_runs()
        self.assertEqual(len(eligible), 0)

    # --------------------------------------------------------------------------
    # CATEGORY B: Prediction Reading & CP-08 Schema Compatibility
    # --------------------------------------------------------------------------

    def test_single_model_artifact_parsing(self) -> None:
        """Verify correct parsing of single-model prediction schema."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_single", is_multi_model=False)
        df = self.reader.load_predictions("detect_run_single")
        self.assertIn("predicted_class", df.columns)
        self.assertIn("model_score", df.columns)
        self.assertIn("source_row_index", df.columns)
        self.assertEqual(len(df), 5)

    def test_multi_model_artifact_parsing(self) -> None:
        """Verify correct parsing of multi-model comparative prediction schema."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_multi", is_multi_model=True)
        df = self.reader.load_predictions("detect_run_multi")
        self.assertIn("svm_prediction", df.columns)
        self.assertIn("mlp_prediction", df.columns)
        self.assertIn("decision_tree_prediction", df.columns)
        self.assertIn("agreement_status", df.columns)
        self.assertEqual(len(df), 5)

    def test_path_traversal_protection(self) -> None:
        """Verify security protection rejecting directory traversal sequences."""
        with self.assertRaises(PermissionError):
            self.reader.load_predictions("../evil_escape")

    def test_attach_reviews_to_predictions(self) -> None:
        """Verify live review fields are joined onto the prediction DataFrame."""
        run_id = "detect_run_review_join"
        _create_mock_detection_run(self.analyses_dir, run_id)

        # Record a review for row 2
        self.review_service.submit_review(
            detection_run_id=run_id,
            source_dataset_id="test_ds",
            source_row_index=2,
            disposition=AnalystDisposition.POTENTIALLY_MALICIOUS,
            analyst_notes="Suspicious SYN flood pattern",
        )

        df = self.reader.load_predictions(run_id, attach_reviews=True)
        self.assertIn("review_status", df.columns)
        self.assertIn("analyst_disposition", df.columns)
        self.assertIn("analyst_notes", df.columns)

        # Row 2 should be Reviewed
        row2 = df[df["source_row_index"] == 2].iloc[0]
        self.assertEqual(row2["review_status"], ReviewStatus.REVIEWED)
        self.assertEqual(row2["analyst_disposition"], AnalystDisposition.POTENTIALLY_MALICIOUS)
        self.assertEqual(row2["analyst_notes"], "Suspicious SYN flood pattern")

        # Row 0 should be Unreviewed
        row0 = df[df["source_row_index"] == 0].iloc[0]
        self.assertEqual(row0["review_status"], ReviewStatus.UNREVIEWED)
        self.assertEqual(row0["analyst_disposition"], "Unreviewed")

    # --------------------------------------------------------------------------
    # CATEGORY C: Multi-Attribute Filtering, Sorting & Pagination
    # --------------------------------------------------------------------------

    def test_class_filter(self) -> None:
        """Verify filtering by predicted attack class."""
        run_id = "detect_run_flt_cls"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        params = PredictionFilterParams(predicted_classes=["PortScan"])
        res_df, count = PredictionQueryEngine.query(df, params)
        self.assertEqual(count, 1)
        self.assertEqual(res_df.iloc[0]["source_row_index"], 1)

    def test_model_specific_filter(self) -> None:
        """Verify model filter inspects the specific model's column."""
        run_id = "detect_run_flt_model"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        # MLP predicts BENIGN for row 2 (SVM predicts DoS Hulk)
        params = PredictionFilterParams(selected_model="mlp", predicted_classes=["BENIGN"])
        res_df, count = PredictionQueryEngine.query(df, params)
        self.assertIn(2, res_df["source_row_index"].values)

    def test_agreement_status_filter(self) -> None:
        """Verify filtering by Agreement and Disagreement status."""
        run_id = "detect_run_flt_agree"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        params = PredictionFilterParams(agreement_statuses=["Disagreement"])
        res_df, count = PredictionQueryEngine.query(df, params)
        self.assertEqual(count, 2)  # rows 2 and 3
        self.assertEqual(set(res_df["source_row_index"]), {2, 3})

    def test_review_status_and_disposition_filters(self) -> None:
        """Verify filtering by ReviewStatus and AnalystDisposition."""
        run_id = "detect_run_flt_rev"
        _create_mock_detection_run(self.analyses_dir, run_id)

        self.review_service.submit_review(
            run_id, "test_ds", 1, AnalystDisposition.LIKELY_BENIGN, "Verified normal scanner"
        )
        df = self.reader.load_predictions(run_id, attach_reviews=True)

        params_rev = PredictionFilterParams(review_statuses=[ReviewStatus.REVIEWED])
        res_df, count = PredictionQueryEngine.query(df, params_rev)
        self.assertEqual(count, 1)
        self.assertEqual(res_df.iloc[0]["source_row_index"], 1)

        params_disp = PredictionFilterParams(analyst_dispositions=[AnalystDisposition.LIKELY_BENIGN])
        res_df2, count2 = PredictionQueryEngine.query(df, params_disp)
        self.assertEqual(count2, 1)

    def test_ground_truth_filter(self) -> None:
        """Verify filtering by ground truth matches and discrepancies."""
        run_id = "detect_run_flt_gt"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        # For MLP, row 2 differs from ground truth (MLP: BENIGN, GT: DoS Hulk)
        params = PredictionFilterParams(selected_model="mlp", ground_truth_filter="differs_from_prediction")
        res_df, count = PredictionQueryEngine.query(df, params)
        self.assertIn(2, res_df["source_row_index"].values)

    def test_score_filtering_safely_isolated(self) -> None:
        """Verify score filtering applies to the specified model only."""
        run_id = "detect_run_flt_score"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        # Filter SVM score >= 2.0 (rows 1 and 4)
        params = PredictionFilterParams(score_filter_model="svm", min_score=2.0)
        res_df, count = PredictionQueryEngine.query(df, params)
        self.assertEqual(count, 2)
        self.assertEqual(set(res_df["source_row_index"]), {1, 4})

    def test_search_substring(self) -> None:
        """Verify substring searching across row indices, classes, and notes."""
        run_id = "detect_run_flt_search"
        _create_mock_detection_run(self.analyses_dir, run_id)

        self.review_service.submit_review(run_id, "test_ds", 4, AnalystDisposition.FALSE_POSITIVE, "Alert_Token_998")
        df = self.reader.load_predictions(run_id, attach_reviews=True)

        params = PredictionFilterParams(search_query="Alert_Token_998")
        res_df, count = PredictionQueryEngine.query(df, params)
        self.assertEqual(count, 1)
        self.assertEqual(res_df.iloc[0]["source_row_index"], 4)

    def test_triage_disagreements_first_sorting(self) -> None:
        """Verify triage sort places Disagreements at the top of the table."""
        run_id = "detect_run_triage"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        params = PredictionFilterParams(disagreement_first=True, sort_by="source_row_index")
        res_df, count = PredictionQueryEngine.query(df, params)
        first_statuses = res_df.iloc[:2]["agreement_status"].tolist()
        self.assertEqual(first_statuses, ["Disagreement", "Disagreement"])

    def test_bounded_pagination(self) -> None:
        """Verify bounded pagination correctly slices records."""
        run_id = "detect_run_pag"
        _create_mock_detection_run(self.analyses_dir, run_id)
        df = self.reader.load_predictions(run_id)

        params = PredictionFilterParams()
        page1, total = PredictionQueryEngine.query(df, params, page=1, page_size=2)
        self.assertEqual(total, 5)
        self.assertEqual(len(page1), 2)
        self.assertEqual(page1.iloc[0]["source_row_index"], 0)

        page3, _ = PredictionQueryEngine.query(df, params, page=3, page_size=2)
        self.assertEqual(len(page3), 1)
        self.assertEqual(page3.iloc[0]["source_row_index"], 4)

    # --------------------------------------------------------------------------
    # CATEGORY D: Individual Record Inspection & Network Telemetry
    # --------------------------------------------------------------------------

    def test_record_inspection_identity_and_predictions(self) -> None:
        """Verify deep inspection of row identity, predictions, and score semantics."""
        run_id = "detect_run_insp_01"
        _create_mock_detection_run(self.analyses_dir, run_id)

        insp = self.inspector.inspect_record(run_id, source_row_index=1)
        self.assertEqual(insp.detection_run_id, run_id)
        self.assertEqual(insp.source_row_index, 1)
        self.assertEqual(insp.source_filename, "sample_flows.csv")
        self.assertEqual(insp.agreement_status, "Agreement")

        # Verify SVM score semantics
        svm_detail = insp.model_predictions["svm"]
        self.assertEqual(svm_detail.predicted_class, "PortScan")
        self.assertEqual(svm_detail.score_type, "decision_margin")
        self.assertIn("Euclidean distance", svm_detail.score_explanation)

        # Verify MLP score semantics
        mlp_detail = insp.model_predictions["mlp"]
        self.assertEqual(mlp_detail.predicted_class, "PortScan")
        self.assertIn("softmax", mlp_detail.score_explanation.lower())

    def test_network_attributes_retrieval_from_raw_csv(self) -> None:
        """Verify extraction of actual flow attributes from raw CSV on disk."""
        run_id = "detect_run_insp_raw"
        _create_mock_detection_run(self.analyses_dir, run_id)

        # Write dummy raw CSV to self.raw_dir
        raw_csv_path = self.raw_dir / "sample_flows.csv"
        raw_content = (
            "Flow ID,Source IP,Destination IP,Source Port,Destination Port,Protocol,Flow Duration\n"
            "flow_0,192.168.1.10,10.0.0.1,4444,80,6,1200\n"
            "flow_1,192.168.1.15,10.0.0.1,5555,22,6,3500\n"
        )
        raw_csv_path.write_text(raw_content, encoding="utf-8")

        insp = self.inspector.inspect_record(run_id, source_row_index=1)
        self.assertTrue(insp.network_attributes_available)
        self.assertEqual(insp.network_attributes.get("Source IP"), "192.168.1.15")
        self.assertEqual(insp.network_attributes.get("Destination Port"), 22)

    def test_missing_raw_csv_handled_gracefully(self) -> None:
        """Verify missing raw CSV sets attributes unavailable without failing."""
        run_id = "detect_run_insp_noraw"
        _create_mock_detection_run(self.analyses_dir, run_id)

        insp = self.inspector.inspect_record(run_id, source_row_index=0)
        self.assertFalse(insp.network_attributes_available)
        self.assertIn("not found in data/raw/", insp.network_attributes_note)

    def test_ground_truth_separation_in_inspector(self) -> None:
        """Verify ground-truth reference is explicitly separated from predictions."""
        run_id = "detect_run_insp_gt"
        _create_mock_detection_run(self.analyses_dir, run_id, has_ground_truth=True)

        insp = self.inspector.inspect_record(run_id, source_row_index=0)
        self.assertTrue(insp.has_ground_truth)
        self.assertEqual(insp.ground_truth_label, "BENIGN")

    # --------------------------------------------------------------------------
    # CATEGORY E: Model Consensus & Disagreement Exploration
    # --------------------------------------------------------------------------

    def test_all_models_agree_consensus(self) -> None:
        """Verify 100% agreement when all models produce matching predictions."""
        agree_df = pd.DataFrame({
            "svm_prediction": ["BENIGN", "PortScan"],
            "mlp_prediction": ["BENIGN", "PortScan"],
            "decision_tree_prediction": ["BENIGN", "PortScan"],
            "agreement_status": ["Agreement", "Agreement"],
        })
        res = DisagreementService.analyze_consensus(agree_df)
        self.assertEqual(res["agreement_pct"], 100.0)
        self.assertEqual(res["disagreement_count"], 0)

    def test_models_disagree_consensus(self) -> None:
        """Verify divergence patterns and pairwise match percentages."""
        disagree_df = pd.DataFrame({
            "svm_prediction": ["BENIGN", "PortScan", "DoS Hulk"],
            "mlp_prediction": ["BENIGN", "BENIGN", "DoS Hulk"],
            "decision_tree_prediction": ["BENIGN", "PortScan", "BENIGN"],
            "agreement_status": ["Agreement", "Disagreement", "Disagreement"],
        })
        res = DisagreementService.analyze_consensus(disagree_df)
        self.assertEqual(res["agreement_count"], 1)
        self.assertEqual(res["disagreement_count"], 2)
        self.assertIn("SVM vs MLP", res["pairwise_agreements"])
        self.assertTrue(len(res["top_divergence_patterns"]) > 0)
        self.assertIn("NOT independently confirm", res["disclaimer"])

    def test_incomplete_model_prediction_handling(self) -> None:
        """Verify missing predictions are classified as Incomplete."""
        inc_df = pd.DataFrame({
            "svm_prediction": ["BENIGN", "PortScan"],
            "mlp_prediction": ["BENIGN", None],
            "decision_tree_prediction": ["BENIGN", "PortScan"],
        })
        res = DisagreementService.analyze_consensus(inc_df)
        self.assertEqual(res["incomplete_count"], 1)
        self.assertEqual(res["agreement_count"], 1)

    # --------------------------------------------------------------------------
    # CATEGORY F: Human Analyst Review Workflow
    # --------------------------------------------------------------------------

    def test_new_review_creation_and_update(self) -> None:
        """Verify new review creation and subsequent update with history preservation."""
        run_id = "detect_run_rev_01"
        _create_mock_detection_run(self.analyses_dir, run_id)

        # 1. First review
        ev1 = self.review_service.submit_review(
            detection_run_id=run_id,
            source_dataset_id="test_ds",
            source_row_index=3,
            disposition=AnalystDisposition.NEEDS_FURTHER_INVESTIGATION,
            analyst_notes="Unusual port activity",
            reviewer_label="Tier-1 Analyst",
        )
        self.assertEqual(ev1.previous_status, ReviewStatus.UNREVIEWED)
        self.assertEqual(ev1.new_status, ReviewStatus.REVIEWED)
        self.assertEqual(ev1.new_disposition, AnalystDisposition.NEEDS_FURTHER_INVESTIGATION)

        # 2. Update review with new disposition
        ev2 = self.review_service.submit_review(
            detection_run_id=run_id,
            source_dataset_id="test_ds",
            source_row_index=3,
            disposition=AnalystDisposition.FALSE_POSITIVE,
            analyst_notes="Confirmed internal load balancer heartbeat",
            reviewer_label="Tier-2 Lead",
        )
        self.assertEqual(ev2.previous_status, ReviewStatus.REVIEWED)
        self.assertEqual(ev2.previous_disposition, AnalystDisposition.NEEDS_FURTHER_INVESTIGATION)
        self.assertEqual(ev2.new_disposition, AnalystDisposition.FALSE_POSITIVE)

        # Verify history holds both events in chronological order
        history = self.review_service.get_history(run_id, source_row_index=3)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].event_id, ev1.event_id)
        self.assertEqual(history[1].event_id, ev2.event_id)

    def test_all_five_allowed_dispositions(self) -> None:
        """Verify all five authorized analyst dispositions can be submitted."""
        run_id = "detect_run_disp_all"
        _create_mock_detection_run(self.analyses_dir, run_id)

        for idx, disp in enumerate(AnalystDisposition.ALL_DISPOSITIONS):
            ev = self.review_service.submit_review(
                detection_run_id=run_id,
                source_dataset_id="test_ds",
                source_row_index=idx,
                disposition=disp,
            )
            self.assertEqual(ev.new_disposition, disp)

        summary = self.review_service.get_summary(run_id, total_eligible_records=5)
        for disp in AnalystDisposition.ALL_DISPOSITIONS:
            self.assertEqual(summary.disposition_counts[disp], 1)
        self.assertEqual(summary.completion_percentage, 100.0)

    def test_unauthorized_disposition_rejection(self) -> None:
        """Verify unauthorized disposition strings are rejected with ValueError."""
        run_id = "detect_run_bad_disp"
        _create_mock_detection_run(self.analyses_dir, run_id)

        with self.assertRaises(ValueError):
            self.review_service.submit_review(
                detection_run_id=run_id,
                source_dataset_id="test_ds",
                source_row_index=0,
                disposition="Confirmed Cyber Attack",  # Unauthorized
            )

    def test_notes_length_validation(self) -> None:
        """Verify analyst notes exceeding 2000 chars are rejected."""
        run_id = "detect_run_notes_len"
        _create_mock_detection_run(self.analyses_dir, run_id)

        with self.assertRaises(ValueError):
            self.review_service.submit_review(
                detection_run_id=run_id,
                source_dataset_id="test_ds",
                source_row_index=0,
                disposition=AnalystDisposition.INCONCLUSIVE,
                analyst_notes="A" * 2001,
            )

    def test_duplicate_submission_protection(self) -> None:
        """Verify identical consecutive review submission does not generate redundant events."""
        run_id = "detect_run_dedup"
        _create_mock_detection_run(self.analyses_dir, run_id)

        ev1 = self.review_service.submit_review(
            detection_run_id=run_id,
            source_dataset_id="test_ds",
            source_row_index=0,
            disposition=AnalystDisposition.POTENTIALLY_MALICIOUS,
            analyst_notes="Same notes",
            reviewer_label="Analyst X",
        )

        # Duplicate submission with identical parameters
        ev2 = self.review_service.submit_review(
            detection_run_id=run_id,
            source_dataset_id="test_ds",
            source_row_index=0,
            disposition=AnalystDisposition.POTENTIALLY_MALICIOUS,
            analyst_notes="Same notes",
            reviewer_label="Analyst X",
        )

        self.assertEqual(ev1.event_id, ev2.event_id)
        history = self.review_service.get_history(run_id, source_row_index=0)
        self.assertEqual(len(history), 1)

    def test_review_isolation_across_detection_runs(self) -> None:
        """Verify reviews for run A do not alter or appear in run B."""
        run_a = "detect_run_iso_a"
        run_b = "detect_run_iso_b"
        _create_mock_detection_run(self.analyses_dir, run_a)
        _create_mock_detection_run(self.analyses_dir, run_b)

        self.review_service.submit_review(run_a, "ds_a", 0, AnalystDisposition.POTENTIALLY_MALICIOUS)

        history_a = self.review_service.get_history(run_a)
        history_b = self.review_service.get_history(run_b)

        self.assertEqual(len(history_a), 1)
        self.assertEqual(len(history_b), 0)

    # --------------------------------------------------------------------------
    # CATEGORY G: Audit Trail, Cryptographic Hash Chain & Zero Mutation
    # --------------------------------------------------------------------------

    def test_cryptographic_hash_chain_verification(self) -> None:
        """Verify hash chaining across sequentially appended review events."""
        run_id = "detect_run_chain"
        _create_mock_detection_run(self.analyses_dir, run_id)

        for i in range(4):
            self.review_service.submit_review(
                detection_run_id=run_id,
                source_dataset_id="test_ds",
                source_row_index=i,
                disposition=AnalystDisposition.LIKELY_BENIGN,
                analyst_notes=f"Note {i}",
            )

        is_valid, errors = self.review_service.verify_integrity(run_id)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_tampered_journal_hash_chain_detection(self) -> None:
        """Verify that altering an event line in review_events.jsonl breaks the hash chain."""
        run_id = "detect_run_tamper"
        _create_mock_detection_run(self.analyses_dir, run_id)

        self.review_service.submit_review(run_id, "test_ds", 0, AnalystDisposition.LIKELY_BENIGN, "Original")
        self.review_service.submit_review(run_id, "test_ds", 1, AnalystDisposition.POTENTIALLY_MALICIOUS, "Target")

        journal_file = self.reviews_dir / run_id / "review_events.jsonl"
        lines = journal_file.read_text(encoding="utf-8").strip().split("\n")

        # Tamper with first line
        tampered_data = json.loads(lines[0])
        tampered_data["analyst_notes"] = "TAMPERED NOTES"
        lines[0] = json.dumps(tampered_data)
        journal_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        is_valid, errors = self.review_service.verify_integrity(run_id)
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_malformed_journal_lines_handled_safely(self) -> None:
        """Verify that a corrupted non-JSON line is logged and does not crash reader."""
        run_id = "detect_run_malformed"
        _create_mock_detection_run(self.analyses_dir, run_id)

        self.review_service.submit_review(run_id, "test_ds", 0, AnalystDisposition.LIKELY_BENIGN)

        journal_file = self.reviews_dir / run_id / "review_events.jsonl"
        with open(journal_file, "a", encoding="utf-8") as f:
            f.write("CORRUPTED_LINE_NOT_JSON\n")

        self.review_service.submit_review(run_id, "test_ds", 1, AnalystDisposition.POTENTIALLY_MALICIOUS)

        history = self.review_service.get_history(run_id)
        self.assertEqual(len(history), 2)  # Safely recovers the two valid events

    def test_zero_mutation_of_cp08_artifacts(self) -> None:
        """Verify CP-08 detection artifacts are NEVER modified during analyst reviews."""
        run_id = "detect_run_zero_mut"
        run_dir = _create_mock_detection_run(self.analyses_dir, run_id)

        pred_file = run_dir / "predictions.csv"
        manifest_file = run_dir / "run_manifest.json"

        pred_sha_before = hashlib.sha256(pred_file.read_bytes()).hexdigest()
        manifest_sha_before = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

        # Perform multiple reviews
        self.review_service.submit_review(run_id, "test_ds", 0, AnalystDisposition.POTENTIALLY_MALICIOUS)
        self.review_service.submit_review(run_id, "test_ds", 1, AnalystDisposition.FALSE_POSITIVE)
        self.review_service.submit_review(run_id, "test_ds", 0, AnalystDisposition.LIKELY_BENIGN)

        pred_sha_after = hashlib.sha256(pred_file.read_bytes()).hexdigest()
        manifest_sha_after = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

        self.assertEqual(pred_sha_before, pred_sha_after)
        self.assertEqual(manifest_sha_before, manifest_sha_after)

    # --------------------------------------------------------------------------
    # CATEGORY H: Dashboard Integration & UI Rendering
    # --------------------------------------------------------------------------

    def test_dashboard_page_render_empty_state(self) -> None:
        """Verify that prediction_explorer.render() executes without error in empty state."""
        from dashboard.pages import prediction_explorer
        try:
            prediction_explorer.render()
        except Exception as e:
            self.fail(f"prediction_explorer.render() raised unexpected exception: {e}")

    def test_dashboard_page_render_with_eligible_run(self) -> None:
        """Verify that prediction_explorer.render() executes without error when an eligible run exists."""
        _create_mock_detection_run(self.analyses_dir, "detect_run_ui_01")
        from dashboard.pages import prediction_explorer

        orig_init = ExplorerRegistry.__init__
        try:
            ExplorerRegistry.__init__ = lambda s, analyses_dir=None: orig_init(s, analyses_dir=self.analyses_dir)
            prediction_explorer.render()
        except Exception as e:
            self.fail(f"prediction_explorer.render() raised unexpected exception with run: {e}")
        finally:
            ExplorerRegistry.__init__ = orig_init


if __name__ == "__main__":
    unittest.main()
