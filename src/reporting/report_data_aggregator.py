"""Report Data Aggregator & Source Verification Engine (CP-11).

Loads, cryptographically verifies, and deterministically aggregates data across
CP-08 Detection Runs, CP-09 Analyst Reviews, and CP-10 MITRE ATT&CK Threat Context.
Enforces strict separation between model predictions, dataset ground truth,
human analyst dispositions, and ATT&CK candidate associations.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np
import pandas as pd

from config.settings import settings
from src.explorer.review_models import AnalystDisposition, ReviewEvent, ReviewStatus, ReviewSummary
from src.explorer.review_store import ReviewStore
from src.reporting.report_models import (
    RedactionConfig,
    ReportClassification,
    ReportDataPackage,
    ReportMetadata,
    ReportStatus,
    ReportType,
)
from src.threat_intelligence.attack_models import MappingStatus
from src.threat_intelligence.mapping_registry import MappingRegistry
from src.threat_intelligence.reference_registry import ReferenceRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Security & Sanitization Utilities
# ==============================================================================

def sanitize_csv_cell(val: Any) -> Any:
    """Sanitize spreadsheet cell against CSV formula injection (DDE/Formula execution).

    If the string begins with '=', '+', '-', or '@', prefix with a single quote "'"
    to neutralize execution in Excel/Calc while preserving human readability.
    """
    if val is None:
        return ""
    if isinstance(val, (int, float, bool)):
        return val
    s = str(val).strip()
    if s and s[0] in ("=", "+", "-", "@"):
        return f"'{s}"
    return s


def redact_ip_address(ip_str: Any, mask_str: str = "xxx") -> str:
    """Redact IPv4 address host portion for privacy and compliance."""
    if not ip_str or pd.isna(ip_str):
        return ""
    s = str(ip_str).strip()
    ipv4_pattern = r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
    match = re.match(ipv4_pattern, s)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}.{mask_str}"
    return s


# ==============================================================================
# Source Artifact Loading & Verification
# ==============================================================================

def verify_and_load_detection_run(
    detection_run_id: str,
    analyses_dir: Optional[Path] = None,
    allow_test_fixtures: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
    """Load and cryptographically verify CP-08 detection artifacts.

    Args:
        detection_run_id: Unique identifier of CP-08 detection run.
        analyses_dir: Optional custom analyses directory.
        allow_test_fixtures: Whether to permit test fixture runs.

    Returns:
        Tuple of (manifest_dict, predictions_df, model_comparison_df_or_None).

    Raises:
        PermissionError: If path traversal is detected.
        FileNotFoundError: If required artifacts do not exist.
        ValueError: If SHA-256 checksum verification fails or status is invalid.
    """
    if ".." in detection_run_id or "/" in detection_run_id or "\\" in detection_run_id:
        raise PermissionError(f"Security violation: path traversal detected in '{detection_run_id}'")

    base_dir = (analyses_dir or settings.analyses_dir).resolve()
    run_dir = (base_dir / detection_run_id).resolve()

    if not str(run_dir).startswith(str(base_dir)):
        raise PermissionError(f"Security violation: directory escape detected for '{detection_run_id}'")

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Detection run directory not found: '{detection_run_id}'")

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest missing in detection run: '{detection_run_id}'")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Check test fixture status
    is_fixture = manifest_data.get("is_test_fixture", False) or manifest_data.get("run_type") in ("test", "fixture", "test_fixture")
    if is_fixture and not allow_test_fixtures:
        raise ValueError(f"Detection run '{detection_run_id}' is a test fixture and cannot be used for production reports.")

    # Check status
    status = manifest_data.get("status", "Unknown")
    if status.lower() not in ("completed", "completed_with_warnings", "success"):
        raise ValueError(f"Detection run '{detection_run_id}' has incomplete or failed status: '{status}'")

    # Load and verify predictions.csv
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions artifact missing: '{pred_path.name}'")

    artifacts = manifest_data.get("output_artifacts", {})
    expected_pred_sha = artifacts.get("predictions.csv")

    with open(pred_path, "rb") as pf:
        actual_pred_sha = hashlib.sha256(pf.read()).hexdigest()

    if expected_pred_sha and actual_pred_sha != expected_pred_sha:
        raise ValueError(
            f"Cryptographic integrity failure: predictions.csv checksum mismatch for '{detection_run_id}'. "
            f"Expected {expected_pred_sha}, got {actual_pred_sha}."
        )

    predictions_df = pd.read_csv(pred_path)

    # Load model_comparison.csv if present
    model_comp_df: Optional[pd.DataFrame] = None
    comp_path = run_dir / "model_comparison.csv"
    if comp_path.exists():
        expected_comp_sha = artifacts.get("model_comparison.csv")
        with open(comp_path, "rb") as cf:
            actual_comp_sha = hashlib.sha256(cf.read()).hexdigest()
        if expected_comp_sha and actual_comp_sha != expected_comp_sha:
            logger.warning(f"model_comparison.csv checksum mismatch in '{detection_run_id}'. Skipping comparison table.")
        else:
            model_comp_df = pd.read_csv(comp_path)

    return manifest_data, predictions_df, model_comp_df


# ==============================================================================
# Aggregation Engine
# ==============================================================================

class ReportDataAggregator:
    """Aggregates and formats forensic data across CP-08, CP-09, and CP-10."""

    def __init__(
        self,
        analyses_dir: Optional[Path] = None,
        reviews_dir: Optional[Path] = None,
        mappings_dir: Optional[Path] = None,
        reference_registry: Optional[ReferenceRegistry] = None,
        evaluations_dir: Optional[Path] = None,
    ):
        """Initialize aggregator with system directories and registries."""
        self.analyses_dir = analyses_dir or settings.analyses_dir
        self.reviews_dir = reviews_dir or getattr(settings, "analyst_reviews_dir", settings.data_dir / "analyst_reviews")
        self.review_store = ReviewStore(reviews_dir=self.reviews_dir)
        self.mapping_registry = MappingRegistry(mappings_dir=mappings_dir)
        self.ref_registry = reference_registry or ReferenceRegistry()
        self.evaluations_dir = evaluations_dir or settings.evaluations_dir

    def aggregate(
        self,
        detection_run_id: str,
        report_type: str = ReportType.EXECUTIVE,
        title: Optional[str] = None,
        classification: str = ReportClassification.ACADEMIC,
        redaction_config: Optional[RedactionConfig] = None,
        allow_test_fixtures: bool = False,
    ) -> ReportDataPackage:
        """Deterministically aggregate all report data for a verified detection run.

        Args:
            detection_run_id: CP-08 run ID.
            report_type: Type of report to generate.
            title: Custom title or default based on report type.
            classification: Sensitivity marking.
            redaction_config: IP and notes redaction options.
            allow_test_fixtures: Whether to allow test fixture runs.

        Returns:
            ReportDataPackage with verified figures and provenance.
        """
        redact = redaction_config or RedactionConfig()
        manifest, df_pred, df_comp = verify_and_load_detection_run(
            detection_run_id=detection_run_id,
            analyses_dir=self.analyses_dir,
            allow_test_fixtures=allow_test_fixtures,
        )

        now_utc = datetime.now(timezone.utc).isoformat()
        report_id = f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        default_title = f"{report_type} — {manifest.get('source_filename', detection_run_id)}"
        final_title = title.strip() if title and title.strip() else default_title

        # 1. Detection Run Summary
        row_accounting = manifest.get("row_accounting", {})
        input_rows = row_accounting.get("input_rows", len(df_pred))
        predicted_rows = row_accounting.get("successfully_predicted_rows", len(df_pred))
        rejected_rows = row_accounting.get("rejected_rows", 0)
        selected_algorithms = manifest.get("selected_algorithms", [])

        # 2. Prediction Distribution
        # Identify primary prediction column
        pred_col = "consensus_prediction" if "consensus_prediction" in df_pred.columns else None
        if not pred_col:
            for cand in ["prediction_label", "prediction", "predicted_label"]:
                if cand in df_pred.columns:
                    pred_col = cand
                    break
        if not pred_col and selected_algorithms:
            for algo in selected_algorithms:
                cand = f"prediction_{algo.lower().replace(' ', '_')}"
                if cand in df_pred.columns:
                    pred_col = cand
                    break

        pred_dist: List[Dict[str, Any]] = []
        benign_count = 0
        attack_count = 0

        if pred_col and pred_col in df_pred.columns:
            counts = df_pred[pred_col].value_counts()
            total_preds = len(df_pred)
            for cls_name, cnt in counts.items():
                pct = round((cnt / total_preds) * 100, 2) if total_preds > 0 else 0.0
                is_benign = str(cls_name).strip().upper() == "BENIGN"
                if is_benign:
                    benign_count += cnt
                else:
                    attack_count += cnt
                pred_dist.append({
                    "class_label": str(cls_name),
                    "count": int(cnt),
                    "percentage": pct,
                    "is_attack": not is_benign,
                })
        else:
            # Fallback if no unified prediction column
            total_preds = len(df_pred)

        # 3. Model Comparisons & Disagreement
        model_comps: List[Dict[str, Any]] = []
        if df_comp is not None and not df_comp.empty:
            for _, row in df_comp.iterrows():
                model_comps.append({
                    "metric_or_aspect": str(row.get("Aspect", row.get("Metric", ""))),
                    "svm_value": str(row.get("SVM", "")),
                    "mlp_value": str(row.get("MLP", "")),
                    "decision_tree_value": str(row.get("Decision Tree", row.get("DecisionTree", ""))),
                })
        else:
            # Generate deterministic model comparison from per-model prediction columns
            for algo in selected_algorithms:
                norm_algo = algo.lower().replace(" ", "_")
                col = f"prediction_{norm_algo}"
                if col in df_pred.columns:
                    a_counts = df_pred[col].value_counts()
                    a_benign = int(a_counts.get("BENIGN", 0))
                    a_attack = int(len(df_pred) - a_benign)
                    model_comps.append({
                        "metric_or_aspect": f"{algo} Predictions",
                        "svm_value": f"Attacks: {a_attack}" if "svm" in norm_algo else "-",
                        "mlp_value": f"Attacks: {a_attack}" if "mlp" in norm_algo else "-",
                        "decision_tree_value": f"Attacks: {a_attack}" if "tree" in norm_algo else "-",
                    })

        # 4. Analyst Review Statistics (CP-09)
        review_summary_obj: ReviewSummary = self.review_store.compute_review_summary(
            detection_run_id=detection_run_id,
            total_eligible_records=len(df_pred),
        )
        review_history_events: List[ReviewEvent] = self.review_store.get_review_history(detection_run_id=detection_run_id)

        clean_events: List[Dict[str, Any]] = []
        for ev in review_history_events:
            notes = ev.analyst_notes
            if not redact.include_analyst_notes:
                notes = "[NOTE EXCLUDED PER PRIVACY POLICY]"
            elif notes and redact.redact_ips:
                # Mask any IP addresses in notes text
                notes = re.sub(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b", r"\1.xxx", notes)

            clean_events.append({
                "event_id": ev.event_id,
                "source_row_index": ev.source_row_index,
                "timestamp_utc": ev.timestamp_utc,
                "disposition": ev.new_disposition,
                "reviewer": ev.reviewer_label or "Security Analyst",
                "notes": notes or "",
            })

        analyst_stats = {
            "total_eligible_records": review_summary_obj.total_eligible_records,
            "reviewed_records": review_summary_obj.reviewed_records,
            "unreviewed_records": review_summary_obj.unreviewed_records,
            "completion_percentage": review_summary_obj.completion_percentage,
            "disposition_counts": review_summary_obj.disposition_counts,
            "total_review_events": review_summary_obj.total_review_events,
            "integrity_verified": review_summary_obj.integrity_verified,
            "recent_events": clean_events[-50:],  # Keep up to 50 recent events for PDF/report
        }

        # 5. MITRE ATT&CK Contextual Mappings (CP-10)
        attack_classes_in_run = [d["class_label"] for d in pred_dist if d["is_attack"]]
        attck_mappings: List[Dict[str, Any]] = []
        all_registered_mappings = self.mapping_registry.list_mappings()

        for m in all_registered_mappings:
            if m.original_dataset_label in attack_classes_in_run or not attack_classes_in_run:
                attck_mappings.append({
                    "mapping_id": m.mapping_id,
                    "dataset_label": m.original_dataset_label,
                    "technique_id": m.technique_id,
                    "technique_name": m.technique_name,
                    "tactic": m.tactic,
                    "status": m.mapping_status,
                    "strength": m.mapping_strength,
                    "rationale": m.mapping_rationale,
                    "available_evidence": ", ".join(m.available_evidence),
                    "missing_evidence": ", ".join(m.missing_evidence),
                    "mitigations": ", ".join(m.mitigation_ids),
                })

        # 6. Evaluation Benchmark Context (CP-07)
        eval_metrics: Optional[Dict[str, Any]] = None
        eng_run_id = manifest.get("engineering_run_id", "unknown")
        eval_meta_file = self.evaluations_dir / f"eval_{eng_run_id}.json"
        if eval_meta_file.exists():
            try:
                with open(eval_meta_file, "r", encoding="utf-8") as ef:
                    eval_data = json.load(ef)
                eval_metrics = {
                    "engineering_run_id": eng_run_id,
                    "evaluation_type": "Offline Test Holdout Benchmark (CP-07)",
                    "note": "Evaluation metrics reflect offline model benchmarking and are NOT operational detection accuracy.",
                    "metrics": eval_data.get("metrics", {}),
                }
            except Exception as e:
                logger.warning(f"Could not load CP-07 evaluation record {eval_meta_file}: {e}")

        # 7. Disclaimers & Governance Notice
        disclaimers = [
            "ACADEMIC & DEFENSIVE USE ONLY: Machine learning model outputs represent statistical inference classifications, NOT confirmed real-world security incidents or compromised hosts.",
            "GOVERNANCE & TERMINOLOGY SEPARATION: Model predictions != dataset ground truth != human analyst dispositions != MITRE ATT&CK candidate mappings.",
            "NON-ATTRIBUTION SAFEGUARD: Technique associations represent behavioral taxonomy categories and do not attribute activity to specific threat actors or nation-states.",
            "LOCAL-FIRST EXECUTION: Generated entirely within local environment using verified artifacts. No data was transmitted to external cloud APIs.",
        ]

        # 8. Report Metadata
        metadata = ReportMetadata(
            report_id=report_id,
            report_type=report_type,
            title=final_title,
            classification=classification,
            detection_run_id=detection_run_id,
            source_dataset_id=manifest.get("source_dataset_id", "unknown"),
            source_filename=manifest.get("source_filename", "unknown"),
            engineering_run_id=eng_run_id,
            created_at_utc=now_utc,
            selected_models=selected_algorithms,
            total_records_analyzed=predicted_rows,
            predicted_attack_records=attack_count,
            predicted_benign_records=benign_count,
            analyst_reviewed_count=review_summary_obj.reviewed_records,
            status=ReportStatus.PENDING,
            redaction_config=redact.to_dict(),
            warnings=manifest.get("warnings", []),
            is_test_fixture=manifest.get("is_test_fixture", False),
        )

        return ReportDataPackage(
            metadata=metadata,
            run_summary={
                "detection_run_id": detection_run_id,
                "input_rows": input_rows,
                "successfully_predicted_rows": predicted_rows,
                "rejected_rows": rejected_rows,
                "source_dataset_id": manifest.get("source_dataset_id", "unknown"),
                "source_filename": manifest.get("source_filename", "unknown"),
                "created_at_utc": manifest.get("created_at_utc", ""),
                "completed_at_utc": manifest.get("completed_at_utc", ""),
                "selected_algorithms": selected_algorithms,
                "status": manifest.get("status", "Completed"),
                "model_artifacts": manifest.get("model_artifacts", {}),
            },
            prediction_distribution=pred_dist,
            model_comparisons=model_comps,
            analyst_summary=analyst_stats,
            attck_mappings=attck_mappings,
            evaluation_metrics=eval_metrics,
            disclaimers=disclaimers,
        )
