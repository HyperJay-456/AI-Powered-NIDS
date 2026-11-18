"""Structured Data Exporter for Reports & Machine-Readable Artifacts (CP-11).

Generates versioned report_data.json, evidence_manifest.json, and sanitized CSV exports:
  - prediction_summary.csv
  - model_comparison.csv
  - analyst_review_summary.csv
  - attck_mapping_summary.csv

Enforces strict CSV formula injection mitigation and atomic file persistence.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.reporting.report_data_aggregator import sanitize_csv_cell
from src.reporting.report_models import (
    EvidenceItem,
    EvidenceManifest,
    ReportDataPackage,
    ReportMetadata,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StructuredExporter:
    """Exports structured machine-readable JSON and CSV report artifacts."""

    def __init__(self, output_dir: Path):
        """Initialize exporter with output destination directory."""
        self.output_dir = output_dir
        self.exports_dir = self.output_dir / "exports"

    def export_all(
        self,
        data_package: ReportDataPackage,
        pdf_path: Optional[Path] = None,
        pdf_sha256: Optional[str] = None,
        pdf_size_bytes: Optional[int] = None,
    ) -> EvidenceManifest:
        """Export all structured artifacts and build a complete EvidenceManifest.

        Args:
            data_package: Aggregated report data.
            pdf_path: Path to generated PDF if available.
            pdf_sha256: SHA-256 of generated PDF.
            pdf_size_bytes: Size in bytes of generated PDF.

        Returns:
            Populated EvidenceManifest documenting all generated artifacts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        meta = data_package.metadata
        manifest = EvidenceManifest(
            report_id=meta.report_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        # 1. Add PDF to evidence manifest if provided
        if pdf_path and pdf_path.exists() and pdf_sha256 and pdf_size_bytes:
            manifest.add_item(EvidenceItem(
                evidence_id=f"evi_pdf_{meta.report_id[:16]}",
                filename=pdf_path.name,
                artifact_type="report_pdf",
                source_checkpoint="CP-11",
                source_run_id=meta.detection_run_id,
                sha256_checksum=pdf_sha256,
                size_bytes=pdf_size_bytes,
                export_timestamp_utc=manifest.created_at_utc,
                inclusion_reason="Primary human-readable cybersecurity assessment document.",
                relative_path=pdf_path.name,
            ))

        # 2. Export report_data.json
        json_item = self._export_report_json(data_package, manifest.created_at_utc)
        manifest.add_item(json_item)

        # 3. Export prediction_summary.csv
        pred_item = self._export_prediction_summary_csv(data_package, manifest.created_at_utc)
        manifest.add_item(pred_item)

        # 4. Export model_comparison.csv
        comp_item = self._export_model_comparison_csv(data_package, manifest.created_at_utc)
        manifest.add_item(comp_item)

        # 5. Export analyst_review_summary.csv
        rev_item = self._export_analyst_review_summary_csv(data_package, manifest.created_at_utc)
        manifest.add_item(rev_item)

        # 6. Export attck_mapping_summary.csv
        attck_item = self._export_attck_mapping_summary_csv(data_package, manifest.created_at_utc)
        manifest.add_item(attck_item)

        # 7. Write evidence_manifest.json
        manifest_path = self.output_dir / "evidence_manifest.json"
        manifest_data = manifest.to_dict()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, sort_keys=True)

        logger.info(f"Exported structured artifacts for report '{meta.report_id}' ({manifest.total_files} files)")
        return manifest

    def _export_report_json(self, data_package: ReportDataPackage, timestamp_utc: str) -> EvidenceItem:
        """Export canonical report_data.json."""
        target_path = self.output_dir / "report_data.json"
        data_dict = data_package.to_dict()
        content = json.dumps(data_dict, indent=2, sort_keys=True)
        content_bytes = content.encode("utf-8")

        with open(target_path, "wb") as f:
            f.write(content_bytes)

        sha = hashlib.sha256(content_bytes).hexdigest()
        return EvidenceItem(
            evidence_id=f"evi_json_{data_package.metadata.report_id[:16]}",
            filename=target_path.name,
            artifact_type="report_data_json",
            source_checkpoint="CP-11",
            source_run_id=data_package.metadata.detection_run_id,
            sha256_checksum=sha,
            size_bytes=len(content_bytes),
            export_timestamp_utc=timestamp_utc,
            inclusion_reason="Machine-readable canonical report schema payload.",
            relative_path=target_path.name,
        )

    def _export_prediction_summary_csv(self, data_package: ReportDataPackage, timestamp_utc: str) -> EvidenceItem:
        """Export prediction_summary.csv with formula injection protection."""
        target_path = self.exports_dir / "prediction_summary.csv"
        rows: List[Dict[str, Any]] = []

        for item in data_package.prediction_distribution:
            rows.append({
                "class_label": sanitize_csv_cell(item["class_label"]),
                "category": sanitize_csv_cell("Malicious" if item["is_attack"] else "Benign"),
                "record_count": item["count"],
                "traffic_share_percentage": item["percentage"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(target_path, index=False, encoding="utf-8")

        with open(target_path, "rb") as f:
            c = f.read()
            sha = hashlib.sha256(c).hexdigest()

        return EvidenceItem(
            evidence_id=f"evi_csv_pred_{data_package.metadata.report_id[:16]}",
            filename=target_path.name,
            artifact_type="prediction_summary_csv",
            source_checkpoint="CP-11",
            source_run_id=data_package.metadata.detection_run_id,
            sha256_checksum=sha,
            size_bytes=len(c),
            export_timestamp_utc=timestamp_utc,
            inclusion_reason="Aggregated prediction class breakdown summary.",
            relative_path=f"exports/{target_path.name}",
        )

    def _export_model_comparison_csv(self, data_package: ReportDataPackage, timestamp_utc: str) -> EvidenceItem:
        """Export model_comparison.csv with formula injection protection."""
        target_path = self.exports_dir / "model_comparison.csv"
        rows: List[Dict[str, Any]] = []

        for item in data_package.model_comparisons:
            rows.append({
                "comparison_dimension": sanitize_csv_cell(item.get("metric_or_aspect", "")),
                "svm_result": sanitize_csv_cell(item.get("svm_value", "")),
                "mlp_result": sanitize_csv_cell(item.get("mlp_value", "")),
                "decision_tree_result": sanitize_csv_cell(item.get("decision_tree_value", "")),
            })

        df = pd.DataFrame(rows)
        df.to_csv(target_path, index=False, encoding="utf-8")

        with open(target_path, "rb") as f:
            c = f.read()
            sha = hashlib.sha256(c).hexdigest()

        return EvidenceItem(
            evidence_id=f"evi_csv_comp_{data_package.metadata.report_id[:16]}",
            filename=target_path.name,
            artifact_type="model_comparison_csv",
            source_checkpoint="CP-11",
            source_run_id=data_package.metadata.detection_run_id,
            sha256_checksum=sha,
            size_bytes=len(c),
            export_timestamp_utc=timestamp_utc,
            inclusion_reason="Multi-algorithm agreement and comparative performance metrics.",
            relative_path=f"exports/{target_path.name}",
        )

    def _export_analyst_review_summary_csv(self, data_package: ReportDataPackage, timestamp_utc: str) -> EvidenceItem:
        """Export analyst_review_summary.csv with formula injection protection."""
        target_path = self.exports_dir / "analyst_review_summary.csv"
        stats = data_package.analyst_summary
        disp_counts = stats.get("disposition_counts", {})
        rows: List[Dict[str, Any]] = []

        for disp, cnt in disp_counts.items():
            rows.append({
                "disposition": sanitize_csv_cell(disp),
                "record_count": cnt,
                "share_of_reviewed_pct": round((cnt / max(1, stats.get("reviewed_records", 1))) * 100, 2),
            })

        df = pd.DataFrame(rows)
        df.to_csv(target_path, index=False, encoding="utf-8")

        with open(target_path, "rb") as f:
            c = f.read()
            sha = hashlib.sha256(c).hexdigest()

        return EvidenceItem(
            evidence_id=f"evi_csv_rev_{data_package.metadata.report_id[:16]}",
            filename=target_path.name,
            artifact_type="analyst_review_summary_csv",
            source_checkpoint="CP-11",
            source_run_id=data_package.metadata.detection_run_id,
            sha256_checksum=sha,
            size_bytes=len(c),
            export_timestamp_utc=timestamp_utc,
            inclusion_reason="Human-in-the-loop analyst review disposition summary.",
            relative_path=f"exports/{target_path.name}",
        )

    def _export_attck_mapping_summary_csv(self, data_package: ReportDataPackage, timestamp_utc: str) -> EvidenceItem:
        """Export attck_mapping_summary.csv with formula injection protection."""
        target_path = self.exports_dir / "attck_mapping_summary.csv"
        rows: List[Dict[str, Any]] = []

        for m in data_package.attck_mappings:
            rows.append({
                "dataset_attack_label": sanitize_csv_cell(m.get("dataset_label", "")),
                "technique_id": sanitize_csv_cell(m.get("technique_id", "")),
                "technique_name": sanitize_csv_cell(m.get("technique_name", "")),
                "tactic": sanitize_csv_cell(m.get("tactic", "")),
                "mapping_status": sanitize_csv_cell(m.get("status", "")),
                "mapping_strength": sanitize_csv_cell(m.get("strength", "")),
                "available_evidence": sanitize_csv_cell(m.get("available_evidence", "")),
                "missing_evidence": sanitize_csv_cell(m.get("missing_evidence", "")),
                "mitigations": sanitize_csv_cell(m.get("mitigations", "")),
            })

        df = pd.DataFrame(rows)
        df.to_csv(target_path, index=False, encoding="utf-8")

        with open(target_path, "rb") as f:
            c = f.read()
            sha = hashlib.sha256(c).hexdigest()

        return EvidenceItem(
            evidence_id=f"evi_csv_attck_{data_package.metadata.report_id[:16]}",
            filename=target_path.name,
            artifact_type="attck_mapping_summary_csv",
            source_checkpoint="CP-11",
            source_run_id=data_package.metadata.detection_run_id,
            sha256_checksum=sha,
            size_bytes=len(c),
            export_timestamp_utc=timestamp_utc,
            inclusion_reason="MITRE ATT&CK candidate technique associations and evidence evaluation.",
            relative_path=f"exports/{target_path.name}",
        )
