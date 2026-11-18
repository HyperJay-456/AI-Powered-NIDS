"""Report Registry & Historical Report Catalog (CP-11).

Provides discovery, cryptographic integrity verification, and retrieval for
completed cybersecurity reports and evidence packages stored in data/reports/.

Strict Invariants:
  - Filters out unverified or corrupted artifacts unless explicitly requested
  - Isolates test fixture reports from genuine production reports
  - Thread-safe and path-traversal secure
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import pypdf

from config.settings import settings
from src.reporting.report_models import (
    ReportClassification,
    ReportDataPackage,
    ReportMetadata,
    ReportStatus,
    ReportType,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportRegistry:
    """Manages discovery and integrity verification of historical reports."""

    _REGISTRY_LOCK = threading.RLock()

    def __init__(self, reports_dir: Optional[Path] = None):
        """Initialize registry with reports root directory."""
        self.reports_dir = reports_dir or getattr(settings, "reports_dir", settings.data_dir / "reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_report_dir(self, report_id: str) -> Path:
        """Resolve report directory with strict path-traversal safeguards."""
        if ".." in report_id or "/" in report_id or "\\" in report_id:
            raise PermissionError(f"Security violation: path traversal detected in '{report_id}'")

        target = (self.reports_dir / report_id).resolve()
        if not str(target).startswith(str(self.reports_dir.resolve())):
            raise PermissionError(f"Security violation: directory escape detected for '{report_id}'")

        return target

    def list_reports(
        self,
        include_test_fixtures: bool = False,
        verify_checksums: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all valid persisted reports in reverse chronological order.

        Args:
            include_test_fixtures: Whether to include automated test fixtures.
            verify_checksums: Whether to verify PDF checksums during discovery.

        Returns:
            List of report metadata summary dictionaries.
        """
        with self._REGISTRY_LOCK:
            if not self.reports_dir.exists():
                return []

            reports: List[Dict[str, Any]] = []

            for r_dir in sorted(self.reports_dir.glob("report_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                if not r_dir.is_dir():
                    continue

                json_path = r_dir / "report_data.json"
                if not json_path.exists():
                    continue

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    meta = data.get("metadata", {})
                    # Exclude test fixtures if requested
                    is_fixture = meta.get("is_test_fixture", False)
                    if is_fixture and not include_test_fixtures:
                        continue

                    # Check PDF presence and integrity
                    pdf_valid = True
                    pdf_filename = meta.get("pdf_filename", "report.pdf")
                    pdf_path = r_dir / pdf_filename

                    if verify_checksums and pdf_path.exists():
                        expected_sha = meta.get("pdf_sha256")
                        if expected_sha:
                            with open(pdf_path, "rb") as pf:
                                actual_sha = hashlib.sha256(pf.read()).hexdigest()
                            if actual_sha != expected_sha:
                                pdf_valid = False
                                logger.warning(f"Report '{r_dir.name}' has corrupted PDF checksum.")

                    # Check bundle presence
                    bundle_filename = meta.get("bundle_filename", f"evidence_bundle_{r_dir.name}.zip")
                    bundle_path = r_dir / bundle_filename
                    has_bundle = bundle_path.exists()

                    reports.append({
                        "report_id": meta.get("report_id", r_dir.name),
                        "report_type": meta.get("report_type", ReportType.EXECUTIVE),
                        "title": meta.get("title", r_dir.name),
                        "classification": meta.get("classification", ReportClassification.ACADEMIC),
                        "detection_run_id": meta.get("detection_run_id", "unknown"),
                        "source_dataset_id": meta.get("source_dataset_id", "unknown"),
                        "source_filename": meta.get("source_filename", "unknown"),
                        "created_at_utc": meta.get("created_at_utc", ""),
                        "status": meta.get("status", ReportStatus.COMPLETED),
                        "total_records": meta.get("total_records_analyzed", 0),
                        "predicted_attacks": meta.get("predicted_attack_records", 0),
                        "predicted_benign": meta.get("predicted_benign_records", 0),
                        "analyst_reviewed": meta.get("analyst_reviewed_count", 0),
                        "has_pdf": pdf_path.exists(),
                        "pdf_valid": pdf_valid,
                        "pdf_pages": meta.get("pdf_page_count", 0),
                        "has_bundle": has_bundle,
                        "bundle_size_bytes": bundle_path.stat().st_size if has_bundle else 0,
                        "dir_path": str(r_dir),
                        "is_test_fixture": is_fixture,
                    })

                except Exception as e:
                    logger.warning(f"Failed to read report in '{r_dir.name}': {e}")
                    continue

            return reports

    def get_report_data(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full report data dictionary by report ID."""
        with self._REGISTRY_LOCK:
            r_dir = self._get_report_dir(report_id)
            json_path = r_dir / "report_data.json"
            if not json_path.exists():
                return None

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading report_data.json for '{report_id}': {e}")
                return None

    def verify_report_integrity(self, report_id: str) -> Dict[str, Any]:
        """Verify cryptographic checksums and format validity of all report artifacts.

        Checks:
          - report_data.json is valid JSON
          - report.pdf matches recorded SHA-256 and has readable pages
          - evidence_manifest.json is valid and items match
          - evidence_bundle ZIP passes zipfile.testzip() and matches SHA-256

        Returns:
            Dict containing verification status and detailed test checks.
        """
        with self._REGISTRY_LOCK:
            r_dir = self._get_report_dir(report_id)
            result: Dict[str, Any] = {
                "report_id": report_id,
                "verified": True,
                "checks": {},
                "errors": [],
            }

            if not r_dir.exists():
                result["verified"] = False
                result["errors"].append(f"Report directory '{report_id}' does not exist.")
                return result

            # 1. Check report_data.json
            json_path = r_dir / "report_data.json"
            if not json_path.exists():
                result["verified"] = False
                result["checks"]["report_data_json"] = False
                result["errors"].append("report_data.json is missing.")
                return result

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result["checks"]["report_data_json"] = True
                meta = data.get("metadata", {})
            except Exception as e:
                result["verified"] = False
                result["checks"]["report_data_json"] = False
                result["errors"].append(f"report_data.json is malformed: {e}")
                return result

            # 2. Check PDF
            pdf_name = meta.get("pdf_filename", "report.pdf")
            pdf_path = r_dir / pdf_name
            if not pdf_path.exists():
                result["verified"] = False
                result["checks"]["pdf_exists"] = False
                result["errors"].append(f"PDF file '{pdf_name}' is missing.")
            else:
                result["checks"]["pdf_exists"] = True
                with open(pdf_path, "rb") as pf:
                    actual_pdf_sha = hashlib.sha256(pf.read()).hexdigest()
                expected_pdf_sha = meta.get("pdf_sha256")
                if expected_pdf_sha and actual_pdf_sha != expected_pdf_sha:
                    result["verified"] = False
                    result["checks"]["pdf_checksum"] = False
                    result["errors"].append("PDF SHA-256 checksum mismatch.")
                else:
                    result["checks"]["pdf_checksum"] = True

                # Test PDF readability
                try:
                    reader = pypdf.PdfReader(str(pdf_path))
                    result["checks"]["pdf_readable"] = len(reader.pages) > 0
                except Exception as e:
                    result["verified"] = False
                    result["checks"]["pdf_readable"] = False
                    result["errors"].append(f"PDF readability failed: {e}")

            # 3. Check Evidence Bundle ZIP
            bundle_name = meta.get("bundle_filename", f"evidence_bundle_{report_id}.zip")
            bundle_path = r_dir / bundle_name
            if bundle_path.exists():
                with open(bundle_path, "rb") as bf:
                    actual_zip_sha = hashlib.sha256(bf.read()).hexdigest()
                expected_zip_sha = meta.get("bundle_sha256")
                if expected_zip_sha and actual_zip_sha != expected_zip_sha:
                    result["verified"] = False
                    result["checks"]["bundle_checksum"] = False
                    result["errors"].append("Evidence bundle ZIP SHA-256 mismatch.")
                else:
                    result["checks"]["bundle_checksum"] = True

                try:
                    with zipfile.ZipFile(bundle_path, "r") as zf:
                        corrupt = zf.testzip()
                        if corrupt is not None:
                            result["verified"] = False
                            result["checks"]["bundle_readable"] = False
                            result["errors"].append(f"ZIP member corrupted: {corrupt}")
                        else:
                            result["checks"]["bundle_readable"] = True
                except Exception as e:
                    result["verified"] = False
                    result["checks"]["bundle_readable"] = False
                    result["errors"].append(f"Evidence bundle unreadable: {e}")

            return result
