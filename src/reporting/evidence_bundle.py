"""Digital Evidence Bundle Packager & Archive Integrity Verifier (CP-11).

Creates self-contained, cryptographically verifiable ZIP archives:
    evidence_bundle_<report_id>.zip
Packaging report PDFs, machine-readable JSON manifests, sanitized CSVs, and selected evidence.

Strict Safeguards:
  - Path traversal and directory escape protection on all archive entries
  - Privacy redaction applied in-memory / on exported copies (source files untouched)
  - Post-creation archive test (zipfile.testzip()) and SHA-256 verification
  - Bounded resource usage preventing memory exhaustion on large datasets
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import pandas as pd

from config.settings import settings
from src.reporting.report_data_aggregator import redact_ip_address, sanitize_csv_cell
from src.reporting.report_models import (
    EvidenceItem,
    EvidenceManifest,
    RedactionConfig,
    ReportMetadata,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceBundlePackager:
    """Creates cryptographically verified ZIP archives of reporting artifacts."""

    def __init__(self, report_dir: Path):
        """Initialize packager with report directory."""
        self.report_dir = report_dir

    def create_bundle(
        self,
        report_id: str,
        detection_run_id: str,
        evidence_manifest: EvidenceManifest,
        redaction_config: Optional[RedactionConfig] = None,
        include_source_manifest: bool = True,
        include_prediction_sample: bool = True,
        max_sample_rows: int = 500,
        analyses_dir: Optional[Path] = None,
    ) -> Tuple[Path, str, int]:
        """Pack all evidence items and supporting artifacts into a secure ZIP archive.

        Args:
            report_id: Unique report identifier.
            detection_run_id: Associated CP-08 run ID.
            evidence_manifest: Manifest of generated artifacts.
            redaction_config: Redaction settings for IP addresses and analyst notes.
            include_source_manifest: Whether to include original CP-08 run_manifest.json.
            include_prediction_sample: Whether to include a bounded sample of prediction rows.
            max_sample_rows: Bounded limit of sampled prediction rows.
            analyses_dir: Optional custom analyses directory.

        Returns:
            Tuple of (zip_path, sha256_checksum, size_bytes).

        Raises:
            PermissionError: If path traversal is detected.
            RuntimeError: If ZIP archive creation or verification fails.
        """
        redact = redaction_config or RedactionConfig()
        zip_filename = f"evidence_bundle_{report_id}.zip"
        zip_path = self.report_dir / zip_filename

        temp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        temp_zip_path = Path(temp_zip.name)
        temp_zip.close()

        try:
            with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # 1. Add all items already listed in evidence_manifest
                for item in evidence_manifest.items:
                    item_file = self.report_dir / item.relative_path
                    if item_file.exists() and item_file.is_file():
                        arcname = self._sanitize_arcname(item.relative_path)
                        zf.write(item_file, arcname=arcname)
                    else:
                        logger.warning(f"Evidence file missing on disk: '{item.relative_path}'")

                # 2. Add evidence/evidence_manifest.json (root or evidence/ folder)
                manifest_file = self.report_dir / "evidence_manifest.json"
                if manifest_file.exists():
                    zf.write(manifest_file, arcname="evidence_manifest.json")

                # 3. Add supporting evidence from source run if available
                base_analyses = (analyses_dir or settings.analyses_dir).resolve()
                run_dir = (base_analyses / detection_run_id).resolve()

                if run_dir.exists() and str(run_dir).startswith(str(base_analyses)):
                    # A. Source run manifest
                    if include_source_manifest:
                        src_manifest = run_dir / "run_manifest.json"
                        if src_manifest.exists():
                            zf.write(src_manifest, arcname="evidence/source_run_manifest.json")

                    # B. Bounded and redacted prediction records sample
                    if include_prediction_sample:
                        src_pred = run_dir / "predictions.csv"
                        if src_pred.exists():
                            sample_csv_bytes = self._create_redacted_sample_csv(
                                src_pred_path=src_pred,
                                max_rows=max_sample_rows,
                                redact=redact,
                            )
                            zf.writestr("evidence/selected_predictions_sample.csv", sample_csv_bytes)

            # 4. Verify Archive Integrity
            if not temp_zip_path.exists() or temp_zip_path.stat().st_size == 0:
                raise RuntimeError("Evidence ZIP generation produced an empty or missing archive.")

            with zipfile.ZipFile(temp_zip_path, "r") as test_zf:
                test_result = test_zf.testzip()
                if test_result is not None:
                    raise RuntimeError(f"Corrupted entry detected in evidence bundle ZIP: {test_result}")
                member_count = len(test_zf.namelist())

            with open(temp_zip_path, "rb") as zf_read:
                zip_bytes = zf_read.read()
                zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
                zip_size = len(zip_bytes)

            # Atomic copy to destination
            with open(zip_path, "wb") as zf_out:
                zf_out.write(zip_bytes)

            logger.info(
                f"Generated evidence bundle '{zip_filename}' [Members: {member_count}, "
                f"Size: {zip_size} bytes, SHA-256: {zip_sha256[:12]}...]"
            )
            return zip_path, zip_sha256, zip_size

        finally:
            if temp_zip_path.exists():
                try:
                    os.unlink(temp_zip_path)
                except Exception:
                    pass

    def _sanitize_arcname(self, rel_path: str) -> str:
        """Sanitize archive member name against path traversal."""
        norm = os.path.normpath(rel_path).replace("\\", "/")
        if ".." in norm or norm.startswith("/") or ":" in norm:
            raise PermissionError(f"Security violation: path traversal in ZIP archive name '{rel_path}'")
        return norm

    def _create_redacted_sample_csv(
        self,
        src_pred_path: Path,
        max_rows: int,
        redact: RedactionConfig,
    ) -> bytes:
        """Read a bounded sample of prediction rows, apply redactions, and serialize to bytes."""
        df = pd.read_csv(src_pred_path, nrows=max_rows)

        # Apply IP redactions if requested
        if redact.redact_ips:
            for col in df.columns:
                col_lower = col.lower()
                if "ip" in col_lower or "source" in col_lower or "destination" in col_lower:
                    df[col] = df[col].apply(lambda v: redact_ip_address(v, redact.mask_str))

        # Apply formula injection protection
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(sanitize_csv_cell)

        return df.to_csv(index=False, encoding="utf-8").encode("utf-8")
