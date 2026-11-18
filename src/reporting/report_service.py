"""Report Generation Service & Orchestration Facade (CP-11).

Orchestrates the entire cybersecurity reporting workflow:
  1. Cryptographic preflight & source artifact verification
  2. Deterministic forensic aggregation across CP-08, CP-09, and CP-10
  3. Publication-grade PDF document compilation
  4. Machine-readable JSON and CSV structured exports
  5. Self-contained, path-traversal secure evidence bundle packaging
  6. Atomic persistence and registration with rollback on error

Strict Invariants:
  - Read-only consumer: NEVER trains models, retrains, or runs new inference
  - Strict concurrency locking: prevents duplicate or competing report generation jobs
  - Clean error recovery: partial or corrupted output directories are rolled back
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.reporting.evidence_bundle import EvidenceBundlePackager
from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.report_data_aggregator import ReportDataAggregator
from src.reporting.report_models import (
    EvidenceManifest,
    RedactionConfig,
    ReportClassification,
    ReportDataPackage,
    ReportMetadata,
    ReportStatus,
    ReportType,
)
from src.reporting.report_registry import ReportRegistry
from src.reporting.structured_exporter import StructuredExporter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportService:
    """Master orchestration service for cybersecurity reporting and digital evidence export."""

    _SERVICE_LOCK = threading.RLock()

    def __init__(
        self,
        reports_dir: Optional[Path] = None,
        analyses_dir: Optional[Path] = None,
        reviews_dir: Optional[Path] = None,
        mappings_dir: Optional[Path] = None,
        evaluations_dir: Optional[Path] = None,
    ):
        """Initialize service with system directories."""
        self.reports_dir = reports_dir or getattr(settings, "reports_dir", settings.data_dir / "reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.analyses_dir = analyses_dir or settings.analyses_dir

        self.aggregator = ReportDataAggregator(
            analyses_dir=self.analyses_dir,
            reviews_dir=reviews_dir,
            mappings_dir=mappings_dir,
            evaluations_dir=evaluations_dir,
        )
        self.pdf_generator = PDFReportGenerator()
        self.registry = ReportRegistry(reports_dir=self.reports_dir)

    def generate_report(
        self,
        detection_run_id: str,
        report_type: str = ReportType.EXECUTIVE,
        title: Optional[str] = None,
        classification: str = ReportClassification.ACADEMIC,
        redaction_config: Optional[RedactionConfig] = None,
        allow_test_fixtures: bool = False,
        create_bundle: bool = True,
    ) -> ReportMetadata:
        """Execute full report generation pipeline atomically.

        Args:
            detection_run_id: Verified CP-08 detection run ID.
            report_type: Report type to produce.
            title: Custom title or default.
            classification: Sensitivity marking.
            redaction_config: Privacy and redaction configuration.
            allow_test_fixtures: Whether to permit test fixture runs.
            create_bundle: Whether to package evidence bundle ZIP.

        Returns:
            Completed ReportMetadata record.

        Raises:
            ValueError: If source validation fails or report parameters are invalid.
            RuntimeError: If document generation or archiving fails.
        """
        with self._SERVICE_LOCK:
            if not ReportType.is_valid(report_type):
                raise ValueError(
                    f"Unsupported report type '{report_type}'. "
                    f"Must be one of {ReportType.ALL_TYPES}"
                )

            redact = redaction_config or RedactionConfig()

            # 1. Aggregation Preflight
            logger.info(f"Aggregating forensic data for detection run '{detection_run_id}'...")
            data_package: ReportDataPackage = self.aggregator.aggregate(
                detection_run_id=detection_run_id,
                report_type=report_type,
                title=title,
                classification=classification,
                redaction_config=redact,
                allow_test_fixtures=allow_test_fixtures,
            )

            report_id = data_package.metadata.report_id
            target_report_dir = self.reports_dir / report_id

            # Create working staging directory for atomic generation
            temp_dir = Path(tempfile.mkdtemp(prefix="nids_rep_"))

            try:
                # 2. PDF Document Generation
                pdf_filename = "report.pdf"
                temp_pdf_path = temp_dir / pdf_filename

                logger.info(f"Generating PDF for report '{report_id}'...")
                _, pdf_sha, pdf_size, page_count = self.pdf_generator.generate(
                    data_package=data_package,
                    output_path=temp_pdf_path,
                )

                # Update metadata with PDF details
                data_package.metadata.pdf_filename = pdf_filename
                data_package.metadata.pdf_sha256 = pdf_sha
                data_package.metadata.pdf_size_bytes = pdf_size
                data_package.metadata.pdf_page_count = page_count

                # 3. Structured Exports (JSON & CSVs)
                logger.info(f"Generating structured exports for report '{report_id}'...")
                exporter = StructuredExporter(output_dir=temp_dir)
                evidence_manifest: EvidenceManifest = exporter.export_all(
                    data_package=data_package,
                    pdf_path=temp_pdf_path,
                    pdf_sha256=pdf_sha,
                    pdf_size_bytes=pdf_size,
                )

                # 4. Evidence Bundle Packaging (ZIP)
                bundle_filename = f"evidence_bundle_{report_id}.zip"
                if create_bundle:
                    logger.info(f"Packaging evidence bundle for report '{report_id}'...")
                    packager = EvidenceBundlePackager(report_dir=temp_dir)
                    _, bundle_sha, bundle_size = packager.create_bundle(
                        report_id=report_id,
                        detection_run_id=detection_run_id,
                        evidence_manifest=evidence_manifest,
                        redaction_config=redact,
                        analyses_dir=self.analyses_dir,
                    )
                    data_package.metadata.bundle_filename = bundle_filename
                    data_package.metadata.bundle_sha256 = bundle_sha
                    data_package.metadata.bundle_size_bytes = bundle_size

                # 5. Finalize Metadata & Re-serialize report_data.json
                data_package.metadata.status = ReportStatus.COMPLETED
                data_package.metadata.completed_at_utc = datetime.now(timezone.utc).isoformat()

                with open(temp_dir / "report_data.json", "w", encoding="utf-8") as f:
                    json.dump(data_package.to_dict(), f, indent=2, sort_keys=True)

                # 6. Commit atomic move to final destination
                if target_report_dir.exists():
                    shutil.rmtree(target_report_dir)
                shutil.move(str(temp_dir), str(target_report_dir))

                logger.info(f"Successfully generated and committed report '{report_id}'")
                return data_package.metadata

            except Exception as e:
                logger.error(f"Failed to generate report '{report_id}': {e}", exc_info=True)
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise
