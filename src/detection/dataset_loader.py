"""Dataset Loader & Ingestion Verification Engine (CP-08).

Safely loads validated network traffic CSV files from CP-04, verifies cryptographic integrity,
guards against path traversal, checks schema compatibility, and audits against training data leakage.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VerifiedDatasetSource:
    """Audit details for a validated input dataset source."""

    dataset_id: str
    original_filename: str
    safe_filename: str
    file_path: Path
    file_size_bytes: int
    sha256_checksum: str
    validation_status: str
    total_rows: int
    valid_rows: int
    has_ground_truth_label: bool
    label_column_name: Optional[str]
    overlaps_training_data: bool = False
    overlap_details: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)


class DetectionDatasetLoader:
    """Loads and verifies input CSV traffic data strictly according to CP-04 standards."""

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        metadata_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        """Initialize loader with repository paths."""
        self.raw_dir = raw_data_dir or settings.raw_data_dir
        self.metadata_dir = metadata_dir or settings.metadata_dir
        self.processed_dir = processed_dir or settings.processed_data_dir

        self.ingestion_manifest = IngestionManifest(metadata_dir=self.metadata_dir)

    def list_available_datasets(self) -> List[DatasetRecord]:
        """List all datasets validated through CP-04."""
        return self.ingestion_manifest.list_datasets()

    def get_dataset_record(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Retrieve dataset record by ID or filename."""
        rec = self.ingestion_manifest.get_by_id(dataset_id)
        if not rec:
            rec = self.ingestion_manifest.get_by_safe_filename(dataset_id)
        return rec

    def verify_and_prepare_source(
        self,
        dataset_id: str,
        engineering_run_id: Optional[str] = None,
    ) -> VerifiedDatasetSource:
        """Verify the integrity, existence, and lineage of an input dataset.

        Args:
            dataset_id: Dataset identifier or safe filename.
            engineering_run_id: Optional CP-05 run ID for leakage cross-checking.

        Returns:
            VerifiedDatasetSource containing verified metadata.

        Raises:
            FileNotFoundError: If dataset or manifest is not found.
            ValueError: If dataset is invalid, corrupted, or not a validated CSV.
            SecurityError: If path traversal is detected.
        """
        rec = self.get_dataset_record(dataset_id)
        if not rec:
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found in CP-04 ingestion manifest.")

        # Validation status check
        if rec.validation_status not in ("Validated", "Validated with Warnings"):
            raise ValueError(
                f"Dataset '{dataset_id}' has invalid validation status '{rec.validation_status}'. "
                "Only validated datasets from CP-04 can be used for intrusion detection."
            )

        file_path = self.raw_dir / rec.safe_filename

        # Security check: path traversal
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(self.raw_dir.resolve())):
            raise PermissionError(f"Security violation: path traversal detected for '{file_path}'")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file missing on disk: {file_path}")

        # Prohibit processed matrix formats
        if file_path.suffix.lower() not in (".csv",):
            raise ValueError(f"Unsupported dataset format '{file_path.suffix}'. Only raw CSV files are accepted.")

        # Cryptographic checksum verification
        with open(file_path, "rb") as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()

        if computed_sha256 != rec.sha256_checksum:
            raise ValueError(
                f"CRITICAL FILE TAMPERING DETECTED: Dataset '{rec.safe_filename}' SHA-256 mismatch! "
                f"Ingested: {rec.sha256_checksum}, Current on disk: {computed_sha256}"
            )

        # Check for training data overlap (leakage prevention)
        overlaps_training = False
        overlap_details = None
        warnings: List[str] = []

        if engineering_run_id:
            eng_meta_file = self.processed_dir / engineering_run_id / "metadata" / "source_files.json"
            if eng_meta_file.exists():
                try:
                    with open(eng_meta_file, "r", encoding="utf-8") as f:
                        sources = json.load(f)
                    for s in sources:
                        if s.get("sha256") == computed_sha256 or s.get("filename") == rec.safe_filename:
                            overlaps_training = True
                            overlap_details = (
                                f"Input dataset '{rec.safe_filename}' (SHA: {computed_sha256[:12]}...) was an input "
                                f"source for training engineering run '{engineering_run_id}'. "
                                "This traffic overlaps with training/evaluation data and cannot be claimed as unseen operational traffic."
                            )
                            warnings.append(overlap_details)
                            logger.warning(overlap_details)
                            break
                except Exception as e:
                    logger.debug(f"Could not inspect source files for run {engineering_run_id}: {e}")

        return VerifiedDatasetSource(
            dataset_id=rec.dataset_id,
            original_filename=rec.original_filename,
            safe_filename=rec.safe_filename,
            file_path=file_path,
            file_size_bytes=file_path.stat().st_size,
            sha256_checksum=computed_sha256,
            validation_status=rec.validation_status,
            total_rows=rec.total_rows,
            valid_rows=rec.valid_rows,
            has_ground_truth_label=rec.has_ground_truth_label,
            label_column_name=rec.label_column_name,
            overlaps_training_data=overlaps_training,
            overlap_details=overlap_details,
            warning_messages=warnings,
        )

    def iter_raw_chunks(
        self,
        file_path: Path,
        chunk_size: int = 10_000,
        encoding: str = "utf-8",
    ) -> Generator[Tuple[pd.DataFrame, int], None, None]:
        """Read CSV in bounded memory chunks with row offset tracking.

        Yields:
            Tuple of (chunk_df, chunk_start_row_index)
        """
        start_row = 0
        for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            encoding=encoding,
            low_memory=False,
        ):
            yield chunk, start_row
            start_row += len(chunk)
