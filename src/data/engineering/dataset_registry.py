"""Dataset Registry Module for Dataset Engineering (CP-05).

Provides verified source dataset discovery, manifest integrity verification,
SHA-256 validation against disk, and eligibility filtering.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetIntegrityCheck:
    """Outcome of verifying a dataset file on disk against its manifest record."""

    dataset_id: str
    filename: str
    is_eligible: bool
    exists_on_disk: bool
    checksum_matches: bool
    registered_checksum: str
    computed_checksum: str
    total_rows: int
    total_columns: int
    validation_status: str
    rejection_reason: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow evaluating integrity check as a boolean."""
        return bool(self.is_eligible and self.checksum_matches)


class DatasetRegistry:
    """Discovers, audits, and gates datasets eligible for dataset engineering."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        metadata_dir: Optional[Path] = None,
    ):
        """Initialize registry with data and metadata locations."""
        self.raw_dir = raw_dir or settings.raw_data_dir
        self.metadata_dir = metadata_dir or settings.metadata_dir
        self.manifest = IngestionManifest(metadata_dir=self.metadata_dir)

    def reload(self) -> None:
        """Reload manifest records from disk."""
        self.manifest.load_all()

    def list_all_records(self) -> List[DatasetRecord]:
        """Return all cataloged dataset records."""
        return self.manifest.list_datasets()

    def get_record(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Retrieve dataset record by identifier."""
        return self.manifest.get_by_id(dataset_id)

    def verify_source_integrity(self, record: Any) -> Any:
        """Verify that a dataset exists, is undamaged, and matches its recorded checksum.

        Args:
            record: DatasetRecord or filename / dataset ID string.

        Returns:
            bool if called with a filename string, or DatasetIntegrityCheck if called with DatasetRecord.
        """
        if isinstance(record, str):
            # Try to resolve by dataset_id or filename
            rec_obj = self.get_record(record)
            if not rec_obj:
                for r in self.list_all_records():
                    if r.safe_filename == record or r.original_filename == record:
                        rec_obj = r
                        break
            if not rec_obj:
                file_path = self.raw_dir / record
                return file_path.exists()

            check = self._verify_record_integrity(rec_obj)
            return bool(check.is_eligible and check.checksum_matches)

        return self._verify_record_integrity(record)

    def _verify_record_integrity(self, record: DatasetRecord) -> DatasetIntegrityCheck:
        """Perform disk and checksum integrity verification for a DatasetRecord."""
        file_path = self.raw_dir / record.safe_filename
        if not file_path.exists():
            return DatasetIntegrityCheck(
                dataset_id=record.dataset_id,
                filename=record.safe_filename,
                is_eligible=False,
                exists_on_disk=False,
                checksum_matches=False,
                registered_checksum=record.sha256_checksum,
                computed_checksum="",
                total_rows=record.total_rows,
                total_columns=record.total_columns,
                validation_status=record.validation_status,
                rejection_reason=f"Target file does not exist on disk: {file_path}",
            )

        # Compute SHA-256 from disk
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    hasher.update(chunk)
            computed_hash = hasher.hexdigest()
        except Exception as e:
            return DatasetIntegrityCheck(
                dataset_id=record.dataset_id,
                filename=record.safe_filename,
                is_eligible=False,
                exists_on_disk=True,
                checksum_matches=False,
                registered_checksum=record.sha256_checksum,
                computed_checksum="",
                total_rows=record.total_rows,
                total_columns=record.total_columns,
                validation_status=record.validation_status,
                rejection_reason=f"Failed to read file from disk for hashing: {e}",
            )

        checksum_matches = computed_hash == record.sha256_checksum
        if not checksum_matches:
            return DatasetIntegrityCheck(
                dataset_id=record.dataset_id,
                filename=record.safe_filename,
                is_eligible=False,
                exists_on_disk=True,
                checksum_matches=False,
                registered_checksum=record.sha256_checksum,
                computed_checksum=computed_hash,
                total_rows=record.total_rows,
                total_columns=record.total_columns,
                validation_status=record.validation_status,
                rejection_reason=(
                    f"Checksum mismatch on '{record.safe_filename}': "
                    f"manifest={record.sha256_checksum[:8]}... vs disk={computed_hash[:8]}..."
                ),
            )

        # Validation status check: Only 'Validated' or 'Validated with Warnings' are eligible
        eligible_statuses = ("Validated", "Validated with Warnings")
        if record.validation_status not in eligible_statuses:
            return DatasetIntegrityCheck(
                dataset_id=record.dataset_id,
                filename=record.safe_filename,
                is_eligible=False,
                exists_on_disk=True,
                checksum_matches=True,
                registered_checksum=record.sha256_checksum,
                computed_checksum=computed_hash,
                total_rows=record.total_rows,
                total_columns=record.total_columns,
                validation_status=record.validation_status,
                rejection_reason=(
                    f"Dataset validation status '{record.validation_status}' is ineligible. "
                    "Must be 'Validated' or 'Validated with Warnings'."
                ),
            )

        # Zero rows check
        if record.valid_rows == 0:
            return DatasetIntegrityCheck(
                dataset_id=record.dataset_id,
                filename=record.safe_filename,
                is_eligible=False,
                exists_on_disk=True,
                checksum_matches=True,
                registered_checksum=record.sha256_checksum,
                computed_checksum=computed_hash,
                total_rows=record.total_rows,
                total_columns=record.total_columns,
                validation_status=record.validation_status,
                rejection_reason="Dataset contains 0 valid data rows (header only).",
            )

        return DatasetIntegrityCheck(
            dataset_id=record.dataset_id,
            filename=record.safe_filename,
            is_eligible=True,
            exists_on_disk=True,
            checksum_matches=True,
            registered_checksum=record.sha256_checksum,
            computed_checksum=computed_hash,
            total_rows=record.total_rows,
            total_columns=record.total_columns,
            validation_status=record.validation_status,
            rejection_reason=None,
        )

    def list_eligible_datasets(self) -> List[Tuple[DatasetRecord, DatasetIntegrityCheck]]:
        """List all datasets that pass integrity and validation checks."""
        self.reload()
        eligible: List[Tuple[DatasetRecord, DatasetIntegrityCheck]] = []
        for rec in self.list_all_records():
            check = self.verify_source_integrity(rec)
            if check.is_eligible:
                eligible.append((rec, check))
        return eligible

    def get_eligible_sources(self) -> List[DatasetRecord]:
        """Return list of eligible DatasetRecords for engineering."""
        return [rec for rec, chk in self.list_eligible_datasets()]
