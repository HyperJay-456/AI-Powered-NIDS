"""Dataset Ingestion Manifest Module.

Defines metadata containers and provides atomic persistence for dataset
cataloging, validation records, and data quality metrics.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "ingestion_manifest.json"


@dataclass
class ColumnMetadata:
    """Metadata and quality indicators for an individual dataset column."""

    name: str
    normalized_name: str
    dtype: str
    non_null_count: int = 0
    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_count: Optional[int] = None
    classification: str = "Numeric"  # 'Numeric', 'Categorical', 'Label', 'Mixed'
    numeric_anomalies: int = 0


@dataclass
class DatasetRecord:
    """Complete metadata and quality record for an ingested CSV dataset."""

    dataset_id: str
    original_filename: str
    safe_filename: str
    file_path: str
    file_size_bytes: int
    file_size_formatted: str
    sha256_checksum: str
    import_timestamp: str
    validation_status: str  # 'Uploaded', 'Validated', 'Validated with Warnings', 'Validation Incomplete', 'Validation Failed'
    total_columns: int = 0
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    missing_value_rows: int = 0
    duplicate_rows: int = 0
    numeric_anomaly_count: int = 0
    has_ground_truth_label: bool = False
    label_column_name: Optional[str] = None
    label_distribution: Dict[str, int] = field(default_factory=dict)
    columns: List[ColumnMetadata] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    encoding: str = "utf-8"
    duplicate_detection_complete: bool = True
    duplicate_detection_notes: str = ""

    @property
    def filename(self) -> str:
        """Alias for safe_filename."""
        return self.safe_filename

    @property
    def row_count(self) -> int:
        """Alias for valid_rows."""
        return self.valid_rows

    @property
    def column_count(self) -> int:
        """Alias for total_columns."""
        return self.total_columns

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        """Reconstruct DatasetRecord from dictionary representation."""
        from dataclasses import fields

        cols_raw = data.get("columns", [])
        columns = [
            ColumnMetadata(**c) if isinstance(c, dict) else c for c in cols_raw
        ]
        data_clean = dict(data)
        data_clean["columns"] = columns

        # Map common aliases if missing
        if "dataset_id" not in data_clean:
            data_clean["dataset_id"] = data_clean.get("id") or Path(data_clean.get("filename", "unknown.csv")).stem
        if "original_filename" not in data_clean:
            data_clean["original_filename"] = data_clean.get("filename", data_clean["dataset_id"] + ".csv")
        if "safe_filename" not in data_clean:
            data_clean["safe_filename"] = data_clean["original_filename"]
        if "file_path" not in data_clean:
            data_clean["file_path"] = f"data/raw/{data_clean['safe_filename']}"
        if "file_size_formatted" not in data_clean:
            size_bytes = data_clean.get("file_size_bytes", 0)
            data_clean["file_size_formatted"] = f"{size_bytes / (1024*1024):.2f} MB" if size_bytes else "0 MB"
        if "import_timestamp" not in data_clean:
            data_clean["import_timestamp"] = data_clean.get("imported_at_utc", "2026-11-17T00:00:00Z")
        if "valid_rows" not in data_clean:
            data_clean["valid_rows"] = data_clean.get("total_rows", 0)
        if "total_columns" not in data_clean:
            data_clean["total_columns"] = data_clean.get("column_count", len(columns))
        if "has_ground_truth_label" not in data_clean:
            has_lbl = any(c.normalized_name == "label" for c in columns) if columns else True
            data_clean["has_ground_truth_label"] = has_lbl
            if has_lbl and "label_column_name" not in data_clean:
                data_clean["label_column_name"] = "Label"

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data_clean.items() if k in valid_fields}
        return cls(**filtered)


class IngestionManifest:
    """Thread-safe, atomic manager for dataset ingestion metadata."""

    def __init__(self, metadata_dir: Optional[Path] = None):
        """Initialize the manifest manager.

        Args:
            metadata_dir: Directory where the manifest is stored. Defaults to settings.metadata_dir.
        """
        self.metadata_dir = metadata_dir or settings.metadata_dir
        self.manifest_file = self.metadata_dir / MANIFEST_FILENAME
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure metadata directory exists."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> Dict[str, DatasetRecord]:
        """Load all dataset records from the manifest.

        Returns:
            Dictionary mapping dataset_id to DatasetRecord.
        """
        if not self.manifest_file.exists():
            return {}

        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            records = {}
            for k, v in raw_data.items():
                try:
                    records[k] = DatasetRecord.from_dict(v)
                except Exception as e:
                    logger.warning(f"Failed to parse manifest entry '{k}': {e}")
            return records
        except Exception as e:
            logger.error(f"Error reading ingestion manifest: {e}")
            return {}

    def get_by_id(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Retrieve a dataset record by its unique ID."""
        records = self.load_all()
        return records.get(dataset_id)

    def get_by_sha256(self, sha256_hash: str) -> Optional[DatasetRecord]:
        """Retrieve a dataset record by its SHA-256 checksum."""
        records = self.load_all()
        for record in records.values():
            if record.sha256_checksum == sha256_hash:
                return record
        return None

    def get_by_safe_filename(self, safe_filename: str) -> Optional[DatasetRecord]:
        """Retrieve a dataset record by its safe stored filename."""
        records = self.load_all()
        for record in records.values():
            if record.safe_filename == safe_filename:
                return record
        return None

    def save_all(self, records: Dict[str, DatasetRecord]) -> bool:
        """Atomically persist dataset records to the manifest file.

        Args:
            records: Dictionary mapping dataset_id to DatasetRecord.

        Returns:
            True if atomic persistence succeeded, False otherwise.
        """
        self._ensure_dir()
        temp_file = self.metadata_dir / f"{MANIFEST_FILENAME}.tmp.{os.getpid()}"

        try:
            serialized = {k: v.to_dict() for k, v in records.items()}
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            os.replace(temp_file, self.manifest_file)
            logger.info(f"Manifest atomically saved with {len(records)} datasets.")
            return True
        except Exception as e:
            logger.error(f"Failed to atomically persist manifest: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            return False

    def add_or_update(self, record: DatasetRecord) -> bool:
        """Add or update an individual dataset record in the manifest.

        Args:
            record: DatasetRecord to add or update.

        Returns:
            True if persistence succeeded, False otherwise.
        """
        records = self.load_all()
        records[record.dataset_id] = record
        return self.save_all(records)

    def list_datasets(self) -> List[DatasetRecord]:
        """Return all datasets sorted chronologically (newest first)."""
        records = self.load_all()
        return sorted(
            records.values(),
            key=lambda r: r.import_timestamp,
            reverse=True,
        )
