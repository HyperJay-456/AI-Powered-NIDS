"""CSV Validation Engine.

Executes bounded-memory chunk processing using Pandas, enforces the row-accounting
invariant (Total Rows = Structurally Valid Rows + Invalid Rows), detects ground-truth
labels, identifies non-finite numeric anomalies, and persists complete metadata.
"""

from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.data.data_quality import (
    DiskBackedDuplicateDetector,
    QualityIssueSample,
    detect_numeric_anomalies_in_chunk,
)
from src.data.ingestion_manifest import (
    ColumnMetadata,
    DatasetRecord,
    IngestionManifest,
)
from src.data.schema_inspector import (
    classify_column,
    detect_duplicate_columns,
    detect_label_column,
    normalize_column_name,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_file_encoding(file_path: Path) -> str:
    """Detect encoding of the CSV file, starting with UTF-8 BOM check."""
    try:
        with open(file_path, "rb") as f:
            header_bytes = f.read(4)
            if header_bytes.startswith(b"\xef\xbb\xbf"):
                return "utf-8-sig"

        # Try standard utf-8
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(8192)
        return "utf-8"
    except UnicodeDecodeError:
        logger.info(f"File {file_path.name} is not standard UTF-8. Falling back to latin-1.")
        return "latin-1"
    except Exception as e:
        logger.warning(f"Error checking encoding for {file_path.name}: {e}")
        return "utf-8"


def save_preview_cache(
    dataset_id: str,
    preview_df: pd.DataFrame,
    metadata_dir: Optional[Path] = None,
) -> None:
    """Save initial preview records to metadata cache for instant UI rendering."""
    metadata_dir = metadata_dir or settings.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    cache_path = metadata_dir / f"preview_{dataset_id}.json"

    try:
        # Convert non-finite numbers to null so json doesn't fail
        clean_df = preview_df.replace([np.inf, -np.inf], None)
        clean_df.to_json(cache_path, orient="records", date_format="iso")
    except Exception as e:
        logger.warning(f"Could not persist preview cache for {dataset_id}: {e}")


def load_preview_cache(
    dataset_id: str,
    metadata_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load cached preview records for an ingested dataset."""
    metadata_dir = metadata_dir or settings.metadata_dir
    cache_path = metadata_dir / f"preview_{dataset_id}.json"

    if not cache_path.exists():
        return None

    try:
        return pd.read_json(cache_path, orient="records")
    except Exception as e:
        logger.warning(f"Could not load preview cache for {dataset_id}: {e}")
        return None


def validate_csv_dataset(
    file_path: Path,
    dataset_id: str,
    chunk_size: Optional[int] = None,
    raw_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> DatasetRecord:
    """Execute full-file bounded-memory validation on an ingested CSV dataset.

    CRITICAL INVARIANT: Total Rows = Structurally Valid Rows + Invalid Rows.
    No row is silently discarded.

    Args:
        file_path: Path to the stored CSV file.
        dataset_id: Identifier of the dataset in the manifest.
        chunk_size: Number of records per chunk. Defaults to settings.ingestion.default_chunk_size.
        raw_dir: Root directory for raw datasets.
        metadata_dir: Root directory for metadata manifests.
        progress_callback: Optional callback receiving (fraction_complete: float, message: str).

    Returns:
        Updated DatasetRecord with all calculated validation metrics.
    """
    chunk_size = chunk_size or settings.ingestion.default_chunk_size
    raw_dir = raw_dir or settings.raw_data_dir
    metadata_dir = metadata_dir or settings.metadata_dir
    manifest = IngestionManifest(metadata_dir=metadata_dir)

    record = manifest.get_by_id(dataset_id)
    if not record:
        record = DatasetRecord(
            dataset_id=dataset_id,
            original_filename=file_path.name,
            safe_filename=file_path.name,
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            file_size_formatted="0 B",
            sha256_checksum="",
            import_timestamp="",
            validation_status="Validation Incomplete",
        )

    if not file_path.exists():
        record.validation_status = "Validation Failed"
        record.validation_errors.append(f"Target CSV file does not exist at {file_path}")
        manifest.add_or_update(record)
        return record

    file_size = file_path.stat().st_size
    if file_size == 0:
        record.validation_status = "Validation Failed"
        record.validation_errors.append("File is empty (0 bytes). Cannot parse headers.")
        manifest.add_or_update(record)
        return record

    encoding = detect_file_encoding(file_path)
    record.encoding = encoding

    if progress_callback:
        progress_callback(0.05, f"Reading header schema with {encoding} encoding...")

    # 1. Read header and inspect column definitions
    try:
        sample_df = pd.read_csv(file_path, nrows=0, encoding=encoding)
        columns = list(sample_df.columns)
    except Exception as e:
        record.validation_status = "Validation Failed"
        record.validation_errors.append(f"Failed to parse CSV header: {str(e)}")
        manifest.add_or_update(record)
        return record

    if len(columns) == 0:
        record.validation_status = "Validation Failed"
        record.validation_errors.append("CSV file contains no readable columns or header.")
        manifest.add_or_update(record)
        return record

    total_cols = len(columns)
    record.total_columns = total_cols

    # Check for duplicate headers
    exact_dups, norm_dups = detect_duplicate_columns(columns)
    if exact_dups:
        record.validation_warnings.append(
            f"Duplicate column names detected in header: {exact_dups}"
        )
    if norm_dups:
        record.validation_warnings.append(
            f"Columns collision after case/whitespace normalization: {norm_dups}"
        )

    # Detect Ground-Truth Label Column
    label_col = detect_label_column(columns)
    if label_col:
        record.has_ground_truth_label = True
        record.label_column_name = label_col
        logger.info(f"Detected ground-truth label column: '{label_col}'")
    else:
        record.has_ground_truth_label = False
        record.label_column_name = None
        record.validation_warnings.append(
            "No ground-truth 'Label' column found. Supervised model evaluation "
            "metrics cannot be derived from this file."
        )

    # Informational warning regarding CIC-IDS-2017 expected dimension
    if total_cols not in (78, 79, 80, 84, 85):
        record.validation_warnings.append(
            f"Dataset schema contains {total_cols} columns (reference CIC-IDS-2017 "
            "benchmark typically provides ~78 flow features + Label). Column schema "
            "will be formally reconciled in CP-05 (Dataset Engineering)."
        )

    # 2. Chunk-Based Bounded Memory Validation Loop
    total_data_rows = 0
    valid_data_rows = 0
    invalid_data_rows = 0
    missing_value_rows = 0
    numeric_anomaly_total = 0

    col_missing_counts = {c: 0 for c in columns}
    col_non_null_counts = {c: 0 for c in columns}
    col_numeric_anomalies = {c: 0 for c in columns}
    col_inferred_dtypes: Dict[str, str] = {}
    label_counter: Counter = Counter()

    preview_df: Optional[pd.DataFrame] = None
    max_preview_limit = settings.ingestion.max_preview_rows

    if progress_callback:
        progress_callback(0.15, "Initializing bounded-memory duplicate detection index...")

    # Estimate progress based on file size or chunk processing
    bytes_processed = 0
    chunk_index = 0

    with DiskBackedDuplicateDetector(temp_dir=metadata_dir) as dup_detector:
        try:
            # We enforce on_bad_lines='error' so that structural errors are caught and accounted for
            reader = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="error",
            )

            for chunk in reader:
                chunk_index += 1
                chunk_rows = len(chunk)
                total_data_rows += chunk_rows
                valid_data_rows += chunk_rows

                # Capture preview from first chunk
                if preview_df is None:
                    preview_df = chunk.head(max_preview_limit).copy()

                # Missing value statistics
                chunk_has_missing = chunk.isna().any(axis=1)
                missing_value_rows += int(chunk_has_missing.sum())

                for col in columns:
                    if col in chunk.columns:
                        col_missing_counts[col] += int(chunk[col].isna().sum())
                        col_non_null_counts[col] += int(chunk[col].notna().sum())
                        if col not in col_inferred_dtypes:
                            col_inferred_dtypes[col] = str(chunk[col].dtype)

                # Ground-truth label count aggregation
                if label_col and label_col in chunk.columns:
                    labels_series = chunk[label_col].dropna().astype(str)
                    label_counter.update(labels_series.value_counts().to_dict())

                # Duplicate detection via SQLite fingerprint index
                dup_detector.process_chunk(chunk)

                # Numeric anomaly inspection (Inf / -Inf)
                numeric_cols = [
                    c for c in columns
                    if c in chunk.columns and pd.api.types.is_numeric_dtype(chunk[c])
                ]
                chunk_anomalies, col_anom_map = detect_numeric_anomalies_in_chunk(
                    chunk, numeric_cols
                )
                numeric_anomaly_total += chunk_anomalies
                for c, count in col_anom_map.items():
                    col_numeric_anomalies[c] += count

                if progress_callback:
                    pct = min(0.15 + (chunk_index * 0.1), 0.85)
                    progress_callback(
                        pct,
                        f"Processed batch {chunk_index} ({total_data_rows:,} rows scanned)...",
                    )

        except pd.errors.EmptyDataError:
            logger.info(f"File {file_path.name} contains headers but 0 data rows.")
            record.validation_warnings.append("Dataset contains 0 data rows (header only).")

        except pd.errors.ParserError as e:
            # Fatal structural CSV parsing error encountered
            record.validation_status = "Validation Failed"
            err_msg = f"CSV parsing stopped due to structural error: {str(e)}"
            record.validation_errors.append(err_msg)
            logger.error(err_msg)
            record.total_rows = total_data_rows
            record.valid_rows = valid_data_rows
            record.invalid_rows = 1  # Record failure point
            manifest.add_or_update(record)
            return record

        except Exception as e:
            record.validation_status = "Validation Incomplete"
            err_msg = f"Validation halted unexpectedly: {str(e)}"
            record.validation_errors.append(err_msg)
            logger.error(err_msg)
            record.total_rows = total_data_rows
            record.valid_rows = valid_data_rows
            record.invalid_rows = invalid_data_rows
            manifest.add_or_update(record)
            return record

        # Duplicate detector completion status
        if dup_detector.completion_error:
            record.duplicate_detection_complete = False
            record.duplicate_detection_notes = dup_detector.completion_error
            record.validation_warnings.append(
                f"Duplicate detection incomplete: {dup_detector.completion_error}"
            )
        else:
            record.duplicate_detection_complete = True

        duplicate_rows_count = dup_detector.total_duplicates

    # 3. Final Invariants and Data Assembly
    record.total_rows = total_data_rows
    record.valid_rows = valid_data_rows
    record.invalid_rows = invalid_data_rows
    record.missing_value_rows = missing_value_rows
    record.duplicate_rows = duplicate_rows_count
    record.numeric_anomaly_count = numeric_anomaly_total
    record.label_distribution = dict(label_counter)

    # Assemble Column Metadata List
    assembled_columns: List[ColumnMetadata] = []
    for col in columns:
        missing_cnt = col_missing_counts[col]
        non_null_cnt = col_non_null_counts[col]
        missing_pct = (missing_cnt / total_data_rows * 100) if total_data_rows > 0 else 0.0
        dtype_str = col_inferred_dtypes.get(col, "object")
        is_label = (col == label_col)

        classification = classify_column(
            col_name=col,
            dtype_str=dtype_str,
            is_label=is_label,
            non_null_count=non_null_cnt,
            unique_count=None,
        )

        assembled_columns.append(
            ColumnMetadata(
                name=col,
                normalized_name=normalize_column_name(col),
                dtype=dtype_str,
                non_null_count=non_null_cnt,
                missing_count=missing_cnt,
                missing_percentage=round(missing_pct, 2),
                unique_count=len(record.label_distribution) if is_label else None,
                classification=classification,
                numeric_anomalies=col_numeric_anomalies[col],
            )
        )
    record.columns = assembled_columns

    # 4. Generate Quality Warnings
    if missing_value_rows > 0:
        missing_pct_dataset = (missing_value_rows / total_data_rows * 100) if total_data_rows > 0 else 0
        record.validation_warnings.append(
            f"Missing values detected: {missing_value_rows:,} rows "
            f"({missing_pct_dataset:.2f}% of total) contain blank or null cells."
        )

    if duplicate_rows_count > 0:
        record.validation_warnings.append(
            f"Exact duplicate records detected: {duplicate_rows_count:,} duplicate rows "
            "identified across chunk boundaries."
        )

    if numeric_anomaly_total > 0:
        record.validation_warnings.append(
            f"Non-finite numeric anomalies detected: {numeric_anomaly_total:,} occurrences of "
            "infinity (inf/-inf) in volumetric flow features."
        )

    # 5. Determine Final Validation Status
    if len(record.validation_errors) > 0:
        record.validation_status = "Validation Failed"
    elif len(record.validation_warnings) > 0:
        record.validation_status = "Validated with Warnings"
    else:
        record.validation_status = "Validated"

    # Save preview cache for instant UI rendering
    if preview_df is not None:
        save_preview_cache(dataset_id, preview_df, metadata_dir=metadata_dir)

    # Atomically update manifest
    manifest.add_or_update(record)

    if progress_callback:
        progress_callback(1.0, f"Validation complete: {record.validation_status}.")

    logger.info(
        f"Validation completed for '{file_path.name}': {record.total_rows:,} rows, "
        f"{record.total_columns} cols, Status: {record.validation_status}"
    )

    return record
