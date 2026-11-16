"""Data Quality and Anomaly Detection Module.

Implements bounded-memory duplicate row detection using disk-backed SQLite
fingerprint indexing, numeric anomaly tracking (non-finite/infinite flows),
and transparent row-level accounting.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityIssueSample:
    """Diagnostic sample representing a row-level data quality issue."""

    row_index: int
    column_name: str
    issue_type: str  # 'Missing Value', 'Infinite Flow Rate', 'Duplicate Record', 'Structural Error'
    description: str


def serialize_row_length_safe(row: Tuple[Any, ...]) -> bytes:
    """Deterministically serialize a tuple of cell values into length-prefixed bytes.

    Eliminates delimiter collision vulnerabilities (e.g., ['ab', 'c'] vs ['a', 'bc'],
    ['a|b', 'c'] vs ['a', 'b|c'], embedded commas, newlines, and distinguishes
    missing/null values from empty strings or literal sentinel strings).

    Format per cell:
      - Null / NaN (missing value): b"N;"
      - String content: b"S<byte_length>:<utf8_bytes>;"
    """
    parts: List[bytes] = []
    for val in row:
        if val is None or pd.isna(val):
            parts.append(b"N;")
        else:
            s_bytes = str(val).encode("utf-8", errors="replace")
            parts.append(f"S{len(s_bytes)}:".encode("ascii") + s_bytes + b";")
    return b"".join(parts)


class DiskBackedDuplicateDetector:
    """Bounded-memory exact duplicate row detector powered by a disk-backed SQLite index.

    Calculates cryptographic SHA-256 fingerprints for each row and tracks collisions
    across chunk boundaries without keeping full dataset tables in system RAM.
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize the disk-backed fingerprint store."""
        self.temp_file: Optional[Path] = None
        self.conn: Optional[sqlite3.Connection] = None
        self.is_active: bool = False
        self.total_duplicates: int = 0
        self.completion_error: Optional[str] = None

        try:
            fd, path_str = tempfile.mkstemp(
                prefix="nids_dup_index_",
                suffix=".sqlite3",
                dir=str(temp_dir) if temp_dir else None,
            )
            os.close(fd)
            self.temp_file = Path(path_str)

            self.conn = sqlite3.connect(str(self.temp_file))
            # Configure SQLite for fast write throughput with bounded cache
            self.conn.execute("PRAGMA synchronous = OFF;")
            self.conn.execute("PRAGMA journal_mode = MEMORY;")
            self.conn.execute("PRAGMA cache_size = -2000;")  # ~2 MB cache

            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS fingerprints (fp_hash BLOB PRIMARY KEY);"
            )
            self.conn.commit()
            self.is_active = True
        except Exception as e:
            self.completion_error = f"Failed to initialize SQLite duplicate index: {e}"
            logger.error(self.completion_error)
            self._cleanup()

    def process_chunk(self, df: pd.DataFrame) -> int:
        """Process a chunk of records and count duplicates both within and across chunks.

        Args:
            df: DataFrame chunk to fingerprint.

        Returns:
            Number of duplicate rows discovered in this chunk.
        """
        if not self.is_active or not self.conn:
            return 0

        if df.empty:
            return 0

        try:
            # Generate unambiguous, length-safe serialized representation of each row
            hashes: List[Tuple[bytes]] = []
            for row in df.itertuples(index=False, name=None):
                row_bytes = serialize_row_length_safe(row)
                hashes.append((hashlib.sha256(row_bytes).digest(),))

            initial_count = self.conn.execute(
                "SELECT COUNT(*) FROM fingerprints;"
            ).fetchone()[0]

            self.conn.executemany(
                "INSERT OR IGNORE INTO fingerprints (fp_hash) VALUES (?);",
                hashes,
            )
            self.conn.commit()

            new_count = self.conn.execute(
                "SELECT COUNT(*) FROM fingerprints;"
            ).fetchone()[0]

            unique_added = new_count - initial_count
            chunk_duplicates = len(hashes) - unique_added
            self.total_duplicates += chunk_duplicates
            return chunk_duplicates

        except Exception as e:
            self.completion_error = f"Duplicate detection interrupted during chunk processing: {e}"
            logger.error(self.completion_error)
            return 0

    def close(self) -> None:
        """Close connection and remove temporary database file."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Internal resource cleanup."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        if self.temp_file and self.temp_file.exists():
            try:
                self.temp_file.unlink()
            except OSError:
                pass
            self.temp_file = None
        self.is_active = False

    def __enter__(self) -> "DiskBackedDuplicateDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def detect_numeric_anomalies_in_chunk(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> Tuple[int, Dict[str, int]]:
    """Detect non-finite numeric anomalies (np.inf, -np.inf) across numeric columns in a chunk.

    In network flow datasets (specifically CIC-IDS-2017), volumetric flow rates
    like 'Flow Bytes/s' and 'Flow Packets/s' frequently contain infinite values.

    Args:
        df: DataFrame chunk.
        numeric_columns: Columns identified as numeric.

    Returns:
        Tuple of (chunk_total_anomalies, anomaly_count_by_column).
    """
    total_anomalies = 0
    column_anomalies: Dict[str, int] = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue
        series = df[col]
        # Only check numeric dtypes
        if pd.api.types.is_numeric_dtype(series):
            # Check for np.isinf
            inf_mask = np.isinf(series)
            inf_count = int(inf_mask.sum())
            if inf_count > 0:
                column_anomalies[col] = inf_count
                total_anomalies += inf_count

    return total_anomalies, column_anomalies
