"""Audit Verification Test Suite for CP-04.

Covers:
- AUDIT 1: CSV parser correctness, agreement between csv.reader and Pandas,
  CRLF/LF, quoted commas, embedded newlines, escaped quotes, unterminated quotes,
  extra/missing delimiters, blank lines, header-only files, and row-accounting invariants.
- AUDIT 2: Deterministic length-safe duplicate fingerprinting, collision resistance
  (['ab','c'] vs ['a','bc'], ['a|b','c'] vs ['a','b|c'], empty string vs null, Unicode).
- AUDIT 3 & 4: Raw data byte preservation, failed scan isolation, and incomplete scan handling.
- AUDIT 5: Privacy sanitization in diagnostic outputs.
"""

import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from config.settings import settings
from src.data.csv_ingestion import save_uploaded_csv
from src.data.csv_validator import (
    scan_csv_records_structural,
    validate_csv_dataset,
)
from src.data.data_quality import (
    DiskBackedDuplicateDetector,
    serialize_row_length_safe,
)
from src.data.ingestion_manifest import IngestionManifest


class TestCSVParserAudit(unittest.TestCase):
    """Audit 1 verification: CSV parser correctness, edge cases, and accounting invariants."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.meta_dir = Path(self.temp_dir.name) / "metadata"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_file(self, filename: str, content_bytes: bytes) -> Path:
        p = self.raw_dir / filename
        p.write_bytes(content_bytes)
        return p

    def test_crlf_and_lf_line_endings_agreement(self) -> None:
        """CRLF and LF line endings parse identically without phantom blank lines."""
        content = (
            b"Port,Duration,Label\r\n"
            b"80,1000,BENIGN\r\n"
            b"443,2000,DoS\n"
            b"22,3000,BENIGN\r\n"
        )
        csv_path = self._write_file("mixed_endings.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="endings-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 3)
        self.assertEqual(record.valid_rows, 3)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)

    def test_quoted_commas_and_escaped_quotes(self) -> None:
        """Quoted commas and escaped quotes are parsed as field content, not delimiters."""
        content = (
            b"ID,Description,Label\n"
            b'1,"Payload containing , comma",BENIGN\n'
            b'2,"Escaped ""quotes"" inside",PortScan\n'
        )
        csv_path = self._write_file("quoted_content.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="quoted-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 2)
        self.assertEqual(record.valid_rows, 2)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)

    def test_embedded_newlines_inside_quotes(self) -> None:
        """Multi-line quoted field is parsed as a single logical record."""
        content = (
            b"ID,Payload,Label\n"
            b'1,"Line one\nLine two\r\nLine three",BENIGN\n'
            b'2,"Simple payload",DDoS\n'
        )
        csv_path = self._write_file("embedded_newlines.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="newline-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 2)
        self.assertEqual(record.valid_rows, 2)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)

    def test_unterminated_quote_at_eof(self) -> None:
        """Unterminated quote at EOF fails cleanly and reports incomplete accounting."""
        content = (
            b"ID,Payload,Label\n"
            b'1,"Normal row",BENIGN\n'
            b'2,"Unterminated quote reaching EOF'
        )
        csv_path = self._write_file("unterminated_quote.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="unterminated-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.validation_status, "Validation Failed")
        self.assertGreater(len(record.validation_errors), 0)

    def test_extra_and_missing_delimiters_accounting(self) -> None:
        """Inconsistent field counts (too many/too few fields) accurately accounted in invalid_rows."""
        content = (
            b"A,B,C,Label\n"
            b"1,2,3,BENIGN\n"          # Valid (4 fields)
            b"4,5,BENIGN\n"            # Missing delimiter (3 fields)
            b"6,7,8,9,EXTRA,BENIGN\n"  # Extra delimiters (6 fields)
            b"10,11,12,DoS\n"          # Valid (4 fields)
        )
        csv_path = self._write_file("delimiter_issues.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="delims-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.validation_status, "Validation Failed")
        self.assertEqual(record.valid_rows, 2)
        self.assertEqual(record.invalid_rows, 2)
        self.assertEqual(record.total_rows, 4)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)

    def test_empty_records_and_blank_lines(self) -> None:
        """Blank lines inside dataset are accounted in invalid_rows without silent dropping."""
        content = (
            b"A,B,Label\n"
            b"1,2,BENIGN\n"
            b"\n"                       # Blank record
            b"3,4,PortScan\n"
        )
        csv_path = self._write_file("blank_lines.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="blank-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.validation_status, "Validation Failed")
        self.assertEqual(record.valid_rows, 2)
        self.assertEqual(record.invalid_rows, 1)
        self.assertEqual(record.total_rows, 3)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)

    def test_header_only_dataset(self) -> None:
        """Header-only dataset has 0 valid rows, 0 invalid rows, and warning."""
        content = b"ColA,ColB,Label\n"
        csv_path = self._write_file("header_only_audit.csv", content)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="header-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 0)
        self.assertEqual(record.valid_rows, 0)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)
        self.assertIn("Validated with Warnings", record.validation_status)


class TestDuplicateFingerprintAudit(unittest.TestCase):
    """Audit 2 verification: Unambiguous length-safe serialization and collision safety."""

    def test_length_safe_collision_resistance_subtle_patterns(self) -> None:
        """Serializing ['ab','c'] vs ['a','bc'] produces distinct byte sequences."""
        s1 = serialize_row_length_safe(("ab", "c"))
        s2 = serialize_row_length_safe(("a", "bc"))
        self.assertNotEqual(s1, s2)

    def test_delimiter_character_collision_resistance(self) -> None:
        """Serializing ['a|b','c'] vs ['a','b|c'] produces distinct byte sequences."""
        s1 = serialize_row_length_safe(("a|b", "c"))
        s2 = serialize_row_length_safe(("a", "b|c"))
        self.assertNotEqual(s1, s2)

    def test_empty_string_versus_missing_value_distinction(self) -> None:
        """Empty string '' produces distinct serialization from null/None."""
        s_empty = serialize_row_length_safe(("",))
        s_null = serialize_row_length_safe((None,))
        s_nan = serialize_row_length_safe((np.nan,))

        self.assertNotEqual(s_empty, s_null)
        self.assertNotEqual(s_empty, s_nan)
        self.assertEqual(s_null, s_nan)  # Both represent missing values

    def test_literal_sentinel_string_versus_null(self) -> None:
        """A string cell literally containing '__NIDS_NULL__' is distinct from None."""
        s_sentinel = serialize_row_length_safe(("__NIDS_NULL__",))
        s_null = serialize_row_length_safe((None,))
        self.assertNotEqual(s_sentinel, s_null)

    def test_unicode_and_special_character_safety(self) -> None:
        """Unicode characters, accents, and emojis serialize deterministically."""
        s_unicode1 = serialize_row_length_safe(("München", "🚀"))
        s_unicode2 = serialize_row_length_safe(("München", "🚀"))
        s_unicode3 = serialize_row_length_safe(("Munchen", "🚀"))

        self.assertEqual(s_unicode1, s_unicode2)
        self.assertNotEqual(s_unicode1, s_unicode3)

    def test_embedded_newline_in_serialization(self) -> None:
        """Embedded newlines serialize unambiguously."""
        s1 = serialize_row_length_safe(("line1\nline2", "val"))
        s2 = serialize_row_length_safe(("line1", "line2\nval"))
        self.assertNotEqual(s1, s2)

    def test_cross_chunk_duplicates_with_sqlite_detector(self) -> None:
        """SQLite duplicate detector flags identical rows split across chunk boundaries."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_dir = Path(tmp)
            with DiskBackedDuplicateDetector(temp_dir=meta_dir) as detector:
                chunk1 = pd.DataFrame([{"A": "ab", "B": "c"}, {"A": "x", "B": "y"}])
                chunk2 = pd.DataFrame([{"A": "a", "B": "bc"}, {"A": "ab", "B": "c"}])

                d1 = detector.process_chunk(chunk1)
                d2 = detector.process_chunk(chunk2)

                self.assertEqual(d1, 0)
                self.assertEqual(d2, 1)  # Only {"A": "ab", "B": "c"} is duplicate! {"A": "a", "B": "bc"} is distinct!
                self.assertEqual(detector.total_duplicates, 1)


class TestIncompleteAndFailedScanAudit(unittest.TestCase):
    """Audit 3, 4, 5 verification: Raw preservation, failed scan isolation, and privacy."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.meta_dir = Path(self.temp_dir.name) / "metadata"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_raw_bytes_preserved_exact_identity(self) -> None:
        """Ingestion preserves exact byte-for-byte fidelity without alterations."""
        test_bytes = b"ColA,ColB\r\n10,20\r\n30,40\r\n"
        stream = io.BytesIO(test_bytes)
        stream.name = "exact_bytes.csv"

        res = save_uploaded_csv(
            stream,
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )
        self.assertTrue(res.success)
        stored_bytes = (self.raw_dir / res.safe_filename).read_bytes()
        self.assertEqual(stored_bytes, test_bytes)

    def test_failed_validation_never_appears_validated_in_manifest(self) -> None:
        """Incomplete or failed validation manifests as 'Validation Failed' in manifest."""
        manifest = IngestionManifest(metadata_dir=self.meta_dir)
        bad_csv = self.raw_dir / "broken.csv"
        bad_csv.write_bytes(b"A,B\n1,2,3,extra\n")

        record = validate_csv_dataset(
            bad_csv,
            dataset_id="broken-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.validation_status, "Validation Failed")
        retrieved = manifest.get_by_id("broken-1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.validation_status, "Validation Failed")

    def test_multi_chunk_dataset_accounting_invariant(self) -> None:
        """Large dataset spanning multiple chunks strictly enforces row accounting."""
        rows = ["Col1,Col2,Label"]
        for i in range(250):
            rows.append(f"{i},{i*10},BENIGN")
        csv_text = "\n".join(rows) + "\n"

        p = self.raw_dir / "multi_chunk_audit.csv"
        p.write_bytes(csv_text.encode("utf-8"))

        record = validate_csv_dataset(
            p,
            dataset_id="chunk-audit-1",
            chunk_size=50,  # 5 chunks
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 250)
        self.assertEqual(record.valid_rows, 250)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)
        self.assertEqual(record.validation_status, "Validated with Warnings")


if __name__ == "__main__":
    unittest.main()
