"""Unit and integration tests for the CSV Validation Engine, duplicate detection, and schema inspector."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.csv_validator import detect_file_encoding, validate_csv_dataset
from src.data.data_quality import DiskBackedDuplicateDetector
from src.data.ingestion_manifest import IngestionManifest
from src.data.schema_inspector import (
    classify_column,
    detect_duplicate_columns,
    detect_label_column,
    normalize_column_name,
)


class TestCSVValidator(unittest.TestCase):
    """Test suite for CSV validation, row accounting, duplicate detection, and anomaly metrics."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.meta_dir = Path(self.temp_dir.name) / "metadata"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_csv(self, filename: str, content: str, encoding: str = "utf-8") -> Path:
        p = self.raw_dir / filename
        p.write_text(content, encoding=encoding)
        return p

    def test_valid_cicids_like_csv_fixture(self) -> None:
        """11. Valid CICIDS-like CSV fixture validates cleanly."""
        csv_text = (
            "Destination Port,Flow Duration,Total Fwd Packets,Total Backward Packets,Flow Bytes/s,Flow Packets/s, Label\n"
            "80,10500,4,3,450.2,66.6,BENIGN\n"
            "443,20000,10,8,1200.0,90.0,DoS Hulk\n"
            "22,500,2,2,100.5,8.0,SSH-Patator\n"
        )
        csv_path = self._create_csv("cicids_sample.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="cicids-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 3)
        self.assertEqual(record.valid_rows, 3)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_columns, 7)
        self.assertTrue(record.has_ground_truth_label)
        self.assertEqual(record.label_column_name, " Label")
        self.assertEqual(record.label_distribution["BENIGN"], 1)
        self.assertEqual(record.label_distribution["DoS Hulk"], 1)
        self.assertEqual(record.label_distribution["SSH-Patator"], 1)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)

    def test_missing_values_detection(self) -> None:
        """12. Accurately detects and counts rows and columns with missing values."""
        csv_text = (
            "Port,Duration,FlowRate,Label\n"
            "80,1000,,BENIGN\n"          # Missing FlowRate
            ",2000,50.0,DDoS\n"          # Missing Port
            "443,3000,100.0,BENIGN\n"    # Complete
            ",,,BENIGN\n"                # Multiple missing
        )
        csv_path = self._create_csv("missing_sample.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="missing-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 4)
        self.assertEqual(record.missing_value_rows, 3)
        self.assertIn("Validated with Warnings", record.validation_status)

        col_dict = {c.name: c for c in record.columns}
        self.assertEqual(col_dict["Port"].missing_count, 2)
        self.assertEqual(col_dict["Duration"].missing_count, 1)
        self.assertEqual(col_dict["FlowRate"].missing_count, 2)
        self.assertEqual(col_dict["Label"].missing_count, 0)

    def test_duplicate_rows_within_chunk(self) -> None:
        """13. Detects duplicate rows within a single chunk."""
        csv_text = (
            "A,B,C\n"
            "1,2,3\n"
            "1,2,3\n"  # Duplicate
            "4,5,6\n"
            "1,2,3\n"  # Duplicate
        )
        csv_path = self._create_csv("dup_single.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="dup-single",
            chunk_size=100,
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 4)
        self.assertEqual(record.duplicate_rows, 2)
        self.assertIn("Validated with Warnings", record.validation_status)

    def test_duplicate_rows_across_different_chunks(self) -> None:
        """14. Detects duplicate rows across different chunk boundaries using SQLite fingerprint index."""
        # 6 rows, processed with chunk_size=2 (3 chunks)
        # Rows 1 and 2 in chunk 1
        # Rows 3 and 4 in chunk 2 (row 3 is duplicate of row 1)
        # Rows 5 and 6 in chunk 3 (row 5 is duplicate of row 2, row 6 duplicate of row 4)
        csv_text = (
            "A,B\n"
            "10,20\n"  # R1 (Chunk 1)
            "30,40\n"  # R2 (Chunk 1)
            "10,20\n"  # R3 = R1 (Chunk 2, cross-chunk duplicate!)
            "50,60\n"  # R4 (Chunk 2)
            "30,40\n"  # R5 = R2 (Chunk 3, cross-chunk duplicate!)
            "50,60\n"  # R6 = R4 (Chunk 3, cross-chunk duplicate!)
        )
        csv_path = self._create_csv("dup_multi.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="dup-multi",
            chunk_size=2,  # Force multiple chunks
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 6)
        self.assertEqual(record.valid_rows, 6)
        self.assertEqual(record.duplicate_rows, 3)
        self.assertTrue(record.duplicate_detection_complete)

    def test_duplicate_column_names(self) -> None:
        """15. Detects raw duplicate headers."""
        cols = ["Port", "Duration", "Port", "Label"]
        exact, _ = detect_duplicate_columns(cols)
        self.assertIn("Port", exact)

    def test_whitespace_case_normalization_headers(self) -> None:
        """16. Detects columns that collide after trimming whitespace and lowercasing."""
        cols = ["Flow Duration", " flow duration ", "Port", "port"]
        exact, norm = detect_duplicate_columns(cols)
        self.assertEqual(len(exact), 0)
        self.assertEqual(len(norm), 4)

    def test_missing_label_column(self) -> None:
        """17. Missing Label column does not reject the dataset, but flags missing ground truth."""
        csv_text = (
            "Port,Duration,Packets\n"
            "80,100,5\n"
            "443,200,8\n"
        )
        csv_path = self._create_csv("unlabeled.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="unlabeled-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 2)
        self.assertFalse(record.has_ground_truth_label)
        self.assertIsNone(record.label_column_name)
        self.assertEqual(len(record.label_distribution), 0)
        self.assertIn("Validated with Warnings", record.validation_status)

    def test_ground_truth_label_detection_and_distribution(self) -> None:
        """18. Case-insensitive Ground-Truth Label column detection and class distribution."""
        csv_text = (
            "FlowID,Duration,  label  \n"
            "1,100,BENIGN\n"
            "2,200,BENIGN\n"
            "3,300,DDoS\n"
            "4,400,PortScan\n"
            "5,500,BENIGN\n"
        )
        csv_path = self._create_csv("labels.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="labels-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertTrue(record.has_ground_truth_label)
        self.assertEqual(record.label_column_name, "  label  ")
        self.assertEqual(record.label_distribution["BENIGN"], 3)
        self.assertEqual(record.label_distribution["DDoS"], 1)
        self.assertEqual(record.label_distribution["PortScan"], 1)

    def test_malformed_csv_structure(self) -> None:
        """19. Structural CSV error triggers validation failure without silent skipping."""
        csv_text = (
            'A,B,C\n'
            '1,2,3\n'
            '4,"unterminated quote,6\n'
            '7,8,9\n'
        )
        csv_path = self._create_csv("malformed.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="malformed-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertIn("Validation Failed", record.validation_status)
        self.assertGreater(len(record.validation_errors), 0)

    def test_inconsistent_field_counts(self) -> None:
        """20. Inconsistent field counts reported cleanly without silent loss."""
        csv_text = (
            "A,B,C\n"
            "1,2,3\n"
            "4,5,6,7,8\n"  # 5 fields instead of 3
            "9,10,11\n"
        )
        csv_path = self._create_csv("inconsistent.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="inconsistent-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertIn("Validation Failed", record.validation_status)
        self.assertGreater(len(record.validation_errors), 0)

    def test_non_finite_numeric_values(self) -> None:
        """21. Detects non-finite numeric values (inf/-inf)."""
        csv_text = (
            "Port,FlowRate,Label\n"
            "80,100.5,BENIGN\n"
            "443,Infinity,DDoS\n"
            "8080,-Infinity,PortScan\n"
            "22,50.0,BENIGN\n"
        )
        csv_path = self._create_csv("infinity_sample.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="inf-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 4)
        self.assertEqual(record.numeric_anomaly_count, 2)
        col_map = {c.name: c for c in record.columns}
        self.assertEqual(col_map["FlowRate"].numeric_anomalies, 2)
        self.assertIn("Validated with Warnings", record.validation_status)

    def test_mixed_type_column_handling(self) -> None:
        """22. Column classification correctly categorizes numeric, categorical, label, and mixed."""
        self.assertEqual(classify_column("Port", "int64", False, 100), "Numeric")
        self.assertEqual(classify_column("Rate", "float64", False, 100), "Numeric")
        self.assertEqual(classify_column("Label", "object", True, 100), "Label")
        self.assertEqual(classify_column("Protocol", "object", False, 100, unique_count=3), "Categorical")

    def test_empty_or_header_only_csv(self) -> None:
        """23. Header-only CSV handled with 0 data rows and warning."""
        csv_text = "Col1,Col2,Col3\n"
        csv_path = self._create_csv("header_only.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="header-only",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 0)
        self.assertEqual(record.total_columns, 3)
        self.assertIn("Validated with Warnings", record.validation_status)

    def test_complete_row_accounting_invariant(self) -> None:
        """24. Invariant: Total Rows == Structurally Valid Rows + Invalid Rows."""
        csv_text = (
            "A,B,C,Label\n"
            "1,2,3,BENIGN\n"
            "4,5,6,DDoS\n"
            "7,8,9,PortScan\n"
            "10,11,12,BENIGN\n"
        )
        csv_path = self._create_csv("invariant_test.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="inv-1",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)
        self.assertEqual(record.total_rows, 4)
        self.assertEqual(record.valid_rows, 4)
        self.assertEqual(record.invalid_rows, 0)

    def test_chunk_boundary_processing(self) -> None:
        """25 & 26 & 27. Process fixture larger than chunk size, counting all rows across chunks."""
        # Generate 150 rows with chunk size 40 -> 4 chunks (40 + 40 + 40 + 30)
        rows = ["ColA,ColB,Label"]
        for i in range(150):
            rows.append(f"{i},{i * 2},BENIGN")
        csv_text = "\n".join(rows) + "\n"

        csv_path = self._create_csv("large_chunk_test.csv", csv_text)
        record = validate_csv_dataset(
            csv_path,
            dataset_id="large-chunk-1",
            chunk_size=40,
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )

        self.assertEqual(record.total_rows, 150)
        self.assertEqual(record.valid_rows, 150)
        self.assertEqual(record.invalid_rows, 0)
        self.assertEqual(record.total_rows, record.valid_rows + record.invalid_rows)
        self.assertEqual(record.label_distribution["BENIGN"], 150)

    def test_validation_failure_reported_when_scan_incomplete(self) -> None:
        """28. Non-existent file marks validation as failed."""
        record = validate_csv_dataset(
            self.raw_dir / "non_existent.csv",
            dataset_id="missing-file",
            raw_dir=self.raw_dir,
            metadata_dir=self.meta_dir,
        )
        self.assertEqual(record.validation_status, "Validation Failed")


if __name__ == "__main__":
    unittest.main()
