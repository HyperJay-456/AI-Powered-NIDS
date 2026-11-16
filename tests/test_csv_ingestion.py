"""Unit and integration tests for CSV ingestion and local file storage subsystem."""

import io
from pathlib import Path
import tempfile
import unittest

from src.data.csv_ingestion import (
    check_disk_space,
    compute_stream_sha256_and_size,
    format_file_size,
    resolve_unique_storage_filename,
    sanitize_filename,
    save_uploaded_csv,
    verify_path_safety,
)
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest


class DummyUploadedFile(io.BytesIO):
    """Mock Streamlit UploadedFile with name attribute."""

    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name
        self.size = len(content)


class TestCSVIngestion(unittest.TestCase):
    """Test suite for file ingestion, sanitization, path security, and deduplication."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.meta_dir = Path(self.temp_dir.name) / "metadata"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_csv_upload_success(self) -> None:
        """1. Successful CSV upload and atomic storage."""
        content = b"Flow ID,Source IP,Label\n1,192.168.1.1,BENIGN\n"
        uploaded = DummyUploadedFile(content, "sample_traffic.csv")

        result = save_uploaded_csv(uploaded, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)

        self.assertTrue(result.success)
        self.assertEqual(result.safe_filename, "sample_traffic.csv")
        self.assertFalse(result.is_duplicate)
        self.assertTrue((self.raw_dir / "sample_traffic.csv").exists())
        self.assertEqual((self.raw_dir / "sample_traffic.csv").read_bytes(), content)

    def test_multiple_csv_uploads(self) -> None:
        """2. Multiple distinct CSV uploads handled and preserved."""
        f1 = DummyUploadedFile(b"Header1,Header2\n1,2\n", "monday.csv")
        f2 = DummyUploadedFile(b"HeaderA,HeaderB\n3,4\n", "tuesday.csv")

        r1 = save_uploaded_csv(f1, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)
        r2 = save_uploaded_csv(f2, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)

        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        self.assertTrue((self.raw_dir / "monday.csv").exists())
        self.assertTrue((self.raw_dir / "tuesday.csv").exists())

    def test_duplicate_file_sha256_detection(self) -> None:
        """3. Duplicate file SHA-256 detection without duplicating storage."""
        content = b"Col1,Col2\nVal1,Val2\n"
        f1 = DummyUploadedFile(content, "original.csv")
        f2 = DummyUploadedFile(content, "copy_of_original.csv")

        r1 = save_uploaded_csv(f1, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)
        r2 = save_uploaded_csv(f2, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)

        self.assertTrue(r1.success)
        self.assertFalse(r1.is_duplicate)
        self.assertTrue(r2.success)
        self.assertTrue(r2.is_duplicate)
        self.assertEqual(r1.sha256_checksum, r2.sha256_checksum)
        # Should not have created a second physical file
        self.assertFalse((self.raw_dir / "copy_of_original.csv").exists())

    def test_same_filename_different_content(self) -> None:
        """4. Uploading different file content with same filename preserves both with unique names."""
        content1 = b"ColA,ColB\n1,2\n"
        content2 = b"ColA,ColB\n3,4\n"
        f1 = DummyUploadedFile(content1, "traffic.csv")
        f2 = DummyUploadedFile(content2, "traffic.csv")

        r1 = save_uploaded_csv(f1, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)
        r2 = save_uploaded_csv(f2, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)

        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        self.assertNotEqual(r1.safe_filename, r2.safe_filename)
        self.assertTrue((self.raw_dir / r1.safe_filename).exists())
        self.assertTrue((self.raw_dir / r2.safe_filename).exists())
        self.assertEqual((self.raw_dir / r1.safe_filename).read_bytes(), content1)
        self.assertEqual((self.raw_dir / r2.safe_filename).read_bytes(), content2)

    def test_filename_sanitization(self) -> None:
        """5. Filename sanitization against forbidden characters and special symbols."""
        self.assertEqual(sanitize_filename("valid_name.csv"), "valid_name.csv")
        self.assertEqual(sanitize_filename("evil:name*with?chars.csv"), "evil_name_with_chars.csv")
        self.assertEqual(sanitize_filename("spaces in filename.csv"), "spaces_in_filename.csv")
        self.assertEqual(sanitize_filename("null\x00byte.csv"), "null_byte.csv")
        self.assertEqual(sanitize_filename("no_ext"), "no_ext.csv")
        self.assertEqual(sanitize_filename(""), "network_traffic.csv")

    def test_path_traversal_protection(self) -> None:
        """6. Path traversal attempts rejected."""
        traversal_name = "../../etc/passwd.csv"
        sanitized = sanitize_filename(traversal_name)
        # Sanitization strips path navigation
        self.assertNotIn("..", sanitized)
        self.assertNotIn("/", sanitized)

        # Path safety verification
        candidate = self.raw_dir / sanitized
        self.assertTrue(verify_path_safety(self.raw_dir, candidate))
        evil_path = self.raw_dir / ".." / "evil.csv"
        self.assertFalse(verify_path_safety(self.raw_dir, evil_path))

    def test_empty_file_rejection(self) -> None:
        """7. Empty 0-byte file rejected."""
        empty_file = DummyUploadedFile(b"", "empty.csv")
        result = save_uploaded_csv(empty_file, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)

        self.assertFalse(result.success)
        self.assertIn("empty", result.message.lower())
        self.assertFalse((self.raw_dir / "empty.csv").exists())

    def test_unsupported_format_rejection(self) -> None:
        """8. Unsupported formats (.pcap, .exe, .zip) rejected."""
        pcap_file = DummyUploadedFile(b"\xd4\xc3\xb2\xa1", "traffic.pcap")
        zip_file = DummyUploadedFile(b"PK\x03\x04", "traffic.zip")

        r_pcap = save_uploaded_csv(pcap_file, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)
        r_zip = save_uploaded_csv(zip_file, raw_dir=self.raw_dir, metadata_dir=self.meta_dir)

        self.assertFalse(r_pcap.success)
        self.assertIn("only .csv", r_pcap.message.lower())
        self.assertFalse(r_zip.success)
        self.assertIn("only .csv", r_zip.message.lower())

    def test_storage_failure_handling(self) -> None:
        """9. Storage failure handling on disk limits."""
        # Test check_disk_space with an absurd required amount
        has_space, _, msg = check_disk_space(self.raw_dir, 10**15, min_margin_mb=500)
        self.assertFalse(has_space)
        self.assertIn("Insufficient disk space", msg)

    def test_manifest_atomicity_and_persistence(self) -> None:
        """10. Manifest atomicity and persistence."""
        manifest = IngestionManifest(metadata_dir=self.meta_dir)
        self.assertEqual(len(manifest.load_all()), 0)

        record = DatasetRecord(
            dataset_id="test-id-123",
            original_filename="sample.csv",
            safe_filename="sample.csv",
            file_path=str(self.raw_dir / "sample.csv"),
            file_size_bytes=1024,
            file_size_formatted="1.00 KB",
            sha256_checksum="abc12345",
            import_timestamp="2026-11-16T12:00:00Z",
            validation_status="Uploaded",
        )
        success = manifest.add_or_update(record)
        self.assertTrue(success)

        # Reload and verify
        reloaded = manifest.get_by_id("test-id-123")
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.original_filename, "sample.csv")
        self.assertEqual(reloaded.file_size_bytes, 1024)


if __name__ == "__main__":
    unittest.main()
