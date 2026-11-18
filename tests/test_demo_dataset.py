"""Unit tests for the Benchmark Demo Dataset generator and Initializer."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.demo_dataset import (
    DEFAULT_BENCHMARK_FILENAME,
    clear_demo_dataset,
    generate_benchmark_sample_dataframe,
    install_benchmark_sample_dataset,
    is_demo_installed,
    is_pipeline_initialized,
)


class TestDemoDataset(unittest.TestCase):
    """Test suite for synthetic benchmark generation and lifecycle management."""

    def test_generate_benchmark_sample_dataframe(self) -> None:
        """Verify generated benchmark sample conforms strictly to CIC-IDS-2017 schema."""
        df = generate_benchmark_sample_dataframe(n_rows=500)
        self.assertEqual(len(df), 500)
        self.assertEqual(df.shape[1], 78)

        # Ground truth label column present and includes expected classes
        self.assertIn("Label", df.columns)
        classes = set(df["Label"].unique())
        self.assertTrue({"BENIGN", "PortScan", "DDoS", "Bot"}.issubset(classes))

        # Check key flow features are present
        self.assertIn("Destination Port", df.columns)
        self.assertIn("Flow Duration", df.columns)
        self.assertIn("Total Fwd Packets", df.columns)
        self.assertIn("Total Backward Packets", df.columns)
        self.assertIn("Flow Bytes/s", df.columns)
        self.assertIn("Flow Packets/s", df.columns)
        self.assertIn("Protocol", df.columns)

    def test_install_and_clear_lifecycle_isolated(self) -> None:
        """Verify installation and clean reset in an isolated temporary directory tree."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "data" / "raw"
            meta_dir = root / "data" / "metadata"
            proc_dir = root / "data" / "processed"
            models_dir = root / "models" / "saved"
            eval_dir = root / "models" / "evaluations"
            analyses_dir = root / "data" / "analyses"

            for d in (raw_dir, meta_dir, proc_dir, models_dir, eval_dir, analyses_dir):
                d.mkdir(parents=True, exist_ok=True)

            # Initially not installed
            self.assertFalse(is_demo_installed(raw_dir=raw_dir))
            self.assertFalse(is_pipeline_initialized(saved_models_dir=models_dir, analyses_dir=analyses_dir))

            # Install sample
            csv_path, rec = install_benchmark_sample_dataset(
                raw_dir=raw_dir,
                metadata_dir=meta_dir,
                filename="test_sample.csv",
                n_rows=100,
            )
            self.assertTrue(csv_path.exists())
            self.assertEqual(rec.total_rows, 100)
            self.assertEqual(rec.validation_status, "Validated with Warnings")

            # Check clear
            counts = clear_demo_dataset(
                raw_dir=raw_dir,
                metadata_dir=meta_dir,
                processed_dir=proc_dir,
                saved_models_dir=models_dir,
                evaluations_dir=eval_dir,
                analyses_dir=analyses_dir,
            )
            self.assertGreaterEqual(counts["raw_files"], 1)
            self.assertFalse((raw_dir / "test_sample.csv").exists())


if __name__ == "__main__":
    unittest.main()
