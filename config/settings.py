"""Centralized Configuration Module for Network Intrusion Detection System (NIDS).

This module defines directory paths, dataset parameters, supported algorithm configurations,
hyperparameter baselines, and runtime settings for the academic NIDS project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ==============================================================================
# Base File System Paths
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
METADATA_DIR: Path = DATA_DIR / "metadata"
MODELS_DIR: Path = BASE_DIR / "models"
SAVED_MODELS_DIR: Path = MODELS_DIR / "saved"
EVALUATIONS_DIR: Path = MODELS_DIR / "evaluations"
ANALYSES_DIR: Path = DATA_DIR / "analyses"
ANALYST_REVIEWS_DIR: Path = DATA_DIR / "analyst_reviews"
LOGS_DIR: Path = BASE_DIR / "logs"
SRC_DIR: Path = BASE_DIR / "src"


# ==============================================================================
# Dataset Specifications (CIC-IDS-2017)
# ==============================================================================
@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for CIC-IDS-2017 benchmark dataset."""

    name: str = "CIC-IDS-2017"
    source: str = "Canadian Institute for Cybersecurity (via Kaggle)"
    target_column_raw: str = " Label"
    target_column_clean: str = "Label"
    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.10

    # Key attack families documented in CIC-IDS-2017
    known_attack_classes: Tuple[str, ...] = (
        "BENIGN",
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "DoS slowloris",
        "DoS Slowhttptest",
        "Bot",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Infiltration",
        "Web Attack - Sql Injection",
        "Heartbleed",
    )

    # Core raw CSV subsets in the Kaggle CIC-IDS-2017 distribution
    raw_files: Tuple[str, ...] = (
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    )


# ==============================================================================
# Machine Learning Model Configurations
# Strictly restricted to: Support Vector Machine (SVM),
# Multilayer Perceptron (MLP), and Decision Tree (DT).
# ==============================================================================
SUPPORTED_ALGORITHMS: Tuple[str, ...] = (
    "Support Vector Machine (SVM)",
    "Multilayer Perceptron (MLP)",
    "Decision Tree",
)

ALGORITHM_KEYS: Tuple[str, ...] = ("svm", "mlp", "decision_tree")


@dataclass
class ModelHyperparameters:
    """Hyperparameter baselines for the three approved NIDS algorithms."""

    # Support Vector Machine (Linear / RBF)
    svm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
            "max_iter": 5000,
        }
    )

    # Multilayer Perceptron (Neural Network)
    mlp_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "max_iter": 200,
            "random_state": 42,
            "early_stopping": True,
            "n_iter_no_change": 10,
        }
    )

    # Decision Tree
    decision_tree_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "criterion": "gini",
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        }
    )


# ==============================================================================
# Evaluation Metrics Configuration
# ==============================================================================
@dataclass(frozen=True)
class EvaluationConfig:
    """Standard academic evaluation metrics for multi-class & binary NIDS."""

    metrics: Tuple[str, ...] = (
        "Accuracy",
        "Balanced Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1-Score (Macro)",
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1-Score (Weighted)",
        "MCC",
        "Cohen's Kappa",
        "ROC-AUC (One-vs-Rest)",
        "Inference Latency (ms)",
    )
    cross_validation_folds: int = 5
    default_batch_size: int = 10_000


# ==============================================================================
# Intrusion Detection Engine Configurations (CP-08)
# ==============================================================================
@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for operational intrusion detection inference."""

    default_batch_size: int = 10_000
    min_batch_size: int = 1_000
    max_batch_size: int = 50_000
    min_ram_free_mb: int = 512
    min_disk_free_mb: int = 500
    max_preview_rows: int = 50


# ==============================================================================
# ==============================================================================
# Ingestion & Large File Batch Configurations
# ==============================================================================
@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for CSV traffic upload, batch chunking, and validation."""

    default_chunk_size: int = 50_000
    max_upload_size_mb: int = 500  # Accommodates largest single CIC-IDS-2017 benchmark CSV file (~450 MB)
    max_batch_upload_size_mb: int = 1200  # Safe aggregate upload threshold for local RAM buffering
    min_disk_free_mb: int = 500
    allowed_extensions: Tuple[str, ...] = (".csv",)
    max_preview_rows: int = 100
    default_preview_rows: int = 25


# ==============================================================================
# Preprocessing & Dataset Engineering Configurations (CP-05)
# ==============================================================================
@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for dataset engineering, splitting, scaling, and encoding."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    default_chunk_size: int = 50_000
    numeric_imputer_strategy: str = "median"  # Median imputation on training split
    categorical_imputer_strategy: str = "most_frequent"
    scaler_strategy: str = "standard"  # Standard scaling for SVM & MLP
    tree_scaler_strategy: str = "none"  # Unscaled for Decision Tree
    duplicate_policy: str = "group_isolated"  # Prevent identical records from crossing partition boundaries
    target_column: str = "Label"
    identifier_keywords: Tuple[str, ...] = (
        "flow_id",
        "flow id",
        "source_ip",
        "source ip",
        "destination_ip",
        "destination ip",
        "src_ip",
        "dst_ip",
        "timestamp",
        "external_ip",
    )
    sparse_output: bool = True


# ==============================================================================
# Streamlit Application Presentation Settings
# ==============================================================================
@dataclass(frozen=True)
class StreamlitConfig:
    """UI presentation settings."""

    app_title: str = "AI-Powered Network Intrusion Detection System"
    app_subtitle: str = "Cybersecurity Threat Analytics & Traffic Classification Engine"
    app_icon: str = "🛡️"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme_color: str = "#0E1117"


# ==============================================================================
# Master Application Settings Aggregator
# ==============================================================================
@dataclass
class AppConfig:
    """Master configuration container for the NIDS platform."""

    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    metadata_dir: Path = METADATA_DIR
    models_dir: Path = MODELS_DIR
    saved_models_dir: Path = SAVED_MODELS_DIR
    evaluations_dir: Path = EVALUATIONS_DIR
    analyses_dir: Path = ANALYSES_DIR
    analyst_reviews_dir: Path = ANALYST_REVIEWS_DIR
    logs_dir: Path = LOGS_DIR
    src_dir: Path = SRC_DIR

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ui: StreamlitConfig = field(default_factory=StreamlitConfig)

    def ensure_directories(self) -> None:
        """Create necessary project runtime directories if they do not exist."""
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.metadata_dir,
            self.saved_models_dir,
            self.evaluations_dir,
            self.analyses_dir,
            self.analyst_reviews_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton instance
settings = AppConfig()
