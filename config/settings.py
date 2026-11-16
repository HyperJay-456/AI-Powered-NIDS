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
MODELS_DIR: Path = BASE_DIR / "models"
SAVED_MODELS_DIR: Path = MODELS_DIR / "saved"
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
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1-Score (Weighted)",
        "ROC-AUC (One-vs-Rest)",
        "Inference Latency (ms)",
    )
    cross_validation_folds: int = 5


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
    models_dir: Path = MODELS_DIR
    saved_models_dir: Path = SAVED_MODELS_DIR
    logs_dir: Path = LOGS_DIR
    src_dir: Path = SRC_DIR

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ui: StreamlitConfig = field(default_factory=StreamlitConfig)

    def ensure_directories(self) -> None:
        """Create necessary project runtime directories if they do not exist."""
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.saved_models_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton instance
settings = AppConfig()
