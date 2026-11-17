"""Hardware Resource Manager & Memory Preflight (CP-06).

Inspects available system memory, estimates sparse matrix footprint and training requirements,
and performs resource preflight checks to prevent out-of-memory crashes on the development laptop.
"""

from dataclasses import dataclass, field
import os
import platform
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreflightCheckResult:
    """Diagnostic outcome of a resource preflight assessment."""

    is_safe: bool
    algorithm: str
    available_ram_bytes: Optional[int]
    available_ram_formatted: str
    estimated_memory_bytes: int
    estimated_memory_formatted: str
    training_rows: int
    training_features: int
    sparse_density_percent: float
    warning_messages: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


class ResourceManager:
    """Monitors local system resources and guards against out-of-memory exceptions."""

    @staticmethod
    def get_available_system_memory() -> Optional[int]:
        """Query available system RAM in bytes without third-party dependencies.

        Uses ctypes with GlobalMemoryStatusEx on Windows, and /proc/meminfo on Linux.
        Returns None if system memory cannot be reliably determined.
        """
        system = platform.system()

        if system == "Windows":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return int(stat.ullAvailPhys)
            except Exception as e:
                logger.debug(f"Windows memory query failed: {e}")
                return None

        elif system == "Linux":
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            parts = line.split()
                            return int(parts[1]) * 1024  # kB to bytes
            except Exception as e:
                logger.debug(f"Linux /proc/meminfo query failed: {e}")
                return None

        return None

    @classmethod
    def get_system_memory_info(cls) -> Dict[str, Any]:
        """Retrieve structured system memory telemetry."""
        avail_bytes = cls.get_available_system_memory()
        if avail_bytes is None:
            return {
                "available_ram_bytes": None,
                "available_ram_formatted": "Measurement Unavailable (System Query Blocked)",
                "is_reliable": False,
            }

        avail_gb = avail_bytes / (1024**3)
        return {
            "available_ram_bytes": avail_bytes,
            "available_ram_formatted": f"{avail_gb:.2f} GB Free",
            "available_ram_gb": round(avail_gb, 2),
            "is_reliable": True,
        }

    @staticmethod
    def format_bytes(num_bytes: int) -> str:
        """Format byte count into human-readable representation."""
        if num_bytes < 1024:
            return f"{num_bytes} B"
        elif num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.2f} KB"
        elif num_bytes < 1024 * 1024 * 1024:
            return f"{num_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"

    @classmethod
    def estimate_sparse_matrix_memory(cls, X: sparse.csr_matrix) -> int:
        """Calculate the exact memory footprint of a SciPy CSR matrix in bytes."""
        if not isinstance(X, sparse.csr_matrix):
            return sys.getsizeof(X)
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)

    @classmethod
    def estimate_training_memory_requirements(
        cls,
        algorithm: str,
        X: sparse.csr_matrix,
        y: np.ndarray,
        config: Optional[Any] = None,
    ) -> int:
        """Estimate the peak memory required during algorithm fitting.

        Args:
            algorithm: One of 'svm', 'mlp', 'decision_tree'.
            X: Training partition feature matrix.
            y: Training partition target vector.
            config: Algorithm training configuration.

        Returns:
            Estimated peak memory in bytes.
        """
        matrix_bytes = cls.estimate_sparse_matrix_memory(X)
        target_bytes = int(y.nbytes)
        n_samples, n_features = X.shape

        algo = algorithm.lower().strip()
        if algo == "svm":
            # LinearSVC creates dual coefficient arrays and temporary sample working buffers
            # Scaled sparse LinearSVC typically consumes 2.5x to 3.5x sparse matrix size
            return int(matrix_bytes * 3.0 + target_bytes * 2)

        elif algo == "mlp":
            # Incremental MLPClassifier processes in mini-batches (default 256)
            # Memory is dominated by batch CSR slice + network weight matrices
            batch_size = getattr(config, "batch_size", 256) if config else 256
            batch_bytes = int((matrix_bytes / max(1, n_samples)) * batch_size)
            hidden_layers = getattr(config, "hidden_layer_sizes", (64, 32)) if config else (64, 32)

            # Estimate weight matrices: (n_features * h1) + (h1 * h2) + (h2 * n_classes)
            n_classes = len(np.unique(y)) if len(y) > 0 else 2
            layer_sizes = [n_features] + list(hidden_layers) + [n_classes]
            weight_params = sum(
                layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1)
            )
            weight_bytes = weight_params * 8  # float64

            # Working buffers for forward/backprop + gradient momentums
            return int(matrix_bytes + (batch_bytes * 4) + (weight_bytes * 5) + target_bytes)

        elif algo == "decision_tree":
            # DecisionTreeClassifier builds binary tree nodes
            # On large sparse matrices, node threshold tests and feature sorting can require 4x to 6x
            max_depth = getattr(config, "max_depth", 15) if config else 15
            depth_multiplier = 4.0 if (max_depth and max_depth <= 20) else 8.0
            return int(matrix_bytes * depth_multiplier + target_bytes * 3)

        return int(matrix_bytes * 4.0)

    @classmethod
    def preflight_check(
        cls,
        algorithm: str,
        X: sparse.csr_matrix,
        y: np.ndarray,
        config: Optional[Any] = None,
        safety_margin_percent: float = 75.0,
    ) -> PreflightCheckResult:
        """Perform comprehensive resource safety preflight check before training execution.

        Args:
            algorithm: Algorithm to check ('svm', 'mlp', 'decision_tree').
            X: Training partition feature matrix.
            y: Training partition target labels.
            config: Specific algorithm configuration.
            safety_margin_percent: Maximum safe percentage of available RAM allowed.

        Returns:
            PreflightCheckResult with pass/fail and telemetry.
        """
        n_samples, n_features = X.shape
        matrix_bytes = cls.estimate_sparse_matrix_memory(X)
        dense_equivalent_bytes = n_samples * n_features * 4  # float32
        density = (X.nnz / max(1, (n_samples * n_features))) * 100.0

        estimated_bytes = cls.estimate_training_memory_requirements(algorithm, X, y, config)
        available_bytes = cls.get_available_system_memory()

        warnings: List[str] = []
        is_safe = True
        rejection_reason: Optional[str] = None

        if available_bytes is not None:
            max_allowed_bytes = available_bytes * (safety_margin_percent / 100.0)

            if estimated_bytes > available_bytes:
                is_safe = False
                rejection_reason = (
                    f"Insufficient System Memory: Estimated training requirements ({cls.format_bytes(estimated_bytes)}) "
                    f"exceed currently available physical RAM ({cls.format_bytes(available_bytes)}). "
                    "Refusing training execution to prevent system freeze or process kill."
                )
            elif estimated_bytes > max_allowed_bytes:
                warnings.append(
                    f"High Memory Utilization Warning: Estimated requirements ({cls.format_bytes(estimated_bytes)}) "
                    f"exceed the configured safety threshold of {safety_margin_percent:.0f}% of available RAM "
                    f"({cls.format_bytes(int(max_allowed_bytes))})."
                )
        else:
            warnings.append(
                "System memory status query is restricted or unavailable on this platform. "
                f"Proceeding under estimated requirement of {cls.format_bytes(estimated_bytes)}."
            )

        # Check Decision Tree unbounded depth risk
        if algorithm.lower() == "decision_tree" and config:
            max_depth = getattr(config, "max_depth", None)
            if max_depth is None or max_depth > 30:
                warnings.append(
                    "Decision Tree max_depth is unconstrained or > 30. This can cause deep tree expansion "
                    "and elevated memory usage on network traffic flow datasets."
                )

        return PreflightCheckResult(
            is_safe=is_safe,
            algorithm=algorithm,
            available_ram_bytes=available_bytes,
            available_ram_formatted=cls.format_bytes(available_bytes) if available_bytes else "Unavailable",
            estimated_memory_bytes=estimated_bytes,
            estimated_memory_formatted=cls.format_bytes(estimated_bytes),
            training_rows=n_samples,
            training_features=n_features,
            sparse_density_percent=round(density, 2),
            warning_messages=warnings,
            rejection_reason=rejection_reason,
        )

    @classmethod
    def estimate_memory_from_dimensions(
        cls,
        n_samples: int,
        n_features: int,
        algorithm: str,
        config: Optional[Any] = None,
        assumed_density: float = 0.50,
    ) -> int:
        """Estimate peak memory required during algorithm fitting using dimension counts.

        Args:
            n_samples: Number of training rows.
            n_features: Number of feature columns.
            algorithm: One of 'svm', 'mlp', 'decision_tree'.
            config: Algorithm training configuration.
            assumed_density: Expected sparsity fraction (default 0.50).

        Returns:
            Estimated peak memory in bytes.
        """
        # CSR format: data (float32=4B), indices (int32=4B), indptr (int32=4B)
        nnz = int(n_samples * n_features * assumed_density)
        matrix_bytes = (nnz * 4) + (nnz * 4) + ((n_samples + 1) * 4)
        target_bytes = n_samples * 4  # int32 labels

        algo = algorithm.lower().strip()
        if algo == "svm":
            return int(matrix_bytes * 3.0 + target_bytes * 2)
        elif algo == "mlp":
            batch_size = getattr(config, "batch_size", 256) if config else 256
            batch_bytes = int((matrix_bytes / max(1, n_samples)) * batch_size)
            hidden_layers = getattr(config, "hidden_layer_sizes", (64, 32)) if config else (64, 32)
            n_classes = 15  # worst-case 15 CIC-IDS-2017 attack classes
            layer_sizes = [n_features] + list(hidden_layers) + [n_classes]
            weight_params = sum(
                layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1)
            )
            weight_bytes = weight_params * 8
            return int(matrix_bytes + (batch_bytes * 4) + (weight_bytes * 5) + target_bytes)
        elif algo == "decision_tree":
            max_depth = getattr(config, "max_depth", 15) if config else 15
            depth_multiplier = 4.0 if (max_depth and max_depth <= 20) else 8.0
            return int(matrix_bytes * depth_multiplier + target_bytes * 3)
        return int(matrix_bytes * 4.0)

    @classmethod
    def preflight_check_dimensions(
        cls,
        n_samples: int,
        n_features: int,
        algorithms: List[str],
        dt_max_depth: Optional[int] = 15,
        safety_margin_percent: float = 75.0,
    ) -> PreflightCheckResult:
        """Perform preflight checks based on dataset dimensions and selected algorithm(s)."""
        available_bytes = cls.get_available_system_memory()
        warnings: List[str] = []
        is_safe = True
        rejection_reason: Optional[str] = None

        total_estimated = 0
        for algo in algorithms:
            est = cls.estimate_memory_from_dimensions(
                n_samples=n_samples,
                n_features=n_features,
                algorithm=algo,
            )
            if est > total_estimated:
                total_estimated = est

        if available_bytes is not None:
            max_allowed = available_bytes * (safety_margin_percent / 100.0)
            if total_estimated > available_bytes:
                is_safe = False
                rejection_reason = (
                    f"Insufficient System Memory: Estimated training requirements ({cls.format_bytes(total_estimated)}) "
                    f"exceed available physical RAM ({cls.format_bytes(available_bytes)})."
                )
            elif total_estimated > max_allowed:
                warnings.append(
                    f"High Memory Utilization Warning: Estimated requirements ({cls.format_bytes(total_estimated)}) "
                    f"exceed safe threshold of {safety_margin_percent:.0f}% of available RAM."
                )

        if "decision_tree" in algorithms:
            if dt_max_depth is None or dt_max_depth > 30:
                warnings.append(
                    "Decision Tree max_depth is unconstrained or > 30. This can cause deep tree expansion "
                    "and elevated memory usage on network traffic flow datasets."
                )

        return PreflightCheckResult(
            is_safe=is_safe,
            algorithm=", ".join(algorithms),
            available_ram_bytes=available_bytes,
            available_ram_formatted=cls.format_bytes(available_bytes) if available_bytes else "Unavailable",
            estimated_memory_bytes=total_estimated,
            estimated_memory_formatted=cls.format_bytes(total_estimated),
            training_rows=n_samples,
            training_features=n_features,
            sparse_density_percent=50.0,
            warning_messages=warnings,
            rejection_reason=rejection_reason,
        )
