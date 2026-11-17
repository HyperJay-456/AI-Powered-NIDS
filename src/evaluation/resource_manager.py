"""Hardware Resource Management & Evaluation Preflight (CP-07).

Inspects available RAM, disk space, and matrix dimensions before running evaluation:
- Estimates prediction batch memory based on sparse matrix representation
- Estimates score array storage (n_samples * n_classes * 8 bytes)
- Determines whether exact ROC/PR curve calculation is safe or needs gating
- Calculates optimal batch size (default 10,000, dynamically adjusted if RAM is constrained)
- Checks disk storage for temporary score files and persistent evaluation artifacts
- Strictly enforces sequential model evaluation to avoid peak memory contention
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationPreflightResult:
    """Outcome of evaluation resource preflight checks."""

    is_safe: bool
    batch_size: int
    available_ram_bytes: Optional[int]
    available_ram_formatted: str
    available_disk_bytes: int
    available_disk_formatted: str
    estimated_batch_memory_bytes: int
    estimated_batch_memory_formatted: str
    estimated_scores_memory_bytes: int
    estimated_scores_memory_formatted: str
    exact_curves_practical: bool
    num_samples: int
    num_features: int
    num_classes: int
    warning_messages: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)


class EvaluationResourceManager:
    """Manages memory and storage gating during model evaluation passes."""

    DEFAULT_BATCH_SIZE: int = 10_000
    MIN_BATCH_SIZE: int = 1_000
    MAX_BATCH_SIZE: int = 50_000
    MIN_DISK_FREE_MB: int = 500
    MIN_RAM_FREE_MB: int = 512

    @staticmethod
    def get_available_system_memory() -> Optional[int]:
        """Query available system RAM in bytes without external packages."""
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
                            return int(parts[1]) * 1024
            except Exception as e:
                logger.debug(f"Linux /proc/meminfo query failed: {e}")
                return None
        return None

    @classmethod
    def get_available_disk_space(cls, target_path: Optional[Path] = None) -> int:
        """Query available free disk space in bytes for the evaluation target path."""
        p = target_path or settings.evaluations_dir
        if not p.exists():
            p = p.parent
        try:
            total, used, free = shutil.disk_usage(p)
            return free
        except Exception as e:
            logger.debug(f"Disk usage check failed: {e}")
            return 10 * 1024 * 1024 * 1024  # Fallback: assume 10 GB

    @classmethod
    def format_bytes(cls, num_bytes: Optional[int]) -> str:
        """Format raw byte counts into human-readable strings."""
        if num_bytes is None:
            return "Unavailable"
        if num_bytes < 1024:
            return f"{num_bytes} B"
        elif num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.2f} KB"
        elif num_bytes < 1024 * 1024 * 1024:
            return f"{num_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"

    @classmethod
    def perform_preflight_check(
        cls,
        num_samples: int,
        num_features: int,
        num_classes: int,
        sparse_density: float = 0.15,
        target_path: Optional[Path] = None,
    ) -> EvaluationPreflightResult:
        """Evaluate resource safety for inference and metric computation.

        Args:
            num_samples: Number of records in evaluation partition.
            num_features: Number of features per sample.
            num_classes: Number of distinct classes.
            sparse_density: Fraction of non-zero entries in CSR matrix.
            target_path: Path where evaluation artifacts will be written.

        Returns:
            EvaluationPreflightResult with safety decision, optimal batch size, and curve flags.
        """
        rejection_reasons: List[str] = []
        warning_messages: List[str] = []

        avail_ram = cls.get_available_system_memory()
        avail_disk = cls.get_available_disk_space(target_path)

        # 1. Determine optimal batch size
        batch_size = cls.DEFAULT_BATCH_SIZE
        if avail_ram is not None:
            avail_ram_mb = avail_ram / (1024 * 1024)
            if avail_ram_mb < cls.MIN_RAM_FREE_MB:
                rejection_reasons.append(
                    f"Critically low system RAM: {avail_ram_mb:.1f} MB available, "
                    f"minimum {cls.MIN_RAM_FREE_MB} MB required for evaluation."
                )
            elif avail_ram_mb < 2048:
                batch_size = cls.MIN_BATCH_SIZE
                warning_messages.append(
                    f"System RAM is constrained ({avail_ram_mb:.1f} MB). Reduced batch size to {batch_size:,}."
                )

        # 2. Check disk space
        avail_disk_mb = avail_disk / (1024 * 1024)
        if avail_disk_mb < cls.MIN_DISK_FREE_MB:
            rejection_reasons.append(
                f"Insufficient disk space: {avail_disk_mb:.1f} MB available, "
                f"minimum {cls.MIN_DISK_FREE_MB} MB required."
            )

        # 3. Estimate memory footprints
        # Batch memory: batch_size rows * num_features * 8 bytes (double) * density
        est_batch_bytes = int(batch_size * num_features * 8 * max(0.05, sparse_density) * 2)

        # Full score array memory: num_samples * num_classes * 8 bytes (float64)
        est_scores_bytes = int(num_samples * num_classes * 8)

        # 4. Assess exact ROC/PR curve practicality
        # If score array exceeds 500 MB or available RAM is below 1 GB, gate exact curve computation
        exact_curves_practical = True
        if est_scores_bytes > 500 * 1024 * 1024:
            exact_curves_practical = False
            warning_messages.append(
                f"Full score matrix requires {cls.format_bytes(est_scores_bytes)}. "
                "ROC/PR curves will use disk-backed score staging or sampled visualization."
            )
        elif avail_ram is not None and est_scores_bytes > (avail_ram * 0.25):
            exact_curves_practical = False
            warning_messages.append(
                "Score matrix would consume more than 25% of available system memory. Gating exact curve display."
            )

        is_safe = len(rejection_reasons) == 0

        return EvaluationPreflightResult(
            is_safe=is_safe,
            batch_size=batch_size,
            available_ram_bytes=avail_ram,
            available_ram_formatted=cls.format_bytes(avail_ram),
            available_disk_bytes=avail_disk,
            available_disk_formatted=cls.format_bytes(avail_disk),
            estimated_batch_memory_bytes=est_batch_bytes,
            estimated_batch_memory_formatted=cls.format_bytes(est_batch_bytes),
            estimated_scores_memory_bytes=est_scores_bytes,
            estimated_scores_memory_formatted=cls.format_bytes(est_scores_bytes),
            exact_curves_practical=exact_curves_practical,
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            warning_messages=warning_messages,
            rejection_reasons=rejection_reasons,
        )
