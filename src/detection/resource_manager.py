"""Resource Manager & Preflight Safety Engine (CP-08).

Enforces strict resource bounds, system memory safety, disk headroom checks,
and safe chunk sizing for operational inference across the AMD Ryzen / 24 GB RAM environment.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import shutil
from typing import List, Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionPreflightResult:
    """Outcome of pre-inference resource safety analysis."""

    is_safe: bool
    batch_size: int
    available_ram_bytes: Optional[int]
    available_disk_bytes: int
    estimated_chunk_ram_bytes: int
    warning_messages: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)


class DetectionResourceManager:
    """Manages memory and storage gating during operational model inference."""

    DEFAULT_BATCH_SIZE: int = 10_000
    MIN_BATCH_SIZE: int = 1_000
    MAX_BATCH_SIZE: int = 50_000
    MIN_DISK_FREE_MB: int = 500
    MIN_RAM_FREE_MB: int = 512

    @staticmethod
    def get_available_system_memory() -> Optional[int]:
        """Query available system RAM in bytes without external non-standard dependencies."""
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
                logger.debug(f"Failed to query Windows memory status: {e}")
        elif system == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except Exception as e:
                logger.debug(f"Failed to query Linux meminfo: {e}")
        return None

    @staticmethod
    def get_available_disk_space(target_path: Optional[Path] = None) -> int:
        """Query free disk space in bytes for the specified path."""
        path = target_path or settings.data_dir
        path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(path))
        return usage.free

    @classmethod
    def perform_preflight_check(
        cls,
        num_features: int = 80,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_models: int = 1,
        target_path: Optional[Path] = None,
    ) -> DetectionPreflightResult:
        """Evaluate memory and disk constraints before executing inference chunks."""
        warnings: List[str] = []
        rejections: List[str] = []

        # Validate and clamp batch size within established bounds
        effective_batch_size = max(cls.MIN_BATCH_SIZE, min(batch_size, cls.MAX_BATCH_SIZE))
        if effective_batch_size != batch_size:
            warnings.append(
                f"Requested batch size {batch_size:,} was clamped to safe bounds [{cls.MIN_BATCH_SIZE:,}, {cls.MAX_BATCH_SIZE:,}]: using {effective_batch_size:,}."
            )

        # Estimate chunk memory:
        # Raw chunk DataFrame + Imputed DataFrame + Sparse matrix + Model prediction arrays
        # ~8 bytes per float64/int64 cell * num_features * batch_size * safety multiplier (4x for copies/sparse overhead)
        bytes_per_sample = max(num_features, 10) * 8
        estimated_chunk_bytes = int(effective_batch_size * bytes_per_sample * 4 * max(1, num_models))

        # Check Available RAM
        avail_ram = cls.get_available_system_memory()
        min_ram_bytes = cls.MIN_RAM_FREE_MB * 1024 * 1024
        if avail_ram is not None:
            if avail_ram < min_ram_bytes:
                rejections.append(
                    f"Insufficient system RAM: {avail_ram / (1024*1024):.1f} MB available, but minimum {cls.MIN_RAM_FREE_MB} MB required."
                )
            elif estimated_chunk_bytes > (avail_ram * 0.5):
                # Auto-downscale batch size if chunk would consume over 50% of available RAM
                safe_batch = max(cls.MIN_BATCH_SIZE, int((avail_ram * 0.25) / (bytes_per_sample * 4 * max(1, num_models))))
                warnings.append(
                    f"Estimated chunk memory ({estimated_chunk_bytes / (1024*1024):.1f} MB) exceeds safe headroom. "
                    f"Auto-scaled batch size from {effective_batch_size:,} to {safe_batch:,}."
                )
                effective_batch_size = safe_batch
                estimated_chunk_bytes = int(effective_batch_size * bytes_per_sample * 4 * max(1, num_models))

        # Check Available Disk
        avail_disk = cls.get_available_disk_space(target_path)
        min_disk_bytes = cls.MIN_DISK_FREE_MB * 1024 * 1024
        if avail_disk < min_disk_bytes:
            rejections.append(
                f"Insufficient disk space: {avail_disk / (1024*1024):.1f} MB free, minimum {cls.MIN_DISK_FREE_MB} MB required."
            )

        is_safe = (len(rejections) == 0)

        return DetectionPreflightResult(
            is_safe=is_safe,
            batch_size=effective_batch_size,
            available_ram_bytes=avail_ram,
            available_disk_bytes=avail_disk,
            estimated_chunk_ram_bytes=estimated_chunk_bytes,
            warning_messages=warnings,
            rejection_reasons=rejections,
        )
