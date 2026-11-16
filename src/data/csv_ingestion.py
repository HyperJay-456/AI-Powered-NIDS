"""CSV Ingestion and Local Storage Subsystem.

Provides secure filename sanitization, path traversal protection, available disk
space validation, atomic persistence, and SHA-256 content-based deduplication.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
from typing import Any, BinaryIO, Optional, Tuple, Union
import uuid

from config.settings import settings
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sanitization regex: allow alphanumeric, underscore, hyphen, and period
SAFE_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9_\-\.]")


@dataclass
class IngestionResult:
    """Outcome of an uploaded CSV ingestion operation."""

    success: bool
    dataset_id: Optional[str] = None
    original_filename: str = ""
    safe_filename: str = ""
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    file_size_formatted: str = ""
    sha256_checksum: str = ""
    is_duplicate: bool = False
    message: str = ""
    validation_status: str = "Uploaded"


def format_file_size(size_bytes: int) -> str:
    """Format byte size into human-readable representation."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal and invalid characters.

    Args:
        filename: Raw user-provided filename.

    Returns:
        Safe filename string ending with .csv.
    """
    if not filename:
        return "network_traffic.csv"

    # Strip directory components, null bytes, and whitespace
    clean = Path(filename.replace("\x00", "_")).name.strip()

    # Normalize Windows and POSIX separators
    clean = clean.split("/")[-1].split("\\")[-1]

    # Replace forbidden / non-standard characters with underscore
    clean = SAFE_CHAR_PATTERN.sub("_", clean)

    # Collapse multiple consecutive underscores
    clean = re.sub(r"_+", "_", clean)

    # Ensure valid .csv extension
    if clean.lower().endswith(".csv"):
        stem = clean[:-4].strip("._")
    else:
        stem = clean.strip("._")

    if not stem:
        stem = "network_traffic"

    return f"{stem}.csv"


def verify_path_safety(base_dir: Path, target_path: Path) -> bool:
    """Verify that the resolved target path strictly resides within base_dir.

    Args:
        base_dir: Enclosing directory root.
        target_path: Path to verify.

    Returns:
        True if target_path is within base_dir, False otherwise.
    """
    try:
        resolved_base = base_dir.resolve()
        resolved_target = target_path.resolve()
        return resolved_target.is_relative_to(resolved_base)
    except (ValueError, RuntimeError):
        return False


def check_disk_space(
    target_dir: Path,
    required_bytes: int,
    min_margin_mb: int = 500,
) -> Tuple[bool, int, str]:
    """Check whether sufficient disk space is available in the target directory.

    Args:
        target_dir: Directory where the file will be stored.
        required_bytes: Number of bytes to be written.
        min_margin_mb: Safety margin in megabytes (default: 500MB).

    Returns:
        Tuple of (has_space, free_bytes, human_message).
    """
    try:
        usage = shutil.disk_usage(target_dir)
        free_bytes = usage.free
        required_total = required_bytes + (min_margin_mb * 1024 * 1024)

        if free_bytes < required_total:
            msg = (
                f"Insufficient disk space. Free: {format_file_size(free_bytes)}, "
                f"Required: {format_file_size(required_total)} (including {min_margin_mb}MB margin)."
            )
            return False, free_bytes, msg
        return True, free_bytes, "Sufficient space available."
    except Exception as e:
        logger.warning(f"Could not verify disk usage for {target_dir}: {e}")
        # If check fails to query system, proceed with cautious confirmation
        return True, 0, "Disk space check unverified."


def compute_stream_sha256_and_size(
    stream: Union[BinaryIO, Any],
) -> Tuple[str, int]:
    """Compute SHA-256 checksum and total byte size of a readable binary stream.

    Preserves seek position if the stream is seekable.

    Args:
        stream: File-like object supporting read() or getbuffer().

    Returns:
        Tuple of (sha256_hex_digest, size_bytes).
    """
    hasher = hashlib.sha256()
    total_bytes = 0

    # If stream has getbuffer() or getvalue() (e.g., BytesIO or Streamlit UploadedFile)
    if hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except Exception:
            pass

    chunk_size = 64 * 1024
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
        total_bytes += len(chunk)

    if hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except Exception:
            pass

    return hasher.hexdigest(), total_bytes


def resolve_unique_storage_filename(
    raw_dir: Path,
    safe_filename: str,
    sha256_hash: str,
) -> Tuple[str, bool]:
    """Resolve storage filename avoiding collisions while preserving distinct files.

    Args:
        raw_dir: Target directory.
        safe_filename: Sanitized candidate filename.
        sha256_hash: SHA-256 checksum of the file to store.

    Returns:
        Tuple of (final_filename, is_exact_duplicate).
    """
    target = raw_dir / safe_filename
    if not target.exists():
        return safe_filename, False

    # Target exists. Check if it's the exact same file content
    try:
        with open(target, "rb") as f:
            existing_hash, _ = compute_stream_sha256_and_size(f)
        if existing_hash == sha256_hash:
            return safe_filename, True
    except Exception as e:
        logger.warning(f"Failed to hash existing file {target}: {e}")

    # Different content with same filename: generate collision-free unique name
    stem = target.stem
    ext = target.suffix or ".csv"
    short_hash = sha256_hash[:8]
    unique_name = f"{stem}_{short_hash}{ext}"

    if not (raw_dir / unique_name).exists():
        return unique_name, False

    # If even that exists, add incremental counter
    counter = 1
    while (raw_dir / f"{stem}_{short_hash}_{counter}{ext}").exists():
        counter += 1
    return f"{stem}_{short_hash}_{counter}{ext}", False


def save_uploaded_csv(
    uploaded_file: Any,
    raw_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
) -> IngestionResult:
    """Securely store an uploaded CSV file with atomic writing and deduplication.

    Args:
        uploaded_file: Streamlit UploadedFile or file-like object with .name and .read().
        raw_dir: Directory for storing raw CSV files. Defaults to settings.raw_data_dir.
        metadata_dir: Directory for ingestion manifests. Defaults to settings.metadata_dir.

    Returns:
        IngestionResult detailing the outcome of the storage operation.
    """
    raw_dir = raw_dir or settings.raw_data_dir
    metadata_dir = metadata_dir or settings.metadata_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(uploaded_file, "name", "upload.csv")

    # 1. Format and extension validation
    if not original_name.lower().endswith(".csv"):
        return IngestionResult(
            success=False,
            original_filename=original_name,
            message=f"Unsupported file format for '{original_name}'. Only .csv files are supported.",
        )

    # 2. Filename sanitization
    safe_name = sanitize_filename(original_name)
    candidate_path = raw_dir / safe_name

    # 3. Path traversal security check
    if not verify_path_safety(raw_dir, candidate_path):
        return IngestionResult(
            success=False,
            original_filename=original_name,
            message=f"Path traversal security violation detected in '{original_name}'.",
        )

    # 4. Stream checksum & size calculation
    sha256_hash, file_size = compute_stream_sha256_and_size(uploaded_file)
    formatted_size = format_file_size(file_size)

    # 5. Reject empty files
    if file_size == 0:
        return IngestionResult(
            success=False,
            original_filename=original_name,
            file_size_bytes=0,
            file_size_formatted="0 B",
            message=f"Uploaded file '{original_name}' is empty (0 bytes). Rejected.",
        )

    # 6. Check size limit
    max_bytes = settings.ingestion.max_upload_size_mb * 1024 * 1024
    if file_size > max_bytes:
        return IngestionResult(
            success=False,
            original_filename=original_name,
            file_size_bytes=file_size,
            file_size_formatted=formatted_size,
            message=(
                f"File '{original_name}' ({formatted_size}) exceeds configured upload "
                f"limit of {settings.ingestion.max_upload_size_mb} MB."
            ),
        )

    # 7. Check available disk space
    has_space, _, space_msg = check_disk_space(
        raw_dir,
        file_size,
        min_margin_mb=settings.ingestion.min_disk_free_mb,
    )
    if not has_space:
        return IngestionResult(
            success=False,
            original_filename=original_name,
            file_size_bytes=file_size,
            file_size_formatted=formatted_size,
            message=space_msg,
        )

    # 8. Check manifest for existing identical SHA-256
    manifest = IngestionManifest(metadata_dir=metadata_dir)
    existing_record = manifest.get_by_sha256(sha256_hash)
    if existing_record and (raw_dir / existing_record.safe_filename).exists():
        logger.info(f"Duplicate file uploaded: {original_name} matches {existing_record.safe_filename}")
        return IngestionResult(
            success=True,
            dataset_id=existing_record.dataset_id,
            original_filename=original_name,
            safe_filename=existing_record.safe_filename,
            file_path=raw_dir / existing_record.safe_filename,
            file_size_bytes=file_size,
            file_size_formatted=formatted_size,
            sha256_checksum=sha256_hash,
            is_duplicate=True,
            message=f"Exact duplicate detected. Stored under '{existing_record.safe_filename}' (SHA-256: {sha256_hash[:8]}...).",
            validation_status=existing_record.validation_status,
        )

    # 9. Resolve safe storage filename (handles name collisions with different content)
    final_safe_name, is_duplicate_on_disk = resolve_unique_storage_filename(
        raw_dir,
        safe_name,
        sha256_hash,
    )
    final_path = raw_dir / final_safe_name

    if is_duplicate_on_disk:
        dataset_id = str(uuid.uuid4())
        return IngestionResult(
            success=True,
            dataset_id=dataset_id,
            original_filename=original_name,
            safe_filename=final_safe_name,
            file_path=final_path,
            file_size_bytes=file_size,
            file_size_formatted=formatted_size,
            sha256_checksum=sha256_hash,
            is_duplicate=True,
            message=f"Identical file already on disk under '{final_safe_name}'.",
        )

    # 10. Atomic write using temporary file
    temp_filename = f".upload_{uuid.uuid4().hex}.tmp"
    temp_path = raw_dir / temp_filename

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        with open(temp_path, "wb") as f_out:
            chunk_size = 64 * 1024
            while True:
                buf = uploaded_file.read(chunk_size)
                if not buf:
                    break
                f_out.write(buf)
            f_out.flush()
            os.fsync(f_out.fileno())

        # Atomic replacement
        os.replace(temp_path, final_path)
        logger.info(f"Atomically committed upload to {final_path} ({formatted_size})")

        dataset_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Create initial manifest record with 'Uploaded' status
        initial_record = DatasetRecord(
            dataset_id=dataset_id,
            original_filename=original_name,
            safe_filename=final_safe_name,
            file_path=str(final_path),
            file_size_bytes=file_size,
            file_size_formatted=formatted_size,
            sha256_checksum=sha256_hash,
            import_timestamp=now_iso,
            validation_status="Uploaded",
            validation_warnings=["Awaiting structural and schema validation."],
        )
        manifest.add_or_update(initial_record)

        return IngestionResult(
            success=True,
            dataset_id=dataset_id,
            original_filename=original_name,
            safe_filename=final_safe_name,
            file_path=final_path,
            file_size_bytes=file_size,
            file_size_formatted=formatted_size,
            sha256_checksum=sha256_hash,
            is_duplicate=False,
            message=f"File successfully stored as '{final_safe_name}' ({formatted_size}).",
            validation_status="Uploaded",
        )

    except Exception as e:
        logger.error(f"Failed to store uploaded file {original_name}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return IngestionResult(
            success=False,
            original_filename=original_name,
            message=f"Storage failure while writing '{original_name}': {str(e)}",
        )
