"""Local MITRE ATT&CK Reference Registry & Import Manager (CP-10).

Manages local, offline-capable versioned MITRE ATT&CK Enterprise STIX 2.1 datasets in:
    data/knowledge/mitre_attack/
        ├── manifest.json
        └── enterprise-attack-20.1.json

Provides:
  - Controlled manual STIX bundle import with schema & integrity validation
  - Atomic staging via .tmp files
  - Cryptographic SHA-256 verification and version tracking
  - In-memory caching for responsive Streamlit reruns
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.threat_intelligence.attack_catalog import AttackCatalog
from src.threat_intelligence.attack_models import AttackBundleMetadata
from src.threat_intelligence.attack_stix_loader import AttackSTIXLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReferenceRegistry:
    """Manages versioned local MITRE ATT&CK STIX 2.1 datasets."""

    _CACHE_LOCK = threading.RLock()
    _CACHED_CATALOG: Optional[AttackCatalog] = None
    _CACHED_METADATA: Optional[AttackBundleMetadata] = None
    _CACHED_MTIME: float = 0.0

    def __init__(self, mitre_dir: Optional[Path] = None):
        """Initialize registry with target knowledge directory."""
        self.mitre_dir = mitre_dir or getattr(
            settings, "mitre_attack_dir", settings.data_dir / "knowledge" / "mitre_attack"
        )
        self.mitre_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.mitre_dir / "manifest.json"

    def _resolve_safe_path(self, filename: str) -> Path:
        """Resolve bundle file path with strict path-traversal validation."""
        if ".." in filename or "/" in filename or "\\" in filename:
            raise PermissionError(f"Security violation: path traversal detected in '{filename}'")
        target = (self.mitre_dir / filename).resolve()
        if not str(target).startswith(str(self.mitre_dir.resolve())):
            raise PermissionError(f"Security violation: path traversal detected for '{filename}'")
        return target

    def get_metadata(self) -> Optional[AttackBundleMetadata]:
        """Load metadata for the active reference bundle from manifest.json."""
        if not self.manifest_path.exists():
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AttackBundleMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load reference manifest: {e}")
            return None

    def get_active_catalog(self) -> Optional[AttackCatalog]:
        """Load and return the active in-memory AttackCatalog.

        Uses thread-safe mtime caching to prevent reparsing large STIX bundles on every UI rerun.
        """
        meta = self.get_metadata()
        if not meta or not meta.source_file:
            # Check if there is an active enterprise-attack JSON file in the directory
            bundles = list(self.mitre_dir.glob("enterprise-attack*.json"))
            if not bundles:
                return None
            target_file = bundles[0]
        else:
            try:
                target_file = self._resolve_safe_path(meta.source_file)
            except Exception:
                return None

        if not target_file.exists():
            return None

        mtime = target_file.stat().st_mtime

        with self._CACHE_LOCK:
            if self._CACHED_CATALOG is not None and self._CACHED_MTIME == mtime:
                return self._CACHED_CATALOG

            try:
                catalog, parsed_meta = AttackSTIXLoader.load_from_file(target_file)
                self._CACHED_CATALOG = catalog
                self._CACHED_METADATA = parsed_meta
                self._CACHED_MTIME = mtime
                return catalog
            except Exception as e:
                logger.error(f"Failed to load ATT&CK catalog from '{target_file}': {e}")
                return None

    def import_stix_bundle(
        self,
        bundle_file_path: Path,
        version_label: str = "v20.1",
        release_date: str = "2026-11-05",
    ) -> Tuple[bool, str, Optional[AttackBundleMetadata]]:
        """Validate, stage, and activate a local MITRE ATT&CK STIX 2.1 bundle.

        Args:
            bundle_file_path: Path to the STIX JSON file.
            version_label: Version tag (e.g. 'v20.1').
            release_date: Official release date string.

        Returns:
            Tuple[bool, str, Optional[AttackBundleMetadata]]: (success, message, metadata)
        """
        src_path = Path(bundle_file_path).resolve()
        if not src_path.exists():
            return False, f"Source STIX bundle file not found at '{src_path}'.", None

        # 1. Validate STIX Bundle Schema & Parse In-Memory
        try:
            catalog, meta = AttackSTIXLoader.load_from_file(src_path)
        except Exception as e:
            return False, f"STIX bundle validation failed: {e}", None

        # 2. Prepare Atomic Staging
        target_filename = f"enterprise-attack-{version_label.lstrip('v')}.json"
        target_path = self.mitre_dir / target_filename
        temp_path = self.mitre_dir / f".tmp_{target_filename}"

        try:
            shutil.copy2(src_path, temp_path)

            # Re-verify checksum of staged copy
            with open(temp_path, "rb") as f:
                staged_sha = hashlib.sha256(f.read()).hexdigest()
            if staged_sha != meta.sha256_checksum:
                if temp_path.exists():
                    temp_path.unlink()
                return False, "Staged file SHA-256 verification mismatch.", None

            # Commit atomically
            os.replace(temp_path, target_path)

            # 3. Update Manifest
            now_iso = datetime.now(timezone.utc).isoformat()
            meta.source_file = target_filename
            meta.version = version_label
            meta.release_date = release_date
            meta.import_timestamp_utc = now_iso
            meta.is_active = True

            with open(self.manifest_path, "w", encoding="utf-8") as mf:
                json.dump(meta.to_dict(), mf, indent=2)

            # 4. Invalidate and refresh cache
            with self._CACHE_LOCK:
                self._CACHED_CATALOG = catalog
                self._CACHED_METADATA = meta
                self._CACHED_MTIME = target_path.stat().st_mtime

            logger.info(
                f"Successfully imported and activated ATT&CK bundle '{target_filename}' "
                f"({meta.techniques_count} techniques, {meta.tactics_count} tactics)."
            )
            return True, f"Successfully imported MITRE ATT&CK {version_label} ({meta.techniques_count} techniques).", meta

        except Exception as e:
            logger.error(f"Failed to import STIX bundle: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False, f"Failed to activate STIX bundle: {e}", None

    def clear_cache(self) -> None:
        """Clear cached catalog to force reload."""
        with self._CACHE_LOCK:
            self._CACHED_CATALOG = None
            self._CACHED_METADATA = None
            self._CACHED_MTIME = 0.0
