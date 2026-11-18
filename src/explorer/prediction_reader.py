"""Prediction Reader & Review Integration Engine (CP-09).

Reads, validates, and caches completed intrusion detection prediction artifacts,
seamlessly joining live analyst review states while maintaining path security
and memory efficiency.
"""

from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.detection.detection_manifest import DetectionRunManifest
from src.explorer.review_models import ReviewStatus
from src.explorer.review_store import ReviewStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionReader:
    """Reads detection artifacts and merges analyst review annotations."""

    _CACHE_LOCK = threading.RLock()
    _PRED_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}

    def __init__(
        self,
        analyses_dir: Optional[Path] = None,
        review_store: Optional[ReviewStore] = None,
    ):
        """Initialize reader with analyses directory and review store."""
        self.analyses_dir = analyses_dir or settings.analyses_dir
        self.analyses_dir.mkdir(parents=True, exist_ok=True)
        self.review_store = review_store or ReviewStore()

    def _resolve_safe_path(self, detection_run_id: str, filename: str) -> Path:
        """Resolve artifact path with strict path-traversal validation."""
        if ".." in detection_run_id or "/" in detection_run_id or "\\" in detection_run_id:
            raise PermissionError(f"Security violation: path traversal detected in '{detection_run_id}'")

        target = (self.analyses_dir / detection_run_id / filename).resolve()
        if not str(target).startswith(str(self.analyses_dir.resolve())):
            raise PermissionError(f"Security violation: path traversal detected for '{detection_run_id}/{filename}'")
        return target

    def load_manifest(self, detection_run_id: str) -> Optional[DetectionRunManifest]:
        """Load and parse DetectionRunManifest for a run."""
        manifest_file = self._resolve_safe_path(detection_run_id, "run_manifest.json")
        if not manifest_file.exists():
            return None

        try:
            import json
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DetectionRunManifest.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load manifest for run '{detection_run_id}': {e}")
            return None

    def load_predictions(
        self,
        detection_run_id: str,
        attach_reviews: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load predictions.csv, optionally joining latest analyst reviews.

        Args:
            detection_run_id: ID of the detection run.
            attach_reviews: Whether to join latest review status and dispositions.
            use_cache: Whether to use cached DataFrame if underlying file is unchanged.

        Returns:
            pd.DataFrame: Formatted prediction records with review annotations.

        Raises:
            FileNotFoundError: If predictions.csv does not exist.
        """
        pred_file = self._resolve_safe_path(detection_run_id, "predictions.csv")
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions artifact not found for run '{detection_run_id}' at {pred_file}")

        mtime = pred_file.stat().st_mtime

        with self._CACHE_LOCK:
            cached_data = self._PRED_CACHE.get(detection_run_id)
            if use_cache and cached_data is not None:
                cached_mtime, base_df = cached_data
                if cached_mtime == mtime:
                    df = base_df.copy()
                else:
                    df = pd.read_csv(pred_file)
                    self._PRED_CACHE[detection_run_id] = (mtime, df.copy())
            else:
                df = pd.read_csv(pred_file)
                self._PRED_CACHE[detection_run_id] = (mtime, df.copy())

        # Standardize source_row_index
        if "source_row_index" in df.columns:
            df["source_row_index"] = df["source_row_index"].astype(int)

        # Standardize agreement_status if missing (e.g. single-model runs)
        if "agreement_status" not in df.columns:
            df["agreement_status"] = "Single Model"

        # Standardize primary predicted_class column if in multi-model format
        if "predicted_class" not in df.columns:
            # Multi-model: look for individual prediction columns
            pred_cols = [c for c in df.columns if c.endswith("_prediction")]
            if pred_cols:
                # Default primary display to the first algorithm's prediction
                df["predicted_class"] = df[pred_cols[0]]
            else:
                df["predicted_class"] = "UNKNOWN"

        # Attach live review annotations from ReviewStore
        if attach_reviews:
            latest_reviews = self.review_store.get_latest_reviews(detection_run_id)

            review_statuses = []
            analyst_dispositions = []
            analyst_notes_list = []
            reviewed_at_list = []
            reviewer_labels = []

            for row_idx in df["source_row_index"]:
                rev = latest_reviews.get(row_idx)
                if rev is not None:
                    review_statuses.append(ReviewStatus.REVIEWED)
                    analyst_dispositions.append(rev.new_disposition)
                    analyst_notes_list.append(rev.analyst_notes or "")
                    reviewed_at_list.append(rev.timestamp_utc)
                    reviewer_labels.append(rev.reviewer_label or "")
                else:
                    review_statuses.append(ReviewStatus.UNREVIEWED)
                    analyst_dispositions.append("Unreviewed")
                    analyst_notes_list.append("")
                    reviewed_at_list.append("")
                    reviewer_labels.append("")

            df["review_status"] = review_statuses
            df["analyst_disposition"] = analyst_dispositions
            df["analyst_notes"] = analyst_notes_list
            df["reviewed_at_utc"] = reviewed_at_list
            df["reviewer_label"] = reviewer_labels

        return df

    def clear_cache(self, detection_run_id: Optional[str] = None) -> None:
        """Clear cached DataFrames."""
        with self._CACHE_LOCK:
            if detection_run_id:
                self._PRED_CACHE.pop(detection_run_id, None)
            else:
                self._PRED_CACHE.clear()
