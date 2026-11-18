"""Analyst Review Workflow Coordinator (CP-09).

Orchestrates human-in-the-loop analyst review submissions, summary metrics,
audit verification, and history inspection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.explorer.review_models import AnalystDisposition, ReviewEvent, ReviewSummary
from src.explorer.review_store import ReviewStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReviewService:
    """High-level service interface for human analyst review workflows."""

    def __init__(self, review_store: Optional[ReviewStore] = None):
        """Initialize review service with ReviewStore."""
        self.store = review_store or ReviewStore()

    def submit_review(
        self,
        detection_run_id: str,
        source_dataset_id: str,
        source_row_index: int,
        disposition: str,
        analyst_notes: Optional[str] = None,
        reviewer_label: Optional[str] = None,
    ) -> ReviewEvent:
        """Submit and persist an analyst review event."""
        return self.store.add_review_event(
            detection_run_id=detection_run_id,
            source_dataset_id=source_dataset_id,
            source_row_index=source_row_index,
            disposition=disposition,
            analyst_notes=analyst_notes,
            reviewer_label=reviewer_label,
        )

    def get_summary(
        self,
        detection_run_id: str,
        total_eligible_records: int,
    ) -> ReviewSummary:
        """Compute review summary statistics for a detection run."""
        return self.store.compute_review_summary(
            detection_run_id=detection_run_id,
            total_eligible_records=total_eligible_records,
        )

    def get_history(
        self,
        detection_run_id: str,
        source_row_index: Optional[int] = None,
    ) -> List[ReviewEvent]:
        """Retrieve chronological review events for a run or specific row."""
        return self.store.get_review_history(
            detection_run_id=detection_run_id,
            source_row_index=source_row_index,
        )

    def verify_integrity(self, detection_run_id: str) -> Tuple[bool, List[str]]:
        """Verify the cryptographic hash chain of review events for a run."""
        return self.store.verify_audit_integrity(detection_run_id)
