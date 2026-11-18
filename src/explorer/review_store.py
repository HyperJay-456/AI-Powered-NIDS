"""Persistent Review Storage & Audit Trail Engine (CP-09).

Manages thread-safe, append-only persistence of analyst review events in:
    data/analyst_reviews/<detection_run_id>/
        ├── review_events.jsonl  # Append-only journal with cryptographic hash chaining
        └── review_metadata.json # Run review metadata and head hash

STRICT INVARIANTS:
    1. NEVER modifies CP-08 detection artifacts (predictions.csv, model_comparison.csv, run_manifest.json)
    2. Chronological audit history is preserved (disposition changes append new events, never overwrite)
    3. Cryptographic hash-chaining across events detects accidental tampering or line deletion
    4. Safe handling of interrupted writes and malformed journal lines
    5. Complete isolation between separate detection runs and test fixtures
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

from config.settings import settings
from src.explorer.review_models import AnalystDisposition, ReviewEvent, ReviewStatus, ReviewSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReviewStore:
    """Manages append-only analyst review journals and cryptographic audit trails."""

    _STORE_LOCK = threading.RLock()

    def __init__(self, reviews_dir: Optional[Path] = None):
        """Initialize review store with target reviews directory."""
        self.reviews_dir = reviews_dir or getattr(settings, "analyst_reviews_dir", settings.data_dir / "analyst_reviews")
        self.reviews_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_review_dir(self, detection_run_id: str) -> Path:
        """Get directory path for a detection run's review records with path traversal protection."""
        if ".." in detection_run_id or "/" in detection_run_id or "\\" in detection_run_id:
            raise PermissionError(f"Security violation: path traversal detected in '{detection_run_id}'")

        target = (self.reviews_dir / detection_run_id).resolve()
        if not str(target).startswith(str(self.reviews_dir.resolve())):
            raise PermissionError(f"Security violation: path traversal detected for '{detection_run_id}'")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def add_review_event(
        self,
        detection_run_id: str,
        source_dataset_id: str,
        source_row_index: int,
        disposition: str,
        analyst_notes: Optional[str] = None,
        reviewer_label: Optional[str] = None,
    ) -> ReviewEvent:
        """Record a new human analyst review event in the append-only journal.

        Args:
            detection_run_id: ID of the detection run being reviewed.
            source_dataset_id: Provenance dataset identifier.
            source_row_index: Row index in the detection artifact.
            disposition: Authorized analyst disposition.
            analyst_notes: Optional notes text (up to 2000 chars).
            reviewer_label: Optional informational reviewer display label.

        Returns:
            ReviewEvent: Persisted review event with verified hash.

        Raises:
            ValueError: If disposition is invalid or notes exceed length limit.
        """
        if not AnalystDisposition.is_valid(disposition):
            raise ValueError(
                f"Unauthorized analyst disposition '{disposition}'. "
                f"Must be one of {AnalystDisposition.ALL_DISPOSITIONS}."
            )

        # Sanitize text
        clean_notes = analyst_notes.strip() if analyst_notes else None
        if clean_notes and len(clean_notes) > 2000:
            raise ValueError(f"Analyst notes exceed 2000 characters limit ({len(clean_notes)} chars).")

        clean_label = reviewer_label.strip() if reviewer_label else None
        if clean_label:
            # Strip control characters from label
            clean_label = "".join(c for c in clean_label if c.isprintable())[:100]

        with self._STORE_LOCK:
            run_dir = self._get_run_review_dir(detection_run_id)
            journal_path = run_dir / "review_events.jsonl"
            metadata_path = run_dir / "review_metadata.json"

            # 1. Fetch current review state for this record to populate previous values
            latest_for_row = self.get_latest_review_for_row(detection_run_id, source_row_index)
            prev_status = ReviewStatus.UNREVIEWED if latest_for_row is None else ReviewStatus.REVIEWED
            prev_disposition = latest_for_row.new_disposition if latest_for_row else None

            # 2. Duplicate submission protection: if identical to latest event, avoid creating duplicate
            if latest_for_row is not None:
                if (
                    latest_for_row.new_disposition == disposition
                    and (latest_for_row.analyst_notes or "") == (clean_notes or "")
                    and (latest_for_row.reviewer_label or "") == (clean_label or "")
                ):
                    logger.info(f"Duplicate review submission ignored for row {source_row_index} in run {detection_run_id}")
                    return latest_for_row

            # 3. Determine previous event hash in the journal for hash chaining
            last_event = self._get_last_journal_event(journal_path)
            prev_event_hash = last_event.event_hash if last_event else None

            # 4. Generate unique event ID and timestamp
            timestamp_str = datetime.now(timezone.utc).isoformat()
            event_id = f"rev_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            event = ReviewEvent(
                event_id=event_id,
                detection_run_id=detection_run_id,
                source_dataset_id=source_dataset_id,
                source_row_index=source_row_index,
                timestamp_utc=timestamp_str,
                previous_status=prev_status,
                new_status=ReviewStatus.REVIEWED,
                previous_disposition=prev_disposition,
                new_disposition=disposition,
                analyst_notes=clean_notes,
                reviewer_label=clean_label,
                schema_version="1.0",
                previous_event_hash=prev_event_hash,
            )

            # 5. Append event to journal
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

            # 6. Update review metadata
            self._update_metadata(run_dir, event)

            logger.info(
                f"Recorded analyst review event '{event.event_id}' for row {source_row_index} "
                f"in run '{detection_run_id}' [Disposition: {disposition}]"
            )
            return event

    def _get_last_journal_event(self, journal_path: Path) -> Optional[ReviewEvent]:
        """Read the last valid event from the journal for hash chaining."""
        if not journal_path.exists():
            return None

        last_event: Optional[ReviewEvent] = None
        try:
            with open(journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        last_event = ReviewEvent.from_dict(data)
                    except Exception as e:
                        logger.warning(f"Skipping malformed journal line: {e}")
        except Exception as e:
            logger.error(f"Error reading journal {journal_path}: {e}")
        return last_event

    def get_review_history(
        self,
        detection_run_id: str,
        source_row_index: Optional[int] = None,
    ) -> List[ReviewEvent]:
        """Retrieve chronological review events for a detection run, optionally filtered by row."""
        with self._STORE_LOCK:
            run_dir = self._get_run_review_dir(detection_run_id)
            journal_path = run_dir / "review_events.jsonl"
            if not journal_path.exists():
                return []

            events: List[ReviewEvent] = []
            try:
                with open(journal_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            event = ReviewEvent.from_dict(data)
                            if source_row_index is None or event.source_row_index == source_row_index:
                                events.append(event)
                        except Exception as e:
                            logger.warning(f"Malformed event line {line_num} in {journal_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to read review history for run '{detection_run_id}': {e}")

            return events

    def get_latest_reviews(self, detection_run_id: str) -> Dict[int, ReviewEvent]:
        """Retrieve a dictionary mapping source_row_index -> latest ReviewEvent."""
        events = self.get_review_history(detection_run_id)
        latest_map: Dict[int, ReviewEvent] = {}
        for ev in events:
            # Since events are chronological, later events overwrite earlier ones
            latest_map[ev.source_row_index] = ev
        return latest_map

    def get_latest_review_for_row(
        self,
        detection_run_id: str,
        source_row_index: int,
    ) -> Optional[ReviewEvent]:
        """Retrieve the latest review event for a specific row, or None if unreviewed."""
        history = self.get_review_history(detection_run_id, source_row_index=source_row_index)
        return history[-1] if history else None

    def verify_audit_integrity(self, detection_run_id: str) -> Tuple[bool, List[str]]:
        """Verify the cryptographic hash chain across all events in the review journal.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_integrity_errors)
        """
        with self._STORE_LOCK:
            run_dir = self._get_run_review_dir(detection_run_id)
            journal_path = run_dir / "review_events.jsonl"
            if not journal_path.exists():
                return True, []

            errors: List[str] = []
            prev_hash: Optional[str] = None

            try:
                with open(journal_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            event = ReviewEvent.from_dict(data)
                        except Exception as e:
                            errors.append(f"Line {line_num}: Corrupted JSON record ({e})")
                            continue

                        # Check previous hash link
                        if event.previous_event_hash != prev_hash:
                            errors.append(
                                f"Line {line_num} (Event {event.event_id}): Hash chain broken. "
                                f"Expected previous hash '{prev_hash}', got '{event.previous_event_hash}'"
                            )

                        # Check self-hash computation
                        recomputed = event.compute_event_hash()
                        if recomputed != event.event_hash:
                            errors.append(
                                f"Line {line_num} (Event {event.event_id}): Self-hash mismatch. "
                                f"Recorded '{event.event_hash}' != recomputed '{recomputed}'"
                            )

                        prev_hash = event.event_hash
            except Exception as e:
                errors.append(f"Fatal error reading journal: {e}")

            is_valid = (len(errors) == 0)
            return is_valid, errors

    def compute_review_summary(
        self,
        detection_run_id: str,
        total_eligible_records: int,
    ) -> ReviewSummary:
        """Compute non-fabricated review progress statistics for a detection run.

        Completion percentage = (Reviewed Unique Records / Total Eligible Records) * 100
        """
        latest_map = self.get_latest_reviews(detection_run_id)
        unique_reviewed = len(latest_map)

        # Count dispositions across unique records
        disposition_counts = {disp: 0 for disp in AnalystDisposition.ALL_DISPOSITIONS}
        last_time: Optional[str] = None

        for event in latest_map.values():
            if event.new_disposition in disposition_counts:
                disposition_counts[event.new_disposition] += 1
            if last_time is None or event.timestamp_utc > last_time:
                last_time = event.timestamp_utc

        unreviewed = max(0, total_eligible_records - unique_reviewed)
        pct = 0.0
        if total_eligible_records > 0:
            pct = round((unique_reviewed / total_eligible_records) * 100.0, 2)

        is_valid, _ = self.verify_audit_integrity(detection_run_id)

        all_events = self.get_review_history(detection_run_id)

        return ReviewSummary(
            detection_run_id=detection_run_id,
            total_eligible_records=total_eligible_records,
            reviewed_records=unique_reviewed,
            unreviewed_records=unreviewed,
            completion_percentage=pct,
            disposition_counts=disposition_counts,
            total_review_events=len(all_events),
            last_reviewed_at_utc=last_time,
            integrity_verified=is_valid,
        )

    def _update_metadata(self, run_dir: Path, latest_event: ReviewEvent) -> None:
        """Update review_metadata.json with head hash and timestamps."""
        metadata_path = run_dir / "review_metadata.json"
        meta: Dict[str, Any] = {
            "detection_run_id": latest_event.detection_run_id,
            "head_event_hash": latest_event.event_hash,
            "last_event_id": latest_event.event_id,
            "last_updated_utc": latest_event.timestamp_utc,
            "schema_version": "1.0",
        }
        try:
            temp_path = metadata_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            os.replace(temp_path, metadata_path)
        except Exception as e:
            logger.error(f"Failed to update review metadata in {run_dir}: {e}")
