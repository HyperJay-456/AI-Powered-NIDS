"""Review Data Structures & Models for Prediction Explorer (CP-09).

Defines the core data models, allowed dispositions, review statuses, and review event
schemas for human-in-the-loop analyst review and persistent audit trails.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


class AnalystDisposition:
    """Authorized analyst dispositions.
    
    CRITICAL: These represent human-selected analyst assessments, NOT automated model predictions.
    """

    POTENTIALLY_MALICIOUS = "Potentially Malicious"
    LIKELY_BENIGN = "Likely Benign"
    FALSE_POSITIVE = "False Positive"
    NEEDS_FURTHER_INVESTIGATION = "Needs Further Investigation"
    INCONCLUSIVE = "Inconclusive"

    ALL_DISPOSITIONS: Tuple[str, ...] = (
        POTENTIALLY_MALICIOUS,
        LIKELY_BENIGN,
        FALSE_POSITIVE,
        NEEDS_FURTHER_INVESTIGATION,
        INCONCLUSIVE,
    )

    @classmethod
    def is_valid(cls, disposition: str) -> bool:
        """Check if a given disposition string is authorized."""
        return disposition in cls.ALL_DISPOSITIONS


class ReviewStatus:
    """Operational review status of a network flow prediction record."""

    UNREVIEWED = "Unreviewed"
    REVIEWED = "Reviewed"

    ALL_STATUSES: Tuple[str, ...] = (UNREVIEWED, REVIEWED)


@dataclass
class ReviewEvent:
    """Individual human-in-the-loop analyst review event.
    
    Represents an immutable, timestamped action taken by a security analyst on an individual
    traffic prediction record from a specific detection run.
    """

    event_id: str
    detection_run_id: str
    source_dataset_id: str
    source_row_index: int
    timestamp_utc: str
    previous_status: str
    new_status: str
    previous_disposition: Optional[str]
    new_disposition: str
    analyst_notes: Optional[str] = None
    reviewer_label: Optional[str] = None
    schema_version: str = "1.0"
    previous_event_hash: Optional[str] = None
    event_hash: str = ""

    def __post_init__(self) -> None:
        """Validate review event attributes upon instantiation."""
        if not AnalystDisposition.is_valid(self.new_disposition):
            raise ValueError(
                f"Unauthorized analyst disposition: '{self.new_disposition}'. "
                f"Must be one of {AnalystDisposition.ALL_DISPOSITIONS}."
            )
        if self.analyst_notes and len(self.analyst_notes) > 2000:
            raise ValueError(
                f"Analyst notes exceed maximum length of 2000 characters ({len(self.analyst_notes)} chars)."
            )
        if not self.event_hash:
            self.event_hash = self.compute_event_hash()

    def compute_event_hash(self) -> str:
        """Compute SHA-256 hash of the canonical JSON representation of this event."""
        payload = {
            "event_id": self.event_id,
            "detection_run_id": self.detection_run_id,
            "source_dataset_id": self.source_dataset_id,
            "source_row_index": self.source_row_index,
            "timestamp_utc": self.timestamp_utc,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "previous_disposition": self.previous_disposition,
            "new_disposition": self.new_disposition,
            "analyst_notes": self.analyst_notes or "",
            "reviewer_label": self.reviewer_label or "",
            "schema_version": self.schema_version,
            "previous_event_hash": self.previous_event_hash or "",
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewEvent":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class ReviewSummary:
    """Summary statistics of analyst review progress for a detection run."""

    detection_run_id: str
    total_eligible_records: int
    reviewed_records: int
    unreviewed_records: int
    completion_percentage: float
    disposition_counts: Dict[str, int]
    total_review_events: int
    last_reviewed_at_utc: Optional[str] = None
    integrity_verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewSummary":
        """Reconstruct from dictionary."""
        return cls(**data)
