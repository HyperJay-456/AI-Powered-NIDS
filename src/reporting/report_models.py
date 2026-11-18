"""Report Data Structures & Models for Reporting & Evidence Export (CP-11).

Defines core data models, report types, evidence manifests, redaction options,
and metadata schemas for academic cybersecurity reports and verifiable evidence bundles.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ReportType:
    """Authorized report types in the NIDS reporting subsystem."""

    EXECUTIVE = "Executive Security Summary"
    TECHNICAL = "Technical Detection Report"
    ANALYST_REVIEW = "Analyst Review Report"
    CONSOLIDATED = "Consolidated Assessment Report"

    ALL_TYPES: Tuple[str, ...] = (
        EXECUTIVE,
        TECHNICAL,
        ANALYST_REVIEW,
        CONSOLIDATED,
    )

    @classmethod
    def is_valid(cls, report_type: str) -> bool:
        """Check if report type is recognized."""
        return report_type in cls.ALL_TYPES


class ReportStatus:
    """Lifecycle status of a report artifact generation task."""

    PENDING = "Pending"
    GENERATING = "Generating"
    COMPLETED = "Completed"
    FAILED = "Failed"

    ALL_STATUSES: Tuple[str, ...] = (PENDING, GENERATING, COMPLETED, FAILED)


class ReportClassification:
    """Traffic Light Protocol (TLP) and academic classification markings."""

    ACADEMIC = "Academic / Internal Research"
    TLP_CLEAR = "TLP:CLEAR"
    TLP_GREEN = "TLP:GREEN"
    TLP_AMBER = "TLP:AMBER"

    ALL_CLASSIFICATIONS: Tuple[str, ...] = (
        ACADEMIC,
        TLP_CLEAR,
        TLP_GREEN,
        TLP_AMBER,
    )


@dataclass
class RedactionConfig:
    """Privacy and sensitivity configuration for report generation and evidence exports."""

    redact_ips: bool = True
    include_analyst_notes: bool = True
    include_network_identifiers: bool = False
    mask_str: str = "xxx"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedactionConfig":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class EvidenceItem:
    """Single digital evidence item included in a report package or ZIP bundle."""

    evidence_id: str
    filename: str
    artifact_type: str
    source_checkpoint: str
    source_run_id: str
    sha256_checksum: str
    size_bytes: int
    export_timestamp_utc: str
    inclusion_reason: str
    redaction_applied: bool = False
    relative_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class EvidenceManifest:
    """Cryptographic manifest documenting all artifacts included in an evidence bundle."""

    report_id: str
    created_at_utc: str
    items: List[EvidenceItem] = field(default_factory=list)
    total_files: int = 0
    total_size_bytes: int = 0
    manifest_version: str = "1.0"

    def __post_init__(self) -> None:
        """Update file and byte totals from items."""
        if self.items:
            self.total_files = len(self.items)
            self.total_size_bytes = sum(item.size_bytes for item in self.items)

    def add_item(self, item: EvidenceItem) -> None:
        """Add evidence item to manifest and update totals."""
        self.items.append(item)
        self.total_files = len(self.items)
        self.total_size_bytes = sum(i.size_bytes for i in self.items)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "manifest_version": self.manifest_version,
            "report_id": self.report_id,
            "created_at_utc": self.created_at_utc,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceManifest":
        """Reconstruct from dictionary."""
        raw_items = data.get("items", [])
        items = [EvidenceItem.from_dict(item) if isinstance(item, dict) else item for item in raw_items]
        return cls(
            report_id=data.get("report_id", ""),
            created_at_utc=data.get("created_at_utc", ""),
            items=items,
            total_files=data.get("total_files", len(items)),
            total_size_bytes=data.get("total_size_bytes", sum(i.size_bytes for i in items)),
            manifest_version=data.get("manifest_version", "1.0"),
        )


@dataclass
class ReportMetadata:
    """Master metadata record for an individual generated report."""

    report_id: str
    report_type: str
    title: str
    classification: str
    detection_run_id: str
    source_dataset_id: str
    source_filename: str
    engineering_run_id: str
    created_at_utc: str
    completed_at_utc: Optional[str] = None
    selected_models: List[str] = field(default_factory=list)
    total_records_analyzed: int = 0
    predicted_attack_records: int = 0
    predicted_benign_records: int = 0
    analyst_reviewed_count: int = 0
    status: str = ReportStatus.PENDING
    pdf_filename: Optional[str] = None
    pdf_sha256: Optional[str] = None
    pdf_size_bytes: Optional[int] = None
    pdf_page_count: Optional[int] = None
    bundle_filename: Optional[str] = None
    bundle_sha256: Optional[str] = None
    bundle_size_bytes: Optional[int] = None
    redaction_config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = "1.0"
    is_test_fixture: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportMetadata":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class ReportDataPackage:
    """Aggregated verified data payload for report compilation and structured exports."""

    metadata: ReportMetadata
    run_summary: Dict[str, Any] = field(default_factory=dict)
    prediction_distribution: List[Dict[str, Any]] = field(default_factory=list)
    model_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    analyst_summary: Dict[str, Any] = field(default_factory=dict)
    attck_mappings: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_metrics: Optional[Dict[str, Any]] = None
    disclaimers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire package to JSON-serializable dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "run_summary": self.run_summary,
            "prediction_distribution": self.prediction_distribution,
            "model_comparisons": self.model_comparisons,
            "analyst_summary": self.analyst_summary,
            "attck_mappings": self.attck_mappings,
            "evaluation_metrics": self.evaluation_metrics,
            "disclaimers": self.disclaimers,
        }
