"""Threat Intelligence & Attack Context Data Models (CP-10).

Defines core schemas for:
  - MITRE ATT&CK Enterprise tactics, techniques, and mitigations
  - Dataset attack label taxonomy
  - Conservative label-to-technique candidate mappings
  - Evidence-aware contextual enrichment and security insights
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

# Official MITRE ATT&CK Attribution Notice & Trademark Disclaimer
MITRE_ATTRIBUTION_NOTICE = (
    "© 2026 The MITRE Corporation. This work is reproduced and distributed "
    "with the permission of The MITRE Corporation."
)

MITRE_DISCLAIMER = (
    "MITRE ATT&CK is a registered trademark of The MITRE Corporation. "
    "MITRE does not endorse or certify this project."
)


# ==============================================================================
# MITRE ATT&CK Reference Schemas
# ==============================================================================

@dataclass
class Tactic:
    """MITRE ATT&CK Enterprise Tactic representation."""

    id: str  # STIX object ID (e.g. x-mitre-tactic--...)
    external_id: str  # ATT&CK Tactic ID (e.g. TA0007)
    name: str  # e.g. Discovery
    shortname: str  # e.g. discovery
    description: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tactic":
        return cls(**data)


@dataclass
class Technique:
    """MITRE ATT&CK Enterprise Technique representation."""

    id: str  # STIX object ID (e.g. attack-pattern--...)
    external_id: str  # ATT&CK Technique ID (e.g. T1046)
    name: str  # e.g. Network Service Discovery
    description: str
    tactics: List[str]  # e.g. ["discovery"]
    platforms: List[str]  # e.g. ["Linux", "Windows", "macOS"]
    is_subtechnique: bool = False
    parent_technique_id: Optional[str] = None  # e.g. T1059 for T1059.001
    detection_guidance: str = ""
    url: str = ""
    is_deprecated: bool = False
    is_revoked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Technique":
        return cls(**data)


@dataclass
class Mitigation:
    """MITRE ATT&CK Mitigation (Course of Action)."""

    id: str  # STIX ID (e.g. course-of-action--...)
    external_id: str  # e.g. M1031
    name: str  # e.g. Network Intrusion Prevention
    description: str
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mitigation":
        return cls(**data)


@dataclass
class AttackRelationship:
    """Relationship between ATT&CK objects (e.g. mitigates, subtechnique-of)."""

    source_ref: str
    target_ref: str
    relationship_type: str  # "mitigates", "subtechnique-of"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackBundleMetadata:
    """Metadata for an installed or imported MITRE ATT&CK STIX bundle."""

    dataset_name: str
    domain: str
    version: str
    release_date: str
    source_repository: str
    source_file: str
    import_timestamp_utc: str
    sha256_checksum: str
    stix_spec_version: str
    total_objects: int
    tactics_count: int
    techniques_count: int
    mitigations_count: int
    validation_status: str  # "Valid", "Corrupted", "Incomplete"
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackBundleMetadata":
        return cls(**data)


# ==============================================================================
# Dataset Label Taxonomy & Mappings
# ==============================================================================

class AttackFamily:
    """Broad threat and attack families."""

    BENIGN = "Benign"
    DENIAL_OF_SERVICE = "Denial of Service"
    DDOS = "Distributed Denial of Service"
    BRUTE_FORCE = "Brute Force"
    PORT_SCANNING = "Port Scanning"
    WEB_ATTACK = "Web Attack"
    BOT_ACTIVITY = "Bot Activity"
    INFILTRATION = "Infiltration"
    EXPLOITATION = "Exploitation"
    OTHER = "Other / Unclassified"

    ALL_FAMILIES: Tuple[str, ...] = (
        BENIGN,
        DENIAL_OF_SERVICE,
        DDOS,
        BRUTE_FORCE,
        PORT_SCANNING,
        WEB_ATTACK,
        BOT_ACTIVITY,
        INFILTRATION,
        EXPLOITATION,
        OTHER,
    )


@dataclass
class DatasetLabelTaxonomyItem:
    """Standardized taxonomy record for a dataset attack classification."""

    label_id: str
    original_label: str
    normalized_label: str
    display_label: str
    broad_family: str
    description: str
    dataset_source: str = "CIC-IDS-2017"
    mapping_status: str = "Candidate"
    analyst_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetLabelTaxonomyItem":
        return cls(**data)


class MappingStatus:
    """Review and acceptance statuses for dataset-to-ATT&CK candidate mappings."""

    CANDIDATE = "Candidate"
    REVIEWED = "Reviewed"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    UNMAPPED = "Unmapped"
    DEPRECATED = "Deprecated"

    ALL_STATUSES: Tuple[str, ...] = (
        CANDIDATE,
        REVIEWED,
        APPROVED,
        REJECTED,
        UNMAPPED,
        DEPRECATED,
    )

    @classmethod
    def is_valid(cls, status: str) -> bool:
        return status in cls.ALL_STATUSES


class MappingStrength:
    """Descriptive evidence-based alignment category."""

    DIRECT = "Direct behavior alignment"
    CONDITIONAL = "Conditional behavior alignment"
    BROAD = "Broad contextual association"
    INSUFFICIENT = "Insufficient evidence"

    ALL_STRENGTHS: Tuple[str, ...] = (
        DIRECT,
        CONDITIONAL,
        BROAD,
        INSUFFICIENT,
    )


@dataclass
class LabelTechniqueMapping:
    """Candidate or approved mapping between a dataset attack label and ATT&CK technique."""

    mapping_id: str
    original_dataset_label: str
    normalized_dataset_label: str
    technique_id: str  # e.g. T1046 or "" if unmapped
    technique_name: str
    domain: str = "Enterprise"
    tactic: str = ""
    mapping_status: str = MappingStatus.CANDIDATE
    mapping_strength: str = MappingStrength.BROAD
    mapping_rationale: str = ""
    required_evidence: List[str] = field(default_factory=list)
    available_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    mitigation_ids: List[str] = field(default_factory=list)
    detection_strategy_notes: str = ""
    source_references: List[str] = field(default_factory=list)
    attack_version: str = "v20.1"
    created_at_utc: str = ""
    last_reviewed_at_utc: Optional[str] = None
    reviewer_label: Optional[str] = None
    analyst_notes: Optional[str] = None
    mapping_hash: str = ""

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            self.created_at_utc = datetime.now(timezone.utc).isoformat()
        if not self.mapping_hash:
            self.mapping_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the canonical mapping fields."""
        payload = {
            "mapping_id": self.mapping_id,
            "original_dataset_label": self.original_dataset_label,
            "technique_id": self.technique_id,
            "mapping_status": self.mapping_status,
            "mapping_strength": self.mapping_strength,
            "mapping_rationale": self.mapping_rationale,
            "reviewer_label": self.reviewer_label or "",
            "analyst_notes": self.analyst_notes or "",
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelTechniqueMapping":
        return cls(**data)


@dataclass
class MappingReviewEvent:
    """Audit record of a human review action performed on a technique mapping."""

    event_id: str
    mapping_id: str
    timestamp_utc: str
    previous_status: str
    new_status: str
    previous_technique_id: str
    new_technique_id: str
    analyst_notes: Optional[str] = None
    reviewer_label: Optional[str] = None
    previous_event_hash: Optional[str] = None
    event_hash: str = ""

    def __post_init__(self) -> None:
        if not self.event_hash:
            self.event_hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = {
            "event_id": self.event_id,
            "mapping_id": self.mapping_id,
            "timestamp_utc": self.timestamp_utc,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "previous_technique_id": self.previous_technique_id,
            "new_technique_id": self.new_technique_id,
            "analyst_notes": self.analyst_notes or "",
            "reviewer_label": self.reviewer_label or "",
            "previous_event_hash": self.previous_event_hash or "",
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MappingReviewEvent":
        return cls(**data)


# ==============================================================================
# Evidence-Aware Threat Context
# ==============================================================================

@dataclass
class EnrichedAttackContext:
    """Forensically enriched context for a network flow prediction."""

    predicted_class: str
    model_name: str
    model_score: Optional[float]
    score_type: str
    source_row_index: int
    taxonomy_item: Optional[DatasetLabelTaxonomyItem]
    candidate_mapping: Optional[LabelTechniqueMapping]
    technique_detail: Optional[Technique]
    relevant_mitigations: List[Mitigation]
    available_evidence_fields: Dict[str, Any]
    missing_evidence_fields: List[str]
    ground_truth_label: Optional[str]
    analyst_disposition: Optional[str]
    disagreement_state: str
    context_disclaimer: str = (
        "Dataset classification indicates statistical model alignment with training patterns. "
        "It is NOT verified proof of an adversary technique, campaign, or real-world threat attribution."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "model_name": self.model_name,
            "model_score": self.model_score,
            "score_type": self.score_type,
            "source_row_index": self.source_row_index,
            "taxonomy_item": self.taxonomy_item.to_dict() if self.taxonomy_item else None,
            "candidate_mapping": self.candidate_mapping.to_dict() if self.candidate_mapping else None,
            "technique_detail": self.technique_detail.to_dict() if self.technique_detail else None,
            "relevant_mitigations": [m.to_dict() for m in self.relevant_mitigations],
            "available_evidence_fields": self.available_evidence_fields,
            "missing_evidence_fields": self.missing_evidence_fields,
            "ground_truth_label": self.ground_truth_label,
            "analyst_disposition": self.analyst_disposition,
            "disagreement_state": self.disagreement_state,
            "context_disclaimer": self.context_disclaimer,
        }
