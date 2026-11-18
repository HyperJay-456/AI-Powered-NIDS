"""Threat Intelligence & Attack Context Subsystem (CP-10).

Connects AI-Powered NIDS predictions to official MITRE ATT&CK Enterprise
taxonomies, conservative candidate mappings, evidence evaluation, and deterministic
security insights without cloud dependencies or unverified threat attribution.
"""

from src.threat_intelligence.attack_catalog import AttackCatalog
from src.threat_intelligence.attack_models import (
    AttackBundleMetadata,
    AttackFamily,
    DatasetLabelTaxonomyItem,
    EnrichedAttackContext,
    LabelTechniqueMapping,
    MappingReviewEvent,
    MappingStatus,
    MappingStrength,
    Mitigation,
    MITRE_ATTRIBUTION_NOTICE,
    MITRE_DISCLAIMER,
    Tactic,
    Technique,
)
from src.threat_intelligence.attack_stix_loader import AttackSTIXLoader
from src.threat_intelligence.context_service import ThreatContextService
from src.threat_intelligence.insight_engine import ThreatInsightEngine
from src.threat_intelligence.mapping_registry import MappingRegistry
from src.threat_intelligence.reference_registry import ReferenceRegistry
from src.threat_intelligence.taxonomy_service import TaxonomyService

__all__ = [
    "AttackBundleMetadata",
    "AttackCatalog",
    "AttackFamily",
    "AttackSTIXLoader",
    "DatasetLabelTaxonomyItem",
    "EnrichedAttackContext",
    "LabelTechniqueMapping",
    "MappingRegistry",
    "MappingReviewEvent",
    "MappingStatus",
    "MappingStrength",
    "Mitigation",
    "MITRE_ATTRIBUTION_NOTICE",
    "MITRE_DISCLAIMER",
    "ReferenceRegistry",
    "Tactic",
    "TaxonomyService",
    "Technique",
    "ThreatContextService",
    "ThreatInsightEngine",
]
