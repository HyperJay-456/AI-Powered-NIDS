"""Evidence-Aware Threat Context Enrichment Service (CP-10).

Enriches model prediction records with structured cybersecurity threat context:
  - Connects predicted attack classes to MITRE ATT&CK candidate mappings
  - Evaluates available network flow evidence vs missing contextual telemetry
  - Retrieves relevant defensive mitigations and detection strategies
  - Enforces explicit non-attribution safeguards and forensic uncertainty disclaimers
"""

from typing import Any, Dict, List, Optional

from src.threat_intelligence.attack_catalog import AttackCatalog
from src.threat_intelligence.attack_models import (
    EnrichedAttackContext,
    LabelTechniqueMapping,
    Mitigation,
    Technique,
)
from src.threat_intelligence.mapping_registry import MappingRegistry
from src.threat_intelligence.reference_registry import ReferenceRegistry
from src.threat_intelligence.taxonomy_service import TaxonomyService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreatContextService:
    """Enriches prediction records with MITRE ATT&CK references and evidence evaluation."""

    def __init__(
        self,
        reference_registry: Optional[ReferenceRegistry] = None,
        mapping_registry: Optional[MappingRegistry] = None,
    ):
        """Initialize service with reference and mapping registries."""
        self.ref_registry = reference_registry or ReferenceRegistry()
        self.map_registry = mapping_registry or MappingRegistry()

    def enrich_prediction_record(
        self,
        predicted_class: str,
        model_name: str = "svm",
        model_score: Optional[float] = None,
        score_type: str = "score",
        source_row_index: int = 0,
        flow_attributes: Optional[Dict[str, Any]] = None,
        ground_truth_label: Optional[str] = None,
        analyst_disposition: Optional[str] = None,
        disagreement_state: str = "Agreement",
    ) -> EnrichedAttackContext:
        """Generate forensic threat context for a single classified network flow.

        Args:
            predicted_class: Predicted attack class label (e.g. 'PortScan').
            model_name: Authorized algorithm name ('svm', 'mlp', 'decision_tree').
            model_score: Raw algorithm decision score.
            score_type: Algorithm score type.
            source_row_index: Row index in detection artifact.
            flow_attributes: Dict of extracted network flow attributes (ports, packets, etc.).
            ground_truth_label: Academic benchmark ground-truth label if available.
            analyst_disposition: CP-09 analyst review disposition if recorded.
            disagreement_state: Consensus state ('Agreement', 'Disagreement', 'Single Model').

        Returns:
            EnrichedAttackContext: Structured context with evidence evaluation and mitigations.
        """
        # 1. Taxonomy categorization
        taxonomy_item = TaxonomyService.get_taxonomy_item(predicted_class)

        # 2. Candidate ATT&CK Mapping
        mapping = self.map_registry.get_mapping_for_label(predicted_class)

        # 3. Retrieve ATT&CK details from active catalog
        catalog = self.ref_registry.get_active_catalog()
        tech_detail: Optional[Technique] = None
        mitigations: List[Mitigation] = []

        if mapping and mapping.technique_id and catalog:
            tech_detail = catalog.get_technique(mapping.technique_id)
            mitigations = catalog.get_mitigations_for_technique(mapping.technique_id)

        # 4. Evidence Evaluation
        avail_evidence: Dict[str, Any] = {}
        missing_evidence: List[str] = []

        raw_attrs = flow_attributes or {}
        if raw_attrs:
            # Check for relevant network indicators
            for key in ("Destination Port", "Source Port", "Protocol", "Flow Duration", "Flow Packets/s", "Flow Bytes/s", "Total Fwd Packets"):
                if key in raw_attrs:
                    avail_evidence[key] = raw_attrs[key]

        if mapping:
            # Inherit standard missing evidence fields documented in the candidate mapping
            missing_evidence.extend(mapping.missing_evidence)
        else:
            missing_evidence.append("No structured threat mapping available for this traffic class.")

        # If payload content was not inspected (always True for netflow CSVs)
        if "Full application payload inspection (DPI)" not in missing_evidence:
            missing_evidence.append("Application layer packet payload / DPI content (flow telemetry only)")

        return EnrichedAttackContext(
            predicted_class=predicted_class,
            model_name=model_name,
            model_score=model_score,
            score_type=score_type,
            source_row_index=source_row_index,
            taxonomy_item=taxonomy_item,
            candidate_mapping=mapping,
            technique_detail=tech_detail,
            relevant_mitigations=mitigations,
            available_evidence_fields=avail_evidence,
            missing_evidence_fields=missing_evidence,
            ground_truth_label=ground_truth_label,
            analyst_disposition=analyst_disposition,
            disagreement_state=disagreement_state,
        )
