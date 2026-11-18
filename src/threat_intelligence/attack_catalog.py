"""MITRE ATT&CK Catalog & Query Engine (CP-10).

Provides fast in-memory indexing, search, filtering, and cross-object relationship
resolution for tactics, techniques, and mitigations.
"""

from typing import Any, Dict, List, Optional
from src.threat_intelligence.attack_models import Mitigation, Tactic, Technique
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttackCatalog:
    """In-memory indexed catalog of official MITRE ATT&CK Enterprise objects."""

    def __init__(
        self,
        tactics: Optional[Dict[str, Tactic]] = None,
        techniques: Optional[Dict[str, Technique]] = None,
        mitigations: Optional[Dict[str, Mitigation]] = None,
        technique_mitigations: Optional[Dict[str, List[str]]] = None,
        tactic_techniques: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize catalog with indexed dictionaries."""
        self.tactics: Dict[str, Tactic] = tactics or {}
        self.techniques: Dict[str, Technique] = techniques or {}
        self.mitigations: Dict[str, Mitigation] = mitigations or {}
        # Mapping from technique_id (external_id e.g. T1046) -> list of mitigation external_ids (e.g. M1031)
        self.technique_mitigations: Dict[str, List[str]] = technique_mitigations or {}
        # Mapping from tactic shortname/external_id -> list of technique external_ids
        self.tactic_techniques: Dict[str, List[str]] = tactic_techniques or {}

    def get_technique(self, technique_id: str) -> Optional[Technique]:
        """Lookup a technique by its external ATT&CK ID (e.g. 'T1046')."""
        clean_id = technique_id.strip().upper()
        return self.techniques.get(clean_id)

    def get_tactic(self, tactic_identifier: str) -> Optional[Tactic]:
        """Lookup a tactic by external ID (e.g. 'TA0007') or shortname ('discovery')."""
        clean_id = tactic_identifier.strip().lower()
        # Check by external_id (uppercased)
        if clean_id.upper() in self.tactics:
            return self.tactics[clean_id.upper()]
        # Check by shortname
        for tac in self.tactics.values():
            if tac.shortname.lower() == clean_id:
                return tac
        return None

    def get_mitigation(self, mitigation_id: str) -> Optional[Mitigation]:
        """Lookup a mitigation by external ID (e.g. 'M1031')."""
        clean_id = mitigation_id.strip().upper()
        return self.mitigations.get(clean_id)

    def get_mitigations_for_technique(self, technique_id: str) -> List[Mitigation]:
        """Retrieve all associated mitigations for a given technique ID."""
        clean_id = technique_id.strip().upper()
        mit_ids = self.technique_mitigations.get(clean_id, [])
        res: List[Mitigation] = []
        for mid in mit_ids:
            mit = self.get_mitigation(mid)
            if mit:
                res.append(mit)
        return res

    def search_techniques(
        self,
        query: Optional[str] = None,
        tactic: Optional[str] = None,
        include_deprecated: bool = False,
        include_revoked: bool = False,
    ) -> List[Technique]:
        """Search techniques by ID, name, description, and tactic filter."""
        results: List[Technique] = []
        q = query.strip().lower() if query else ""
        tac_filter = tactic.strip().lower() if tactic else ""

        for tech in self.techniques.values():
            if tech.is_deprecated and not include_deprecated:
                continue
            if tech.is_revoked and not include_revoked:
                continue

            if tac_filter:
                tech_tacs = [t.lower() for t in tech.tactics]
                if tac_filter not in tech_tacs:
                    continue

            if q:
                match_id = q in tech.external_id.lower()
                match_name = q in tech.name.lower()
                match_desc = q in tech.description.lower()
                if not (match_id or match_name or match_desc):
                    continue

            results.append(tech)

        return sorted(results, key=lambda t: t.external_id)

    def search_tactics(self, query: Optional[str] = None) -> List[Tactic]:
        """Search tactics by ID or name."""
        results: List[Tactic] = []
        q = query.strip().lower() if query else ""

        for tac in self.tactics.values():
            if q:
                match_id = q in tac.external_id.lower()
                match_name = q in tac.name.lower()
                match_desc = q in tac.description.lower()
                if not (match_id or match_name or match_desc):
                    continue
            results.append(tac)

        return sorted(results, key=lambda t: t.external_id)

    def search_mitigations(self, query: Optional[str] = None) -> List[Mitigation]:
        """Search mitigations by ID or name."""
        results: List[Mitigation] = []
        q = query.strip().lower() if query else ""

        for mit in self.mitigations.values():
            if q:
                match_id = q in mit.external_id.lower()
                match_name = q in mit.name.lower()
                match_desc = q in mit.description.lower()
                if not (match_id or match_name or match_desc):
                    continue
            results.append(mit)

        return sorted(results, key=lambda m: m.external_id)
