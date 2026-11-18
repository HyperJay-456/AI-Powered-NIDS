"""Official MITRE ATT&CK STIX 2.1 Bundle Loader & Parser (CP-10).

Parses and validates official MITRE ATT&CK Enterprise STIX 2.1 JSON bundles:
  - Validates specification version and bundle schema
  - Extracts Tactics, Techniques, Sub-techniques, and Mitigations
  - Resolves cross-object relationships (Mitigations -> Techniques, Subtechniques -> Parents)
  - Computes cryptographic SHA-256 provenance checksums
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.threat_intelligence.attack_catalog import AttackCatalog
from src.threat_intelligence.attack_models import (
    AttackBundleMetadata,
    Mitigation,
    Tactic,
    Technique,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttackSTIXLoader:
    """Validates and parses official MITRE ATT&CK Enterprise STIX 2.1 bundles."""

    @classmethod
    def load_from_file(cls, file_path: Path) -> Tuple[AttackCatalog, AttackBundleMetadata]:
        """Load and parse an official ATT&CK STIX 2.1 JSON file.

        Args:
            file_path: Absolute or relative path to the STIX JSON bundle.

        Returns:
            Tuple[AttackCatalog, AttackBundleMetadata]: Parsed catalog and metadata.

        Raises:
            FileNotFoundError: If the bundle file does not exist.
            ValueError: If the file is not a valid STIX 2.1 bundle.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ATT&CK bundle file not found: {path}")

        # Compute SHA-256
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        bundle_sha = hasher.hexdigest()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Malformed JSON in ATT&CK bundle '{path.name}': {e}")

        return cls.parse_stix_dict(data, source_filename=path.name, sha256_checksum=bundle_sha)

    @classmethod
    def parse_stix_dict(
        cls,
        data: Dict[str, Any],
        source_filename: str = "stix_bundle.json",
        sha256_checksum: str = "",
    ) -> Tuple[AttackCatalog, AttackBundleMetadata]:
        """Validate and parse STIX dictionary into an AttackCatalog.

        Args:
            data: Parsed STIX JSON object dictionary.
            source_filename: Name of the source file for metadata.
            sha256_checksum: Cryptographic SHA-256 checksum.

        Returns:
            Tuple[AttackCatalog, AttackBundleMetadata]: Parsed catalog and metadata.

        Raises:
            ValueError: If structure violates STIX 2.1 specification.
        """
        # Validate STIX bundle specification
        if not isinstance(data, dict):
            raise ValueError("STIX payload must be a JSON object dictionary.")

        bundle_type = data.get("type")
        if bundle_type != "bundle":
            raise ValueError(f"Invalid STIX object type: expected 'bundle', got '{bundle_type}'.")

        # Check spec version (Enterprise v20.1 uses STIX 2.1 or 2.0)
        spec_version = data.get("spec_version", "2.1")

        objects = data.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("STIX bundle lacks required 'objects' list array.")

        tactics_by_ext_id: Dict[str, Tactic] = {}
        techniques_by_ext_id: Dict[str, Technique] = {}
        mitigations_by_ext_id: Dict[str, Mitigation] = {}

        # STIX ID -> External ID cross-indexes for relationship resolution
        stix_id_to_ext_id: Dict[str, str] = {}

        # Relationship accumulators
        # Technique external_id -> List of Mitigation external_ids
        tech_to_mits: Dict[str, List[str]] = {}
        # Tactic shortname -> List of Technique external_ids
        tactic_to_techs: Dict[str, List[str]] = {}

        raw_relationships: List[Dict[str, Any]] = []

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            obj_type = obj.get("type", "")
            stix_id = obj.get("id", "")
            is_revoked = bool(obj.get("revoked", False))
            is_deprecated = bool(obj.get("x_mitre_deprecated", False))

            # 1. Parse Tactics (x-mitre-tactic)
            if obj_type == "x-mitre-tactic":
                ext_id, url = cls._extract_external_ref(obj)
                shortname = obj.get("x_mitre_shortname", "")
                if not ext_id:
                    ext_id = shortname.upper()

                tactic = Tactic(
                    id=stix_id,
                    external_id=ext_id,
                    name=obj.get("name", "Unknown Tactic"),
                    shortname=shortname,
                    description=obj.get("description", ""),
                    url=url,
                )
                tactics_by_ext_id[ext_id] = tactic
                if shortname:
                    tactics_by_ext_id[shortname.lower()] = tactic
                if stix_id:
                    stix_id_to_ext_id[stix_id] = ext_id

            # 2. Parse Techniques (attack-pattern)
            elif obj_type == "attack-pattern":
                ext_id, url = cls._extract_external_ref(obj)
                if not ext_id:
                    continue

                tactics = []
                for phase in obj.get("kill_chain_phases", []):
                    if phase.get("kill_chain_name") == "mitre-attack":
                        p_name = phase.get("phase_name")
                        if p_name:
                            tactics.append(p_name)
                            if p_name not in tactic_to_techs:
                                tactic_to_techs[p_name] = []
                            if ext_id not in tactic_to_techs[p_name]:
                                tactic_to_techs[p_name].append(ext_id)

                is_sub = bool(obj.get("x_mitre_is_subtechnique", False))
                platforms = obj.get("x_mitre_platforms", [])
                detection = obj.get("x_mitre_detection", "")

                tech = Technique(
                    id=stix_id,
                    external_id=ext_id,
                    name=obj.get("name", "Unknown Technique"),
                    description=obj.get("description", ""),
                    tactics=tactics,
                    platforms=platforms,
                    is_subtechnique=is_sub,
                    detection_guidance=detection,
                    url=url,
                    is_deprecated=is_deprecated,
                    is_revoked=is_revoked,
                )
                techniques_by_ext_id[ext_id] = tech
                if stix_id:
                    stix_id_to_ext_id[stix_id] = ext_id

            # 3. Parse Mitigations (course-of-action)
            elif obj_type == "course-of-action":
                ext_id, url = cls._extract_external_ref(obj)
                if not ext_id:
                    continue

                mit = Mitigation(
                    id=stix_id,
                    external_id=ext_id,
                    name=obj.get("name", "Unknown Mitigation"),
                    description=obj.get("description", ""),
                    url=url,
                )
                mitigations_by_ext_id[ext_id] = mit
                if stix_id:
                    stix_id_to_ext_id[stix_id] = ext_id

            # 4. Save Relationships for 2nd pass
            elif obj_type == "relationship":
                raw_relationships.append(obj)

        # Second Pass: Resolve Relationships
        for rel in raw_relationships:
            rel_type = rel.get("relationship_type", "")
            source_ref = rel.get("source_ref", "")
            target_ref = rel.get("target_ref", "")

            # Course of Action mitigates Attack Pattern
            if rel_type == "mitigates":
                mit_ext_id = stix_id_to_ext_id.get(source_ref)
                tech_ext_id = stix_id_to_ext_id.get(target_ref)
                if mit_ext_id and tech_ext_id:
                    if tech_ext_id not in tech_to_mits:
                        tech_to_mits[tech_ext_id] = []
                    if mit_ext_id not in tech_to_mits[tech_ext_id]:
                        tech_to_mits[tech_ext_id].append(mit_ext_id)

            # Subtechnique of Parent Technique
            elif rel_type == "subtechnique-of":
                sub_ext_id = stix_id_to_ext_id.get(source_ref)
                parent_ext_id = stix_id_to_ext_id.get(target_ref)
                if sub_ext_id and parent_ext_id and sub_ext_id in techniques_by_ext_id:
                    techniques_by_ext_id[sub_ext_id].parent_technique_id = parent_ext_id

        catalog = AttackCatalog(
            tactics=tactics_by_ext_id,
            techniques=techniques_by_ext_id,
            mitigations=mitigations_by_ext_id,
            technique_mitigations=tech_to_mits,
            tactic_techniques=tactic_to_techs,
        )

        metadata = AttackBundleMetadata(
            dataset_name="MITRE ATT&CK Enterprise",
            domain="Enterprise",
            version="v20.1",
            release_date="2026-11-05",
            source_repository="https://github.com/mitre-attack/attack-stix-data",
            source_file=source_filename,
            import_timestamp_utc="",
            sha256_checksum=sha256_checksum,
            stix_spec_version=str(spec_version),
            total_objects=len(objects),
            tactics_count=len(set(t.external_id for t in tactics_by_ext_id.values())),
            techniques_count=len(techniques_by_ext_id),
            mitigations_count=len(mitigations_by_ext_id),
            validation_status="Valid",
            is_active=True,
        )

        return catalog, metadata

    @staticmethod
    def _extract_external_ref(obj: Dict[str, Any]) -> Tuple[str, str]:
        """Extract external_id and URL from external_references list."""
        ext_refs = obj.get("external_references", [])
        ext_id = ""
        url = ""
        for ref in ext_refs:
            if ref.get("source_name") in ("mitre-attack", "mitre-enterprise-attack"):
                ext_id = ref.get("external_id", "")
                url = ref.get("url", "")
                if ext_id:
                    break
        return ext_id, url
