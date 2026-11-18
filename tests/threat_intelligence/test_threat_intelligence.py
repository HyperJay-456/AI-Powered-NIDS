"""Exhaustive Verification & Regression Suite for Threat Intelligence (CP-10).

Tests all required functional verification categories:
  Category A: STIX Reference Import & Schema Validation
  Category B: ATT&CK Catalogue Indexing & Querying
  Category C: Dataset Attack Label Taxonomy & Normalization
  Category D: Label-to-Technique Candidate Mappings
  Category E: Evidence-Aware Context & Non-Attribution Disclaimers
  Category F: Mitigation & Detection Reference Retrieval
  Category G: Human Mapping Review Workflow & Audit Trail
  Category H: Deterministic Security Insights
  Category I: UI Integration & Attribution Verification
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.threat_intelligence.attack_catalog import AttackCatalog
from src.threat_intelligence.attack_models import (
    AttackFamily,
    MappingStatus,
    MappingStrength,
    MITRE_ATTRIBUTION_NOTICE,
    MITRE_DISCLAIMER,
)
from src.threat_intelligence.attack_stix_loader import AttackSTIXLoader
from src.threat_intelligence.context_service import ThreatContextService
from src.threat_intelligence.insight_engine import ThreatInsightEngine
from src.threat_intelligence.mapping_registry import MappingRegistry
from src.threat_intelligence.reference_registry import ReferenceRegistry
from src.threat_intelligence.taxonomy_service import TaxonomyService


def _create_mock_stix_bundle(
    file_path: Path,
    include_invalid_json: bool = False,
    include_bad_spec: bool = False,
    include_revoked: bool = False,
    include_deprecated: bool = False,
) -> Path:
    """Helper creating a test STIX 2.1 JSON file."""
    if include_invalid_json:
        file_path.write_text("CORRUPTED_NOT_JSON", encoding="utf-8")
        return file_path

    bundle_dict = {
        "type": "other" if include_bad_spec else "bundle",
        "id": "bundle--test-bundle",
        "spec_version": "1.0" if include_bad_spec else "2.1",
        "objects": [
            {
                "type": "x-mitre-tactic",
                "id": "x-mitre-tactic--ta0007-disc",
                "name": "Discovery",
                "description": "Discovery tactic description",
                "x_mitre_shortname": "discovery",
                "external_references": [
                    {
                        "source_name": "mitre-attack",
                        "external_id": "TA0007",
                        "url": "https://attack.mitre.org/tactics/TA0007",
                    }
                ],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t1046-scan",
                "name": "Network Service Discovery",
                "description": "Probe listening services on remote hosts.",
                "x_mitre_platforms": ["Linux", "Windows"],
                "x_mitre_detection": "Monitor closed port probe spikes.",
                "x_mitre_is_subtechnique": False,
                "revoked": include_revoked,
                "x_mitre_deprecated": include_deprecated,
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "discovery"}
                ],
                "external_references": [
                    {
                        "source_name": "mitre-attack",
                        "external_id": "T1046",
                        "url": "https://attack.mitre.org/techniques/T1046",
                    }
                ],
            },
            {
                "type": "course-of-action",
                "id": "course-of-action--m1031-nids",
                "name": "Network Intrusion Prevention",
                "description": "Block probe traffic at the perimeter.",
                "external_references": [
                    {
                        "source_name": "mitre-attack",
                        "external_id": "M1031",
                        "url": "https://attack.mitre.org/mitigations/M1031",
                    }
                ],
            },
            {
                "type": "relationship",
                "id": "relationship--rel-1",
                "relationship_type": "mitigates",
                "source_ref": "course-of-action--m1031-nids",
                "target_ref": "attack-pattern--t1046-scan",
            },
        ],
    }

    file_path.write_text(json.dumps(bundle_dict, indent=2), encoding="utf-8")
    return file_path


class TestThreatIntelligenceSuite(unittest.TestCase):
    """Exhaustive test suite for CP-10 Threat Intelligence & Attack Context."""

    def setUp(self) -> None:
        """Create isolated temporary workspaces for each test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.tmp_dir.name)
        self.mitre_dir = self.root_path / "data" / "knowledge" / "mitre_attack"
        self.mappings_dir = self.root_path / "data" / "knowledge" / "mappings"

        self.mitre_dir.mkdir(parents=True, exist_ok=True)
        self.mappings_dir.mkdir(parents=True, exist_ok=True)

        self.ref_registry = ReferenceRegistry(mitre_dir=self.mitre_dir)
        self.mapping_registry = MappingRegistry(mappings_dir=self.mappings_dir)
        self.context_service = ThreatContextService(
            reference_registry=self.ref_registry,
            mapping_registry=self.mapping_registry,
        )
        self.insight_engine = ThreatInsightEngine(mapping_registry=self.mapping_registry)

    def tearDown(self) -> None:
        """Clean up temporary workspaces."""
        self.tmp_dir.cleanup()

    # --------------------------------------------------------------------------
    # CATEGORY A: STIX Reference Import & Schema Validation
    # --------------------------------------------------------------------------

    def test_valid_stix_bundle_parsing(self) -> None:
        """Verify parsing of valid official-format STIX 2.1 bundle."""
        bundle_file = self.root_path / "valid_bundle.json"
        _create_mock_stix_bundle(bundle_file)

        catalog, meta = AttackSTIXLoader.load_from_file(bundle_file)
        self.assertEqual(meta.validation_status, "Valid")
        self.assertEqual(meta.stix_spec_version, "2.1")
        self.assertIn("T1046", catalog.techniques)
        self.assertIn("TA0007", catalog.tactics)
        self.assertIn("M1031", catalog.mitigations)

    def test_invalid_json_stix_bundle_rejection(self) -> None:
        """Verify malformed JSON raises ValueError."""
        corrupt_file = self.root_path / "corrupt.json"
        _create_mock_stix_bundle(corrupt_file, include_invalid_json=True)

        with self.assertRaises(ValueError):
            AttackSTIXLoader.load_from_file(corrupt_file)

    def test_invalid_bundle_type_rejection(self) -> None:
        """Verify non-bundle STIX object type raises ValueError."""
        bad_spec_file = self.root_path / "bad_spec.json"
        _create_mock_stix_bundle(bad_spec_file, include_bad_spec=True)

        with self.assertRaises(ValueError):
            AttackSTIXLoader.load_from_file(bad_spec_file)

    def test_import_stix_bundle_atomic_staging(self) -> None:
        """Verify ReferenceRegistry imports, stages, and verifies bundle atomically."""
        src_bundle = self.root_path / "src_bundle.json"
        _create_mock_stix_bundle(src_bundle)

        ok, msg, meta = self.ref_registry.import_stix_bundle(src_bundle, version_label="v20.1")
        self.assertTrue(ok)
        self.assertIsNotNone(meta)
        self.assertTrue(self.ref_registry.manifest_path.exists())

        active_cat = self.ref_registry.get_active_catalog()
        self.assertIsNotNone(active_cat)
        self.assertIn("T1046", active_cat.techniques)

    def test_missing_bundle_file_handling(self) -> None:
        """Verify import of non-existent file returns failure cleanly."""
        missing = self.root_path / "does_not_exist.json"
        ok, msg, meta = self.ref_registry.import_stix_bundle(missing)
        self.assertFalse(ok)
        self.assertIn("not found", msg)
        self.assertIsNone(meta)

    # --------------------------------------------------------------------------
    # CATEGORY B: ATT&CK Catalogue Indexing & Querying
    # --------------------------------------------------------------------------

    def test_tactic_and_technique_retrieval(self) -> None:
        """Verify retrieval by external ID and shortname."""
        bundle_file = self.root_path / "bundle_b.json"
        _create_mock_stix_bundle(bundle_file)
        catalog, _ = AttackSTIXLoader.load_from_file(bundle_file)

        # Lookup technique
        tech = catalog.get_technique("T1046")
        self.assertIsNotNone(tech)
        self.assertEqual(tech.name, "Network Service Discovery")

        # Lookup tactic by ID and shortname
        tac_by_id = catalog.get_tactic("TA0007")
        tac_by_name = catalog.get_tactic("discovery")
        self.assertIsNotNone(tac_by_id)
        self.assertEqual(tac_by_id.name, "Discovery")
        self.assertEqual(tac_by_id, tac_by_name)

    def test_cross_object_mitigation_relationship(self) -> None:
        """Verify technique-to-mitigation relationship resolution."""
        bundle_file = self.root_path / "bundle_rel.json"
        _create_mock_stix_bundle(bundle_file)
        catalog, _ = AttackSTIXLoader.load_from_file(bundle_file)

        mits = catalog.get_mitigations_for_technique("T1046")
        self.assertEqual(len(mits), 1)
        self.assertEqual(mits[0].external_id, "M1031")
        self.assertEqual(mits[0].name, "Network Intrusion Prevention")

    def test_search_techniques_and_filtering(self) -> None:
        """Verify technique search by keyword and tactic."""
        bundle_file = self.root_path / "bundle_search.json"
        _create_mock_stix_bundle(bundle_file)
        catalog, _ = AttackSTIXLoader.load_from_file(bundle_file)

        results = catalog.search_techniques(query="Service Discovery")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].external_id, "T1046")

        tactic_results = catalog.search_techniques(tactic="discovery")
        self.assertEqual(len(tactic_results), 1)

        empty_results = catalog.search_techniques(query="NonexistentTerm")
        self.assertEqual(len(empty_results), 0)

    # --------------------------------------------------------------------------
    # CATEGORY C: Dataset Attack Label Taxonomy & Normalization
    # --------------------------------------------------------------------------

    def test_taxonomy_normalization_and_whitespace_safety(self) -> None:
        """Verify case-insensitivity and whitespace normalization."""
        item1 = TaxonomyService.get_taxonomy_item("PortScan")
        item2 = TaxonomyService.get_taxonomy_item("  portscan  ")
        item3 = TaxonomyService.get_taxonomy_item("PORTSCAN")

        self.assertEqual(item1.broad_family, AttackFamily.PORT_SCANNING)
        self.assertEqual(item1.broad_family, item2.broad_family)
        self.assertEqual(item1.broad_family, item3.broad_family)

    def test_known_classes_family_mappings(self) -> None:
        """Verify correct threat family mappings for CIC-IDS-2017 classes."""
        self.assertEqual(TaxonomyService.get_taxonomy_item("BENIGN").broad_family, AttackFamily.BENIGN)
        self.assertEqual(TaxonomyService.get_taxonomy_item("DoS Hulk").broad_family, AttackFamily.DENIAL_OF_SERVICE)
        self.assertEqual(TaxonomyService.get_taxonomy_item("DDoS").broad_family, AttackFamily.DDOS)
        self.assertEqual(TaxonomyService.get_taxonomy_item("FTP-Patator").broad_family, AttackFamily.BRUTE_FORCE)
        self.assertEqual(TaxonomyService.get_taxonomy_item("SSH-Patator").broad_family, AttackFamily.BRUTE_FORCE)
        self.assertEqual(TaxonomyService.get_taxonomy_item("Web Attack - Sql Injection").broad_family, AttackFamily.WEB_ATTACK)
        self.assertEqual(TaxonomyService.get_taxonomy_item("Heartbleed").broad_family, AttackFamily.EXPLOITATION)

    def test_unknown_attack_class_handling(self) -> None:
        """Verify unknown attack classes are assigned to Other / Unclassified without failure."""
        item = TaxonomyService.get_taxonomy_item("UnseenCustomAttackVariant")
        self.assertEqual(item.broad_family, AttackFamily.OTHER)
        self.assertEqual(item.mapping_status, "Unmapped")

    # --------------------------------------------------------------------------
    # CATEGORY D: Label-to-Technique Candidate Mappings
    # --------------------------------------------------------------------------

    def test_default_baseline_mappings_initialization(self) -> None:
        """Verify initialization of default conservative candidate mappings."""
        mappings = self.mapping_registry.list_mappings()
        self.assertEqual(len(mappings), 15)

        portscan_map = self.mapping_registry.get_mapping_for_label("PortScan")
        self.assertIsNotNone(portscan_map)
        self.assertEqual(portscan_map.technique_id, "T1046")
        self.assertEqual(portscan_map.mapping_status, MappingStatus.CANDIDATE)
        self.assertEqual(portscan_map.mapping_strength, MappingStrength.DIRECT)

    def test_unmapped_status_for_ambiguous_classes(self) -> None:
        """Verify broad or normal traffic classes remain Unmapped by default."""
        benign_map = self.mapping_registry.get_mapping_for_label("BENIGN")
        self.assertIsNotNone(benign_map)
        self.assertEqual(benign_map.mapping_status, MappingStatus.UNMAPPED)

        bot_map = self.mapping_registry.get_mapping_for_label("Bot")
        self.assertIsNotNone(bot_map)
        self.assertEqual(bot_map.mapping_status, MappingStatus.UNMAPPED)

    # --------------------------------------------------------------------------
    # CATEGORY E: Evidence-Aware Context & Non-Attribution Disclaimers
    # --------------------------------------------------------------------------

    def test_evidence_aware_context_enrichment(self) -> None:
        """Verify context generation, available evidence, and missing evidence."""
        # Install mock STIX bundle first
        src_bundle = self.root_path / "bundle_ctx.json"
        _create_mock_stix_bundle(src_bundle)
        self.ref_registry.import_stix_bundle(src_bundle)

        ctx = self.context_service.enrich_prediction_record(
            predicted_class="PortScan",
            model_name="svm",
            model_score=2.45,
            score_type="decision_margin",
            source_row_index=10,
            flow_attributes={"Destination Port": 80, "Flow Packets/s": 1500},
        )

        self.assertEqual(ctx.predicted_class, "PortScan")
        self.assertEqual(ctx.candidate_mapping.technique_id, "T1046")
        self.assertIn("Destination Port", ctx.available_evidence_fields)
        self.assertTrue(len(ctx.missing_evidence_fields) > 0)
        self.assertIn("NOT verified proof", ctx.context_disclaimer)

    def test_ground_truth_separation_in_context(self) -> None:
        """Verify ground truth is preserved as academic benchmark without conflation."""
        ctx = self.context_service.enrich_prediction_record(
            predicted_class="DoS Hulk",
            ground_truth_label="BENIGN",
            disagreement_state="Disagreement",
        )
        self.assertEqual(ctx.predicted_class, "DoS Hulk")
        self.assertEqual(ctx.ground_truth_label, "BENIGN")
        self.assertEqual(ctx.disagreement_state, "Disagreement")

    # --------------------------------------------------------------------------
    # CATEGORY F: Mitigation & Detection Reference Retrieval
    # --------------------------------------------------------------------------

    def test_mitigation_retrieval_for_mapped_technique(self) -> None:
        """Verify retrieval of associated mitigations for mapped technique."""
        src_bundle = self.root_path / "bundle_mit.json"
        _create_mock_stix_bundle(src_bundle)
        self.ref_registry.import_stix_bundle(src_bundle)

        ctx = self.context_service.enrich_prediction_record(predicted_class="PortScan")
        self.assertTrue(len(ctx.relevant_mitigations) > 0)
        self.assertEqual(ctx.relevant_mitigations[0].external_id, "M1031")

    # --------------------------------------------------------------------------
    # CATEGORY G: Human Mapping Review Workflow & Audit Trail
    # --------------------------------------------------------------------------

    def test_human_mapping_approval_and_rejection(self) -> None:
        """Verify human approval and rejection actions update status and record history."""
        target_map = self.mapping_registry.get_mapping_for_label("PortScan")
        self.assertIsNotNone(target_map)

        # 1. Approve mapping
        approved = self.mapping_registry.approve_mapping(
            mapping_id=target_map.mapping_id,
            reviewer_label="Lead Analyst",
            analyst_notes="Confirmed port scanning telemetry alignment",
        )
        self.assertEqual(approved.mapping_status, MappingStatus.APPROVED)
        self.assertEqual(approved.reviewer_label, "Lead Analyst")

        # 2. Reject mapping
        rejected = self.mapping_registry.reject_mapping(
            mapping_id=target_map.mapping_id,
            reviewer_label="Review Board",
            analyst_notes="Rejected due to policy update",
        )
        self.assertEqual(rejected.mapping_status, MappingStatus.REJECTED)

        # Verify audit history
        hist = self.mapping_registry.get_history(target_map.mapping_id)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0].new_status, MappingStatus.APPROVED)
        self.assertEqual(hist[1].new_status, MappingStatus.REJECTED)

    def test_mapping_history_hash_chain_integrity(self) -> None:
        """Verify cryptographic hash chaining across sequential mapping reviews."""
        target_map = self.mapping_registry.get_mapping_for_label("FTP-Patator")
        self.mapping_registry.approve_mapping(target_map.mapping_id, "Reviewer 1", "Notes 1")
        self.mapping_registry.reject_mapping(target_map.mapping_id, "Reviewer 2", "Notes 2")

        is_valid, errors = self.mapping_registry.verify_audit_integrity()
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_tampered_mapping_history_detection(self) -> None:
        """Verify that modifying mapping_history.jsonl breaks the hash chain."""
        target_map = self.mapping_registry.get_mapping_for_label("SSH-Patator")
        self.mapping_registry.approve_mapping(target_map.mapping_id, "Reviewer 1", "Original")

        history_path = self.mappings_dir / "mapping_history.jsonl"
        lines = history_path.read_text(encoding="utf-8").strip().split("\n")
        tampered_entry = json.loads(lines[0])
        tampered_entry["analyst_notes"] = "TAMPERED NOTES"
        lines[0] = json.dumps(tampered_entry)
        history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        is_valid, errors = self.mapping_registry.verify_audit_integrity()
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    # --------------------------------------------------------------------------
    # CATEGORY H: Deterministic Security Insights
    # --------------------------------------------------------------------------

    def test_mapping_coverage_computation(self) -> None:
        """Verify deterministic calculation of ATT&CK mapping coverage."""
        cov = self.insight_engine.compute_mapping_coverage()
        self.assertEqual(cov["total_classes"], 15)
        self.assertTrue(cov["mapped_classes"] > 0)
        self.assertTrue(cov["coverage_pct"] > 0)
        self.assertIn("BENIGN", cov["unmapped_list"])

    def test_family_distribution_computation(self) -> None:
        """Verify calculation of attack family distributions."""
        dummy_df = pd.DataFrame({
            "predicted_class": ["PortScan", "PortScan", "DoS Hulk", "BENIGN"],
        })
        families = self.insight_engine.compute_family_distribution(dummy_df)
        self.assertEqual(len(families), 3)
        self.assertEqual(families[0]["attack_family"], AttackFamily.PORT_SCANNING)
        self.assertEqual(families[0]["record_count"], 2)

    def test_empty_dataset_insight_safety(self) -> None:
        """Verify empty DataFrames do not cause division by zero or crashes."""
        empty_df = pd.DataFrame()
        families = self.insight_engine.compute_family_distribution(empty_df)
        self.assertEqual(families, [])

        disagree = self.insight_engine.compute_disagreement_by_class(empty_df)
        self.assertEqual(disagree, [])

    # --------------------------------------------------------------------------
    # CATEGORY I: UI Integration & Attribution Verification
    # --------------------------------------------------------------------------

    def test_threat_intelligence_page_renders_cleanly(self) -> None:
        """Verify that dashboard.pages.threat_intelligence.render() executes without error."""
        from dashboard.pages import threat_intelligence
        try:
            threat_intelligence.render()
        except Exception as e:
            self.fail(f"threat_intelligence.render() raised unexpected exception: {e}")

    def test_mitre_attribution_constants(self) -> None:
        """Verify that required official MITRE attribution notice and disclaimer are present."""
        self.assertIn("The MITRE Corporation", MITRE_ATTRIBUTION_NOTICE)
        self.assertIn("2026", MITRE_ATTRIBUTION_NOTICE)
        self.assertIn("does not endorse", MITRE_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
