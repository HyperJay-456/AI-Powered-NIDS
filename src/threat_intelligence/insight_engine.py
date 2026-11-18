"""Deterministic Security Insights Engine (CP-10).

Calculates verifiable cybersecurity insights from actual detection runs,
dataset taxonomy, and MITRE ATT&CK mapping coverage:
  - Attack family distributions across classified flows
  - ATT&CK technique mapping coverage and approval progress
  - Model disagreement cross-tabulation across attack classes
  - Evidence availability metrics without fabricated risk scores
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.threat_intelligence.attack_models import AttackFamily, MappingStatus
from src.threat_intelligence.mapping_registry import MappingRegistry
from src.threat_intelligence.taxonomy_service import TaxonomyService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreatInsightEngine:
    """Computes deterministic cybersecurity insights from actual detection artifacts."""

    def __init__(self, mapping_registry: Optional[MappingRegistry] = None):
        """Initialize insight engine with mapping registry."""
        self.mapping_registry = mapping_registry or MappingRegistry()

    def compute_mapping_coverage(self, active_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute mapping coverage metrics across active or benchmark dataset classes.

        Args:
            active_classes: Optional list of unique attack classes present in a run.
                           If None, computes across all 15 benchmark CIC-IDS-2017 classes.

        Returns:
            Dict[str, Any]: Mapping coverage metrics.
        """
        all_mappings = {m.normalized_dataset_label: m for m in self.mapping_registry.list_mappings()}

        classes_to_check = active_classes or [
            "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
            "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
            "Bot", "Web Attack - Brute Force", "Web Attack - XSS",
            "Infiltration", "Web Attack - Sql Injection", "Heartbleed",
        ]

        total_classes = len(classes_to_check)
        if total_classes == 0:
            return {
                "total_classes": 0,
                "mapped_classes": 0,
                "approved_classes": 0,
                "reviewed_classes": 0,
                "unmapped_classes": 0,
                "coverage_pct": 0.0,
                "approval_pct": 0.0,
                "unmapped_list": [],
            }

        mapped_count = 0
        approved_count = 0
        reviewed_count = 0
        unmapped_list = []

        for cls_name in classes_to_check:
            norm_key = TaxonomyService.normalize_label(cls_name)
            mapping = all_mappings.get(norm_key)

            if mapping and mapping.technique_id and mapping.mapping_status != MappingStatus.UNMAPPED:
                mapped_count += 1
                if mapping.mapping_status == MappingStatus.APPROVED:
                    approved_count += 1
                elif mapping.mapping_status in (MappingStatus.REVIEWED, MappingStatus.APPROVED):
                    reviewed_count += 1
            else:
                unmapped_list.append(cls_name)

        coverage_pct = round((mapped_count / max(1, total_classes)) * 100.0, 2)
        approval_pct = round((approved_count / max(1, total_classes)) * 100.0, 2)

        return {
            "total_classes": total_classes,
            "mapped_classes": mapped_count,
            "approved_classes": approved_count,
            "reviewed_classes": reviewed_count,
            "unmapped_classes": len(unmapped_list),
            "coverage_pct": coverage_pct,
            "approval_pct": approval_pct,
            "unmapped_list": unmapped_list,
        }

    def compute_family_distribution(
        self,
        predictions_df: pd.DataFrame,
        class_column: str = "predicted_class",
    ) -> List[Dict[str, Any]]:
        """Compute distribution of predicted flows grouped by broad attack family.

        Args:
            predictions_df: DataFrame of detection predictions.
            class_column: Column name containing class labels.

        Returns:
            List[Dict[str, Any]]: Grouped counts and percentages by AttackFamily.
        """
        if predictions_df.empty or class_column not in predictions_df.columns:
            return []

        total_rows = len(predictions_df)
        counts_by_family: Dict[str, int] = {}

        for cls_label in predictions_df[class_column]:
            tax_item = TaxonomyService.get_taxonomy_item(cls_label)
            family = tax_item.broad_family
            counts_by_family[family] = counts_by_family.get(family, 0) + 1

        results = []
        for family, count in sorted(counts_by_family.items(), key=lambda x: x[1], reverse=True):
            pct = round((count / max(1, total_rows)) * 100.0, 2)
            results.append({
                "attack_family": family,
                "record_count": count,
                "percentage": pct,
            })

        return results

    def compute_disagreement_by_class(
        self,
        predictions_df: pd.DataFrame,
        class_column: str = "predicted_class",
        agreement_column: str = "agreement_status",
    ) -> List[Dict[str, Any]]:
        """Cross-tabulate model disagreement against predicted attack classes.

        Args:
            predictions_df: DataFrame with prediction and agreement columns.
            class_column: Column name for class labels.
            agreement_column: Column name for consensus status.

        Returns:
            List[Dict[str, Any]]: Breakdown per attack class.
        """
        if (
            predictions_df.empty
            or class_column not in predictions_df.columns
            or agreement_column not in predictions_df.columns
        ):
            return []

        grouped = predictions_df.groupby([class_column, agreement_column]).size().unstack(fill_value=0)

        results = []
        for cls_name, row in grouped.iterrows():
            agreed = int(row.get("Agreement", 0))
            disagreed = int(row.get("Disagreement", 0))
            incomplete = int(row.get("Incomplete", 0))
            total_cls = agreed + disagreed + incomplete
            disagree_pct = round((disagreed / max(1, total_cls)) * 100.0, 2)

            results.append({
                "predicted_class": str(cls_name),
                "total_flows": total_cls,
                "agreement_count": agreed,
                "disagreement_count": disagreed,
                "incomplete_count": incomplete,
                "disagreement_pct": disagree_pct,
            })

        return sorted(results, key=lambda x: x["disagreement_count"], reverse=True)
