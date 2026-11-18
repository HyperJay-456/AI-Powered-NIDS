"""Disagreement Analysis & Model Consensus Engine (CP-08).

Analyzes row-level prediction concordance across multiple models, identifies disagreement patterns,
and computes consensus metrics without building an unauthorized ensemble or majority-voting detector.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.detection.detection_manifest import DisagreementSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DisagreementAnalyzer:
    """Analyzes consensus and disagreement across multiple independent models."""

    @staticmethod
    def analyze_row_consensus(
        predictions_by_algo: Dict[str, List[str]],
    ) -> Tuple[List[str], List[List[str]], DisagreementSummary]:
        """Compute row-level agreement and consolidated disagreement summary.

        Args:
            predictions_by_algo: Dict mapping algorithm keys to lists of predicted labels.

        Returns:
            Tuple of:
              - agreement_statuses: List of 'Agreement', 'Disagreement', or 'Incomplete'
              - differing_models_per_row: List of lists indicating which algorithms differed
              - summary: DisagreementSummary dataclass
        """
        algos = list(predictions_by_algo.keys())
        if not algos:
            return [], [], DisagreementSummary()

        n_rows = len(predictions_by_algo[algos[0]])
        if len(algos) == 1:
            # Single model has trivial 100% agreement with itself
            statuses = ["Agreement"] * n_rows
            diffs = [[] for _ in range(n_rows)]
            summary = DisagreementSummary(
                total_compared_rows=n_rows,
                agreement_count=n_rows,
                disagreement_count=0,
                agreement_percentage=100.0,
                disagreement_percentage=0.0,
                incomplete_count=0,
                pairwise_agreements={f"{algos[0]}_vs_{algos[0]}": 100.0},
                top_disagreements=[],
            )
            return statuses, diffs, summary

        agreement_statuses: List[str] = []
        differing_models_per_row: List[List[str]] = []
        disagreement_patterns: Counter = Counter()

        agreement_count = 0
        disagreement_count = 0
        incomplete_count = 0

        # Pairwise agreement trackers: (algo_a, algo_b) -> matching count
        pairwise_matches: Dict[Tuple[str, str], int] = {}
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                pairwise_matches[(algos[i], algos[j])] = 0

        for r in range(n_rows):
            row_preds = {}
            for a in algos:
                val = predictions_by_algo[a][r] if r < len(predictions_by_algo[a]) else None
                row_preds[a] = val

            if any(v is None for v in row_preds.values()):
                agreement_statuses.append("Incomplete")
                differing_models_per_row.append([a for a, v in row_preds.items() if v is None])
                incomplete_count += 1
                continue

            unique_classes = set(row_preds.values())
            if len(unique_classes) == 1:
                agreement_statuses.append("Agreement")
                differing_models_per_row.append([])
                agreement_count += 1
            else:
                agreement_statuses.append("Disagreement")
                disagreement_count += 1

                # Find which models differed from the most common class in this row
                counts = Counter(row_preds.values())
                majority_class = counts.most_common(1)[0][0]
                differing = [a for a, v in row_preds.items() if v != majority_class]
                differing_models_per_row.append(differing)

                # Record pattern: e.g. "svm:BENIGN | mlp:DoS | decision_tree:BENIGN"
                pattern_str = " | ".join(f"{a}:{row_preds[a]}" for a in sorted(row_preds.keys()))
                disagreement_patterns[pattern_str] += 1

            # Update pairwise matches
            for (a1, a2) in pairwise_matches:
                if row_preds[a1] == row_preds[a2]:
                    pairwise_matches[(a1, a2)] += 1

        total_compared = n_rows
        agr_pct = (agreement_count / total_compared * 100.0) if total_compared > 0 else 0.0
        disagr_pct = (disagreement_count / total_compared * 100.0) if total_compared > 0 else 0.0

        # Compile pairwise percentages
        pairwise_agreements = {}
        for (a1, a2), matches in pairwise_matches.items():
            key = f"{a1}_vs_{a2}"
            pairwise_agreements[key] = round((matches / total_compared * 100.0) if total_compared > 0 else 0.0, 2)

        # Top 5 disagreement patterns
        top_disagreements = [
            {"pattern": pat, "count": cnt, "percentage": round(cnt / total_compared * 100.0, 2)}
            for pat, cnt in disagreement_patterns.most_common(5)
        ]

        summary = DisagreementSummary(
            total_compared_rows=total_compared,
            agreement_count=agreement_count,
            disagreement_count=disagreement_count,
            agreement_percentage=round(agr_pct, 2),
            disagreement_percentage=round(disagr_pct, 2),
            incomplete_count=incomplete_count,
            pairwise_agreements=pairwise_agreements,
            top_disagreements=top_disagreements,
        )

        return agreement_statuses, differing_models_per_row, summary
