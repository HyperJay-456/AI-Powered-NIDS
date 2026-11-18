"""Model Disagreement & Consensus Analytics Subsystem (CP-09).

Provides deep multi-model consensus and divergence analysis without introducing
unauthorized ensembles, majority-voting classifiers, or automated threat escalations.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DisagreementService:
    """Computes cross-model consensus, divergence matrices, and pattern summaries."""

    DISAGREEMENT_DISCLAIMER = (
        "Model disagreement indicates algorithmic divergence and classification uncertainty; "
        "it does NOT independently confirm or refute an intrusion, nor does it create an ensemble."
    )

    @classmethod
    def analyze_consensus(
        cls,
        df: pd.DataFrame,
        selected_algorithms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute consensus metrics and pairwise agreement across available model predictions.

        Args:
            df: DataFrame containing model prediction columns.
            selected_algorithms: Optional list of algorithms to compare.

        Returns:
            Dict[str, Any]: Comprehensive consensus analytics dictionary.
        """
        if df.empty:
            return {
                "total_rows": 0,
                "agreement_count": 0,
                "disagreement_count": 0,
                "incomplete_count": 0,
                "agreement_pct": 0.0,
                "disagreement_pct": 0.0,
                "incomplete_pct": 0.0,
                "pairwise_agreements": {},
                "top_divergence_patterns": [],
                "disclaimer": cls.DISAGREEMENT_DISCLAIMER,
            }

        total_rows = len(df)

        # Identify model prediction columns
        available_algos = selected_algorithms or []
        if not available_algos:
            for algo in ("svm", "mlp", "decision_tree"):
                if f"{algo}_prediction" in df.columns:
                    available_algos.append(algo)

        # If only single model
        if len(available_algos) <= 1:
            return {
                "total_rows": total_rows,
                "agreement_count": total_rows,
                "disagreement_count": 0,
                "incomplete_count": 0,
                "agreement_pct": 100.0,
                "disagreement_pct": 0.0,
                "incomplete_pct": 0.0,
                "pairwise_agreements": {},
                "top_divergence_patterns": [],
                "disclaimer": "Single model evaluated: consensus analysis requires 2 or more authorized algorithms.",
            }

        # Check existing agreement_status column if present
        if "agreement_status" in df.columns:
            status_counts = df["agreement_status"].value_counts().to_dict()
            agree_count = status_counts.get("Agreement", 0)
            disagree_count = status_counts.get("Disagreement", 0)
            incomplete_count = status_counts.get("Incomplete", 0)
        else:
            # Recompute agreement status row by row
            agree_count = 0
            disagree_count = 0
            incomplete_count = 0
            for _, row in df.iterrows():
                preds = [row.get(f"{a}_prediction") for a in available_algos]
                if any(pd.isna(p) or str(p).strip() in ("", "None", "UNKNOWN") for p in preds):
                    incomplete_count += 1
                elif len(set(str(p).strip() for p in preds)) == 1:
                    agree_count += 1
                else:
                    disagree_count += 1

        agree_pct = round((agree_count / total_rows) * 100.0, 2)
        disagree_pct = round((disagree_count / total_rows) * 100.0, 2)
        incomplete_pct = round((incomplete_count / total_rows) * 100.0, 2)

        # Pairwise agreement percentages
        pairwise: Dict[str, float] = {}
        for i in range(len(available_algos)):
            for j in range(i + 1, len(available_algos)):
                a1 = available_algos[i]
                a2 = available_algos[j]
                c1 = f"{a1}_prediction"
                c2 = f"{a2}_prediction"
                if c1 in df.columns and c2 in df.columns:
                    match_count = (df[c1] == df[c2]).sum()
                    pct = round((match_count / max(1, total_rows)) * 100.0, 2)
                    pair_label = f"{a1.upper()} vs {a2.upper()}"
                    pairwise[pair_label] = pct

        # Divergence patterns (where models disagree)
        patterns_counter: Counter = Counter()
        for _, row in df.iterrows():
            if row.get("agreement_status") == "Disagreement" or (
                len(available_algos) > 1 and len(set(str(row.get(f"{a}_prediction", "")).strip() for a in available_algos)) > 1
            ):
                pattern_parts = [f"{a.upper()}: {row.get(f'{a}_prediction', 'N/A')}" for a in available_algos]
                pattern_str = " | ".join(pattern_parts)
                patterns_counter[pattern_str] += 1

        top_divergences = [
            {"pattern": pat, "count": count, "percentage": round((count / max(1, total_rows)) * 100.0, 2)}
            for pat, count in patterns_counter.most_common(10)
        ]

        return {
            "total_rows": total_rows,
            "agreement_count": agree_count,
            "disagreement_count": disagree_count,
            "incomplete_count": incomplete_count,
            "agreement_pct": agree_pct,
            "disagreement_pct": disagree_pct,
            "incomplete_pct": incomplete_pct,
            "pairwise_agreements": pairwise,
            "top_divergence_patterns": top_divergences,
            "disclaimer": cls.DISAGREEMENT_DISCLAIMER,
        }
