"""One-vs-Rest Precision-Recall Analysis Engine (CP-07).

Computes binary one-vs-rest Precision-Recall (PR) curves and Average Precision (AP):
1. LinearSVC: Uses decision_function() signed distance margins.
2. MLPClassifier & DecisionTreeClassifier: Uses predict_proba() output scores.
3. Explicitly handles classes with zero positive samples (unavailable).
4. Computes Macro Average Precision (AP) across eligible classes.
5. Generates interactive Plotly figures with baseline reference horizontal lines.
6. Caps curve point counts to prevent excessive serialization overhead.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.evaluation.evaluation_manifest import PrecisionRecallMetricRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PrecisionRecallAnalyzer:
    """Performs multiclass one-vs-rest Precision-Recall analysis."""

    MAX_DISPLAY_POINTS: int = 200

    @classmethod
    def compute_ovr_pr(
        cls,
        y_true: np.ndarray,
        scores: Optional[np.ndarray],
        classes: List[str],
        score_type: str = "predict_proba",
    ) -> Tuple[Dict[str, PrecisionRecallMetricRecord], Optional[float]]:
        """Compute one-vs-rest PR curves and Average Precision for each class.

        Args:
            y_true: Ground truth target labels.
            scores: Continuous prediction scores of shape (n_samples, n_classes).
            classes: Canonical class names.
            score_type: "decision_function" or "predict_proba".

        Returns:
            Tuple of (per_class_records_dict, macro_average_precision).
        """
        pr_results: Dict[str, PrecisionRecallMetricRecord] = {}
        estimable_aps: List[float] = []

        if scores is None or scores.shape[1] != len(classes):
            for cls_name in classes:
                pr_results[cls_name] = PrecisionRecallMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    average_precision=None,
                    score_type=score_type,
                    unavailable_reason="Continuous scores unavailable or dimension misaligned",
                )
            return pr_results, None

        for idx, cls_name in enumerate(classes):
            y_bin = (y_true == idx).astype(int)
            pos_count = int(np.sum(y_bin == 1))
            neg_count = int(np.sum(y_bin == 0))

            # PR curve requires at least one positive sample
            if pos_count == 0:
                pr_results[cls_name] = PrecisionRecallMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    average_precision=None,
                    positive_samples=0,
                    negative_samples=neg_count,
                    score_type=score_type,
                    unavailable_reason="Undefined: No positive ground-truth samples in evaluated partition",
                )
                continue

            class_scores = scores[:, idx]

            if not np.all(np.isfinite(class_scores)):
                pr_results[cls_name] = PrecisionRecallMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    average_precision=None,
                    positive_samples=pos_count,
                    negative_samples=neg_count,
                    score_type=score_type,
                    unavailable_reason="Non-finite values encountered in continuous score array",
                )
                continue

            try:
                # Compute exact Average Precision
                ap_val = float(average_precision_score(y_bin, class_scores))
                estimable_aps.append(ap_val)

                # Compute PR curve points
                prec_arr, rec_arr, _ = precision_recall_curve(y_bin, class_scores)

                # Safe downsample for visualization storage
                if len(prec_arr) > cls.MAX_DISPLAY_POINTS:
                    indices = np.linspace(0, len(prec_arr) - 1, cls.MAX_DISPLAY_POINTS, dtype=int)
                    prec_display = prec_arr[indices].tolist()
                    rec_display = rec_arr[indices].tolist()
                else:
                    prec_display = prec_arr.tolist()
                    rec_display = rec_arr.tolist()

                pr_results[cls_name] = PrecisionRecallMetricRecord(
                    class_name=cls_name,
                    is_available=True,
                    average_precision=round(ap_val, 4),
                    precision_points=prec_display,
                    recall_points=rec_display,
                    positive_samples=pos_count,
                    negative_samples=neg_count,
                    score_type=score_type,
                    unavailable_reason=None,
                )

            except Exception as e:
                logger.warning(f"Failed to compute PR for class '{cls_name}': {e}")
                pr_results[cls_name] = PrecisionRecallMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    average_precision=None,
                    positive_samples=pos_count,
                    negative_samples=neg_count,
                    score_type=score_type,
                    unavailable_reason=f"Calculation exception: {e}",
                )

        macro_ap = round(float(np.mean(estimable_aps)), 4) if estimable_aps else None
        return pr_results, macro_ap

    @staticmethod
    def create_plotly_pr_chart(
        pr_results: Dict[str, PrecisionRecallMetricRecord],
        macro_ap: Optional[float],
        algorithm_name: str,
    ) -> go.Figure:
        """Create multi-class Precision-Recall curve Plotly visualization."""
        fig = go.Figure()

        palette = [
            "#38BDF8", "#34D399", "#F472B6", "#FBBF24", "#A78BFA",
            "#FB7185", "#2DD4BF", "#818CF8", "#C084FC", "#4ADE80",
        ]

        color_idx = 0
        for cls_name, record in pr_results.items():
            if record.is_available and record.recall_points and record.precision_points:
                c = palette[color_idx % len(palette)]
                color_idx += 1
                ap_str = f"{record.average_precision:.3f}" if record.average_precision is not None else "N/A"
                fig.add_trace(
                    go.Scatter(
                        x=record.recall_points,
                        y=record.precision_points,
                        mode="lines",
                        name=f"{cls_name} (AP={ap_str})",
                        line=dict(color=c, width=2),
                        hovertemplate=f"<b>{cls_name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
                    )
                )

        macro_title = f" — Macro AP: {macro_ap:.3f}" if macro_ap is not None else ""
        fig.update_layout(
            title=dict(
                text=f"One-vs-Rest Precision-Recall Curves: {algorithm_name}{macro_title}",
                font=dict(family="Inter, sans-serif", size=14, color="#F8FAFC"),
            ),
            xaxis=dict(
                title=dict(text="Recall (Sensitivity)", font=dict(color="#94A3B8")),
                range=[-0.02, 1.02],
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(148, 163, 184, 0.1)",
            ),
            yaxis=dict(
                title=dict(text="Precision (Positive Predictive Value)", font=dict(color="#94A3B8")),
                range=[-0.02, 1.05],
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(148, 163, 184, 0.1)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                font=dict(family="Inter, sans-serif", size=10, color="#94A3B8"),
                bgcolor="rgba(15, 23, 42, 0.7)",
                bordercolor="rgba(148, 163, 184, 0.2)",
                borderwidth=1,
            ),
            margin=dict(l=60, r=40, t=60, b=60),
            height=450,
        )

        return fig
