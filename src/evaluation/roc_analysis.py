"""One-vs-Rest ROC Analysis Engine (CP-07).

Computes binary one-vs-rest Receiver Operating Characteristic (ROC) curves and AUC:
1. LinearSVC: Uses decision_function() signed distance margins (not probabilities).
2. MLPClassifier & DecisionTreeClassifier: Uses predict_proba() output scores.
3. Explicitly flags unavailable curves when a class has no positive or negative examples.
4. Computes Macro ROC-AUC across mathematically estimable classes.
5. Generates interactive Plotly figures with random chance reference diagonals.
6. Caps visualization points to prevent bloated JSON payloads while computing exact AUC.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.evaluation_manifest import ROCMetricRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ROCAnalyzer:
    """Performs multiclass one-vs-rest ROC analysis."""

    MAX_DISPLAY_POINTS: int = 200

    @classmethod
    def compute_ovr_roc(
        cls,
        y_true: np.ndarray,
        scores: Optional[np.ndarray],
        classes: List[str],
        score_type: str = "predict_proba",
    ) -> Tuple[Dict[str, ROCMetricRecord], Optional[float]]:
        """Compute one-vs-rest ROC curves and AUC for each class.

        Args:
            y_true: Ground truth target labels.
            scores: Continuous prediction scores of shape (n_samples, n_classes).
            classes: Canonical class names.
            score_type: "decision_function" or "predict_proba".

        Returns:
            Tuple of (per_class_records_dict, macro_roc_auc).
        """
        roc_results: Dict[str, ROCMetricRecord] = {}
        estimable_aucs: List[float] = []

        if scores is None or scores.shape[1] != len(classes):
            # Scores unavailable or misaligned
            for cls_name in classes:
                roc_results[cls_name] = ROCMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    auc=None,
                    score_type=score_type,
                    unavailable_reason="Continuous scores unavailable or dimension misaligned",
                )
            return roc_results, None

        for idx, cls_name in enumerate(classes):
            y_bin = (y_true == idx).astype(int)
            pos_count = int(np.sum(y_bin == 1))
            neg_count = int(np.sum(y_bin == 0))

            # ROC requires at least one positive and one negative sample
            if pos_count == 0:
                roc_results[cls_name] = ROCMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    auc=None,
                    score_type=score_type,
                    unavailable_reason="Undefined: No positive ground-truth samples in evaluated partition",
                )
                continue

            if neg_count == 0:
                roc_results[cls_name] = ROCMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    auc=None,
                    score_type=score_type,
                    unavailable_reason="Undefined: All samples in partition belong to this class (no negative examples)",
                )
                continue

            class_scores = scores[:, idx]

            # Check for non-finite values
            if not np.all(np.isfinite(class_scores)):
                roc_results[cls_name] = ROCMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    auc=None,
                    score_type=score_type,
                    unavailable_reason="Non-finite values encountered in continuous score array",
                )
                continue

            try:
                # Compute exact AUC on full dataset
                auc_val = float(roc_auc_score(y_bin, class_scores))
                estimable_aucs.append(auc_val)

                # Compute ROC curve points
                fpr, tpr, _ = roc_curve(y_bin, class_scores)

                # Safe downsample for visualization storage if point count is large
                if len(fpr) > cls.MAX_DISPLAY_POINTS:
                    indices = np.linspace(0, len(fpr) - 1, cls.MAX_DISPLAY_POINTS, dtype=int)
                    fpr_display = fpr[indices].tolist()
                    tpr_display = tpr[indices].tolist()
                    # Ensure endpoints (0,0) and (1,1) are preserved
                    fpr_display[0], tpr_display[0] = 0.0, 0.0
                    fpr_display[-1], tpr_display[-1] = 1.0, 1.0
                else:
                    fpr_display = fpr.tolist()
                    tpr_display = tpr.tolist()

                roc_results[cls_name] = ROCMetricRecord(
                    class_name=cls_name,
                    is_available=True,
                    auc=round(auc_val, 4),
                    fpr=fpr_display,
                    tpr=tpr_display,
                    score_type=score_type,
                    unavailable_reason=None,
                )

            except Exception as e:
                logger.warning(f"Failed to compute ROC for class '{cls_name}': {e}")
                roc_results[cls_name] = ROCMetricRecord(
                    class_name=cls_name,
                    is_available=False,
                    auc=None,
                    score_type=score_type,
                    unavailable_reason=f"Calculation exception: {e}",
                )

        macro_auc = round(float(np.mean(estimable_aucs)), 4) if estimable_aucs else None
        return roc_results, macro_auc

    @staticmethod
    def create_plotly_roc_chart(
        roc_results: Dict[str, ROCMetricRecord],
        macro_auc: Optional[float],
        algorithm_name: str,
    ) -> go.Figure:
        """Create multi-class ROC curve Plotly visualization."""
        fig = go.Figure()

        # Random baseline diagonal
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.50)",
                line=dict(color="#64748B", dash="dash", width=1.5),
                hoverinfo="skip",
            )
        )

        palette = [
            "#38BDF8", "#34D399", "#F472B6", "#FBBF24", "#A78BFA",
            "#FB7185", "#2DD4BF", "#818CF8", "#C084FC", "#4ADE80",
        ]

        color_idx = 0
        for cls_name, record in roc_results.items():
            if record.is_available and record.fpr and record.tpr:
                c = palette[color_idx % len(palette)]
                color_idx += 1
                auc_str = f"{record.auc:.3f}" if record.auc is not None else "N/A"
                fig.add_trace(
                    go.Scatter(
                        x=record.fpr,
                        y=record.tpr,
                        mode="lines",
                        name=f"{cls_name} (AUC={auc_str})",
                        line=dict(color=c, width=2),
                        hovertemplate=f"<b>{cls_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                    )
                )

        macro_title = f" — Macro AUC: {macro_auc:.3f}" if macro_auc is not None else ""
        fig.update_layout(
            title=dict(
                text=f"One-vs-Rest ROC Curves: {algorithm_name}{macro_title}",
                font=dict(family="Inter, sans-serif", size=14, color="#F8FAFC"),
            ),
            xaxis=dict(
                title=dict(text="False Positive Rate (1 - Specificity)", font=dict(color="#94A3B8")),
                range=[-0.02, 1.02],
                tickfont=dict(color="#94A3B8"),
                gridcolor="rgba(148, 163, 184, 0.1)",
            ),
            yaxis=dict(
                title=dict(text="True Positive Rate (Sensitivity / Recall)", font=dict(color="#94A3B8")),
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
