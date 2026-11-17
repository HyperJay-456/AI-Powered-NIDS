"""Confusion Matrix Engine (CP-07).

Generates exact multiclass raw-count and row-normalized confusion matrices:
1. Ground-truth labels on rows (vertical axis).
2. Predicted labels on columns (horizontal axis).
3. Stable ordering from canonical CP-05 label mapping.
4. Row-normalization: normalized by true class support (sum of row).
5. Exact accounting check: sum of all matrix cells == total sample count.
6. Generates Plotly figure objects ready for dashboard rendering.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.evaluation.evaluation_manifest import ConfusionMatrixRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfusionMatrixEngine:
    """Computes and formats multiclass confusion matrices."""

    @staticmethod
    def compute_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str],
    ) -> ConfusionMatrixRecord:
        """Compute raw and row-normalized confusion matrices across all canonical classes.

        Args:
            y_true: Ground truth target labels.
            y_pred: Model predictions.
            classes: Canonical class names list.

        Returns:
            ConfusionMatrixRecord with raw counts, row-normalized percentages, and total count.
        """
        n_classes = len(classes)
        labels = list(range(n_classes))

        # Compute raw counts
        raw = confusion_matrix(y_true, y_pred, labels=labels)

        # Row normalize: row sum is the ground truth support for that class
        row_sums = raw.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.true_divide(raw, row_sums)
            normalized[~np.isfinite(normalized)] = 0.0  # Handle zero support rows

        # Accounting check
        total_samples = int(np.sum(raw))
        if total_samples != len(y_true):
            raise ValueError(
                f"Confusion matrix accounting error: matrix sum {total_samples} != len(y_true) {len(y_true)}"
            )

        return ConfusionMatrixRecord(
            classes=classes,
            raw_matrix=raw.astype(int).tolist(),
            normalized_matrix=np.round(normalized * 100.0, 2).tolist(),  # In percent
            total_samples=total_samples,
        )

    @staticmethod
    def create_plotly_heatmap(
        cm_record: ConfusionMatrixRecord,
        normalized: bool = False,
        title: str = "Multiclass Confusion Matrix",
    ) -> go.Figure:
        """Generate interactive Plotly heatmap figure for Streamlit dashboard.

        Args:
            cm_record: ConfusionMatrixRecord instance.
            normalized: If True, display row-normalized percentages; else raw counts.
            title: Chart title.

        Returns:
            Plotly go.Figure.
        """
        classes = cm_record.classes
        matrix_data = cm_record.normalized_matrix if normalized else cm_record.raw_matrix

        z_values = np.array(matrix_data)
        text_values = []
        for r_idx in range(len(classes)):
            row_text = []
            for c_idx in range(len(classes)):
                val = z_values[r_idx, c_idx]
                if normalized:
                    raw_val = cm_record.raw_matrix[r_idx][c_idx]
                    row_text.append(f"{val:.1f}%<br>({raw_val:,})")
                else:
                    row_text.append(f"{int(val):,}")
            text_values.append(row_text)

        colorscale = "Blues" if not normalized else "Teal"

        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=classes,
                y=classes,
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 11, "family": "Inter, sans-serif"},
                colorscale=colorscale,
                colorbar=dict(
                    title="%" if normalized else "Count",
                    tickfont=dict(color="#94A3B8"),
                    titlefont=dict(color="#F8FAFC"),
                ),
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(family="Inter, sans-serif", size=15, color="#F8FAFC"),
            ),
            xaxis=dict(
                title=dict(text="Predicted Label", font=dict(color="#94A3B8")),
                tickangle=-35,
                tickfont=dict(family="Inter, sans-serif", size=10, color="#94A3B8"),
                showgrid=False,
            ),
            yaxis=dict(
                title=dict(text="Actual Ground Truth Label", font=dict(color="#94A3B8")),
                autorange="reversed",
                tickfont=dict(family="Inter, sans-serif", size=10, color="#94A3B8"),
                showgrid=False,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=40, t=60, b=80),
            height=max(450, len(classes) * 35),
        )

        return fig
