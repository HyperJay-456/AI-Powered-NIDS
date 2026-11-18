"""Prediction Query, Filtering, Sorting & Pagination Engine (CP-09).

Provides robust, memory-conscious querying over prediction records:
  - Multi-attribute filtering (class, model, agreement, review, disposition, ground truth)
  - Safe score-filtering isolated by model family
  - Substring search across identifiers and text fields
  - Stable sorting and bounded pagination
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.explorer.review_models import AnalystDisposition, ReviewStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionFilterParams:
    """Filter parameters for querying prediction records."""

    predicted_classes: Optional[List[str]] = None
    selected_model: Optional[str] = None  # "svm", "mlp", "decision_tree"
    agreement_statuses: Optional[List[str]] = None  # "Agreement", "Disagreement", "Incomplete"
    review_statuses: Optional[List[str]] = None  # "Unreviewed", "Reviewed"
    analyst_dispositions: Optional[List[str]] = None  # Any of the 5 dispositions or "Unreviewed"
    ground_truth_filter: Optional[str] = "all"  # "all", "has_ground_truth", "differs_from_prediction", "matches_prediction"
    search_query: Optional[str] = None
    score_filter_model: Optional[str] = None  # Must match a specific model to avoid cross-algorithm distortion
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    sort_by: str = "source_row_index"
    sort_ascending: bool = True
    disagreement_first: bool = False  # Triage convenience view (not an ML risk score)

    def is_default(self) -> bool:
        """Check whether filter params are in their default state."""
        return (
            not self.predicted_classes
            and not self.selected_model
            and not self.agreement_statuses
            and not self.review_statuses
            and not self.analyst_dispositions
            and (self.ground_truth_filter in (None, "all"))
            and not self.search_query
            and not self.score_filter_model
            and self.min_score is None
            and self.max_score is None
            and self.sort_by == "source_row_index"
            and self.sort_ascending is True
            and not self.disagreement_first
        )


class PredictionQueryEngine:
    """Executes filtered, sorted, and paginated queries on prediction DataFrames."""

    @staticmethod
    def query(
        df: pd.DataFrame,
        params: PredictionFilterParams,
        page: int = 1,
        page_size: int = 25,
    ) -> Tuple[pd.DataFrame, int]:
        """Apply filters, sorting, and pagination to the prediction DataFrame.

        Args:
            df: Input DataFrame containing prediction and review columns.
            params: PredictionFilterParams containing filter criteria.
            page: 1-based page number.
            page_size: Number of records per page (bounded between 5 and 500).

        Returns:
            Tuple[pd.DataFrame, int]: (paginated_slice_df, total_matching_records)
        """
        if df.empty:
            return df.copy(), 0

        # Work on filtered index mask to avoid large data copying
        mask = pd.Series(True, index=df.index)

        # 1. Predicted Class Filter
        if params.predicted_classes:
            classes_to_match = set(params.predicted_classes)
            # Check target model prediction column or general predicted_class
            target_col = "predicted_class"
            if params.selected_model and f"{params.selected_model}_prediction" in df.columns:
                target_col = f"{params.selected_model}_prediction"

            if target_col in df.columns:
                mask &= df[target_col].astype(str).isin(classes_to_match)

        # 2. Agreement Status Filter
        if params.agreement_statuses and "agreement_status" in df.columns:
            mask &= df["agreement_status"].isin(params.agreement_statuses)

        # 3. Review Status Filter
        if params.review_statuses and "review_status" in df.columns:
            mask &= df["review_status"].isin(params.review_statuses)

        # 4. Analyst Disposition Filter
        if params.analyst_dispositions and "analyst_disposition" in df.columns:
            mask &= df["analyst_disposition"].isin(params.analyst_dispositions)

        # 5. Ground Truth Filter
        if "ground_truth_label" in df.columns and params.ground_truth_filter not in (None, "all"):
            gt_series = df["ground_truth_label"]
            has_gt_mask = gt_series.notna() & (gt_series != "") & (gt_series != "None")

            pred_col = "predicted_class"
            if params.selected_model and f"{params.selected_model}_prediction" in df.columns:
                pred_col = f"{params.selected_model}_prediction"

            if params.ground_truth_filter == "has_ground_truth":
                mask &= has_gt_mask
            elif params.ground_truth_filter == "differs_from_prediction":
                # Show flows where prediction != ground truth
                mask &= (has_gt_mask & (df[pred_col].astype(str) != gt_series.astype(str)))
            elif params.ground_truth_filter == "matches_prediction":
                # Show flows where prediction == ground truth
                mask &= (has_gt_mask & (df[pred_col].astype(str) == gt_series.astype(str)))

        # 6. Score Filter (Safely isolated to a single model family)
        if params.score_filter_model:
            score_col = f"{params.score_filter_model}_score"
            if score_col not in df.columns and "model_score" in df.columns:
                score_col = "model_score"

            if score_col in df.columns:
                if params.min_score is not None:
                    mask &= (df[score_col].notna() & (df[score_col] >= params.min_score))
                if params.max_score is not None:
                    mask &= (df[score_col].notna() & (df[score_col] <= params.max_score))

        # 7. Substring Search
        if params.search_query:
            query_str = str(params.search_query).strip().lower()
            if query_str:
                search_mask = pd.Series(False, index=df.index)

                # Search in source_row_index
                if "source_row_index" in df.columns:
                    search_mask |= df["source_row_index"].astype(str).str.contains(query_str, case=False, na=False)

                # Search in predicted_class
                if "predicted_class" in df.columns:
                    search_mask |= df["predicted_class"].astype(str).str.lower().str.contains(query_str, na=False)

                # Search in model prediction columns
                for col in df.columns:
                    if col.endswith("_prediction") or col in ("analyst_notes", "reviewer_label", "ground_truth_label"):
                        search_mask |= df[col].astype(str).str.lower().str.contains(query_str, na=False)

                mask &= search_mask

        # Apply filtering mask
        filtered_df = df[mask]
        total_matching = len(filtered_df)

        if total_matching == 0:
            return filtered_df.copy(), 0

        # 8. Sorting
        res_df = filtered_df.copy()

        if params.disagreement_first and "agreement_status" in res_df.columns:
            # Sort Disagreements to top, then apply sort_by
            is_disagree = (res_df["agreement_status"] == "Disagreement").astype(int)
            res_df["_sort_disagree"] = is_disagree
            if params.sort_by in res_df.columns:
                res_df = res_df.sort_values(
                    by=["_sort_disagree", params.sort_by],
                    ascending=[False, params.sort_ascending],
                )
            else:
                res_df = res_df.sort_values(by="_sort_disagree", ascending=False)
            res_df = res_df.drop(columns=["_sort_disagree"])
        elif params.sort_by in res_df.columns:
            res_df = res_df.sort_values(by=params.sort_by, ascending=params.sort_ascending)

        # 9. Bounded Pagination
        clamped_page_size = max(1, min(500, int(page_size)))
        total_pages = max(1, math.ceil(total_matching / clamped_page_size))
        clamped_page = max(1, min(total_pages, int(page)))

        start_idx = (clamped_page - 1) * clamped_page_size
        end_idx = start_idx + clamped_page_size

        page_slice = res_df.iloc[start_idx:end_idx].copy()
        return page_slice, total_matching
