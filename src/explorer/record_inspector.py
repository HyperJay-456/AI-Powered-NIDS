"""Individual Traffic Record Inspector (CP-09).

Retrieves complete multi-angle forensic telemetry for an individual network flow:
  1. Record Identity & Data Provenance
  2. Network Traffic Attributes (from source CSV if available on disk)
  3. Per-Model Predictions with Algorithm-Specific Score Semantics
  4. Explicit Ground-Truth Separation
  5. Complete Chronological Analyst Review Audit Trail
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.detection.detection_manifest import DetectionRunManifest
from src.explorer.prediction_reader import PredictionReader
from src.explorer.review_models import ReviewEvent
from src.explorer.review_store import ReviewStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelPredictionDetail:
    """Individual model prediction and score semantics for a flow."""

    algorithm: str
    algorithm_display_name: str
    predicted_class: str
    raw_score: Optional[float]
    score_type: str
    score_explanation: str


@dataclass
class DetailedRecordInspection:
    """Consolidated forensic inspection view for an individual traffic record."""

    detection_run_id: str
    source_dataset_id: str
    source_row_index: int
    source_filename: str
    has_ground_truth: bool
    ground_truth_label: Optional[str]
    agreement_status: str
    differing_models: Optional[str]
    model_predictions: Dict[str, ModelPredictionDetail]
    network_attributes: Dict[str, Any]
    network_attributes_available: bool
    network_attributes_note: str
    review_history: List[ReviewEvent]
    latest_review: Optional[ReviewEvent]


class RecordInspector:
    """Retrieves deep forensic details for individual network flow records."""

    def __init__(
        self,
        prediction_reader: Optional[PredictionReader] = None,
        review_store: Optional[ReviewStore] = None,
        raw_data_dir: Optional[Path] = None,
    ):
        """Initialize inspector with reader, review store, and raw data directory."""
        self.reader = prediction_reader or PredictionReader()
        self.review_store = review_store or ReviewStore()
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir

    def inspect_record(
        self,
        detection_run_id: str,
        source_row_index: int,
    ) -> DetailedRecordInspection:
        """Inspect a specific network flow record by its source row index.

        Args:
            detection_run_id: ID of the detection run.
            source_row_index: 0-based source row index.

        Returns:
            DetailedRecordInspection: Forensic record details.

        Raises:
            KeyError: If the specified row index does not exist in predictions.
        """
        manifest = self.reader.load_manifest(detection_run_id)
        preds_df = self.reader.load_predictions(detection_run_id, attach_reviews=True)

        row_match = preds_df[preds_df["source_row_index"] == int(source_row_index)]
        if row_match.empty:
            raise KeyError(
                f"Row index {source_row_index} not found in predictions for run '{detection_run_id}'."
            )

        row_data = row_match.iloc[0]

        source_dataset_id = manifest.source_dataset_id if manifest else "unknown"
        source_filename = manifest.source_filename if manifest else "unknown"

        # 1. Multi-Model Predictions & Score Semantics
        model_predictions: Dict[str, ModelPredictionDetail] = {}

        # Authorized algorithms to inspect
        algo_names = {
            "svm": "Support Vector Machine (LinearSVC)",
            "mlp": "Multilayer Perceptron (MLP)",
            "decision_tree": "Decision Tree",
        }

        # Check single-model columns
        if "model_score" in row_data.index and "predicted_class" in row_data.index:
            single_algo = "svm"
            if manifest and manifest.selected_algorithms:
                single_algo = manifest.selected_algorithms[0]

            score_val = float(row_data["model_score"]) if pd.notna(row_data.get("model_score")) else None
            score_type = str(row_data.get("score_type", "score"))
            explanation = self._get_score_semantic_explanation(single_algo)

            model_predictions[single_algo] = ModelPredictionDetail(
                algorithm=single_algo,
                algorithm_display_name=algo_names.get(single_algo, single_algo.upper()),
                predicted_class=str(row_data["predicted_class"]),
                raw_score=score_val,
                score_type=score_type,
                score_explanation=explanation,
            )

        # Check multi-model columns
        for algo_key in ("svm", "mlp", "decision_tree"):
            pred_col = f"{algo_key}_prediction"
            score_col = f"{algo_key}_score"
            if pred_col in row_data.index:
                pred_val = str(row_data[pred_col])
                score_val = float(row_data[score_col]) if score_col in row_data.index and pd.notna(row_data[score_col]) else None
                score_type = "decision_margin" if algo_key == "svm" else "uncalibrated_probabilities"
                explanation = self._get_score_semantic_explanation(algo_key)

                model_predictions[algo_key] = ModelPredictionDetail(
                    algorithm=algo_key,
                    algorithm_display_name=algo_names.get(algo_key, algo_key.upper()),
                    predicted_class=pred_val,
                    raw_score=score_val,
                    score_type=score_type,
                    score_explanation=explanation,
                )

        # 2. Agreement Status
        agreement_status = str(row_data.get("agreement_status", "Single Model"))
        differing = str(row_data.get("differing_models", "None")) if "differing_models" in row_data.index else None

        # 3. Ground Truth
        gt_label: Optional[str] = None
        has_gt = False
        if "ground_truth_label" in row_data.index and pd.notna(row_data["ground_truth_label"]):
            gt_val = str(row_data["ground_truth_label"]).strip()
            if gt_val and gt_val != "None":
                gt_label = gt_val
                has_gt = True

        # 4. Network Attributes from Source CSV
        net_attrs, attrs_avail, attrs_note = self._extract_raw_network_attributes(
            source_filename=source_filename,
            source_row_index=int(source_row_index),
        )

        # 5. Review History
        history = self.review_store.get_review_history(
            detection_run_id=detection_run_id,
            source_row_index=int(source_row_index),
        )
        latest_rev = history[-1] if history else None

        return DetailedRecordInspection(
            detection_run_id=detection_run_id,
            source_dataset_id=source_dataset_id,
            source_row_index=int(source_row_index),
            source_filename=source_filename,
            has_ground_truth=has_gt,
            ground_truth_label=gt_label,
            agreement_status=agreement_status,
            differing_models=differing,
            model_predictions=model_predictions,
            network_attributes=net_attrs,
            network_attributes_available=attrs_avail,
            network_attributes_note=attrs_note,
            review_history=history,
            latest_review=latest_rev,
        )

    def _extract_raw_network_attributes(
        self,
        source_filename: str,
        source_row_index: int,
    ) -> Tuple[Dict[str, Any], bool, str]:
        """Extract original traffic attributes from raw CSV on disk if available."""
        if not source_filename or source_filename == "unknown":
            return {}, False, "Source filename not recorded in detection manifest."

        raw_file = self.raw_data_dir / source_filename
        if not raw_file.exists():
            return {}, False, f"Source CSV '{source_filename}' not found in data/raw/."

        try:
            # Read single row: header + row at source_row_index
            # skiprows takes 1-based row numbers to skip: skip rows 1 to source_row_index
            if source_row_index == 0:
                raw_df = pd.read_csv(raw_file, nrows=1)
            else:
                raw_df = pd.read_csv(raw_file, skiprows=range(1, source_row_index + 1), nrows=1)

            if raw_df.empty:
                return {}, False, f"Row index {source_row_index} out of bounds in '{source_filename}'."

            # Normalize column names for clean display
            clean_attrs: Dict[str, Any] = {}
            for col in raw_df.columns:
                norm_key = col.strip()
                val = raw_df[col].iloc[0]
                # Format floats cleanly
                if isinstance(val, float) and pd.notna(val):
                    clean_attrs[norm_key] = round(val, 4) if abs(val) < 1000 else f"{val:,.2f}"
                elif pd.isna(val):
                    clean_attrs[norm_key] = "NaN"
                else:
                    clean_attrs[norm_key] = val

            return clean_attrs, True, f"Extracted from {source_filename} [Row {source_row_index:,}]"

        except Exception as e:
            logger.warning(f"Failed to extract raw attributes from {raw_file} at index {source_row_index}: {e}")
            return {}, False, f"Error reading raw CSV: {e}"

    @staticmethod
    def _get_score_semantic_explanation(algorithm: str) -> str:
        """Provide transparent, algorithm-specific score semantics."""
        if algorithm == "svm":
            return (
                "Signed Euclidean distance to the separating decision hyperplane (LinearSVC margin). "
                "Class is assigned based on the largest decision margin. Distances are uncalibrated and "
                "do NOT represent posterior probabilities."
            )
        elif algorithm == "mlp":
            return (
                "Uncalibrated softmax activation output. Reflects neural network layer activations, "
                "NOT a calibrated Bayesian probability of an intrusion. High scores on out-of-distribution "
                "traffic often exhibit overconfidence."
            )
        elif algorithm == "decision_tree":
            return (
                "Empirical training class frequency proportion at the terminal leaf node. Uncalibrated "
                "discrete distribution heuristic; NOT a true attack probability."
            )
        return "Model decision score."
