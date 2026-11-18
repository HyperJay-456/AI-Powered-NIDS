"""Inference Pipeline & Detection Orchestrator (CP-08).

Master workflow coordinator for operational intrusion detection:
  - Enforces model eligibility and candidate freeze verification
  - Enforces CP-04 dataset verification and training data overlap audits
  - Reuses exact CP-05 preprocessing transformations without refitting
  - Performs bounded-chunk sparse matrix inference
  - Enforces row accounting and consensus analysis
  - Atomically records audit manifests and prediction tables
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import ALGORITHM_KEYS, settings
from src.detection.dataset_loader import DetectionDatasetLoader, VerifiedDatasetSource
from src.detection.detection_manifest import (
    DetectionRunManifest,
    DisagreementSummary,
    ModelInferenceMetadata,
    RowAccountingRecord,
    capture_detection_environment,
)
from src.detection.detection_registry import DetectionRegistry
from src.detection.disagreement_analyzer import DisagreementAnalyzer
from src.detection.input_validator import DetectionInputValidator
from src.detection.model_eligibility import EligibleModelInfo, ModelEligibilityVerifier
from src.detection.prediction_engine import DetectionPredictionEngine
from src.detection.prediction_mapper import DetectionPredictionMapper
from src.detection.preprocessing_loader import DetectionPreprocessingLoader
from src.detection.resource_manager import DetectionResourceManager
from src.detection.result_writer import DetectionResultWriter
from src.detection.score_interpreter import DetectionScoreInterpreter
from src.models.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionInferencePipeline:
    """Master coordinator for CSV-based network intrusion detection."""

    _CONCURRENCY_LOCK = threading.Lock()

    def __init__(
        self,
        saved_models_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        evaluations_dir: Optional[Path] = None,
        analyses_dir: Optional[Path] = None,
        raw_data_dir: Optional[Path] = None,
        metadata_dir: Optional[Path] = None,
    ):
        """Initialize pipeline with subsystem paths."""
        self.saved_dir = saved_models_dir or settings.saved_models_dir
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.eval_dir = evaluations_dir or settings.evaluations_dir
        self.analyses_dir = analyses_dir or settings.analyses_dir
        self.raw_dir = raw_data_dir or settings.raw_data_dir
        self.metadata_dir = metadata_dir or settings.metadata_dir

        self.verifier = ModelEligibilityVerifier(
            saved_models_dir=self.saved_dir,
            processed_dir=self.processed_dir,
            evaluations_dir=self.eval_dir,
        )
        self.dataset_loader = DetectionDatasetLoader(
            raw_data_dir=self.raw_dir,
            metadata_dir=self.metadata_dir,
            processed_dir=self.processed_dir,
        )
        self.model_registry = ModelRegistry(saved_models_dir=self.saved_dir)
        self.result_writer = DetectionResultWriter(analyses_dir=self.analyses_dir)
        self.detection_registry = DetectionRegistry(analyses_dir=self.analyses_dir)

    def run_inference(
        self,
        dataset_id: str,
        algorithm: Optional[str] = None,
        training_run_id: Optional[str] = None,
        algorithms: Optional[List[str]] = None,
        batch_size: int = 10_000,
        allow_test_fixtures: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> DetectionRunManifest:
        """Execute single-model or multi-model comparative intrusion detection.

        Args:
            dataset_id: Validated CP-04 dataset identifier or filename.
            algorithm: Single algorithm key ('svm', 'mlp', 'decision_tree').
            training_run_id: Optional specific training run ID. If omitted, uses latest eligible.
            algorithms: List of algorithm keys for multi-model comparison.
            batch_size: Bounded chunk size for batch processing.
            allow_test_fixtures: Whether to permit test fixture models (e.g. in test suites).
            progress_callback: Optional reporter function (current_rows, total_rows, message).

        Returns:
            DetectionRunManifest with complete provenance, accounting, and metrics.
        """
        # Concurrency protection
        if not self._CONCURRENCY_LOCK.acquire(blocking=False):
            raise RuntimeError("Another intrusion detection inference run is currently active.")

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            # 1. Resolve Target Algorithms
            target_algos: List[str] = []
            if algorithms:
                target_algos = [a.lower().strip() for a in algorithms]
            elif algorithm:
                target_algos = [algorithm.lower().strip()]
            else:
                # Default to all eligible models
                eligible_all = self.verifier.discover_eligible_models(allow_test_fixtures=allow_test_fixtures)
                target_algos = [m.algorithm for m in eligible_all]

            if not target_algos:
                raise ValueError(
                    "No eligible production model is currently available. "
                    "Train a model and complete the required evaluation and candidate-freeze steps first."
                )

            # 2. Verify Model Eligibility
            model_infos: Dict[str, EligibleModelInfo] = {}
            for algo in target_algos:
                if training_run_id:
                    info = self.verifier.verify_model(
                        training_run_id=training_run_id,
                        algorithm=algo,
                        allow_test_fixtures=allow_test_fixtures,
                    )
                else:
                    # Find latest eligible model for this algorithm
                    eligible_list = [
                        m for m in self.verifier.discover_eligible_models(allow_test_fixtures=allow_test_fixtures)
                        if m.algorithm == algo
                    ]
                    if not eligible_list:
                        raise ValueError(
                            f"No eligible production model available for algorithm '{algo}'. "
                            "Complete CP-06 training and CP-07 candidate freezing first."
                        )
                    info = eligible_list[0]

                if not info.is_eligible:
                    raise ValueError(
                        f"Model '{algo}' (Run: {info.training_run_id}) is not eligible for inference: "
                        f"{'; '.join(info.rejection_reasons)}"
                    )
                model_infos[algo] = info

            # Verify that all selected models share the SAME CP-05 engineering run lineage
            eng_runs = {info.engineering_run_id for info in model_infos.values()}
            if len(eng_runs) > 1:
                raise ValueError(
                    f"Selected models originate from different engineering runs: {eng_runs}. "
                    "Multi-model comparison requires models from the same dataset engineering lineage."
                )
            common_eng_run_id = next(iter(eng_runs))

            # 3. Verify and Prepare Input Dataset
            source_info: VerifiedDatasetSource = self.dataset_loader.verify_and_prepare_source(
                dataset_id=dataset_id,
                engineering_run_id=common_eng_run_id,
            )

            # 4. Resource Preflight
            preflight = DetectionResourceManager.perform_preflight_check(
                num_features=model_infos[target_algos[0]].feature_count or 80,
                batch_size=batch_size,
                num_models=len(target_algos),
                target_path=self.analyses_dir,
            )
            if not preflight.is_safe:
                raise RuntimeError(f"Resource preflight check failed: {'; '.join(preflight.rejection_reasons)}")
            effective_batch_size = preflight.batch_size

            # 5. Initialize Preprocessing & Validation
            prep_loader = DetectionPreprocessingLoader(
                engineering_run_id=common_eng_run_id,
                processed_dir=self.processed_dir,
            )
            input_validator = DetectionInputValidator(feature_schema=prep_loader.feature_schema)

            # Verify schema compatibility against sample
            sample_df = pd.read_csv(source_info.file_path, nrows=5, low_memory=False)
            is_compat, missing_cols = input_validator.validate_schema_compatibility(list(sample_df.columns))
            if not is_compat:
                raise ValueError(
                    f"Dataset '{source_info.safe_filename}' is incompatible with model schema. "
                    f"Missing required feature columns: {missing_cols[:5]}..."
                )

            # Build Prediction Mappers (using CP-05 label mapping)
            classes_list = prep_loader.label_mapping.get("classes", [])
            idx_to_label = prep_loader.label_mapping.get("index_to_label", {})
            prediction_mapper = DetectionPredictionMapper(
                index_to_label=idx_to_label,
                classes=classes_list,
            )

            # 6. Load Model Estimators Safely
            estimators: Dict[str, Any] = {}
            for algo, info in model_infos.items():
                estimators[algo] = self.model_registry.load_model(
                    training_run_id=info.training_run_id,
                    algorithm=algo,
                )

            # 7. Execute Chunk-Based Inference Loop
            total_expected_rows = source_info.total_rows or source_info.valid_rows
            processed_rows = 0
            successfully_predicted_rows = 0
            rejected_rows = 0
            skipped_rows = 0

            # Accumulators
            preds_by_algo: Dict[str, List[str]] = {a: [] for a in target_algos}
            scores_by_algo: Dict[str, List[Optional[float]]] = {a: [] for a in target_algos}
            score_types_by_algo: Dict[str, str] = {}
            ground_truth_records: List[Optional[str]] = []
            source_row_indices: List[int] = []
            class_distributions: Dict[str, Counter] = {a: Counter() for a in target_algos}

            if progress_callback:
                progress_callback(0, total_expected_rows, f"Starting inference across {len(target_algos)} model(s)...")

            for chunk_df, start_idx in self.dataset_loader.iter_raw_chunks(
                file_path=source_info.file_path,
                chunk_size=effective_batch_size,
            ):
                chunk_len = len(chunk_df)
                processed_rows += chunk_len

                # Validate and strip ground-truth label
                val_res = input_validator.process_chunk(
                    chunk_df=chunk_df,
                    label_column_name=source_info.label_column_name,
                )

                if val_res.rejected_indices:
                    rejected_rows += len(val_res.rejected_indices)

                n_valid = len(val_res.features_df)
                if n_valid == 0:
                    skipped_rows += chunk_len
                    continue

                successfully_predicted_rows += n_valid

                # Record source indices and ground-truth references
                for i in range(n_valid):
                    source_row_indices.append(start_idx + i)
                    if val_res.ground_truth_series is not None and i < len(val_res.ground_truth_series):
                        ground_truth_records.append(str(val_res.ground_truth_series.iloc[i]))
                    else:
                        ground_truth_records.append(None)

                # Execute Inference per Algorithm
                for algo in target_algos:
                    estimator = estimators[algo]
                    # Transform chunk with algorithm-specific pipeline
                    X_chunk_csr = prep_loader.transform(val_res.features_df, algorithm=algo)

                    # Predict
                    y_pred, raw_scores, score_type = DetectionPredictionEngine.predict_chunk(
                        estimator=estimator,
                        algorithm=algo,
                        X_chunk=X_chunk_csr,
                    )
                    score_types_by_algo[algo] = score_type

                    # Map to class labels
                    mapped_labels = prediction_mapper.map_predictions(y_pred)
                    winning_scores = DetectionScoreInterpreter.extract_winning_scores(
                        algorithm=algo,
                        scores=raw_scores,
                        predicted_indices=y_pred,
                    )

                    preds_by_algo[algo].extend(mapped_labels)
                    scores_by_algo[algo].extend(winning_scores)

                    for lbl in mapped_labels:
                        class_distributions[algo][lbl] += 1

                if progress_callback:
                    pct = int(min(100, (processed_rows / max(1, total_expected_rows)) * 100))
                    progress_callback(
                        processed_rows,
                        total_expected_rows,
                        f"Processed {processed_rows:,} rows ({pct}%)...",
                    )

            # 8. Consensus & Disagreement Analysis (when multiple models)
            disagreement_summary: Optional[DisagreementSummary] = None
            agreement_statuses: List[str] = []
            differing_models: List[List[str]] = []

            if len(target_algos) > 1:
                agreement_statuses, differing_models, disagreement_summary = DisagreementAnalyzer.analyze_row_consensus(
                    predictions_by_algo=preds_by_algo,
                )

            # 9. Assemble Output DataFrames
            has_gt = any(gt is not None for gt in ground_truth_records)
            predictions_data: Dict[str, Any] = {"source_row_index": source_row_indices}

            if len(target_algos) == 1:
                single_algo = target_algos[0]
                predictions_data["predicted_class"] = preds_by_algo[single_algo]
                predictions_data["model_score"] = scores_by_algo[single_algo]
                predictions_data["score_type"] = score_types_by_algo.get(single_algo, "score")
            else:
                for algo in target_algos:
                    predictions_data[f"{algo}_prediction"] = preds_by_algo[algo]
                    predictions_data[f"{algo}_score"] = scores_by_algo[algo]
                predictions_data["agreement_status"] = agreement_statuses
                predictions_data["differing_models"] = [", ".join(d) if d else "None" for d in differing_models]

            if has_gt:
                predictions_data["ground_truth_label"] = ground_truth_records

            predictions_df = pd.DataFrame(predictions_data)

            # Multi-model comparison DataFrame
            comparison_df: Optional[pd.DataFrame] = None
            if len(target_algos) > 1:
                comp_data: Dict[str, Any] = {"source_row_index": source_row_indices}
                for algo in target_algos:
                    comp_data[f"{algo}_prediction"] = preds_by_algo[algo]
                comp_data["agreement_status"] = agreement_statuses
                comp_data["differing_models"] = [", ".join(d) if d else "None" for d in differing_models]
                if has_gt:
                    comp_data["ground_truth_label"] = ground_truth_records
                comparison_df = pd.DataFrame(comp_data)

            # 10. Enforce Row Accounting Invariant
            accounting = RowAccountingRecord(
                input_rows=processed_rows,
                successfully_predicted_rows=successfully_predicted_rows,
                rejected_rows=rejected_rows,
                skipped_rows=skipped_rows,
            )

            if not accounting.accounting_invariant_holds:
                raise ValueError(
                    f"Row accounting invariant violation: {processed_rows} input != "
                    f"{successfully_predicted_rows} pred + {rejected_rows} rej + {skipped_rows} skip."
                )

            # 11. Compile Master Run Manifest
            completed_at = datetime.now(timezone.utc).isoformat()
            run_id = DetectionResultWriter.generate_detection_run_id()

            model_metas: Dict[str, ModelInferenceMetadata] = {}
            for algo in target_algos:
                info = model_infos[algo]
                model_metas[algo] = ModelInferenceMetadata(
                    algorithm=algo,
                    algorithm_display_name=info.algorithm_display_name,
                    training_run_id=info.training_run_id,
                    model_checksum=info.model_sha256,
                    engineering_run_id=info.engineering_run_id,
                    is_primary_candidate=info.is_primary_candidate,
                    candidate_freeze_time_utc=info.candidate_freeze_time_utc,
                    candidate_selection_criterion=info.candidate_selection_criterion,
                    status="Completed",
                    predicted_rows=successfully_predicted_rows,
                    score_type=score_types_by_algo.get(algo, "score"),
                    error_message=None,
                )

            is_fixture = any(info.is_test_fixture for info in model_infos.values()) or allow_test_fixtures

            manifest = DetectionRunManifest(
                detection_run_id=run_id,
                created_at_utc=started_at,
                completed_at_utc=completed_at,
                status="Completed with Warnings" if source_info.overlaps_training_data else "Completed",
                source_dataset_id=source_info.dataset_id,
                source_filename=source_info.safe_filename,
                source_sha256=source_info.sha256_checksum,
                source_validation_status=source_info.validation_status,
                source_overlaps_training=source_info.overlaps_training_data,
                source_overlap_details=source_info.overlap_details,
                has_ground_truth_target=has_gt,
                ground_truth_column_name=source_info.label_column_name if has_gt else None,
                selected_algorithms=target_algos,
                models=model_metas,
                engineering_run_id=common_eng_run_id,
                feature_schema_fingerprint=prep_loader.feature_schema.get("feature_schema_hash", common_eng_run_id),
                batch_size=effective_batch_size,
                row_accounting=accounting,
                class_distribution={a: dict(c) for a, c in class_distributions.items()},
                disagreement_summary=disagreement_summary,
                warnings=source_info.warning_messages + preflight.warning_messages,
                errors=[],
                notes=[f"Successfully executed inference across {len(target_algos)} model(s)."],
                run_type="test_fixture" if is_fixture else "production",
                is_test_fixture=is_fixture,
                environment=capture_detection_environment(),
            )

            # 12. Persist Outputs Atomically
            self.result_writer.write_results(
                manifest=manifest,
                predictions_df=predictions_df,
                comparison_df=comparison_df,
            )

            if progress_callback:
                progress_callback(processed_rows, total_expected_rows, "Inference completed successfully!")

            return manifest

        finally:
            self._CONCURRENCY_LOCK.release()
