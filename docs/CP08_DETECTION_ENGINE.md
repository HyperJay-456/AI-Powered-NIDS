# Checkpoint CP-08: Intrusion Detection Engine — Technical Architecture & Specification

**Project:** AI-Powered Network Intrusion Detection System (NIDS)  
**Checkpoint:** CP-08 — Intrusion Detection Engine  
**Previous Checkpoint:** CP-07 — Model Evaluation Laboratory  
**Accepted Baseline Commit:** `0404518`  
**Current Status:** Completed & Audited (180/180 Tests Passing)  
**Date:** November 18, 2026  

---

## 1. System Overview & Scope Boundaries

Checkpoint **CP-08 (Intrusion Detection Engine)** introduces a modular, resource-aware, offline, CSV-based operational inference subsystem for the AI-Powered NIDS platform.

### Non-Negotiable Operational Invariants
1. **Exactly Three Authorized Algorithms:**
   - **Support Vector Machine (`LinearSVC`)**
   - **Multilayer Perceptron (`MLPClassifier`)**
   - **Decision Tree (`DecisionTreeClassifier`)**
   - *Strict Prohibition:* No Random Forest, XGBoost, LightGBM, KNN, Isolation Forest, Autoencoders, or pseudo-ensemble detectors.
2. **Local-First & Offline Architecture:**
   - Purely local CPU-optimized batch inference. No cloud inference, external AI APIs, or internet connectivity required.
3. **CSV-Based Detection Only:**
   - Operates strictly on validated CSV flow files ingested through CP-04.
   - *Explicitly Out of Scope:* Live PCAP sniffing, real-time packet interception, automatic firewall rules, automated IP blocking.
4. **No Fabricated Telemetry or Predictions:**
   - Every prediction, class distribution, and consensus metric originates directly from actual model inference against disk-backed artifacts.
5. **Raw Data Immutability:**
   - Ingested CSV files in `data/raw/` are treated as read-only. Predictions and manifests are committed to `data/analyses/<run_id>/`.
6. **Ground-Truth Target Protection:**
   - Ground-truth target columns (e.g. `Label`) are stripped from feature matrices before preprocessing and inference. Ground truth is preserved solely as reference metadata and never exposed to model estimators.

---

## 2. Detection Engine Architecture & Data Flow

The subsystem is implemented under `src/detection/` with clean separation of concerns:

```text
src/detection/
├── __init__.py                <- Package exports
├── dataset_loader.py          <- CP-04 ingestion manifest validation & chunk streaming
├── detection_manifest.py      <- Dataclass schemas for provenance, accounting, and manifests
├── detection_registry.py      <- History catalog, SHA-256 artifact verification, discovery
├── disagreement_analyzer.py   <- Multi-model consensus and row-by-row disagreement triage
├── inference_pipeline.py      <- Master workflow orchestrator & concurrency lock
├── input_validator.py         <- Schema reconciliation, numeric sanitization, label stripping
├── model_eligibility.py       <- Production registry & CP-07 candidate freeze verifier
├── prediction_engine.py       <- Estimator execution (predict, decision_function, predict_proba)
├── prediction_mapper.py       <- Attack class label mapping & index translation
├── preprocessing_loader.py    <- Zero-refit reuse of CP-05 imputers, encoders, and scalers
├── resource_manager.py        <- RAM/disk preflight gating & batch size boundary enforcement
└── score_interpreter.py       <- Model-specific score semantics (margins vs uncalibrated scores)
```

### End-to-End Inference Data Flow

```text
[ Validated Raw CSV in data/raw/ ]
              │
              ▼
[ DetectionDatasetLoader ] ──────> Validates SHA-256 & CP-04 Ingestion Manifest
              │                   Audits Training Data Overlap (Source Fingerprints)
              ▼
[ DetectionInputValidator ] ─────> Strips Ground-Truth Label (Preserves Reference Only)
              │                   Segregates Network Identifiers to Provenance
              │                   Verifies Schema Compatibility against CP-05 feature_schema.json
              ▼
[ DetectionPreprocessingLoader ] ─> Applies Training-Fitted Imputer (Median/Mode)
              │                   Applies Training-Fitted OneHotEncoder (handle_unknown='ignore')
              │                   Applies Training-Fitted Scaler:
              │                     - Scaled Sparse CSR for LinearSVC & MLP
              │                     - Unscaled Sparse CSR for Decision Tree
              ▼
[ DetectionPredictionEngine ] ───> Executes predict(), decision_function(), predict_proba()
              │                   Sequential CPU-Safe Execution (Default 10,000 Rows/Batch)
              ▼
[ DetectionPredictionMapper ] ───> Reconstructs Attack Class Strings from label_mapping.json
              │
              ▼
[ DisagreementAnalyzer ] ────────> Computes Row-Level Consensus: Agreement / Disagreement
              │                   Disagreement = Prediction Uncertainty (Forensic Triage)
              ▼
[ DetectionResultWriter ] ───────> Validates Row Accounting Invariant
                                  Atomically Commits:
                                    - data/analyses/<run_id>/predictions.csv
                                    - data/analyses/<run_id>/model_comparison.csv
                                    - data/analyses/<run_id>/run_manifest.json
```

---

## 3. Production Model Eligibility & CP-07 Integration

Before any model is permitted to execute operational inference, `ModelEligibilityVerifier` validates:
1. **Registry Existence:** Model artifact exists in `models/saved/<training_run_id>/<algo>/model.joblib`.
2. **Authorized Algorithm:** Matches one of the three approved algorithm keys (`svm`, `mlp`, `decision_tree`).
3. **Fixture Isolation:** Automated test fixtures (`is_test_fixture=True` or `run_type="test_fixture"`) are strictly rejected in production inference mode.
4. **Cryptographic Checksum:** On-disk SHA-256 matches the manifest checksum. Any byte tampering raises an immediate error.
5. **Preprocessing Provenance:** Referenced CP-05 `engineering_run_id` exists in `data/processed/`, containing valid feature schemas and fitted joblib transformers.
6. **CP-07 Candidate Freeze Verification:** In production mode, the model must be registered in an evaluation manifest under `frozen_candidates`, confirming that validation benchmarking and candidate freezing occurred prior to deployment.

If no eligible model exists, the console informs the user:
> *"No eligible production model is currently available. Train a model and complete the required evaluation and candidate-freeze steps first."*

---

## 4. Preprocessing Pipeline Reuse & Zero Refitting

The inference subsystem reuses the exact, immutable fitted preprocessing transformers produced during CP-05:
- **Missing Value Imputer (`LeakageFreeImputer`):** Uses medians (numeric) and modes (categorical) learned strictly on the training partition.
- **Categorical Encoder (`LeakageFreeCategoricalEncoder`):** Reuses the fitted `OneHotEncoder`. Unseen categorical values in operational traffic are mapped to all-zero one-hot vectors (`handle_unknown='ignore'`) without expanding the matrix dimension.
- **Dual Representation Scaler (`DualRepresentationScaler`):**
  - **SVM & MLP:** Scaled numerical representation (`StandardScaler(with_mean=False)`) stacked with categorical sparse CSR.
  - **Decision Tree:** Unscaled raw numerical values stacked with categorical sparse CSR, preserving physical flow interpretability.
- **Strict Invariant:** Inference never calls `fit()`, `fit_transform()`, or `partial_fit()`. Feature counts and column orders are verified against `output_features` before calling model estimators.

---

## 5. Input Validation, Leakage Auditing & Ground-Truth Protection

### Ground-Truth Target Label Protection
Incoming network traffic CSV files may contain a target column (e.g. `Label`, `label`, `attack_cat`).
- The validator extracts the column into a separate `ground_truth_series`.
- The column is dropped from the features DataFrame before any imputer or transformer executes.
- In output tables, it is displayed as `"Dataset Ground Truth"`, never as a model prediction.

### Data Leakage & Source Overlap Detection
The loader inspects `metadata/source_files.json` from the model's CP-05 engineering run:
- If an input CSV's filename or SHA-256 matches a file in the training/validation/test split, `source_overlaps_training` is flagged as `True`.
- The run status is marked as `"Completed with Warnings"`, and a prominent warning is rendered:
  > *"Input dataset was part of the training/evaluation dataset engineering run. This traffic overlaps with training data and cannot be claimed as unseen operational traffic."*

---

## 6. Model-Specific Score Semantics

Model prediction scores are presented with semantic transparency:
- **Support Vector Machine (`LinearSVC`):**
  - Score type: `decision_margin`
  - Output: Signed geometric distance to the separating hyperplane. Positive values indicate class affinity.
  - *Disclaimer:* Margins are uncalibrated geometric distances, not probabilities.
- **Multilayer Perceptron (`MLPClassifier`):**
  - Score type: `uncalibrated_probabilities`
  - Output: Softmax distribution over output units.
  - *Disclaimer:* Softmax values are uncalibrated model heuristic scores, not verified likelihoods of an attack.
- **Decision Tree (`DecisionTreeClassifier`):**
  - Score type: `uncalibrated_probabilities`
  - Output: Fraction of training samples belonging to each class in the terminating leaf node.
  - *Disclaimer:* Empirical leaf frequencies, not calibrated intrusion probabilities.

---

## 7. Disagreement Handling & Multi-Model Consensus

When multiple models are selected:
- Each model classifies every row independently.
- The `DisagreementAnalyzer` records:
  - **Agreement:** All selected models output the exact same predicted class label.
  - **Disagreement:** At least one model diverges from the consensus.
  - **Incomplete:** One or more models failed or produced an unmapped result.
- **Scientific Interpretation Policy:**
  > *"Model disagreement indicates prediction uncertainty and warrants forensic analyst triage; it is NOT an automatic confirmation of an intrusion."*
- *Strict Prohibition:* The engine never resolves disagreements via majority voting or creates an artificial ensemble detector.

---

## 8. Row Accounting Invariant

Every input record is deterministically accounted for:
$$\text{Input Rows} = \text{Successfully Predicted Rows} + \text{Explicitly Rejected Rows} + \text{Explicitly Skipped Rows}$$
$$\text{Output Prediction Rows} = \text{Successfully Predicted Rows}$$
Any violation of this accounting equation halts persistence and raises a `ValueError`.

---

## 9. Output Artifact Structure & Provenance

Inference results are committed atomically to `data/analyses/<detection_run_id>/`:
```text
data/analyses/<detection_run_id>/
├── run_manifest.json          <- Master audit manifest with cryptographic checksums
├── predictions.csv            <- Row-by-row predicted classes, scores, and ground truth
└── model_comparison.csv      <- Multi-model comparative table (when >= 2 models selected)
```

Each run produces a collision-resistant unique identifier: `detect_run_<timestamp>_<token>`. Existing analysis runs are never overwritten.

---

## 10. Hardware Resource Management

Designed for stable CPU execution on the developer workstation (AMD Ryzen 5 5600H, 24 GB RAM, Windows 11):
- **Bounded Chunk Processing:** Bounded batch iterator reads CSV in chunks (default: 10,000 rows).
- **RAM Preflight Analysis:** `DetectionResourceManager` queries physical available RAM via Windows ctypes and scales down batch sizes if chunk memory exceeds 50% of free RAM.
- **Storage Headroom:** Requires at least 500 MB free disk space before initiating writes.
- **Sequential Model Execution:** Models run sequentially across chunks, bounding simultaneous memory usage.
- **Concurrency Lock:** Thread-safe re-entrancy lock prevents parallel conflicting inference jobs.

---

## 11. Known Limitations & Production Readiness

1. **Benchmark Telemetry Availability:** Production directories present clean, truthful empty states until a real dataset is uploaded by the user and processed through CP-04 to CP-07.
2. **Offline Batch Focus:** Real-time PCAP packet capture and active network interface sniffing are explicitly out of scope.
3. **Probability Calibration:** Output scores are uncalibrated margins and heuristics; isotonic or Platt calibration is reserved for future research iterations.

---

## 12. Verification & Test Coverage Summary

- **CP-08 Test Suite:** `tests/test_detection_engine.py` (25/25 tests passing).
- **Full Regression Suite:** 180/180 automated tests passing across the repository.
- **Zero Broken Features:** All previous checkpoints (CP-01 through CP-07) remain fully operational and verified.
