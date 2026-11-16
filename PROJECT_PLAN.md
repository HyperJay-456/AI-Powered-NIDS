# Project Execution Plan: AI-Powered Network Intrusion Detection System (NIDS)

This document formalizes the phased checkpoint roadmap for the design, implementation, validation, and presentation of the academic AI-Powered Network Intrusion Detection System.

---

## Checkpoint Overview

```
+---------------------------------------------------------------------------------------+
|  CP-01: Foundation & Modular Architecture (CURRENT)                                   |
|  - Workspace layout, centralized configuration, app.py, docs, Git ignore rules        |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-02: Data Pipeline & Preprocessing                                                 |
|  - CIC-IDS-2017 loader, cleaning (inf/NaN), feature scaling, train/test splitting      |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-03: Model Engineering & Training                                                  |
|  - Pipelines for SVM, MLP, and Decision Tree; hyperparameter tuning & serialization   |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-04: Rigorous Evaluation & Benchmarking                                            |
|  - Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, latency benchmarks   |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-05: Streamlit Interactive Dashboard & Live Inspection                             |
|  - Telemetry UI, model comparison dashboard, interactive flow classification shell    |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-06: Verification, Final Academic Audit & Delivery                                 |
|  - End-to-end regression testing, comprehensive documentation, final deliverables     |
+---------------------------------------------------------------------------------------+
```

---

## Detailed Checkpoint Specifications

### CP-01: Project Foundation & Architecture
- **Status:** **COMPLETED** (Pending Final User Acceptance)
- **Primary Objective:** Establish an enterprise-grade, reproducible project foundation, directory structure, centralized configuration, dependency manifests, and documentation baseline.
- **Key Deliverables:**
  - Workspace inspection and clean directory layout (`config/`, `src/`, `data/`, `models/`, `tests/`).
  - `config/settings.py` for centralized project configuration and path management.
  - `app.py` executable Streamlit entry point.
  - `requirements.txt` with locked version ranges for Streamlit, pandas, numpy, scikit-learn, joblib, and matplotlib/seaborn.
  - Comprehensive `.gitignore` protecting datasets, binaries, secrets, and virtual environments.
  - Professional `README.md`, `ARCHITECTURE.md`, and `PROJECT_PLAN.md`.
  - Initialized Git repository and local verification tests (`tests/test_config.py`).
- **Acceptance Criteria:**
  - All target folders and configuration modules exist and resolve absolute paths correctly.
  - Unit tests pass with zero errors.
  - `app.py` executes without syntax or runtime import errors.
  - No datasets, model binaries, fake statistics, or unauthorized algorithms exist in the repository.

---

### CP-02: Data Ingestion, Cleansing & Preprocessing (CIC-IDS-2017)
- **Status:** **PLANNED** (Blocked on Human Approval for CP-01)
- **Primary Objective:** Build an automated, memory-efficient data engineering pipeline for the Kaggle CIC-IDS-2017 dataset.
- **Key Deliverables:**
  - `src/data/loader.py`: Safe, chunked CSV ingestion for multi-gigabyte traffic flow logs.
  - `src/data/preprocessor.py`: Column sanitization, numeric parsing, handling positive/negative infinities (common in `Flow Bytes/s`), and missing value imputation.
  - `src/features/scaler.py`: Robust / Standard scaling fit strictly on training splits.
  - `src/features/encoder.py`: Multi-class and binary attack categorization mappings.
  - `src/data/split.py`: Stratified train-validation-test data partitioning.
  - `tests/test_data_pipeline.py`: Unit tests validating zero data leakage, schema conformity, and absence of NaNs/infs.
- **Acceptance Criteria:**
  - Data ingestion pipeline processes all 8 CIC-IDS-2017 traffic subsets cleanly.
  - Processed data outputs contain 0 NaNs and 0 infinite values.
  - Stratified split accurately maintains class proportions across rare attack families.
  - Preprocessing artifacts (fitted scalers and label encoders) serialized to disk.

---

### CP-03: Model Engineering & Training
- **Status:** **PLANNED** (Blocked on CP-02 completion)
- **Primary Objective:** Implement, train, tune, and serialize the three authorized machine learning models.
- **Authorized Models:**
  1. **Support Vector Machine (SVM):** Linear and RBF kernel classifiers with probability estimation.
  2. **Multilayer Perceptron (MLP):** Feedforward artificial neural network with early stopping.
  3. **Decision Tree (DT):** CART classifier with pre-pruned depth and leaf constraints.
- **Key Deliverables:**
  - `src/models/base.py`: Abstract base class enforcing uniform estimator interface (`fit`, `predict`, `predict_proba`, `save`, `load`).
  - `src/models/svm_model.py`: SVM implementation and cross-validation trainer.
  - `src/models/mlp_model.py`: Neural network implementation and convergence monitor.
  - `src/models/decision_tree.py`: Decision tree implementation with tree depth profiling.
  - `models/saved/`: Serialized models and training metadata saved via `joblib`.
  - `tests/test_models.py`: Unit tests asserting model instantiation, prediction shapes, and serialization round-trips.
- **Acceptance Criteria:**
  - Strictly 3 models implemented (no auxiliary or unauthorized models).
  - All models train to convergence without numerical instability.
  - Serialized model artifacts load and reproduce identical predictions.

---

### CP-04: Model Evaluation & Benchmarking
- **Status:** **PLANNED** (Blocked on CP-03 completion)
- **Primary Objective:** Conduct an empirical, academically rigorous evaluation of all three models on held-out test data.
- **Key Deliverables:**
  - `src/evaluation/metrics.py`: Calculation of Accuracy, Precision (weighted/macro), Recall (weighted/macro), F1-Score, and ROC-AUC.
  - `src/evaluation/visualizations.py`: Generation of Confusion Matrices, ROC curves, and Precision-Recall curves.
  - `src/evaluation/benchmarks.py`: Latency profiling (wall-clock inference time per 1,000 packets) and memory utilization profiling.
  - Comprehensive benchmarking report comparing SVM vs. MLP vs. Decision Tree.
- **Acceptance Criteria:**
  - Zero fabricated or mock statistics; all metrics calculated directly from model evaluation runs on test splits.
  - Multi-class confusion matrices generated and saved.
  - Clear trade-off analysis documented (accuracy vs. training time vs. line-rate inference latency).

---

### CP-05: Streamlit Interactive Dashboard & Live Inspection
- **Status:** **PLANNED** (Blocked on CP-04 completion)
- **Primary Objective:** Upgrade `app.py` into a dynamic, production-grade cybersecurity dashboard.
- **Key Deliverables:**
  - Interactive multi-page / tab navigation:
    1. Threat Intelligence & Executive Summary.
    2. Dataset Explorer & Distribution Analytics.
    3. Model Architectures & Hyperparameter Controls.
    4. Comparative Performance Leaderboard & Interactive Confusion Matrix.
    5. Real-Time Packet Classification Simulator (upload CSV / manual flow parameter input).
  - Visual charts rendered with Matplotlib / Seaborn / native Streamlit charts.
- **Acceptance Criteria:**
  - Dashboard runs seamlessly without exceptions via `streamlit run app.py`.
  - Live packet inspection executes real inference through serialized models from `models/saved/`.
  - Fast response time (< 500ms UI interaction latency).

---

### CP-06: Verification, Final Academic Audit & Delivery
- **Status:** **PLANNED** (Blocked on CP-05 completion)
- **Primary Objective:** Execute full end-to-end regression testing, code cleanup, documentation audit, and final project packaging.
- **Key Deliverables:**
  - Complete automated test suite running with high coverage (`pytest --cov=src`).
  - PEP 8 compliance and code formatting audit.
  - Final Academic Project Report detailing methodology, experimental results, and defense preparations.
  - Verified repository ready for archival and academic evaluation.
- **Acceptance Criteria:**
  - All automated tests pass 100%.
  - Zero unused dependencies or dead code.
  - Academic report matches all experimental findings exactly.

---

## Strict Implementation Guardrails

- **Zero Data Leakage:** Preprocessing transformations fit strictly on training partitions.
- **Algorithmic Boundary:** Exactly SVM, MLP, and Decision Tree. No gradient boosting, random forests, or unauthorized models.
- **Empirical Authenticity:** No mock predictions, placeholder results, or simulated metric outputs.
- **Checkpoint Gates:** Work on subsequent checkpoints commences **only** after explicit human review and approval.
