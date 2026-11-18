# Project Execution Plan: AI-Powered Network Intrusion Detection System (NIDS)

This document formalizes the comprehensive 12-checkpoint execution roadmap for the design, implementation, validation, and production deployment of the academic AI-Powered Network Intrusion Detection System.

---

## 12-Checkpoint Executive Roadmap

```
+---------------------------------------------------------------------------------------+
|  CP-01: Project Foundation (COMPLETED)                                                |
|  - Workspace layout, centralized settings, documentation, app.py, Git initialization  |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-02: UI Design System                                                              |
|  - Cyber-themed design tokens, color palette, responsive typography, custom components |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-03: Application Dashboard                                                         |
|  - Executive dashboard, telemetry overview, pipeline status, system health monitoring |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-04: Traffic Upload & Data Explorer                                                |
|  - Validated CSV flow uploader, schema validator, dynamic filtering, telemetry        |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-05: Dataset Engineering                                                           |
|  - CIC-IDS-2017 ingestion, infinity/NaN remediation, feature scaling, stratified split|
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-06: ML Training Pipeline                                                          |
|  - Unified pipeline for SVM, MLP, and Decision Tree; hyperparameter tuning & saving   |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-07: Model Evaluation Laboratory                                                   |
|  - Cross-validation, Confusion Matrices, ROC-AUC, Precision, Recall, F1, Latency profile|
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-08: Intrusion Detection Engine                                                    |
|  - CSV-based batch detection engine, feature alignment, three-model predictions       |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-09: Threat Intelligence & Prediction Explorer                                     |
|  - Deep-dive prediction analysis, attack taxonomy mapping, feature importance/SHAP    |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-10: Reports & Exports                                                             |
|  - Automated PDF/CSV diagnostic reports, audit trails, benchmark export generator    |
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-11: Integration, Testing & Optimization                                           |
|  - End-to-end integration tests, latency optimization, memory profiling, test coverage|
+-------------------------------------------+-------------------------------------------+
                                            | Human Approval Gate
                                            v
+---------------------------------------------------------------------------------------+
|  CP-12: Deployment & Release                                                          |
|  - Containerization / release packaging, user guides, academic defense deliverables    |
+---------------------------------------------------------------------------------------+
```

---

## Detailed Checkpoint Specifications

### CP-01: Project Foundation
- **Status:** **COMPLETED** (Baseline Established)
- **Objective:** Establish the clean, modular Python repository architecture, centralized configuration management, dependency specifications, and version control rules.
- **Key Deliverables:**
  - Standard directory hierarchy: `config/`, `src/` (`data/`, `features/`, `models/`, `evaluation/`, `utils/`), `data/` (`raw/`, `processed/`), `models/saved/`, `tests/`.
  - [config/settings.py](file:///c:/Projects/ML%20Mini%20Project/config/settings.py) centralized configuration module.
  - [app.py](file:///c:/Projects/ML%20Mini%20Project/app.py) main Streamlit entry point.
  - [requirements.txt](file:///c:/Projects/ML%20Mini%20Project/requirements.txt) with locked dependency ranges.
  - [.gitignore](file:///c:/Projects/ML%20Mini%20Project/.gitignore) excluding datasets, models, virtual environments, and secrets.
  - [README.md](file:///c:/Projects/ML%20Mini%20Project/README.md), [ARCHITECTURE.md](file:///c:/Projects/ML%20Mini%20Project/ARCHITECTURE.md), and [PROJECT_PLAN.md](file:///c:/Projects/ML%20Mini%20Project/PROJECT_PLAN.md).
  - Git repository initialization and baseline verification tests.
- **Acceptance Criteria:**
  - Workspace structure conforms to enterprise Python standards.
  - `tests/test_config.py` passes with zero failures.
  - `app.py` executes cleanly without unhandled syntax or import errors.
  - No datasets, model binaries, mock logic, or unauthorized algorithms exist in Git.

---

### CP-02: UI Design System
- **Status:** **PLANNED** (Next Checkpoint — Awaiting Human Approval)
- **Objective:** Establish a cohesive, professional cybersecurity design system, styling architecture, and reusable UI components in Streamlit.
- **Key Deliverables:**
  - Custom CSS theme and design tokens (dark-mode cybersecurity palette, high-contrast threat indicators, typography).
  - Reusable Streamlit UI components: KPI metric cards, status callout banners, status pills, and data containers.
  - Consistent layout scaffolding across sidebar and main viewports.
- **Acceptance Criteria:**
  - Consistent styling across all viewports.
  - Clear visual hierarchy with zero layout shifts or broken responsive elements.

---

### CP-03: Application Dashboard
- **Status:** **PLANNED**
- **Objective:** Implement the central executive application dashboard presenting real-time system state, telemetry overview, and pipeline orchestration.
- **Key Deliverables:**
  - System telemetry overview: active environment, hardware utilization, dataset readiness, and model registry status.
  - Interactive pipeline status board mapping data and model progression.
  - Navigation controls linking into downstream exploration and inspection modules.
- **Acceptance Criteria:**
  - Dashboard loads instantaneously without mock data or fabricated metrics.
  - Accurately reports presence or absence of underlying dataset and model assets.

---

### CP-04: Traffic Upload & Data Explorer
- **Status:** **COMPLETED**
- **Objective:** Build an interactive network flow data exploration interface with validated CSV network-flow ingestion and dynamic tabular inspection.
- **Key Deliverables:**
  - Validated CSV network-flow ingestion engine supporting chunked streaming and schema validation.
  - Interactive DataFrame explorer with column filtering, search, and sorting.
  - Summary statistics panel: flow duration distributions, packet length aggregates, protocol breakdown.
- **Acceptance Criteria:**
  - Safely handles uploads up to configured size limits with informative error messages.
  - Validates flow schema against standard CIC-IDS-2017 feature definitions.

---

### CP-05: Dataset Engineering
- **Status:** **PLANNED**
- **Objective:** Build the automated, memory-efficient data engineering pipeline for the CIC-IDS-2017 benchmark dataset.
- **Key Deliverables:**
  - Memory-safe chunked loader for CIC-IDS-2017 CSV flow records.
  - Data sanitization pipeline: column whitespace stripping, numeric conversion, infinity (`inf` / `-inf`) replacement, and missing value imputation.
  - Feature transformation: `RobustScaler` / `StandardScaler` fitted strictly on training data.
  - Label encoding for both binary (Benign vs. Attack) and multi-class (15 attack categories) taxonomies.
  - Stratified train-validation-test split (70% train, 10% validation, 20% test).
- **Acceptance Criteria:**
  - Preprocessing generates zero NaNs and zero infinite values.
  - Zero data leakage between train and test partitions.
  - Serialized preprocessor artifacts saved in `models/saved/`.

---

### CP-06: ML Training Pipeline
- **Status:** **PLANNED**
- **Objective:** Implement, train, optimize, and serialize the three authorized machine learning models.
- **Authorized Models:**
  1. **Support Vector Machine (SVM):** Maximum-margin linear/RBF kernel classifier.
  2. **Multilayer Perceptron (MLP):** Feedforward artificial neural network with backpropagation and early stopping.
  3. **Decision Tree (CART):** Recursive partitioning tree with depth and leaf constraints.
- **Key Deliverables:**
  - Unified base estimator interface (`fit`, `predict`, `predict_proba`, `save`, `load`).
  - Model trainers with cross-validation and hyperparameter tuning.
  - Model serialization engine saving verified artifacts to `models/saved/`.
- **Acceptance Criteria:**
  - Strictly the 3 authorized algorithms implemented (no random forests, gradient boosting, or unapproved models).
  - All models train to convergence and serialize cleanly via `joblib`.

---

### CP-07: Model Evaluation Laboratory
- **Status:** **PLANNED**
- **Objective:** Conduct rigorous empirical evaluation and comparative benchmarking across all three trained models.
- **Key Deliverables:**
  - Calculation of classification metrics: Accuracy, Precision (weighted/macro), Recall (weighted/macro), F1-Score, and ROC-AUC (OvR).
  - Visualization artifacts: Confusion matrices, ROC curves, Precision-Recall curves.
  - Operational benchmark suite: Wall-clock inference latency (ms per 1,000 flows) and memory footprint.
- **Acceptance Criteria:**
  - 100% authentic metrics derived directly from held-out test data. Zero fabricated statistics.
  - Comparative benchmark table highlighting accuracy vs. throughput trade-offs.

---

### CP-08: Intrusion Detection Engine
- **Status:** **COMPLETED**
- **Objective:** Deliver the operational CSV-based batch detection engine, feature alignment, three-model predictions, and algorithm disagreement analysis.
- **Key Deliverables:**
  - Offline CSV-based batch detection engine applying saved preprocessing transforms and three-model estimators.
  - Feature alignment verification between ingested flows and trained pipeline expectations.
  - Independent three-model predictions (LinearSVC, MLPClassifier, DecisionTreeClassifier) and algorithm disagreement analysis.
- **Acceptance Criteria:**
  - Accurately classifies incoming flow vectors into Benign or specific attack classes.
  - Gracefully handles malformed or out-of-range flow vectors.

---

### CP-09: Threat Intelligence & Prediction Explorer
- **Status:** **PLANNED**
- **Objective:** Provide deep diagnostic interpretability and threat intelligence analytics for classified network flows.
- **Key Deliverables:**
  - Deep-dive prediction viewer detailing feature contributions and confidence distributions.
  - Attack taxonomy explorer mapping detected attacks to MITRE ATT&CK or CIC threat families.
  - Feature importance analysis highlighting primary flow discriminators (e.g., packet length variance, flag counts).
- **Acceptance Criteria:**
  - Enables security analysts to inspect why a specific flow was flagged as malicious.
  - Visualizes model confidence and probability calibration.

---

### CP-10: Reports & Exports
- **Status:** **PLANNED**
- **Objective:** Build automated report generation and audit trail export capabilities.
- **Key Deliverables:**
  - Diagnostic executive report generator (ReportLab PDF / JSON / CSV).
  - Forensic CSV/JSON export for classified intrusion events.
  - Model card and benchmark data export for academic auditing.
- **Acceptance Criteria:**
  - Reports export cleanly with timestamped telemetry, evaluation figures, and audit metadata.

---

### CP-11: Integration, Testing & Optimization
- **Status:** **PLANNED**
- **Objective:** Perform comprehensive end-to-end integration testing, profiling, and performance optimization.
- **Key Deliverables:**
  - Automated integration test suite validating the complete pipeline from CSV ingestion to Streamlit inference.
  - Memory and latency profiling report.
  - Performance optimizations (vectorized feature transforms, caching of model objects via `@st.cache_resource`).
- **Acceptance Criteria:**
  - Comprehensive test suite passing with high test coverage.
  - Zero memory leaks during repeated inference cycles.

---

### CP-12: Deployment & Release
- **Status:** **PLANNED**
- **Objective:** Finalize the project for academic presentation, peer review, and operational release.
- **Key Deliverables:**
  - Production deployment configuration (Docker / Streamlit Community Cloud readiness).
  - Comprehensive user and developer manual.
  - Academic defense presentation deck and demonstration script.
  - Final tagged Git release.
- **Acceptance Criteria:**
  - System installs and runs on a clean environment with single command: `streamlit run app.py`.
  - All academic project requirements, documentation, and defense deliverables complete.

---

## Strict Guardrails & Verification Gates

- **Zero Fabricated Logic:** All statistics, metrics, and classifications must derive strictly from empirical data and trained model estimators.
- **Algorithmic Constraint:** Only SVM, MLP, and Decision Tree are authorized.
- **Data Leakage Safeguards:** Preprocessing statistics are strictly computed on training partitions.
- **Checkpoint Progression Gate:** Advancing to any subsequent checkpoint requires explicit human sign-off.
