# System Architecture: AI-Powered Network Intrusion Detection System (NIDS)

## 1. Executive Architectural Overview

The AI-Powered Network Intrusion Detection System (NIDS) is architected as a modular, layered machine learning pipeline paired with an interactive Streamlit presentation engine. Designed specifically for academic rigor and engineering reproducibility, the platform analyzes high-dimensional network flow records from the **CIC-IDS-2017** benchmark corpus to identify both broad anomalous traffic and specific intrusion signatures.

The platform strictly incorporates three foundational, complementary machine learning paradigms:
1. **Support Vector Machine (SVM):** Maximum-margin hyperplanes for geometric separation in high-dimensional feature spaces.
2. **Multilayer Perceptron (MLP):** Feedforward artificial neural networks capturing non-linear network traffic relationships.
3. **Decision Tree (DT):** Recursive partition trees providing rapid, interpretable, deterministic inference.

---

## 2. Layered Architecture Diagram

```
+----------------------------------------------------------------------------------------------------+
|                                    PRESENTATION LAYER (Streamlit)                                  |
|                                                                                                    |
|   +-----------------------+  +------------------------+  +--------------------+  +--------------+  |
|   | Overview & Blueprint  |  | Pipeline Status View   |  | Model Parameters   |  | Benchmarks   |  |
|   +-----------------------+  +------------------------+  +--------------------+  +--------------+  |
+-------------------------------------------------+--------------------------------------------------+
                                                  | API Calls / Model Loader
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    INFERENCE & EVALUATION LAYER                                    |
|                                                                                                    |
|   +---------------------------------------------+  +-------------------------------------------+   |
|   | src/evaluation/metrics.py                   |  | src/evaluation/benchmark.py               |   |
|   | - Accuracy, Precision, Recall, F1 (Weighted)|  | - Confusion Matrix & ROC-AUC curves       |   |
|   | - Cross-validation aggregator               |  | - Inference Latency & Resource Profiler   |   |
|   +---------------------------------------------+  +-------------------------------------------+   |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                     MODEL SUBSYSTEM LAYER                                          |
|                                                                                                    |
|      BaseModel Pipeline Interface (src/models/base.py)                                             |
|        ├── src/models/svm_model.py          -> Support Vector Machine (Linear / RBF SVC)           |
|        ├── src/models/mlp_model.py          -> Multilayer Perceptron (MLPClassifier)               |
|        └── src/models/decision_tree.py      -> Decision Tree (CART DecisionTreeClassifier)         |
|                                                                                                    |
|      Serialization Engine (joblib -> models/saved/*.joblib)                                        |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Clean Feature Tensors
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               FEATURE ENGINEERING LAYER (src/features)                             |
|                                                                                                    |
|   +------------------------------------+  +------------------------------------+                   |
|   | src/features/scaler.py             |  | src/features/encoder.py            |                   |
|   | - Robust / Standard Scaling        |  | - Label Encoding & Binarization    |                   |
|   | - Fit-on-train / Transform-on-test |  | - Class mapping (15 -> Multi-class)|                   |
|   +------------------------------------+  +------------------------------------+                   |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Validated Clean DataFrames
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                DATA INGESTION & CLEANSING (src/data)                               |
|                                                                                                    |
|   +------------------------------------+  +------------------------------------+                   |
|   | src/data/loader.py                 |  | src/data/preprocessor.py           |                   |
|   | - Stream/Chunked CSV Reader        |  | - Column strip & standard naming   |                   |
|   | - CIC-IDS-2017 File Aggregator     |  | - Infinite (inf/-inf) remediation  |                   |
|   | - Integrity & Schema Validation    |  | - Missing value (NaN) imputation   |                   |
|   +------------------------------------+  +------------------------------------+                   |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Read Raw CSVs
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                  PHYSICAL STORAGE / DATA ASSETS                                    |
|   ├── data/raw/          <- Kaggle CIC-IDS-2017 CSV flow archives (gitignored)                     |
|   ├── data/processed/    <- Parquet / Feather cleaned splits (gitignored)                          |
|   └── models/saved/      <- Serialized estimators, scalers, and label maps (gitignored)            |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. End-to-End Data Flow Lifecycle

```
[ CIC-IDS-2017 Raw CSVs ]
            │
            ▼
┌───────────────────────────────┐
│ 1. Ingestion & Sanitization   │  • Header stripping (" Label" -> "Label")
│    (src/data/loader.py)       │  • Validation of expected 78 flow features
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ 2. Preprocessing & Cleansing  │  • Replace inf / -inf with feature maximums or NaN
│    (src/data/preprocessor.py) │  • Impute or drop NaN flow records
│                               │  • Remove zero-variance / constant flow attributes
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ 3. Feature Transformation     │  • Fit Robust/StandardScaler on Training split only
│    (src/features/scaler.py)   │  • Transform Test/Validation partitions
│                               │  • Label encode Benign vs. Multi-attack taxonomy
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ 4. Stratified Partitioning    │  • 70% Train, 10% Validation, 20% Holdout Test
│    (src/data/split.py)        │  • Preserve class balance across rare attack vectors
└───────────┬───────────────────┘
            │
            ├─────────────────────────────────────────┬────────────────────────────────────────┐
            ▼                                         ▼                                        ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐        ┌───────────────────────────────┐
│ Support Vector Machine (SVM)  │         │ Multilayer Perceptron (MLP)   │        │ Decision Tree (CART)          │
│ • Kernel: RBF / Linear        │         │ • Architecture: (64, 32)      │        │ • Criterion: Gini / Entropy   │
│ • Hyperplane margin           │         │ • Activation: ReLU            │        │ • Max Depth: 15               │
│ • Regularization: C           │         │ • Solver: Adam (adaptive)     │        │ • Min Leaf: 5                 │
└───────────┬───────────────────┘         └───────────┬───────────────────┘        └───────────┬───────────────────┘
            │                                         │                                        │
            └─────────────────────────────────────────┼────────────────────────────────────────┘
                                                      │
                                                      ▼
                                  ┌───────────────────────────────────────┐
                                  │ 5. Model Serialization                │
                                  │    (joblib.dump -> models/saved/)     │
                                  └───────────────────┬───────────────────┘
                                                      │
                                                      ▼
                                  ┌───────────────────────────────────────┐
                                  │ 6. Evaluation & Benchmarking          │
                                  │    • Confusion Matrix & ROC-AUC       │
                                  │    • Precision / Recall / F1-Score    │
                                  │    • Empirical Inference Latency (ms) │
                                  └───────────────────┬───────────────────┘
                                                      │
                                                      ▼
                                  ┌───────────────────────────────────────┐
                                  │ 7. Streamlit Control Dashboard        │
                                  │    • Diagnostic Visualizations        │
                                  │    • Live Network Flow Inspection     │
                                  └───────────────────────────────────────┘
```

---

## 4. Component-by-Component Specifications

### 4.1 Configuration Core (`config/settings.py`)
- **Responsibility:** Serves as the single source of truth for runtime directories, algorithm hyperparameters, dataset paths, and UI constants.
- **Key Classes:** `AppConfig`, `DatasetConfig`, `ModelHyperparameters`, `EvaluationConfig`, `StreamlitConfig`.
- **Invariants:** Guaranteed absolute path resolution via `pathlib.Path`; frozen data classes preventing accidental runtime mutation.

### 4.2 Data Subsystem (`src/data`)
- **`loader.py`:** Handles robust parsing of CIC-IDS-2017 CSV files. Manages chunking to conserve memory when loading gigabyte-scale flow tables.
- **`preprocessor.py`:** Standardizes column naming (removes leading/trailing whitespaces), enforces numeric casting, detects and mitigates infinite values generated by flow rate calculations (e.g., `Flow Bytes/s`), and executes missing value imputation.
- **`split.py`:** Executes stratified train-validation-test partitioning to prevent data leakage.

### 4.3 Feature Subsystem (`src/features`)
- **`scaler.py`:** Fits statistical scalers strictly on training splits to prevent lookahead bias. Supports `StandardScaler` and `RobustScaler` (resistant to extreme network packet outliers).
- **`encoder.py`:** Encodes attack categories into integer labels and one-hot representations. Handles both binary detection (Benign vs. Malicious) and multi-class categorization (15 discrete attack categories).

### 4.4 Model Subsystem (`src/models`)
- **Unified Interface:** Each model wraps a scikit-learn estimator conforming to `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `save(path)`, and `load(path)`.
- **Approved Models:**
  1. `SVMModel`: Support Vector Classification with probability calibration.
  2. `MLPModel`: Multi-layer neural network with early stopping and adaptive learning.
  3. `DecisionTreeModel`: Recursive binary tree with pre-pruning (`max_depth`, `min_samples_leaf`) to mitigate overfitting.

### 4.5 Evaluation Subsystem (`src/evaluation`)
- **`metrics.py`:** Computes weighted and macro Precision, Recall, F1-Score, and multi-class ROC-AUC using One-vs-Rest (OvR).
- **`benchmarks.py`:** Measures wall-clock inference latency per sample and memory footprint to benchmark operational viability in line-rate intrusion detection environments.

### 4.6 Presentation Subsystem (`app.py`)
- Built with Streamlit to deliver responsive dashboarding.
- Provides modular views for system health, dataset diagnostics, model configurations, empirical benchmark tables, and live simulated packet classification.

---

## 5. Architectural Principles & Best Practices

1. **Separation of Concerns (SoC):** Data ingestion, feature transformations, model logic, and UI rendering reside in strictly separated packages.
2. **Deterministic Reproducibility:** Global random seeds (`random_state = 42`) are uniformly configured across dataset splitting, neural weight initialization, and decision tree splits.
3. **Data Leakage Prevention:** Scalers and encoders are strictly fitted on training splits, with transformations applied symmetrically to validation and test partitions.
4. **Defensive Programming:** Explicit verification of column schemas, infinite value remediation, and null value sanitization before any data reaches model estimators.
5. **No Mock Logic or Fabricated Statistics:** All views and metrics reflect genuine underlying model outputs or clearly state their awaiting stage.
