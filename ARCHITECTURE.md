# System Architecture: AI-Powered Network Intrusion Detection System (NIDS)
## Version 1.0.0 Production Architecture Specification

---

## 1. Executive Architectural Overview & Core Invariants

The **AI-Powered Network Intrusion Detection System (NIDS)** is architected as an offline-first, modular Python platform coupled with an interactive Streamlit presentation layer. Designed for academic rigor and verifiable engineering reproducibility, the platform analyzes high-dimensional network flow records from the **CIC-IDS-2017** benchmark corpus to detect cyber intrusions, benchmark machine learning paradigms, and generate cryptographically verifiable evidence.

### Non-Negotiable Invariants
1. **Strictly Three ML Algorithms:** Classification is restricted strictly to:
   - **Support Vector Machine (LinearSVC):** Maximum-margin hyperplanes for geometric separation in high-dimensional feature spaces.
   - **Multilayer Perceptron (MLPClassifier):** Feedforward artificial neural networks with adaptive mini-batch iteration.
   - **Decision Tree (DecisionTreeClassifier):** Transparent, deterministic recursive binary splitting on physical flow attributes.
   *No fourth algorithm, voting ensemble, or external model is permitted or used.*
2. **Zero Data Leakage:** Preprocessing transformations (median imputation, standard scaling, label encoding) are fitted strictly on the training partition. Final test holdouts are protected by persistent evaluation locks.
3. **Zero Data Fabrication:** Unpopulated runtime directories display genuine empty states. Synthetic test fixtures are strictly isolated from production registries.
4. **Local-First Runtime:** The application executes entirely locally without cloud dependencies, external AI APIs, or remote data transmission.

---

## 2. Comprehensive Layered Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PRESENTATION LAYER (Streamlit)                            │
│  dashboard/                                                                            │
│  ├── theme.py (CSS Tokens)        ├── navigation.py (8 Routes)                          │
│  ├── components/ (Headers, MetricCards, ContentCards, Badges, Panels)                  │
│  └── pages/                                                                            │
│      ├── overview.py              ├── traffic_analysis.py      ├── detection_center.py  │
│      ├── model_laboratory.py      ├── threat_intelligence.py   ├── prediction_explorer.py│
│      ├── reports.py               └── about.py                                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Dispatches Actions / Reads Registries
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CORE APPLICATION & FORENSIC SUBSYSTEMS                          │
│                                                                                        │
│ ┌───────────────────────────┐  ┌───────────────────────────┐  ┌──────────────────────┐ │
│ │  REPORTING & EVIDENCE     │  │   THREAT INTELLIGENCE     │  │ PREDICTION EXPLORER  │ │
│ │  src/reporting/           │  │   src/threat_intelligence/│  │ src/explorer/        │ │
│ │  - PDFReportGenerator     │  │   - AttackCatalog (STIX)  │  │ - ExplorerRegistry   │ │
│ │  - StructuredExporter     │  │   - MappingRegistry       │  │ - PredictionReader   │ │
│ │  - EvidenceBundlePackager │  │   - ThreatContextService  │  │ - ReviewStore        │ │
│ │  - ReportRegistry         │  │   - ThreatInsightEngine   │  │ - RecordInspector    │ │
│ └─────────────▲─────────────┘  └─────────────▲─────────────┘  └──────────▲───────────┘ │
│               │                              │                           │             │
│               └──────────────────────────────┼───────────────────────────┘             │
│                                              │ Reads Run Manifest & Predictions        │
│                                              ▼                                         │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ INTRUSION DETECTION ENGINE (src/detection/)                                        │ │
│ │ - ModelEligibilityVerifier    - DetectionDatasetLoader   - DetectionRegistry       │ │
│ │ - DetectionInferencePipeline  - ScoreInterpreter         - DisagreementAnalyzer    │ │
│ └────────────────────────────────────────────▲───────────────────────────────────────┘ │
│                                              │ Reuses Models & Preprocessors           │
│                       ┌──────────────────────┴──────────────────────┐                  │
│                       ▼                                             ▼                  │
│ ┌───────────────────────────────────────────┐ ┌──────────────────────────────────────┐ │
│ │ MODEL TRAINING & REGISTRY                 │ │ MODEL EVALUATION LABORATORY          │ │
│ │ src/models/                               │ │ src/evaluation/                      │ │
│ │ - SVMTrainingEngine (LinearSVC)           │ │ - EvaluationOrchestrator             │ │
│ │ - MLPTrainingEngine (MLPClassifier)       │ │ - ConfusionMatrixEngine              │ │
│ │ - DecisionTreeTrainingEngine              │ │ - ROC & Precision-Recall Analyzers   │ │
│ │ - TrainingOrchestrator & ModelRegistry    │ │ - CandidateFreeze & TestHoldoutLock  │ │
│ └─────────────────────▲─────────────────────┘ └──────────────────▲───────────────────┘ │
│                       │ Train Split                              │ Val / Test Splits   │
│                       └──────────────────────┬───────────────────┘                     │
│                                              ▼                                         │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ DATASET ENGINEERING & PREPROCESSING (src/data/engineering/)                        │ │
│ │ - DatasetSplitter (70/15/15) - MissingValuesImputer     - NumericalScaler (Standard)│ │
│ │ - CategoricalEncoder         - DuplicateIsolationPolicy - ArtifactManager          │ │
│ └────────────────────────────────────────────▲───────────────────────────────────────┘ │
│                                              │ Validated Datasets                      │
│                                              ▼                                         │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ CSV INGESTION & DATA QUALITY (src/data/)                                           │ │
│ │ - CSVIngestionEngine (Chunked) - CSVValidator (Structural) - DiskBackedDuplicate   │ │
│ └────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Specifications & Contracts

### 3.1 Data Ingestion & Quality (`src/data/`)
- **Ingestion Pipeline:** Reads large CSVs (up to 500 MB per file) using bounded chunk streaming (`chunk_size=50_000`).
- **Structural Validation:** Analyzes delimiter consistency, non-ASCII characters, empty lines, and column schema compliance without loading entire files into memory.
- **Deduplication:** `DiskBackedDuplicateDetector` serializes records into length-prefixed bytes and tracks SHA-256 fingerprints to identify intra-file and cross-file duplicate flows.
- **Raw File Immutability:** Uploaded files are copied to `data/raw/` with read-only permissions and verified with SHA-256 checksums in `IngestionManifest`.

### 3.2 Dataset Engineering & Preprocessing (`src/data/engineering/`)
- **Partitioning:** Stratified splitting into 70% Training, 15% Validation, and 15% Test holdout partitions.
- **Leakage Prevention:** Numerical imputers (median) and scalers (`StandardScaler`) fit strictly on the training partition. The fitted transformers are persisted in `data/processed/` for downstream inference.
- **Duplicate Isolation:** Duplicate flow records spanning partition boundaries are isolated to prevent cross-partition leakage.
- **Feature Policy:** Preserves flow identifiers (Source/Destination IP, Timestamp) in metadata provenance while excluding them from model feature matrices.

### 3.3 Model Subsystem & Training (`src/models/`)
- **SVM Engine:** `LinearSVC` with dual optimization formulation, configurable margin regularization $C$, and balanced class weights.
- **MLP Engine:** `MLPClassifier` with mini-batch incremental learning (`partial_fit`), adaptive learning rate schedules, and convergence stopping.
- **Decision Tree Engine:** `DecisionTreeClassifier` utilizing unscaled physical flow attributes with Gini impurity / entropy criteria and maximum depth bounding.
- **Resource Preflight:** Enforces hardware safety checks before training, validating that minimum free RAM (512 MB) and disk space are available.

### 3.4 Model Evaluation Laboratory (`src/evaluation/`)
- **Multiclass Metrics:** Accuracy, Balanced Accuracy, Precision (Macro/Weighted), Recall, F1-Score, and Cohen's Kappa.
- **Visual Diagnostics:** Normalized multiclass confusion matrices, One-vs-Rest ROC curves, and Precision-Recall curves.
- **Candidate Freezing:** Candidate model artifacts and SHA-256 checksums are frozen prior to test-holdout evaluation.
- **Test Holdout Lock Governance:** A persistent file-based lock prevents rerunning test-holdout evaluation on altered candidate sets, stopping data snooping.

### 3.5 Intrusion Detection Engine (`src/detection/`)
- **Batch Multi-Model Inference:** Executes single-model or comparative multi-model evaluation across chunks of 10,000 rows.
- **Model Disagreement Analysis:** Computes unanimous agreement, majority consensus, and algorithm divergence without pseudo-ensembling.
- **Row Accounting:** Reconciles input rows against successfully classified, rejected, and skipped records.

### 3.6 Prediction Explorer & Analyst Review (`src/explorer/`)
- **Attribute Filtering:** Multi-column search, score thresholding, attack family filtering, and bounded pagination.
- **Human-in-the-Loop Triage:** Authorized dispositions: *Potentially Malicious*, *Likely Benign*, *False Positive*, *Needs Further Investigation*, *Inconclusive*.
- **Append-Only Journal:** Review events are appended to `data/analyst_reviews/<run_id>/review_events.jsonl` with SHA-256 event hash chaining.

### 3.7 Threat Intelligence & ATT&CK Context (`src/threat_intelligence/`)
- **STIX 2.1 Pinned Catalogue:** Official MITRE ATT&CK Enterprise v20.1 catalogue stored locally in `data/knowledge/mitre_attack/`.
- **Candidate Label Mappings:** Connects dataset attack labels to ATT&CK techniques with explicit status governance (`Candidate`, `Approved`, `Rejected`).
- **Evidence Evaluation:** Assesses available network flow telemetry (ports, packet lengths, flow rates) against missing forensic telemetry (host process trees, memory dumps).

### 3.8 Reporting & Evidence Export (`src/reporting/`)
- **PDF Generator:** ReportLab compiler with two-pass `NumberedCanvas` rendering publication-grade documents with headers, footers, and page numbers.
- **Four Archetypes:** Executive Security Summary, Technical Detection Report, Analyst Review Report, and Consolidated Assessment Report.
- **Structured Exports:** `report_data.json` and sanitized CSV tables (`prediction_summary.csv`, `model_comparison.csv`, etc.).
- **Security Protections:** CSV formula injection mitigation and IPv4 host address redaction (`192.168.1.xxx`).
- **Evidence Bundles:** Cryptographically verified `evidence_bundle_<report_id>.zip` packages with path-traversal safeguards.

---

## 4. Threat Boundary & Security Matrix

| Security Threat | Vulnerability Vector | Implemented Architectural Defense |
|---|---|---|
| **CSV Formula Injection** | Opening exported CSVs in Microsoft Excel or Calc | Prepending single quote (`'`) to cells starting with `=`, `+`, `-`, `@` |
| **Path Traversal / Escape** | Malicious directory names or archive member paths | Strict path normalization and boundary validation against root directories |
| **Data Snooping** | Repeated tuning against final test holdout | Persistent evaluation locks preventing re-evaluation on test split |
| **Untrusted Deserialization** | Loading arbitrary `.joblib` / `.pkl` binaries | Restricting loading strictly to trusted internal models with verified SHA-256 |
| **Privacy / Telemetry Leakage**| Sensitive IP addresses in exported reports | Automatic masking of IPv4 host octets (`192.168.1.xxx`) and note redactions |
| **Network Exposure** | Streamlit binding to public interfaces | Local loopback binding (`127.0.0.1` / `localhost`) without external cloud APIs |
