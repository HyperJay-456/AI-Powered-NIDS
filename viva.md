# CYBER-NIDS — Comprehensive Viva Voce Examination Guide
## AI-Powered Network Intrusion Detection System
**Project Version:** `v1.0.0 Production` | **Target Benchmark:** `CIC-IDS-2017` | **Architecture:** Local-First Modular Python

---

## 1. Executive Project Summary (30-Second Elevator Pitch)

> **"What is this project?"**
>
> *"This project is an **AI-Powered Network Intrusion Detection System (CYBER-NIDS)** that analyzes statistical network flow telemetry to detect, classify, and audit cyber attacks. It uses the benchmark **CIC-IDS-2017** dataset (~2.8M flows, 15 attack families) and evaluates **three foundational machine learning algorithms** representing distinct mathematical paradigms: **LinearSVC**, **MLPClassifier**, and **DecisionTreeClassifier**. 
>
> The system operates **100% offline (air-gapped)**, features strict **data leakage prevention** and **test-holdout governance**, maps detections to **MITRE ATT&CK v20.1**, provides a **human-in-the-loop analyst triage journal**, and generates **cryptographically verified digital evidence packages (ReportLab PDFs & SHA-256 ZIPs)**. The entire system is validated by **270 automated tests (0 regressions)**."*

---

## 2. Why Exactly These Three ML Algorithms?

By architectural mandate, classification is strictly restricted to **three complementary paradigms** to compare distinct learning mechanics without black-box ensemble bloat or cloud dependencies:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3 FOUNDATIONAL ML PARADIGMS                            │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ 1. Margin-Based Geometry │ 2. Neural Representation │ 3. Rule Partitioning  │
│    Support Vector Machine│    Multilayer Perceptron │    Decision Tree      │
│         (LinearSVC)      │       (MLPClassifier)    │ (DecisionTreeClassifier│
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

### 1. Support Vector Machine (`LinearSVC`) — *Geometric / Margin-Based*
- **Mathematical Principle:** Finds the optimal hyperplane that maximizes the geometric margin ($2/\|w\|$) between benign and malicious flow vectors in standardized continuous feature space.
- **Why it is used:** Highly effective in high-dimensional continuous spaces (~78 statistical flow features). Convex optimization guarantees a global optimum without local-minima traps.
- **Operational Strength:** Exceptional at separating volumetric flooding (DDoS/DoS) from normal traffic baselines; fast inference via linear dot products.

### 2. Multilayer Perceptron (`MLPClassifier`) — *Representation-Based / Neural*
- **Mathematical Principle:** Feedforward artificial neural network with non-linear activation functions ($f(x) = \text{ReLU}(Wx + b)$), optimized via backpropagation and Adam.
- **Why it is used:** Automatically captures complex, non-linear interactions and cross-correlations across subtle temporal flow metrics (e.g., inter-arrival times vs. packet length variances).
- **Operational Strength:** Detects sophisticated multi-stage intrusions (Botnets, Infiltration, Web Attacks) that evade linear hyperplanes.

### 3. Decision Tree (`DecisionTreeClassifier` / CART) — *Rule-Based / Partitioning*
- **Mathematical Principle:** Deterministic recursive binary splitting using Gini Impurity ($\text{Gini} = 1 - \sum p_i^2$) on physical flow attributes.
- **Why it is used:** Offers **100% white-box explainability**. Security analysts can trace the exact logical threshold path (e.g., `Destination Port == 80 AND Flow Duration < 500ms AND Packet Length Variance > 1200`).
- **Operational Strength:** Evaluates unscaled physical units directly; impervious to monotonic feature transformations.

### Why not an Ensemble (Random Forest / XGBoost / Stacking)?
- Ensembles obscure model explainability and bloat inference overhead. 
- Instead of hiding predictions behind an opaque average, CYBER-NIDS performs **Tri-Model Consensus & Disagreement Analysis**:
  - **Unanimous Agreement (3/3):** High confidence threat detection.
  - **Majority Consensus (2/3):** Elevated threat requiring review.
  - **Model Divergence (All 3 disagree):** Ambiguous boundary case routed to human analyst triage.

---

## 3. Dashboard Pages Walkthrough (All 8 Routes in Short)

```
[01. Overview] ──> [02. Traffic Analysis] ──> [03. Detection Center] ──> [04. Model Laboratory]
                                                                                │
[08. About]    <── [07. Reports & Evidence] <── [06. Prediction Explorer] <── [05. Threat Intelligence]
```

| Route | Page Title | Primary Function in 1-2 Sentences |
| :--- | :--- | :--- |
| **01. `overview`** | **Security Operations Center (SOC) Console** | Executive telemetry flight deck showing system health, dataset checklist, calibrated model statuses, and 1-click pipeline actions (`⚡ Seed Demo`, `🚀 Run Pipeline`, `🧹 Reset`). |
| **02. `traffic_analysis`** | **Traffic Ingestion & Telemetry Explorer** | Ingests validated flow-level CSV files in 50,000-row chunks; runs delimiter/schema validation and SHA-256 duplicate flow detection without live packet sniffing (no PCAP). |
| **03. `detection_center`** | **Intrusion Detection Center** | CSV batch inference engine (10,000 rows/chunk); runs feature alignment checks, executes all 3 models independently, and computes consensus vs. algorithm disagreement metrics. |
| **04. `model_laboratory`** | **Model Training & Evaluation Lab** | Sequential training orchestrator with RAM/disk preflights, training loss tracking, confusion matrices, ROC/PR curves, candidate freezing, and persistent one-time test evaluation locks. |
| **05. `threat_intelligence`**| **Threat Intelligence & Attack Context** | Pinned offline MITRE ATT&CK Enterprise v20.1 STIX catalogue; maps detected attack classes to adversary tactics/techniques and audits available network telemetry vs. missing host telemetry. |
| **06. `prediction_explorer`**| **Prediction Explorer & Forensic Triage** | Multi-attribute search and bounded pagination for classified flows; provides an analyst triage workbench saving dispositions (e.g., False Positive) to an append-only, hash-chained journal. |
| **07. `reports`** | **Reporting & Evidence Export** | Two-pass ReportLab PDF compiler ("Page X of Y") generating 4 report archetypes (Executive, Technical, Analyst, Consolidated), CSV formula injection mitigation, and SHA-256 evidence ZIP bundles. |
| **08. `about`** | **Theoretical Foundations & Specs** | Formal mathematical documentation of SVM, MLP, and CART separation mechanics, dataset characteristics, and software release specifications. |

---

## 4. High-Probability Viva Questions & Crisp Answers

### Category A: Data & Preprocessing

#### Q1: What dataset does this project use, and what is its structure?
**Answer:**
We use the **CIC-IDS-2017** benchmark corpus created by the Canadian Institute for Cybersecurity (UNB). It contains **over 2.8 million bidirectional network flows** collected across 5 business days, comprising **~78-80 statistical flow features** (packet lengths, inter-arrival times, flags, flow durations) extracted via CICFlowMeter, covering **15 attack families** plus normal Benign traffic.

#### Q2: Does this project capture live network packets (PCAP) or sniff traffic?
**Answer:**
**No.** The project is architected strictly around **validated flow-level CSV telemetry**. Live packet capture (PCAP) requires raw packet reassembly, root socket hooks, and captures sensitive payload bytes. Flow telemetry summarizes traffic into directional statistical moments, which is the enterprise industry standard (NetFlow/IPFIX) and avoids payload privacy concerns.

#### Q3: What is Data Leakage, and how did you prevent it?
**Answer:**
Data leakage occurs when information from the validation or test splits accidentally contaminates the training phase, leading to artificially inflated performance. We prevent it by:
1. Performing **stratified 70% Train / 15% Validation / 15% Test** splitting *first*.
2. Fitting all transformers (`MissingValuesImputer` with median, `StandardScaler`, `CategoricalEncoder`) **strictly on the 70% training partition**.
3. Applying only `.transform()` (never `.fit()`) to validation and test partitions.
4. Isolating cross-partition duplicate flows via `DuplicateIsolationPolicy`.

#### Q4: Why is class imbalance a major problem in NIDS, and how do you handle it?
**Answer:**
In realistic network traffic, Benign traffic accounts for over 80-90% of all flows, while attacks like Heartbleed, Infiltration, or Botnet are rare (<1%). If you optimize for raw accuracy, a dummy model that predicts 100% Benign will achieve 90% accuracy but 0% threat detection. We handle this via:
1. **Stratified splitting** to preserve class distributions across splits.
2. Setting `class_weight='balanced'` in LinearSVC.
3. Evaluating performance primarily on **Macro-F1** and **Weighted-F1** scores rather than raw accuracy.

---

### Category B: Machine Learning & Modeling

#### Q5: What is the difference between Macro-F1 and Weighted-F1?
**Answer:**
- **Macro-F1:** Calculates the F1-score independently for each class and takes an unweighted average. It treats all classes equally, penalizing models that fail on rare attack classes.
- **Weighted-F1:** Weights the F1-score of each class by its support (frequency in the dataset). It reflects overall volume performance.
- We report **both** to guarantee that rare intrusions are not masked by large benign volumes.

#### Q6: What is the "Test Holdout Lock" and "Candidate Freeze Gate"?
**Answer:**
In academic research, repeated evaluation and tuning against the test set leads to **data snooping** (overfitting to the test set). We implemented:
1. **Candidate Freeze Gate:** Before test evaluation, candidate model artifacts, hyperparameters, and their SHA-256 hashes are frozen and committed.
2. **Persistent Test Holdout Lock:** A disk-persisted cryptographic lock file (`test_evaluation_lock.json`). If anyone alters candidate models and attempts to re-evaluate the test split, the engine aborts execution.

#### Q7: Why scale features for LinearSVC and MLP, but not for Decision Tree?
**Answer:**
- **LinearSVC:** Calculates Euclidean distances to find the maximum margin. Features with large scales (e.g., Flow Duration in microseconds: $10^7$) will dominate features with small scales (e.g., Flag Counts: 0-1) without `StandardScaler`.
- **MLPClassifier:** Gradient descent optimization (Adam) requires well-scaled inputs to prevent exploding/vanishing gradients and ensure rapid convergence.
- **Decision Tree:** Splits are based on ranking/ordering of individual features ($\leq \theta$). Monotonic scaling does not change order; preserving raw units keeps the tree rules transparent and explainable.

#### Q8: How does the Disagreement Analyzer work in the Detection Center?
**Answer:**
When batch inference runs, all three models independently classify each flow. The engine computes:
- **Unanimous Agreement ($3/3$ match):** Complete consensus among margin, neural, and rule models.
- **Majority Consensus ($2/3$ match):** Two models agree; the dissenting model is logged.
- **Complete Divergence ($0/3$ match):** All three models predict different classes; flagged for forensic review.

---

### Category C: Cybersecurity, MITRE ATT&CK & Forensics

#### Q9: What is MITRE ATT&CK, and how is it integrated into your project?
**Answer:**
MITRE ATT&CK is the globally recognized cybersecurity knowledge base of adversary Tactics, Techniques, and Procedures (TTPs). Our application includes an **offline pinned STIX 2.1 catalogue (Enterprise v20.1)**. When an attack class is identified (e.g., PortScan, FTP-Patator, DoS Hulk), the system associates it with candidate ATT&CK techniques (e.g., `T1046: Network Service Discovery`, `T1110: Brute Force`), displaying detection rationales and defensive mitigations.

#### Q10: What is the "Evidence Evaluation" concept in your Threat Intelligence module?
**Answer:**
Flow telemetry only observes packet dynamics across the wire (IPs, ports, packet sizes). It **cannot see host-level internal events** (process trees, memory injections, registry modifications). Our Evidence Evaluator explicitly audits what network evidence is *present* versus what host evidence is *missing*, preventing false conclusions during forensic investigations.

#### Q11: How does the Prediction Explorer ensure audit trail integrity?
**Answer:**
Analyst reviews (e.g., labeling a flow as *False Positive* or *Potentially Malicious*) are appended to an **immutable review journal** (`review_events.jsonl`). Each event contains the timestamp, analyst ID, review disposition, and a **SHA-256 cryptographic hash of the previous event**, forming a tamper-evident audit chain.

---

### Category D: Security Engineering & System Defenses

#### Q12: What specific security vulnerabilities did you defend against in the software?
**Answer:**
We built defenses directly into our code:
1. **CSV Formula Injection (DDE):** When exporting CSVs, any cell beginning with `=`, `+`, `-`, or `@` is prepended with a single quote (`'`) to prevent command execution in Microsoft Excel.
2. **Path Traversal / Directory Escape:** All file upload and archive member paths are normalized and verified to remain strictly within workspace boundaries.
3. **Host IP Address Exposure:** IPv4 addresses in generated PDF reports and evidence archives are automatically masked (e.g., `192.168.1.xxx`) via configurable privacy redaction.
4. **Untrusted Deserialization:** Model loading strictly validates SHA-256 checksums from the registered manifest before executing `joblib.load()`.
5. **Network Exposure:** The Streamlit dashboard binds strictly to local loopback (`localhost:8501`) with zero remote telemetry or external cloud calls.

#### Q13: How is reporting handled, and what are the evidence bundles?
**Answer:**
Reports are compiled offline using **ReportLab** into professional multi-page PDFs using a two-pass `NumberedCanvas` ("Page X of Y"). The system exports four archetypes (Executive, Technical, Analyst, Consolidated) and packages them alongside sanitized CSVs and SHA-256 checksum manifests into a verifiable `.zip` evidence bundle.

---

### Category E: Software Engineering, Testing & Operations

#### Q14: How many automated tests exist, and what do they cover?
**Answer:**
The repository contains **270 automated tests** executed via `pytest`, achieving **100% pass rate (0 failures, 0 regressions)**. Test categories include:
- Structural CSV validation and duplicate detection tests.
- Leakage-free dataset splitting and preprocessor tests.
- Model training, convergence, and registry tests.
- Evaluation metrics, confusion matrix, and candidate freeze lock tests.
- Detection engine inference, batching, and disagreement tests.
- MITRE ATT&CK STIX parsing and mapping tests.
- Prediction explorer filtering and hash-chain journal tests.
- ReportLab PDF compilation, IPv4 masking, and formula injection tests.
- UI design tokens, navigation routes, and dashboard state tests.

#### Q15: How do you start, operate, and shut down the application?
**Answer:**
The project includes automated Windows batch utilities:
- **`init_demo.bat`**: Seeds 1,500 sample benchmark flows, fits preprocessors, trains the 3 models, and runs batch detection in ~2 seconds for instant demonstration.
- **`start.bat`**: Validates the Python 3.13 `.venv`, checks port 8501, boots the Streamlit server in a dedicated window, and launches the default browser.
- **`stop.bat`**: Detects and cleanly terminates the server process listening on port 8501.

---

## 5. Quick-Glance Formula Reference

| Concept | Mathematical Formulation | Practical Significance in NIDS |
| :--- | :--- | :--- |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures alarm reliability: *"Of all flagged intrusions, how many were real attacks?"* (Minimizes false alarms). |
| **Recall (Sensitivity)**| $\frac{TP}{TP + FN}$ | Measures threat detection coverage: *"Of all actual attacks, how many did we catch?"* (Minimizes missed intrusions). |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing alarm precision against intrusion coverage. |
| **Gini Impurity (CART)**| $1 - \sum_{i=1}^C p_i^2$ | Metric used by Decision Tree to choose the split that creates the purest subsets of traffic. |
| **SVM Margin Objective**| $\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum \xi_i$ | Finds maximum-margin hyperplane balancing separation margin against classification slack penalties $\xi_i$. |
| **Standard Normalization**| $z = \frac{x - \mu}{\sigma}$ | Standardizes features to mean 0, variance 1; fitted strictly on training partition $\mu_{\text{train}}, \sigma_{\text{train}}$. |

---

## 6. Golden Rules for the Viva Presentation

1. **Be Confident About the Invariants:** Never say *"we could add Random Forest or a neural voting ensemble."* Explain that **strict architectural constraints** (exactly 3 distinct paradigms) allow scientific comparison without black-box opacity.
2. **Emphasize Leakage Prevention:** Teachers love asking about data leakage. Explicitly mention that **transformers were fitted strictly on the 70% training split**, and that test evaluation is guarded by a **persistent cryptographic lock**.
3. **Know Your Dataset:** When asked about input data, say **"CIC-IDS-2017 bidirectional network flows extracted via CICFlowMeter, covering 15 attack families across ~78 statistical features."**
4. **Highlight Verifiable Integrity:** Emphasize that all outputs (models, evaluation reports, evidence archives) are anchored by **SHA-256 cryptographic checksums** and **hash-chained review logs**.
5. **Emphasize Offline-First:** Point out that the system requires **zero cloud APIs, zero external telemetry, and zero live root packet capture**, making it safe for air-gapped enterprise networks.
