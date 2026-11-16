# AI-Powered Network Intrusion Detection System (NIDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset: CIC-IDS-2017](https://img.shields.io/badge/Dataset-CIC--IDS--2017-orange.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License: Academic/MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic, production-grade Machine Learning system designed to identify, categorize, and mitigate malicious network activities using the benchmark **CIC-IDS-2017** dataset. Built with a modular Python architecture and an interactive **Streamlit** control dashboard, the project rigorously implements and benchmarks three foundational machine learning algorithms: **Support Vector Machine (SVM)**, **Multilayer Perceptron (MLP)**, and **Decision Tree (DT)**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Specification](#dataset-specification)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Repository Structure](#repository-structure)
- [System Architecture](#system-architecture)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Testing and Verification](#testing-and-verification)
- [Project Roadmap (Checkpoints)](#project-roadmap-checkpoints)
- [Academic Integrity & Ethical Disclaimer](#academic-integrity--ethical-disclaimer)

---

## Project Overview

Modern enterprise and cloud networks face an evolving spectrum of sophisticated cyber threats, ranging from Distributed Denial of Service (DDoS) and Port Scans to targeted Web Attacks and Infiltration. Traditional signature-based Intrusion Detection Systems (IDS) frequently fail to detect zero-day anomalies and polymorphic attacks.

This project implements an anomaly- and signature-hybrid AI-Powered Network Intrusion Detection System. It bridges statistical learning theory with real-time network flow telemetry, providing:
- High-fidelity traffic classification (Benign vs. specific cyber-attack families).
- Comparative algorithmic benchmarking across distinct algorithmic paradigms (margin-based, neural, tree-based).
- Interactive, non-fabricated Streamlit user interface for telemetry inspection and diagnostic model evaluation.

---

## Dataset Specification

The system is engineered around the **CIC-IDS-2017** dataset developed by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB).

### Key Characteristics
- **Network Traffic Capture:** Over 2.8 million flow records across 5 consecutive business days (Monday through Friday).
- **Features:** 78 statistical network traffic features extracted via CICFlowMeter (including flow duration, inter-arrival times, packet length distributions, flag counts, and sub-flow rates).
- **Target Attack Families:**
  - **Benign:** Normal operational baseline traffic.
  - **Denial of Service (DoS):** DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest.
  - **Distributed Denial of Service (DDoS):** High-volume volumetric floods.
  - **Reconnaissance:** PortScan.
  - **Brute Force:** FTP-Patator, SSH-Patator.
  - **Web Attacks:** SQL Injection, Cross-Site Scripting (XSS), Brute Force Web.
  - **Advanced Threats:** Botnet traffic, Infiltration, Heartbleed.

> **Data Storage Notice:** In compliance with repository standards, raw dataset CSVs are excluded from Git version control. Files must reside in `data/raw/` prior to running the ingestion pipeline in Checkpoint CP-02.

---

## Machine Learning Algorithms

The project restricts classification strictly to three distinct algorithmic paradigms:

1. **Support Vector Machine (SVM):**
   - *Paradigm:* Structural risk minimization via maximum-margin hyperplane separation.
   - *Role:* Evaluates margin-based boundaries in high-dimensional feature spaces using linear and radial basis function (RBF) kernels.
2. **Multilayer Perceptron (MLP):**
   - *Paradigm:* Multi-layer feedforward artificial neural network with backpropagation.
   - *Role:* Captures complex non-linear interactions across flow features through deep representation learning.
3. **Decision Tree (CART):**
   - *Paradigm:* Non-parametric recursive binary splitting based on Gini impurity / Shannon entropy.
   - *Role:* Provides deterministic, transparent, and ultra-low-latency rule-based decision paths suitable for line-rate network packet filtering.

---

## Repository Structure

```
ML Mini Project/
├── .env.example              # Sample environment variable definitions
├── .gitignore                # Comprehensive ignore rules (data, models, venv, secrets)
├── app.py                    # Main Streamlit dashboard entry point
├── requirements.txt          # Python project dependencies
├── README.md                 # Project documentation and user guide
├── ARCHITECTURE.md           # System architecture, data flow, and module contracts
├── PROJECT_PLAN.md           # Phased checkpoint roadmap (CP-01 through CP-06)
├── config/
│   ├── __init__.py           # Package exports for configuration
│   └── settings.py           # Centralized configuration (paths, dataset, models, UI)
├── data/
│   ├── raw/                  # Raw CIC-IDS-2017 CSV flow files (gitignored)
│   │   └── .gitkeep
│   └── processed/            # Cleaned, normalized, and split datasets (gitignored)
│       └── .gitkeep
├── models/
│   └── saved/                # Trained model artifacts and weights (gitignored)
│       └── .gitkeep
├── src/
│   ├── __init__.py           # Core source package
│   ├── data/                 # Ingestion, validation, and imputation
│   │   └── __init__.py
│   ├── features/             # Scaling, encoding, and feature selection
│   │   └── __init__.py
│   ├── models/               # SVM, MLP, and Decision Tree training pipelines
│   │   └── __init__.py
│   ├── evaluation/           # Metrics computation and visualization utilities
│   │   └── __init__.py
│   └── utils/                # Logging and system helpers
│       ├── __init__.py
│       └── logger.py
└── tests/
    ├── __init__.py
    └── test_config.py        # Configuration and structural regression tests
```

---

## System Architecture

The architecture enforces strict separation of concerns across five cohesive layers:
1. **Data Ingestion Layer (`src/data`):** Validates schema, handles infinite/NaN values, and standardizes column headers.
2. **Feature Engineering Layer (`src/features`):** Scales numerical features (Robust/StandardScaler) and encodes attack labels.
3. **Model Subsystem (`src/models`):** Encapsulates the three authorized algorithms (SVM, MLP, Decision Tree) behind a unified training and inference contract.
4. **Evaluation Engine (`src/evaluation`):** Computes cross-validated performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) without fabrication.
5. **Presentation Layer (`app.py`):** Interactive Streamlit dashboard providing intuitive system telemetry, model comparisons, and real-time inference simulations.

See [ARCHITECTURE.md](file:///c:/Projects/ML%20Mini%20Project/ARCHITECTURE.md) for full architectural specifications.

---

## Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git version control

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "ML Mini Project"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```

---

## Running the Application

Launch the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

The application will start and open automatically in your browser at `http://localhost:8501`.

---

## Testing and Verification

Run the automated test suite to ensure structural integrity and configuration validation:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or using `pytest`:

```bash
pytest tests/ -v
```

---

## Project Roadmap (Checkpoints)

| Checkpoint | Scope | Status |
| :--- | :--- | :--- |
| **CP-01** | **Project Foundation** | **Completed** |
| **CP-02** | UI Design System | Scheduled |
| **CP-03** | Application Dashboard | Scheduled |
| **CP-04** | Traffic Upload & Data Explorer | Scheduled |
| **CP-05** | Dataset Engineering | Scheduled |
| **CP-06** | ML Training Pipeline | Scheduled |
| **CP-07** | Model Evaluation Laboratory | Scheduled |
| **CP-08** | Intrusion Detection Engine | Scheduled |
| **CP-09** | Threat Intelligence & Prediction Explorer | Scheduled |
| **CP-10** | Reports & Exports | Scheduled |
| **CP-11** | Integration, Testing & Optimization | Scheduled |
| **CP-12** | Deployment & Release | Scheduled |

See [PROJECT_PLAN.md](file:///c:/Projects/ML%20Mini%20Project/PROJECT_PLAN.md) for detailed deliverables and verification gates.

---

## Academic Integrity & Ethical Disclaimer

This software is developed strictly for educational, scientific, and defensive cybersecurity research. All network attack categories analyzed originate from sanitized, publicly released academic datasets (CIC-IDS-2017). The project does not contain tools for malicious exploitation or unauthorized penetration testing.
