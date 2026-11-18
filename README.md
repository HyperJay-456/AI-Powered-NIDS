# AI-Powered Network Intrusion Detection System (NIDS)

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-green.svg)](docs/RELEASE_NOTES_v1.0.0.md)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset: CIC-IDS-2017](https://img.shields.io/badge/Dataset-CIC--IDS--2017-orange.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Tests: 268 Passing](https://img.shields.io/badge/tests-268%20passing-brightgreen.svg)](tests/)
[![License: Academic/MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic, production-grade Machine Learning system designed to ingest, validate, classify, and audit network traffic flow telemetry using the benchmark **CIC-IDS-2017** dataset. Built with an offline-first modular Python architecture and an interactive **Streamlit** control console, the platform implements, benchmarks, and operationalizes three authorized machine learning algorithms: **Support Vector Machine (LinearSVC)**, **Multilayer Perceptron (MLPClassifier)**, and **Decision Tree (DecisionTreeClassifier)**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Authorized ML Algorithms](#authorized-ml-algorithms)
- [Benchmark Dataset Specification](#benchmark-dataset-specification)
- [System Architecture & Capabilities](#system-architecture--capabilities)
- [Interactive Dashboard Views (8 Routes)](#interactive-dashboard-views-8-routes)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Project Roadmap (CP-01 through CP-12)](#project-roadmap-cp-01-through-cp-12)
- [Security, Privacy & Compliance Safeguards](#security-privacy--compliance-safeguards)
- [Academic Integrity & Ethical Disclaimer](#academic-integrity--ethical-disclaimer)

---

## Project Overview

Modern enterprise and cloud networks encounter an evolving spectrum of cyber threats, ranging from volumetric Distributed Denial of Service (DDoS) and aggressive Port Scanning to credential brute-forcing, Web Attacks, and lateral Infiltration. Traditional signature-based Intrusion Detection Systems (IDS) often struggle with emerging polymorphic patterns and zero-day variations.

This project implements a hybrid machine learning-powered Network Intrusion Detection System that bridges statistical learning theory with realistic network flow telemetry:
- **High-Fidelity Traffic Classification:** Identifies Benign baseline traffic vs specific cyber-attack families across multiclass dimensions.
- **Strict Invariant Governance:** Exactly three authorized ML algorithms representing distinct learning paradigms (margin-based, neural representation, recursive binary partitioning).
- **Zero Data Leakage:** Preprocessing transformations (imputation, scaling, encoding) fitted strictly on training partitions; final test partitions protected by one-time evaluation locks.
- **Human-in-the-Loop Analyst Triage:** Persistent cryptographic review journals documenting human validation, false positive tagging, and disposition audits.
- **MITRE ATT&CK Contextual Enrichment:** Pinned STIX 2.1 catalogue (Enterprise v20.1) associating detected attacks with candidate tactics, techniques, and defensive mitigations.
- **Publication-Grade Reporting:** Local PDF report generation ("Page X of Y"), formula-injection safe CSV exports, and cryptographically verified evidence ZIP packages.

---

## Authorized ML Algorithms

In strict adherence to the project's architectural invariants, classification is restricted to exactly three distinct algorithmic paradigms:

| Algorithm | Scikit-Learn Class | Learning Paradigm | Primary Operational Role |
|---|---|---|---|
| **Support Vector Machine (SVM)** | `LinearSVC` | Structural Risk Minimization | Geometric maximum-margin hyperplanes separating high-dimensional continuous flow statistics. |
| **Multilayer Perceptron (MLP)** | `MLPClassifier` | Feedforward Artificial Neural Network | Deep non-linear representation learning with adaptive mini-batch training and convergence stopping. |
| **Decision Tree (CART)** | `DecisionTreeClassifier` | Recursive Binary Partitioning | Interpretable, transparent hierarchical decision paths on physical flow attributes. |

*No 4th algorithm, voting ensemble, or external cloud inference service is permitted or used.*

---

## Benchmark Dataset Specification

The platform is designed around the **CIC-IDS-2017** benchmark corpus developed by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB):
- **Capture Volume:** Over 2.8 million flow records across 5 consecutive business days.
- **Feature Space:** ~78 statistical network flow features extracted via CICFlowMeter (flow durations, inter-arrival times, packet length moments, TCP flag counts, sub-flow rates).
- **Target Attack Families:**
  - **Benign:** Normal baseline operational traffic.
  - **DoS Attacks:** DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest.
  - **DDoS Attacks:** Volumetric denial-of-service floods.
  - **Reconnaissance:** PortScan probes.
  - **Credential Access:** FTP-Patator, SSH-Patator.
  - **Web Attacks:** SQL Injection, Cross-Site Scripting (XSS), Brute Force Web.
  - **Advanced Threats:** Botnet command-and-control, Infiltration, Heartbleed.

> **Data Storage Notice:** Raw CSV flow files reside in `data/raw/` and are strictly excluded from Git version control. Raw files remain completely immutable during ingestion.

---

## System Architecture & Capabilities

```text
[ Raw CSV Uploads ] ──> ( CSV Ingestion & Validation - CP-04 )
                                  │
                                  ▼
                     ( Dataset Engineering - CP-05 )
                     [Train / Validation / Test Splits]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  [ LinearSVC ]            [ MLPClassifier ]      [ DecisionTreeClassifier ]
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │ ( Model Training & Registry - CP-06 )
                                  ▼
                     ( Model Evaluation Lab - CP-07 )
                     [Validation Tuning & Test Gate]
                                  │
                                  ▼
                    ( Intrusion Detection Engine - CP-08 )
                    [Batch Multi-Model Inference & CSV Output]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
( Prediction Explorer - CP-09 )            ( Threat Intelligence - CP-10 )
[Analyst Review & Audit Trail]             [MITRE ATT&CK v20.1 Enrichment]
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  │
                                  ▼
                 ( Reporting & Evidence Export - CP-11 )
                 [PDF Reports, Sanitized CSVs & ZIP Bundles]
```

---

## Interactive Dashboard Views (8 Routes)

1. **Overview (`overview`):** Executive telemetry dashboard, system status checker, directory inventory, and benchmark checklist.
2. **Traffic Analysis (`traffic_analysis`):** Validated CSV network-flow ingestion, structural CSV validation, disk-backed duplicate detection, and schema explorer (no PCAP or live network capture).
3. **Detection Center (`detection_center`):** CSV-based batch detection engine supporting feature alignment, three-model predictions (LinearSVC, MLPClassifier, DecisionTreeClassifier), and algorithm disagreement analysis (no threat severity scoring or alert generation).
4. **Model Laboratory (`model_laboratory`):** Training orchestrator for SVM, MLP, and Decision Tree with resource preflights and CP-07 evaluation benchmarking.
5. **Threat Intelligence (`threat_intelligence`):** Pinned MITRE ATT&CK Enterprise v20.1 catalogue, dataset taxonomy, and candidate technique mapping review.
6. **Prediction Explorer (`prediction_explorer`):** Multi-attribute search, filtering, individual flow inspector, and human analyst review triage.
7. **Reports (`reports`):** Publication-grade ReportLab PDF generator (Executive, Technical, Analyst, Consolidated), structured JSON/CSV exports, and SHA-256-verified ZIP evidence bundles (no HTML reports or digital signatures).
8. **About (`about`):** Theoretical foundations, algorithmic justification, academic context, and release specifications.

---

## Installation and Setup

### Prerequisites
- Windows 11 (64-bit), Windows 10, or Linux (Ubuntu 22.04+)
- Python 3.13.14 (or Python 3.10+)
- Git 2.40+

### Setup Commands (PowerShell)
```powershell
# 1. Navigate to project root
cd "C:\Projects\ML Mini Project"

# 2. Create clean virtual environment
py -3.13 -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

Launch the local Streamlit dashboard:
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## Testing & Quality Assurance

Run the complete 268-test automated test suite:
```powershell
pytest tests/ -q
```
*Current Result: `268 passed, 49 warnings in 15.08s (0 failures, 0 regressions)`.*

---

## Project Roadmap (CP-01 through CP-12)

| Checkpoint | Scope | Status | Acceptance Gate |
| :--- | :--- | :--- | :--- |
| **CP-01** | Project Foundation | **Completed** | Clean modular architecture, centralized settings, Git versioning |
| **CP-02** | UI Design System | **Completed** | Streamlit design tokens, custom cybersecurity CSS, atomic components |
| **CP-03** | Application Dashboard | **Completed** | Executive overview, pipeline health checker, benchmark checklist |
| **CP-04** | Traffic Upload & Data Explorer | **Completed** | Validated CSV network-flow ingestion, structural validator, duplicate detection |
| **CP-05** | Dataset Engineering | **Completed** | Leakage-free train/val/test splitting, training-only imputation, scaling |
| **CP-06** | ML Training Pipeline | **Completed** | Resource-aware training for LinearSVC, MLP, and Decision Tree |
| **CP-07** | Model Evaluation Laboratory | **Completed** | Multiclass confusion matrices, ROC/PR curves, candidate freezing, test locks |
| **CP-08** | Intrusion Detection Engine | **Completed** | CSV-based batch detection engine, feature alignment, three-model predictions, disagreement analysis |
| **CP-09** | Prediction Explorer & Analyst Review | **Completed** | Multi-attribute search, flow inspector, append-only review journals |
| **CP-10** | Threat Intelligence & Attack Context | **Completed** | MITRE ATT&CK v20.1 STIX catalogue, candidate mappings, evidence context |
| **CP-11** | Reporting & Evidence Export | **Completed** | ReportLab PDF generation, structured JSON/CSV exports, SHA-256-verified ZIP evidence bundles |
| **CP-12** | Final Integration, Testing & Release | **Completed** | End-to-end integration audit, v1.0.0 release manifest, local release tag |

---

## Security, Privacy & Compliance Safeguards

- **Local-First & Offline:** 100% local execution. Zero network calls to cloud inference services or external APIs.
- **CSV Formula Injection Mitigation:** All spreadsheet downloads prefix formula triggers (`=`, `+`, `-`, `@`) with single quotes to prevent DDE code execution.
- **Host IP Redaction:** Automatic masking of IPv4 host octets (`192.168.1.xxx`) across PDF documents and evidence archives.
- **Cryptographic Provenance:** SHA-256 checksums verify the authenticity of all dataset manifests, model binaries, detection outputs, and report bundles.
- **Test-Holdout Governance:** Persistent test-evaluation locks prevent repeated holdout evaluations and data snooping.
- **Path-Traversal Defense:** Normalized path verification ensures zero directory escape across file loaders and ZIP members.

---

## Academic Integrity & Ethical Disclaimer

This software is developed strictly for educational, scientific, and defensive cybersecurity research. All network attack categories analyzed originate from sanitized, publicly released academic datasets (CIC-IDS-2017). The project does not contain tools for malicious exploitation, unauthorized penetration testing, or offensive cyber operations.
