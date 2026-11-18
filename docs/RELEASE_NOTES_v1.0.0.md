# Release Notes — AI-Powered Network Intrusion Detection System (NIDS)
## Version 1.0.0 (Production Release)

**Release Date:** 2026-11-18  
**Target Architecture:** Local-First, Offline-Capable  
**Python Runtime:** Python 3.13.14 on Windows 11  
**Git Tag:** `v1.0.0`  
**Test Suite Status:** 268 Passed, 0 Failures, 0 Regressions  

---

## 1. Overview & Major Milestones

Version 1.0.0 marks the final production-ready milestone for the **AI-Powered Network Intrusion Detection System (NIDS)**. The project consolidates 12 engineering checkpoints into a unified, reliable, and auditable cybersecurity platform that analyzes network flow traffic, detects cyber-attack vectors, evaluates machine learning models, enriches findings with MITRE ATT&CK threat intelligence, and produces publication-grade evidence reports.

### The 12 Completed Checkpoints:
1. **CP-01 (Project Foundation):** Clean modular Python architecture, centralized settings, and Git ignore rules.
2. **CP-02 (UI Design System):** Custom cybersecurity design system, atomic UI components, and accessible color tokens.
3. **CP-03 (Application Dashboard):** High-level executive telemetry, pipeline status checker, and benchmark checklist.
4. **CP-04 (Traffic Upload & Data Explorer):** Validated CSV network-flow ingestion engine, structural validator, schema inspector, and disk-backed deduplication (no PCAP or live network capture).
5. **CP-05 (Dataset Engineering & Preprocessing):** Zero-leakage train/validation/test splitting, training-only imputation, feature engineering, and robust scaling.
6. **CP-06 (ML Training Pipeline & Model Registry):** Resource-aware training for the three authorized algorithms (LinearSVC, MLPClassifier, DecisionTreeClassifier) with persistent training manifests.
7. **CP-07 (Model Evaluation Laboratory):** Multiclass confusion matrices, ROC/PR curves, per-class attack metrics, candidate freezing, and test-holdout evaluation locks.
8. **CP-08 (Intrusion Detection Engine):** CSV-based batch detection engine, feature alignment verification, three-model predictions (LinearSVC, MLPClassifier, DecisionTreeClassifier), and algorithm disagreement analysis (no threat severity scoring or alert generation).
9. **CP-09 (Prediction Explorer & Analyst Review):** Bounded search, multi-attribute filtering, flow inspector, and human-in-the-loop review journals with hash-chained audit trails.
10. **CP-10 (Threat Intelligence & Attack Context):** Pinned MITRE ATT&CK Enterprise v20.1 STIX 2.1 catalogue, conservative candidate technique mappings, evidence evaluation, and mapping approval workflows.
11. **CP-11 (Reporting & Evidence Export):** Publication-grade ReportLab PDF generation ("Page X of Y"), structured JSON/CSV exports, CSV formula injection protection, IPv4 host redaction, and SHA-256-verified ZIP evidence bundles (no HTML reports or digital signature certificates).
12. **CP-12 (Final Integration, Testing & Release):** Full cross-checkpoint integration audit, performance and security verification, documentation finalization, and local `v1.0.0` release manifest.

---

## 2. Authorized Machine Learning Algorithms

In strict adherence to project constraints, classification is restricted to exactly three distinct algorithmic paradigms:

| Algorithm | Scikit-Learn Class | Primary Architectural Role |
|---|---|---|
| **Support Vector Machine (SVM)** | `LinearSVC` | Geometric maximum-margin classification on standardized continuous flow features. |
| **Multilayer Perceptron (MLP)** | `MLPClassifier` | Deep representation learning with adaptive mini-batch iteration and early stopping. |
| **Decision Tree (CART)** | `DecisionTreeClassifier` | Transparent, deterministic recursive binary splitting on physical flow attributes. |

*No 4th algorithm, voting ensemble, or external cloud inference service is permitted or used.*

---

## 3. Key Security & Privacy Safeguards

- **Local-First Runtime:** The application runs 100% locally on the user's workstation. No network flow data or telemetry is transmitted to external cloud APIs or remote servers.
- **CSV Formula Injection Mitigation:** All CSV downloads neutralize formula execution in spreadsheet software (Microsoft Excel, LibreOffice Calc) by prepending formula-triggering characters (`=`, `+`, `-`, `@`) with a single quote.
- **Privacy Controls:** Configurable masking of IPv4 host octets (`192.168.1.xxx`) and optional redaction of personal analyst notes.
- **Cryptographic Integrity:** SHA-256 checksums verify the integrity of all datasets, trained models, detection outputs, and report bundles.
- **Test-Holdout Governance:** One-time evaluation lock protects final test partitions against repeated testing and data snooping.
- **Path-Traversal Defense:** Normalization and boundary checking prevent directory traversal across all file operations.

---

## 4. System Requirements & Installation

- **Operating System:** Windows 11 (64-bit)
- **Processor:** AMD Ryzen 5 5600H (or equivalent x86-64 CPU)
- **Memory:** 24 GB RAM (minimum 8 GB recommended)
- **Python Version:** 3.13.14 (in `.venv`)

### Launch Command:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## 5. Automated Test Verification

- **Total Tests Discovered:** 268
- **Tests Passing:** 268 (100%)
- **Failures / Errors:** 0
- **Execution Time:** ~15.1 seconds
- **Regression Count:** 0
