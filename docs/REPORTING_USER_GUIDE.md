# Cybersecurity Reporting & Digital Evidence User Guide

This user guide documents how to operate the **Reporting & Evidence Export** subsystem (Checkpoint CP-11) within the AI-Powered Network Intrusion Detection System (NIDS).

---

## 1. Overview & Core Capabilities

The Reporting & Evidence Export subsystem enables security practitioners, auditors, evaluators, and researchers to transform raw network traffic intrusion classifications into publication-grade reports and cryptographically verified digital evidence packages.

### Supported Report Archetypes
1. **Executive Security Summary:** High-level threat overview tailored for evaluators, instructors, and non-technical stakeholders. Highlights traffic volume, attack proportions, consensus distributions, and key defensive observations.
2. **Technical Detection Report:** In-depth forensic report containing per-class frequency tables, multi-algorithm comparative agreement (LinearSVC vs MLPClassifier vs DecisionTreeClassifier), and algorithm performance dimensions.
3. **Analyst Review Report:** Audit trail documenting human-in-the-loop analyst decisions, disposition breakdowns (Potentially Malicious, Likely Benign, False Positive, Needs Further Investigation, Inconclusive), and sanitized analyst notes.
4. **Consolidated Assessment Report:** Unified multi-section document combining executive, technical, analyst review, and ATT&CK threat context sections into a single comprehensive PDF report.

---

## 2. Navigating to the Reporting Console

1. Launch the NIDS dashboard application (`streamlit run app.py`).
2. Using the left-hand navigation sidebar, select **📑 Reports**.
3. The Reporting & Evidence Export page renders with Sections A through G.

---

## 3. Step-by-Step Reporting Workflow

### Step 1: Review Subsystem Overview (Section A)
- Inspect the top KPI cards: Total Reports, Executive Summaries, Technical Reports, Analyst Audits, and Evidence Bundles.
- If no reports have been generated, an informational panel guides you to begin.

### Step 2: Select a Source Detection Run (Section B)
- Under **Select Detection Run**, select a completed CP-08 intrusion detection run from the dropdown.
- The console displays:
  - Detection Run ID
  - Original Source Dataset Filename
  - Analyzed Network Flow Records
  - Selected Authorized Algorithms
- *Evaluation / Demonstration Mode:* If no production detection runs are currently stored, check **Include automated test fixtures** to inspect sample templates and test data.

### Step 3: Configure Report Format & Classification (Section C)
- **Report Archetype:** Choose from *Executive Security Summary*, *Technical Detection Report*, *Analyst Review Report*, or *Consolidated Assessment Report*.
- **Custom Report Title (Optional):** Enter a custom descriptive title (e.g., `Q4 Enterprise Perimeter Security Audit`).
- **Information Sensitivity Label:** Select the appropriate classification marking:
  - `Academic / Internal Research` (Default)
  - `TLP:CLEAR`
  - `TLP:GREEN`
  - `TLP:AMBER`

### Step 4: Configure Privacy Controls & Evidence Selection (Section D)
- **Redact Host IP Addresses (Recommended: Enabled):** Automatically masks the host octet of IPv4 addresses (e.g., `192.168.1.xxx` instead of `192.168.1.105`) across all PDF tables and exported CSVs.
- **Include Analyst Notes:** Toggle whether human analyst review comments appear in the report audit trail.
- **Package Evidence Bundle ZIP:** When checked, creates a self-contained, downloadable ZIP archive containing the PDF, JSON manifest, and sanitized CSV tables.

### Step 5: Generate Report (Section E)
- Click **🚀 Generate Cybersecurity Report**.
- The platform executes:
  1. Cryptographic preflight verification of CP-08 predictions.
  2. Forensic aggregation of predictions, CP-09 reviews, and CP-10 MITRE ATT&CK associations.
  3. PDF compilation with dynamic headers, footers, and page numbers.
  4. Structured machine-readable JSON and CSV exports with formula injection safeguards.
  5. Cryptographic evidence bundle ZIP packaging.
- Upon completion, a green confirmation banner confirms the Report ID, total pages, and SHA-256 checksum.

### Step 6: Preview & Download Artifacts (Section F)
- **Document Metadata Tab:** View Report ID, generation timestamp, detection run reference, and SHA-256 checksums.
- **Traffic & Predictions Tab:** Inspect tabular predicted class distributions and attack proportions.
- **Analyst Audit & ATT&CK Tab:** Review human review disposition counts and mapped MITRE ATT&CK techniques.
- **Download Artifacts Tab:**
  - Click **📥 Download Report PDF** to download the publication-grade document.
  - Click **📥 Download report_data.json** for machine-readable JSON integration.
  - Click **📦 Download Evidence ZIP** to download the full evidence bundle.
  - Expand **Download Sanitized Forensic CSV Tables** to download individual spreadsheets (`prediction_summary.csv`, `model_comparison.csv`, `analyst_review_summary.csv`, `attck_mapping_summary.csv`).

### Step 7: Historical Reports & Integrity Verification (Section G)
- Browse historical reports generated in earlier sessions.
- Click **🔍 Verify Integrity** on any report to verify that its PDF, JSON, and ZIP bundle checksums match the recorded cryptographic hashes.
- Download previously generated PDFs or ZIP archives directly without having to re-run generation.

---

## 4. Security & Compliance Safeguards

- **Strict Separation of Telemetry:** Model predictions $\neq$ Dataset Ground Truth $\neq$ Human Analyst Dispositions $\neq$ MITRE ATT&CK Mappings.
- **CSV Formula Injection Mitigation:** All exported CSV cells starting with formula-triggering characters (`=`, `+`, `-`, `@`) are prefixed with a single quote to prevent code execution in spreadsheet software.
- **Non-Attribution Disclaimers:** All reports print standard non-attribution disclaimers clarifying that machine learning classifications and MITRE ATT&CK techniques describe behavioral patterns and do not attribute activity to specific threat actors.
- **Local-First & Offline:** No data leaves the local workstation. No external cloud reporting APIs or third-party web services are called.
