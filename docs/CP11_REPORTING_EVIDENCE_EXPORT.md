# Checkpoint CP-11 — Reporting & Evidence Export Subsystem Documentation

## 1. Architectural Overview & Design Philosophy

The **Reporting & Evidence Export** subsystem (Checkpoint CP-11) provides automated generation of publication-grade cybersecurity reports, structured machine-readable forensic exports, and cryptographically verified digital evidence packages.

The subsystem operates under a strict **read-only consumer** architecture:
- It consumes verified artifacts from Checkpoint CP-08 (Intrusion Detection Engine), CP-09 (Prediction Explorer & Analyst Review), and CP-10 (Threat Intelligence & Attack Context).
- It **never** trains or retrains machine learning models, executes new inference pipelines, or mutates any existing prediction records, analyst journals, or STIX reference bundles.
- It operates completely offline (local-first) with zero telemetry leakage and no cloud API dependencies.

```text
data/analyses/detect_run_*/          data/analyst_reviews/           data/knowledge/mappings/
   ├── run_manifest.json               └── review_events.jsonl         └── label_mappings.json
   └── predictions.csv                         │                               │
            │                                  │                               │
            └─────────────────────────┬────────┴───────────────────────────────┘
                                      │
                         [ ReportDataAggregator ]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [ PDFReportGenerator ]                [ StructuredExporter ]
                   │                                     │
         (report.pdf via ReportLab)            (report_data.json & CSVs)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                         [ EvidenceBundlePackager ]
                                      │
                     (evidence_bundle_<report_id>.zip)
                                      ▼
                        data/reports/<report_id>/
```

---

## 2. Supported Report Archetypes

| Report Archetype | Target Audience | Primary Focus | Content Components |
|---|---|---|---|
| **Executive Security Summary** | Evaluators, Instructors, Leadership | High-level risk posture & operational findings | Traffic totals, attack proportions, model consensus breakdown, key observations, governance limitations |
| **Technical Detection Report** | Cybersecurity Analysts, Engineers | Forensic flow classification & model comparison | In-depth per-class frequency tables, multi-algorithm comparison (LinearSVC vs MLP vs Decision Tree), flow telemetry |
| **Analyst Review Report** | SOC Leads, Quality Assurance Auditors | Human-in-the-loop validation & audit trail | Reviewed vs unreviewed stats, disposition breakdowns (Potentially Malicious, False Positive, etc.), review events log |
| **Consolidated Assessment Report** | Technical Reviewers, Formal Auditors | Unified end-to-end cybersecurity assessment | Multi-section compilation combining Executive, Technical, Analyst Review, and MITRE ATT&CK contextual sections |

---

## 3. Source Artifact Verification & Cryptographic Provenance

Before generating any report, the aggregator verifies the source detection run:
1. **Path-Traversal Defense:** Detection run IDs and report paths are normalized and checked to prevent directory escape attacks.
2. **Status Gate:** Runs must have a status of `Completed` or `Completed_with_warnings`. Failed or corrupted runs cannot be used for reports.
3. **Fixture Isolation:** Automated test fixture runs (`is_test_fixture: True`) are rejected by default from production reporting and only accessible when the user explicitly enables test fixtures in demonstration mode.
4. **Cryptographic Checksum Verification:** The SHA-256 hash of `predictions.csv` and `model_comparison.csv` must match the hashes recorded in `run_manifest.json`.

---

## 4. Deterministic Aggregation Rules & Metric Governance

The reporting subsystem enforces strict separation of security concepts:
- $\text{Model Predictions} \neq \text{Dataset Ground Truth} \neq \text{Analyst Dispositions} \neq \text{ATT\&CK Mappings}$.
- **Consensus Prediction Distribution:** Counts are derived from the unified prediction column or consensus votes across the three authorized algorithms (LinearSVC, MLPClassifier, DecisionTreeClassifier).
- **Analyst Review Summaries:** Computed from the append-only journal in `data/analyst_reviews/<run_id>/review_events.jsonl` using the latest disposition per row index. Historical edits are preserved in the chronological event history.
- **MITRE ATT&CK Context:** Mapped techniques maintain explicit status markings: `Candidate` vs `Approved` vs `Rejected`. Mappings are never represented as confirmed real-world security incidents.

---

## 5. Professional PDF Document Compilation

The PDF engine (`src/reporting/pdf_generator.py`) is built using **ReportLab** and incorporates:
- **Two-Pass `NumberedCanvas`:** Tracks total pages dynamically to render running headers and footers with exact `"Page X of Y"` numbering, Report ID, and timestamp.
- **Academic Typography & Palette:** Clean styling using Deep Slate (`#0f172a`), Primary Blue (`#2563eb`), Neutral Light (`#f8fafc`), and Border Slate (`#cbd5e1`).
- **Flowable Tables with Text Wrapping:** All table cells wrap strings in `Paragraph` elements to prevent horizontal clipping, supporting multi-page table flows.
- **Non-Attribution Disclaimers:** Every page footer prints the mandatory academic/defensive research notice:
  > *"ACADEMIC / DEFENSIVE USE ONLY — Model predictions do not constitute confirmed real-world security incidents. Non-attribution disclaimer applied."*
- **Post-Generation Verification:** Verifies file existence, tests document readability with `pypdf`, computes the SHA-256 checksum, and records page count and byte size.

---

## 6. Structured Exports & Digital Evidence Bundle

Each report generates a standardized directory structure under `data/reports/<report_id>/`:

```text
data/reports/<report_id>/
├── report.pdf
├── report_data.json
├── evidence_manifest.json
├── evidence_bundle_<report_id>.zip
└── exports/
    ├── prediction_summary.csv
    ├── model_comparison.csv
    ├── analyst_review_summary.csv
    └── attck_mapping_summary.csv
```

### CSV Formula Injection Safeguard
All exported CSV cells are sanitized against CSV Formula Injection (DDE). Any cell beginning with `=`, `+`, `-`, or `@` is prefixed with a single quote (`'`) to neutralize automatic spreadsheet formula execution while preserving the underlying text.

### Privacy Redaction
- **IPv4 Address Redaction:** Host octets are masked (e.g., `192.168.1.xxx` instead of `192.168.1.105`).
- **Analyst Notes Redaction:** Toggling off notes strips personal comments from exported summaries (`"[NOTE EXCLUDED PER PRIVACY POLICY]"`).
- Original source datasets and prediction CSVs remain strictly immutable.

### Evidence Bundle ZIP Archive
- Packages the PDF, JSON manifest, and sanitized CSV exports into a single ZIP archive.
- Implements strict path normalization preventing directory traversal (`..` or leading `/`).
- Includes a bounded sample of prediction flows with applied privacy redactions.
- Validates the resulting ZIP using `zipfile.ZipFile.testzip()`.

---

## 7. Report Registry & Historical Verification

The `ReportRegistry` (`src/reporting/report_registry.py`) provides persistent cataloging and on-demand verification:
- Discovers all historical reports in `data/reports/`.
- Verifies artifact existence and SHA-256 consistency.
- Enables direct downloading of previously generated PDFs, JSON files, CSV tables, and ZIP bundles without needing to regenerate them.

---

## 8. Verification & Test Coverage Matrix

The CP-11 test suite (`tests/reporting/test_reporting.py`) verifies:
- **Category A:** Source verification, checksum mismatch detection, fixture isolation, and path-traversal prevention.
- **Category B:** Deterministic report aggregation, class distribution counts, analyst review statistics.
- **Category C & D:** Generation of all 4 report archetypes, PDF readability via `pypdf`, page count verification, Unicode text handling, and long note wrapping.
- **Category E:** Structured exports, JSON serialization, and CSV formula injection protection.
- **Category F:** Evidence bundle ZIP creation, `testzip()` validation, and member path sanitization.
- **Category G:** Report registry discovery and cryptographic integrity verification.
- **Category H:** Privacy controls, IPv4 host redaction, and analyst note exclusion policies.
- **Category I:** Streamlit UI smoke testing (`dashboard/pages/reports.py`).
- **Category J:** Full regression verification against CP-01 through CP-10 baselines (263 / 263 tests passing).
