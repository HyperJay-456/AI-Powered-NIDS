# Installation & First-Run Setup Guide — AI-Powered NIDS v1.0.0

This guide provides end-to-end instructions for deploying and verifying the **AI-Powered Network Intrusion Detection System (NIDS)** in a local environment.

---

## 1. System Requirements

### Hardware Requirements
- **Operating System:** Windows 11 (64-bit) / Windows 10 (64-bit) / Linux (Ubuntu 22.04+)
- **CPU:** AMD Ryzen 5 5600H (or equivalent multicore x86-64 CPU)
- **RAM:** Minimum 8 GB (24 GB recommended for large CIC-IDS-2017 batch files)
- **Disk Space:** Minimum 2 GB free disk space for dependencies and runtime datasets
- **GPU (Optional):** NVIDIA GeForce RTX 3050 Laptop GPU (all algorithms run on CPU by default)

### Software Prerequisites
- **Python:** Python 3.13.14 (or Python 3.10+)
- **Git:** Git 2.40+
- **Terminal:** PowerShell (Windows) or Bash (Linux / macOS)

---

## 2. Step-by-Step Installation

### Step 1: Open Terminal & Navigate to Project Directory
```powershell
cd "C:\Projects\ML Mini Project"
```

### Step 2: Create a Clean Virtual Environment
Ensure you are using Python 3.13:
```powershell
py -3.13 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks script execution, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 4: Upgrade Pip & Install Dependencies
Install all required runtime and development libraries:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verified dependencies installed:
- `streamlit>=1.35.0` (Interactive dashboard presentation shell)
- `numpy>=1.26.0`, `pandas>=2.1.0`, `scipy>=1.11.0` (Data manipulation)
- `scikit-learn>=1.4.0` (LinearSVC, MLPClassifier, DecisionTreeClassifier)
- `joblib>=1.3.0` (Deterministic model serialization)
- `plotly>=5.20.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0` (Data visualization)
- `reportlab>=5.0.0`, `pypdf>=5.0.0` (PDF generation & verification)
- `pytest>=8.0.0`, `pytest-cov>=5.0.0` (Automated testing)

### Step 5: Verify Installation with the Test Suite
Run the full automated test suite to confirm all 12 checkpoints pass:
```powershell
pytest tests/ -q
```
*Expected output: `268 passed, 49 warnings in ~15s`.*

---

## 3. Launching the Application

Start the Streamlit web dashboard:
```powershell
streamlit run app.py
```

The application will bind locally and automatically open in your default browser at:
```text
http://localhost:8501
```

---

## 4. First-Run Behavior & Genuine Empty States

When launching the application for the first time without uploaded traffic or pre-trained models, the application strictly presents **genuine empty states**:
- **Overview (01):** Displays real-time directory presence, indicating raw flow files are awaiting upload.
- **Traffic Analysis (02):** Shows empty file drop-zone ready to receive CIC-IDS-2017 flow CSVs.
- **Detection Center (03):** Shows "Awaiting Models" badge until models are trained in the Model Laboratory.
- **Model Laboratory (04):** Shows "No Preprocessed Datasets Detected" until Dataset Engineering is executed.
- **Threat Intelligence (05):** Fully operational offline catalogue indexing MITRE ATT&CK Enterprise v20.1 techniques.
- **Prediction Explorer (06):** Displays informational notice explaining prerequisite detection runs.
- **Reports (07):** Displays "No Production Reports Generated Yet" with controls to compile new reports.
- **About (08):** Outlines academic context, theoretical algorithm rationale, and release specifications.

---

## 5. Offline Operation & Security

- **Zero Cloud Dependence:** The entire application runs offline on your local loopback interface (`127.0.0.1` / `localhost`).
- **No External APIs:** No network connections are initiated to external cloud inference, report generation, or analytics providers.
- **Data Protection:** Raw network flow CSVs placed in `data/raw/` are immutable and strictly ignored by Git version control.
