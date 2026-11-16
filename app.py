"""AI-Powered Network Intrusion Detection System (NIDS) - Streamlit Dashboard.

This is the primary presentation layer for the NIDS platform. It provides an
interactive interface for monitoring architecture, configuring preprocessing,
inspecting ML models (SVM, MLP, Decision Tree), and visualizing evaluation metrics.
"""

from pathlib import Path
import sys

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from config.settings import (
    ALGORITHM_KEYS,
    SUPPORTED_ALGORITHMS,
    settings,
)


def render_sidebar() -> str:
    """Render the sidebar controls and navigation menu.

    Returns:
        str: Selected navigation view.
    """
    st.sidebar.image(
        "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/shield-halved.svg",
        width=64,
    )
    st.sidebar.title("NIDS Control Center")
    st.sidebar.caption("CIC-IDS-2017 Threat Detection")
    st.sidebar.markdown("---")

    navigation_options = [
        "1. Project Overview & Architecture",
        "2. Dataset & Preprocessing Pipeline",
        "3. Model Configuration & Parameters",
        "4. Evaluation & Benchmarking",
        "5. Live Traffic Inspection Shell",
    ]

    selected_view = st.sidebar.radio(
        "Navigation",
        options=navigation_options,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    st.sidebar.info(
        f"**Environment:** Foundation\n\n"
        f"**Active Milestone:** CP-01\n\n"
        f"**Target Dataset:** {settings.dataset.name}\n\n"
        f"**Algorithms:** SVM, MLP, DT"
    )

    return selected_view


def view_overview() -> None:
    """Render Project Overview and Architectural blueprint view."""
    st.header("Project Overview & Architecture", divider="blue")
    st.markdown(
        """
        The **AI-Powered Network Intrusion Detection System (NIDS)** is an advanced machine learning
        framework designed to detect, classify, and mitigate malicious network anomalies in high-throughput
        network environments using the benchmark **CIC-IDS-2017** dataset.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Target Dataset", value=settings.dataset.name)
    with col2:
        st.metric(label="Core Algorithms", value=f"{len(SUPPORTED_ALGORITHMS)} Models")
    with col3:
        st.metric(label="Current Stage", value="CP-01 Foundation")

    st.subheader("System Architecture & Data Flow", divider="gray")
    st.markdown(
        """
        ```
        +-----------------------------------------------------------------------------------+
        |                               Presentation Layer                                  |
        |                       (Streamlit Interactive Web Dashboard)                       |
        +-----------------------------------------+-----------------------------------------+
                                                  |
        +-----------------------------------------v-----------------------------------------+
        |                          Inference & Analytics Engine                             |
        |      (Multi-class Threat Classifier: Benign vs. Multi-attack Signature Families)  |
        +-----------------------------------------+-----------------------------------------+
                                                  |
        +-----------------------------------------v-----------------------------------------+
        |                          Model Pipeline Subsystem                                 |
        |     +-------------------------+---------------------+-----------------------+     |
        |     | Support Vector Machine  | Multilayer          | Decision Tree         |     |
        |     | (Linear / RBF SVC)      | Perceptron (MLP-NN) | (CART Classifier)     |     |
        |     +-------------------------+---------------------+-----------------------+     |
        +-----------------------------------------+-----------------------------------------+
                                                  |
        +-----------------------------------------v-----------------------------------------+
        |                        Feature Engineering & Cleansing                            |
        |       (Handling Infinity/NaNs, Robust/MinMax Scaling, Label Normalization)        |
        +-----------------------------------------+-----------------------------------------+
                                                  |
        +-----------------------------------------v-----------------------------------------+
        |                              Data Ingestion Layer                                 |
        |           (CIC-IDS-2017 PCAP-Generated Flow Files - Canadian Cyber Center)         |
        +-----------------------------------------------------------------------------------+
        ```
        """
    )

    st.subheader("Environment Directory Health", divider="gray")
    paths_data = [
        {"Component": "Root Workspace", "Path": str(settings.base_dir), "Status": "Initialized"},
        {"Component": "Raw Data Store", "Path": str(settings.raw_data_dir), "Status": "Configured"},
        {"Component": "Processed Data Store", "Path": str(settings.processed_data_dir), "Status": "Configured"},
        {"Component": "Model Registry", "Path": str(settings.saved_models_dir), "Status": "Configured"},
        {"Component": "Runtime Logs", "Path": str(settings.logs_dir), "Status": "Configured"},
    ]
    st.table(paths_data)


def view_dataset_pipeline() -> None:
    """Render Dataset specifications and Preprocessing pipeline status."""
    st.header("Dataset & Preprocessing Pipeline", divider="blue")
    st.markdown(
        f"""
        This system targets the **{settings.dataset.name}** dataset, provided by the **{settings.dataset.source}**.
        The dataset captures realistic network traffic over 5 consecutive days, incorporating modern
        benign behaviors and known cyber-attack vectors.
        """
    )

    st.subheader("Benchmark Traffic Subsets", divider="gray")
    file_status = []
    for raw_file in settings.dataset.raw_files:
        path = settings.raw_data_dir / raw_file
        file_status.append(
            {
                "Target Flow File": raw_file,
                "Local Status": "Present" if path.exists() else "Awaiting Download (CP-02)",
                "Expected Path": str(path.relative_to(settings.base_dir)),
            }
        )
    st.table(file_status)

    st.subheader("Target Attack Taxonomies", divider="gray")
    st.write(
        "The CIC-IDS-2017 corpus contains the following verified threat categories:"
    )
    cols = st.columns(3)
    for idx, attack in enumerate(settings.dataset.known_attack_classes):
        with cols[idx % 3]:
            st.code(attack, language="text")

    st.info(
        "**Compliance Notice:** Dataset files are intentionally not downloaded in CP-01. "
        "The data ingestion, missing value imputation, infinity remediation, and train/test split "
        "pipelines will be implemented and validated during Checkpoint CP-02."
    )


def view_model_configuration() -> None:
    """Render Model architectures and baseline hyperparameters."""
    st.header("Model Configuration & Parameters", divider="blue")
    st.markdown(
        """
        The NIDS architecture is strictly standardized around three diverse machine learning algorithms,
        representing geometric margin classification, artificial neural networks, and hierarchical
        recursive partitioning.
        """
    )

    tab1, tab2, tab3 = st.tabs([
        "Support Vector Machine (SVM)",
        "Multilayer Perceptron (MLP)",
        "Decision Tree",
    ])

    with tab1:
        st.subheader("Support Vector Machine (SVM)")
        st.markdown(
            """
            - **Algorithm Category:** Maximum Margin Separating Hyperplane Classifier.
            - **Strengths:** Robust against high-dimensional feature spaces, effective generalization.
            - **Configured Parameters:**
            """
        )
        st.json(settings.models.svm_params)

    with tab2:
        st.subheader("Multilayer Perceptron (MLP)")
        st.markdown(
            """
            - **Algorithm Category:** Feedforward Deep Artificial Neural Network.
            - **Strengths:** Capable of modeling complex non-linear network traffic patterns.
            - **Configured Parameters:**
            """
        )
        st.json(settings.models.mlp_params)

    with tab3:
        st.subheader("Decision Tree")
        st.markdown(
            """
            - **Algorithm Category:** Recursive Binary Splitting / CART.
            - **Strengths:** Highly interpretable decision boundaries, deterministic low-latency inference.
            - **Configured Parameters:**
            """
        )
        st.json(settings.models.decision_tree_params)

    st.warning(
        "**Training Guardrail Active:** In adherence to Checkpoint CP-01 restrictions, model training "
        "and serialization will take place under Checkpoint CP-03 following dataset preprocessing."
    )


def view_evaluation_benchmarking() -> None:
    """Render Evaluation methodology and metrics blueprint."""
    st.header("Evaluation & Benchmarking", divider="blue")
    st.markdown(
        """
        To guarantee academic rigor, model performance will be evaluated using standard classification
        metrics with 5-fold stratified cross-validation and independent test partition validation.
        """
    )

    st.subheader("Target Evaluation Metrics", divider="gray")
    metric_cols = st.columns(3)
    for idx, metric in enumerate(settings.evaluation.metrics):
        with metric_cols[idx % 3]:
            st.markdown(f"- **{metric}**")

    st.subheader("Benchmarking Status", divider="gray")
    st.info(
        "No models have been trained yet. No fabricated statistics or mock metrics are presented. "
        "Empirical benchmarks will be populated upon completion of model training and validation in CP-04."
    )


def view_live_inspection() -> None:
    """Render Live Network Traffic Simulation shell."""
    st.header("Live Traffic Inspection Shell", divider="blue")
    st.markdown(
        """
        The live inspection interface provides an interactive simulation environment to feed single or
        batched network flow vectors into trained NIDS models for real-time intrusion classification.
        """
    )

    st.subheader("Engine Status: Idle", divider="gray")
    st.info(
        "**Status:** Model inference pipeline is uninitialized. "
        "Model binaries must be trained and registered in `models/saved/` before live prediction becomes active."
    )


def main() -> None:
    """Application entry point and router."""
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in the current Python environment.")
        print("Please install requirements using: pip install -r requirements.txt")
        print("Run the application using: streamlit run app.py")
        return

    # Ensure required runtime folders exist
    settings.ensure_directories()

    # Configure Streamlit page layout
    st.set_page_config(
        page_title=settings.ui.app_title,
        page_icon=settings.ui.app_icon,
        layout=settings.ui.layout,
        initial_sidebar_state=settings.ui.initial_sidebar_state,
    )

    selected_view = render_sidebar()

    if selected_view.startswith("1."):
        view_overview()
    elif selected_view.startswith("2."):
        view_dataset_pipeline()
    elif selected_view.startswith("3."):
        view_model_configuration()
    elif selected_view.startswith("4."):
        view_evaluation_benchmarking()
    elif selected_view.startswith("5."):
        view_live_inspection()


if __name__ == "__main__":
    main()
