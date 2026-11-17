"""Model Laboratory: ML Training Pipeline and Model Registry (CP-06).

Provides resource-aware, zero-leakage training workflows across the three authorized algorithms:
1. Support Vector Machine (LinearSVC)
2. Multilayer Perceptron (MLP with mini-batch incremental learning)
3. Decision Tree (CART on unscaled physical flow attributes)
"""

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from dashboard.components.badges import badge_html
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import error_panel, info_panel, success_panel, warning_panel
from src.data.engineering.artifact_manager import ArtifactManager
from src.models.artifact_loader import CP05ArtifactLoader
from src.models.model_config import (
    AuthorizedAlgorithm,
    DecisionTreeTrainingConfig,
    MasterTrainingConfiguration,
    MLPTrainingConfig,
    SVMTrainingConfig,
)
from src.models.model_registry import ModelRegistry
from src.models.resource_manager import PreflightCheckResult, ResourceManager
from src.models.training_orchestrator import TrainingOrchestrator
from src.models.training_progress import TrainingProgressTracker, TrainingStage


def _render_empty_runs_view() -> None:
    """Render empty state when no CP-05 engineering runs exist."""
    info_panel(
        title="No Preprocessed Datasets Detected",
        message=(
            "Model training requires a completed and validated Dataset Engineering run from Checkpoint CP-05. "
            "No preprocessed flow partitions were found in 'data/processed/'. "
            "Please navigate to Traffic Analysis and execute the Dataset Engineering pipeline first."
        ),
        icon="ℹ️",
    )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    col_nav, _ = st.columns([1, 2])
    with col_nav:
        if st.button("Go to Traffic Analysis / Preprocessing 📊", key="btn_goto_preprocessing"):
            from dashboard.navigation import set_active_route

            st.session_state["traffic_analysis_tab_index"] = 1
            set_active_route("traffic_analysis")
            st.rerun()

    section_header(
        title="Authorized Algorithm Architectures (Planned Baseline)",
        description="Standardized model baselines configured in config/settings.py ready for execution once data is prepared:",
    )

    tab_svm, tab_mlp, tab_dt = st.tabs([
        "Support Vector Machine (LinearSVC)",
        "Multilayer Perceptron (MLP)",
        "Decision Tree (CART)",
    ])

    with tab_svm:
        st.caption("Maximum-margin hyperplane classifier optimized for high-dimensional flow statistics (scaled input).")
        st.json(settings.models.svm_params)

    with tab_mlp:
        st.caption("Feedforward artificial neural network trained via mini-batch partial_fit with backpropagation (scaled input).")
        st.json(settings.models.mlp_params)

    with tab_dt:
        st.caption("Recursive binary partition tree preserving physical flow threshold interpretabiltiy (unscaled input).")
        st.json(settings.models.decision_tree_params)


def _render_preflight_panel(preflight: PreflightCheckResult, sys_mem: Dict[str, Any]) -> None:
    """Render resource telemetry and preflight safety verification."""
    avail_gb = sys_mem.get("available_ram_gb", 0.0)
    avail_color = (
        "var(--color-status-success)"
        if avail_gb >= 4.0
        else ("var(--color-status-warning)" if avail_gb >= 2.0 else "var(--color-status-error)")
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            label="Total System RAM",
            value=f"{sys_mem.get('total_gb', 23.34):.2f} GB",
            subtitle="Host physical memory",
            status_color="var(--color-text-primary)",
        )
    with c2:
        metric_card(
            label="Available RAM",
            value=f"{avail_gb:.2f} GB",
            subtitle=sys_mem.get("available_ram_formatted", "Available RAM"),
            status_color=avail_color,
        )
    with c3:
        metric_card(
            label="Estimated Training Memory",
            value=preflight.estimated_memory_formatted,
            subtitle="Working set estimate",
            status_color="var(--color-accent-primary)",
        )
    with c4:
        pf_status = "Passed" if preflight.is_safe else "Blocked"
        pf_status_color = (
            "var(--color-status-success)"
            if preflight.is_safe and not preflight.warning_messages
            else ("var(--color-status-warning)" if preflight.is_safe else "var(--color-status-error)")
        )
        metric_card(
            label="Preflight Gate",
            value=pf_status,
            subtitle="Resource Safety Check",
            status_color=pf_status_color,
        )

    if preflight.warning_messages:
        for w in preflight.warning_messages:
            warning_panel("Resource Preflight Warning", w)
    if not preflight.is_safe and preflight.rejection_reason:
        error_panel("Resource Preflight Blocked", preflight.rejection_reason)
    elif preflight.is_safe and not preflight.warning_messages:
        st.caption("✓ Resource safety margin verified. CPU execution strictly within bounds.")


def _render_loss_curve(loss_history: List[float]) -> None:
    """Render Plotly training loss curve for MLP."""
    if not loss_history:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(loss_history) + 1)),
            y=loss_history,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="#00D2FF", width=2.5),
            marker=dict(size=6, color="#00E5FF"),
        )
    )

    fig.update_layout(
        title="MLP Incremental Training Loss History",
        xaxis_title="Epoch",
        yaxis_title="Cross-Entropy / Log Loss",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94A3B8"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_training_tab(
    artifact_mgr: ArtifactManager,
    resource_mgr: ResourceManager,
    registry: ModelRegistry,
    runs: List[Any],
) -> None:
    """Render the Model Training Engine tab."""
    section_header(
        title="1. Source Preprocessed Dataset (CP-05)",
        description="Select a completed CP-05 dataset engineering run containing partitioned, clean training features:",
    )

    run_options = {
        r.run_id: f"{r.run_id} ({r.train_rows:,} train rows • {r.num_features_out} features • {r.num_classes} classes • {r.created_at[:19]})"
        for r in runs
    }
    run_ids = list(run_options.keys())

    default_idx = 0
    if "active_engineering_run_id" in st.session_state and st.session_state["active_engineering_run_id"] in run_ids:
        default_idx = run_ids.index(st.session_state["active_engineering_run_id"])

    selected_run_id = st.selectbox(
        "Available Engineering Runs",
        options=run_ids,
        format_func=lambda rid: run_options[rid],
        index=default_idx,
        key="sb_selected_eng_run",
    )

    selected_run = next((r for r in runs if r.run_id == selected_run_id), runs[0])

    # Verify run integrity
    loader = CP05ArtifactLoader(artifact_mgr=artifact_mgr)
    is_valid, validation_errors = loader.verify_engineering_run_integrity(selected_run_id)

    if not is_valid:
        error_panel(
            title="Selected Run Failed Integrity Verification",
            message="; ".join(validation_errors),
        )
        return

    # Display dataset partition metrics
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        metric_card(
            label="Training Samples",
            value=f"{selected_run.train_rows:,}",
            subtitle="Fit partition (Zero Leakage)",
            status_color="var(--color-status-success)",
        )
    with col_d2:
        metric_card(
            label="Feature Dimension",
            value=f"{selected_run.num_features_out} Columns",
            subtitle="Sparse CSR dual-scaled",
            status_color="var(--color-accent-primary)",
        )
    with col_d3:
        metric_card(
            label="Attack Classes",
            value=f"{selected_run.num_classes} Categories",
            subtitle=", ".join(selected_run.classes[:2]) + ("..." if len(selected_run.classes) > 2 else ""),
            status_color="var(--color-text-primary)",
        )
    with col_d4:
        metric_card(
            label="Held-Out Partitions",
            value=f"{selected_run.val_rows + selected_run.test_rows:,} Rows",
            subtitle="Untouched (Val + Test)",
            status_color="var(--color-status-neutral)",
        )

    st.caption("🛡️ **Zero Data Leakage Guarantee:** Only `datasets/train/X_*.npz` and `datasets/train/y.npy` will be loaded. Validation and test partitions remain strictly sealed on disk for CP-07 evaluation.")

    section_header(
        title="2. Algorithm Selection & Training Scope",
        description="Choose an authorized model to train individually, or train all three authorized algorithms sequentially:",
    )

    training_target = st.radio(
        "Select Target Training Routine",
        options=[
            "Support Vector Machine (LinearSVC)",
            "Multilayer Perceptron (MLP)",
            "Decision Tree (CART)",
            "All Three Models (Sequential Full Training Run)",
        ],
        index=0,
        horizontal=True,
        key="rb_training_target",
    )

    # Determine selected algorithms
    is_train_all = "All Three" in training_target
    target_algos: List[str] = (
        ["svm", "mlp", "decision_tree"]
        if is_train_all
        else (
            ["svm"]
            if "Support Vector" in training_target
            else (["mlp"] if "Multilayer" in training_target else ["decision_tree"])
        )
    )

    section_header(
        title="3. Hyperparameter Configuration",
        description="Tune hyperparameters for the selected algorithm(s) or use verified baseline defaults:",
    )

    svm_config = SVMTrainingConfig()
    mlp_config = MLPTrainingConfig()
    dt_config = DecisionTreeTrainingConfig()

    # Hyperparameter tuning tabs / expanders
    if is_train_all or "svm" in target_algos:
        with st.expander("⚙️ Support Vector Machine (LinearSVC) Hyperparameters", expanded=(not is_train_all and "svm" in target_algos)):
            c_svm1, c_svm2, c_svm3 = st.columns(3)
            with c_svm1:
                c_val = st.number_input("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="svm_c")
                loss_val = st.selectbox("Loss Function", options=["squared_hinge"], index=0, key="svm_loss")
            with c_svm2:
                max_iter_svm = st.number_input("Maximum Solver Iterations", min_value=500, max_value=20000, value=2000, step=500, key="svm_max_iter")
                dual_val = st.selectbox("Dual Formulation", options=["auto", False, True], index=0, key="svm_dual")
            with c_svm3:
                tol_val = st.number_input("Tolerance (tol)", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f", key="svm_tol")
                random_state_svm = st.number_input("Random Seed", value=42, step=1, key="svm_seed")

            svm_config = SVMTrainingConfig(
                C=float(c_val),
                loss=loss_val,
                max_iter=int(max_iter_svm),
                dual=dual_val,
                tol=float(tol_val),
                random_state=int(random_state_svm),
            )

    if is_train_all or "mlp" in target_algos:
        with st.expander("⚙️ Multilayer Perceptron (MLP) Hyperparameters", expanded=(not is_train_all and "mlp" in target_algos)):
            c_mlp1, c_mlp2, c_mlp3 = st.columns(3)
            with c_mlp1:
                hidden_arch = st.selectbox(
                    "Hidden Layer Architecture",
                    options=["(64, 32)", "(128, 64)", "(32, 16)", "(64,)", "(128, 64, 32)"],
                    index=0,
                    key="mlp_hidden",
                )
                hidden_tuple = eval(hidden_arch)
                activation_fn = st.selectbox("Activation Function", options=["relu", "tanh"], index=0, key="mlp_act")
            with c_mlp2:
                epochs_mlp = st.number_input("Training Epochs", min_value=1, max_value=50, value=10, step=1, key="mlp_epochs")
                batch_size_mlp = st.number_input("Mini-Batch Size", min_value=128, max_value=8192, value=1024, step=128, key="mlp_batch")
            with c_mlp3:
                lr_mlp = st.number_input("Learning Rate (init)", min_value=1e-5, max_value=0.1, value=0.001, format="%.5f", key="mlp_lr")
                alpha_mlp = st.number_input("L2 Regularization (alpha)", min_value=1e-6, max_value=1.0, value=0.0001, format="%.6f", key="mlp_alpha")

            mlp_config = MLPTrainingConfig(
                hidden_layer_sizes=hidden_tuple,
                activation=activation_fn,
                max_iter=int(epochs_mlp),
                batch_size=int(batch_size_mlp),
                learning_rate_init=float(lr_mlp),
                alpha=float(alpha_mlp),
                random_state=42,
            )

    if is_train_all or "decision_tree" in target_algos:
        with st.expander("⚙️ Decision Tree (CART) Hyperparameters", expanded=(not is_train_all and "decision_tree" in target_algos)):
            c_dt1, c_dt2, c_dt3 = st.columns(3)
            with c_dt1:
                criterion_dt = st.selectbox("Split Criterion", options=["gini", "entropy"], index=0, key="dt_criterion")
                max_depth_choice = st.selectbox("Max Depth Constraint", options=["15 (Recommended)", "10", "20", "30", "Unlimited (Unconstrained)"], index=0, key="dt_depth_choice")
                max_depth_val = None if "Unlimited" in max_depth_choice else int(max_depth_choice.split()[0])
            with c_dt2:
                min_split = st.number_input("Min Samples Split", min_value=2, max_value=100, value=10, step=1, key="dt_min_split")
                min_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=50, value=5, step=1, key="dt_min_leaf")
            with c_dt3:
                max_features_dt = st.selectbox("Max Features per Split", options=["sqrt (Recommended for high-dim)", "log2", "all (None)"], index=0, key="dt_max_features")
                max_feat_val = None if "all" in max_features_dt else ("sqrt" if "sqrt" in max_features_dt else "log2")
                random_state_dt = st.number_input("Random Seed (DT)", value=42, step=1, key="dt_seed")

            dt_config = DecisionTreeTrainingConfig(
                criterion=criterion_dt,
                max_depth=max_depth_val,
                min_samples_split=int(min_split),
                min_samples_leaf=int(min_leaf),
                max_features=max_feat_val,
                random_state=int(random_state_dt),
            )

    master_config = MasterTrainingConfiguration(
        svm_config=svm_config,
        mlp_config=mlp_config,
        decision_tree_config=dt_config,
    )

    section_header(
        title="4. System Resource Preflight Telemetry",
        description="Inspect real-time physical memory and algorithmic working set safety margins:",
    )

    # Perform preflight analysis for the selected configuration
    sys_mem = resource_mgr.get_system_memory_info()
    preflight = resource_mgr.preflight_check_dimensions(
        n_samples=selected_run.train_rows,
        n_features=selected_run.num_features_out,
        algorithms=target_algos,
        dt_max_depth=dt_config.max_depth,
    )

    _render_preflight_panel(preflight, sys_mem)

    section_header(
        title="5. Model Training Execution",
        description="Trigger training pipeline. Fitted models will be atomically serialized to models/saved/<run_id>/:",
    )

    col_btn, col_info = st.columns([1, 2])
    can_train = preflight.is_safe and not st.session_state.get("training_in_progress", False)

    with col_btn:
        start_btn = st.button(
            "🚀 Start Training Pipeline",
            type="primary",
            disabled=not can_train,
            use_container_width=True,
            key="btn_start_training",
        )

    with col_info:
        if not can_train and not preflight.is_safe:
            st.error("Training blocked: Critical memory deficit detected in preflight check.")
        elif st.session_state.get("training_in_progress", False):
            st.info("A training execution is actively running...")
        else:
            st.caption("Click to fit models on CPU using sparse CSR representations. Training runs in background with live stage telemetry.")

    # Execution handling
    if start_btn:
        st.session_state["training_in_progress"] = True
        st.session_state["training_last_error"] = None
        st.session_state["training_last_result"] = None

        orchestrator = TrainingOrchestrator(
            processed_dir=artifact_mgr.processed_dir,
            saved_models_dir=registry.saved_dir,
        )

        progress_bar = st.progress(0, text="Initializing training pipeline...")
        status_text = st.empty()
        loss_curve_placeholder = st.empty()
        loss_history_collector: List[float] = []

        def progress_callback(update: Any) -> None:
            pct = int(update.progress_percentage)
            progress_bar.progress(
                min(max(pct, 0), 100),
                text=f"[{update.algorithm.upper()}] {update.stage.value}: {update.message}",
            )
            status_text.markdown(
                f"**Stage:** `{update.stage.value}` • **Algorithm:** `{update.algorithm.upper()}` • **Elapsed:** `{update.elapsed_seconds:.1f}s`"
            )
            if update.loss is not None:
                loss_history_collector.append(update.loss)
                with loss_curve_placeholder:
                    _render_loss_curve(loss_history_collector)

        try:
            if is_train_all:
                res = orchestrator.train_all_models_sequentially(
                    engineering_run_id=selected_run_id,
                    master_config=master_config,
                    progress_callback=progress_callback,
                )
            else:
                algo_key = target_algos[0]
                selected_cfg = (
                    svm_config if algo_key == "svm" else (mlp_config if algo_key == "mlp" else dt_config)
                )
                res = orchestrator.train_single_model(
                    engineering_run_id=selected_run_id,
                    algorithm=algo_key,
                    config=selected_cfg,
                    progress_callback=progress_callback,
                )

            st.session_state["training_in_progress"] = False
            st.session_state["training_last_result"] = res
            progress_bar.progress(100, text="Training pipeline completed successfully!")
            st.success("✓ Training pipeline successfully completed and persisted to model registry!")
            st.rerun()

        except Exception as e:
            st.session_state["training_in_progress"] = False
            st.session_state["training_last_error"] = str(e)
            progress_bar.empty()
            st.error(f"Training failed: {e}")

    # Display results of last run if available
    last_res = st.session_state.get("training_last_result")
    if last_res:
        section_header(
            title="Latest Training Execution Diagnostics",
            description="Summary metrics and convergence telemetry from the completed run:",
        )
        # Normalize into a dict of manifests
        if isinstance(last_res, dict):
            manifest_map = last_res
        else:
            manifest_map = {last_res.algorithm: last_res}

        for a_key, a_man in manifest_map.items():
            with st.expander(f"Diagnostics: {a_man.algorithm_display_name} ({a_man.estimator_class})", expanded=True):
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                with r_c1:
                    metric_card("Duration", f"{a_man.training_duration_seconds:.2f}s", "Training time")
                with r_c2:
                    metric_card("Samples Fitted", f"{a_man.training_samples:,}", "Train partition")
                with r_c3:
                    metric_card("Iterations / Epochs", str(a_man.solver_iterations or "N/A"), "Convergence steps")
                with r_c4:
                    conv_color = (
                        "var(--color-status-success)"
                        if a_man.convergence_status == "Converged"
                        else "var(--color-status-warning)"
                    )
                    metric_card("Convergence", a_man.convergence_status, "Solver state", status_color=conv_color)

                loss_hist = a_man.diagnostics.get("loss_history") if a_man.diagnostics else None
                if loss_hist:
                    _render_loss_curve(loss_hist)

                if a_man.warnings:
                    for w in a_man.warnings:
                        warning_panel("Solver Warning", w)


def _render_registry_tab(registry: ModelRegistry) -> None:
    """Render the Model Registry and Saved Artifacts tab."""
    section_header(
        title="Persisted Training Sessions",
        description="All registered training sessions stored in models/saved/:",
    )

    runs = registry.list_training_runs()

    if not runs:
        info_panel(
            title="Model Registry is Empty",
            message="No trained models have been persisted yet. Execute a training run from the 'Model Training Engine' tab to generate verified model artifacts.",
            icon="📦",
        )
        return

    # Render training runs summary table
    run_table_data = []
    for r in runs:
        run_table_data.append({
            "Training Run ID": r["session_run_id"],
            "Source CP-05 Run": r["engineering_run_id"],
            "Status": r["overall_status"],
            "Trained Models": ", ".join(r["trained_algorithms"]).upper(),
            "Models Count": r["model_count"],
            "Completed At (UTC)": r["completed_at_utc"][:19] if r["completed_at_utc"] else "N/A",
        })
    st.dataframe(run_table_data, use_container_width=True)

    section_header(
        title="Registered Model Artifacts & Cryptographic Verification",
        description="Inspect individual model weights, file sizes, SHA-256 checksums, and metadata integrity:",
    )

    models = registry.list_all_trained_models()

    if not models:
        st.info("No individual model artifacts discovered.")
        return

    for m in models:
        is_intact = registry.verify_model_integrity(m.training_run_id, m.algorithm)
        intact_badge = badge_html("Verified (SHA-256 Valid)", variant="success") if is_intact else badge_html("Checksum Mismatch", variant="error")

        with st.expander(f"📦 {m.algorithm_display_name} — Run `{m.training_run_id}`"):
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                st.markdown(
                    f"""
                    <div class="nids-card" style="margin-bottom: 0;">
                        <div class="nids-card-title">Artifact Telemetry</div>
                        <div style="font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.8;">
                            <div><strong>Algorithm Key:</strong> <code>{m.algorithm}</code></div>
                            <div><strong>Estimator Class:</strong> <code>{m.estimator_class}</code></div>
                            <div><strong>Artifact Path:</strong> <code>{m.artifact_path}</code></div>
                            <div><strong>Model Size:</strong> {m.model_size_bytes / (1024*1024):.2f} MB ({m.model_size_bytes:,} bytes)</div>
                            <div><strong>Integrity Check:</strong> {intact_badge}</div>
                            <div><strong>SHA-256:</strong> <code style="font-size: 0.75rem;">{m.model_file_sha256}</code></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    f"""
                    <div class="nids-card" style="margin-bottom: 0;">
                        <div class="nids-card-title">Training Provenance</div>
                        <div style="font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.8;">
                            <div><strong>CP-05 Source Run:</strong> <code>{m.engineering_run_id}</code></div>
                            <div><strong>Training Samples:</strong> {m.training_samples:,} rows</div>
                            <div><strong>Feature Dimensions:</strong> {m.feature_count} features</div>
                            <div><strong>Classes:</strong> {m.class_count} categories ({", ".join(m.classes)})</div>
                            <div><strong>Training Time:</strong> {m.training_duration_seconds:.2f} seconds</div>
                            <div><strong>Convergence:</strong> {m.convergence_status}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Read metadata JSON if present
            meta_path = Path(m.artifact_path).parent / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
                with st.expander("📄 View Serialized Manifest (metadata.json)"):
                    st.json(meta_json)


def render() -> None:
    """Render the master Model Laboratory view."""
    page_header(
        title="Model Training Laboratory",
        subtitle="Resource-aware training pipeline, hyperparameter tuning & atomic model registry",
        icon="🧪",
        badges=[
            ("CP-06: Active", "success"),
            ("3 Authorized Algorithms", "info"),
            ("Train Only • Zero Leakage", "neutral"),
        ],
    )

    # Architectural Scope & Stage Guardrail Notice
    info_panel(
        title="CP-06 Scope & Evaluation Isolation Guardrail",
        message=(
            "This module executes model fitting, parameter tuning, resource estimation, and atomic serialization into models/saved/. "
            "In strict adherence to the project plan, 5-fold cross-validation, confusion matrices, ROC-AUC curves, and test split evaluation "
            "are strictly isolated and scheduled for Checkpoint CP-07 (Model Evaluation Laboratory). "
            "Validation and test partitions remain completely untouched."
        ),
        icon="🛡️",
    )

    # Initialize managers
    artifact_mgr = ArtifactManager()
    resource_mgr = ResourceManager()
    registry = ModelRegistry()

    # Discover CP-05 runs
    completed_runs = artifact_mgr.list_runs()

    if not completed_runs:
        _render_empty_runs_view()
        return

    # Render Tabs
    tab_training, tab_registry = st.tabs([
        "🚀 Model Training Engine",
        "📦 Model Registry & Saved Artifacts",
    ])

    with tab_training:
        _render_training_tab(
            artifact_mgr=artifact_mgr,
            resource_mgr=resource_mgr,
            registry=registry,
            runs=completed_runs,
        )

    with tab_registry:
        _render_registry_tab(registry=registry)
