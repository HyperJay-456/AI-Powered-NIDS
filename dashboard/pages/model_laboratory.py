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
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from dashboard.components.badges import badge_html
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import error_panel, info_panel, success_panel, warning_panel
from src.data.engineering.artifact_manager import ArtifactManager
from src.evaluation import (
    ArtifactVerifier,
    ConfusionMatrixEngine,
    EvaluationOrchestrator,
    EvaluationRegistry,
    EvaluationResourceManager,
    ModelComparator,
    PrecisionRecallAnalyzer,
    ROCAnalyzer,
    SecurityMetricsEngine,
)
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


def _render_evaluation_tab(
    artifact_mgr: ArtifactManager,
    registry: ModelRegistry,
    completed_runs: List[Any],
) -> None:
    """Render the Model Evaluation Laboratory tab (CP-07)."""
    verifier = ArtifactVerifier(
        saved_models_dir=registry.saved_dir,
        processed_dir=artifact_mgr.processed_dir,
    )
    eval_registry = EvaluationRegistry(evaluations_dir=settings.evaluations_dir)
    orchestrator = EvaluationOrchestrator(
        saved_models_dir=registry.saved_dir,
        processed_dir=artifact_mgr.processed_dir,
        evaluations_dir=settings.evaluations_dir,
    )

    # Section A: Evaluation Readiness
    section_header(
        title="Section A — Evaluation Readiness & Asset Verification",
        description="Verifying CP-05 engineering run provenance, partition availability, and eligible production model artifacts:",
    )

    eligible_models = verifier.discover_eligible_production_models(allow_test_fixtures=False)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        metric_card(
            label="Eligible Production Models",
            value=f"{len(eligible_models)} Models" if eligible_models else "0 Models",
            subtitle="Verified SHA-256 & Schema",
            status_color="var(--color-status-success)" if eligible_models else "var(--color-status-warning)",
        )
    with col_r2:
        eng_run_id = eligible_models[0].engineering_run_id if eligible_models else (completed_runs[0].run_id if completed_runs else "None")
        metric_card(
            label="Source CP-05 Run",
            value=eng_run_id[:16] + "..." if len(eng_run_id) > 16 else eng_run_id,
            subtitle="Engineered partition provenance",
            status_color="var(--color-accent-primary)",
        )
    with col_r3:
        has_val = False
        if completed_runs:
            val_path = Path(completed_runs[0].artifact_directory) / "datasets" / "validation"
            has_val = val_path.exists() and (val_path / "X_scaled.npz").exists()
        metric_card(
            label="Validation Partition",
            value="Available" if has_val else "Missing",
            subtitle="Holdout for model selection",
            status_color="var(--color-status-success)" if has_val else "var(--color-status-error)",
        )
    with col_r4:
        has_test = False
        if completed_runs:
            test_path = Path(completed_runs[0].artifact_directory) / "datasets" / "test"
            has_test = test_path.exists() and (test_path / "X_scaled.npz").exists()
        metric_card(
            label="Test Holdout Partition",
            value="Available (Locked)" if has_test else "Missing",
            subtitle="Gated final holdout partition",
            status_color="var(--color-status-info)" if has_test else "var(--color-status-error)",
        )

    # If no eligible production models exist, show informative empty state and return
    if not eligible_models:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        info_panel(
            title="No Verified Production Models Available",
            message=(
                "The Model Evaluation Laboratory requires trained, verified production models in 'models/saved/'. "
                "No completed production models were detected. "
                "Automated test fixtures are strictly excluded from production evaluation. "
                "Please navigate to the 'Model Training Engine' tab to train and register SVM, MLP, or Decision Tree models first."
            ),
            icon="ℹ️",
        )
        return

    # Section B: Model Selection
    section_header(
        title="Section B — Model Selection for Evaluation",
        description="Select verified production models to benchmark and evaluate on held-out validation telemetry:",
    )

    available_algo_keys = [m.algorithm for m in eligible_models]
    algo_display_map = {
        "svm": "Support Vector Machine (LinearSVC)",
        "mlp": "Multilayer Perceptron (MLP)",
        "decision_tree": "Decision Tree (CART)",
    }

    selected_algos = st.multiselect(
        "Select Algorithms to Evaluate:",
        options=available_algo_keys,
        default=available_algo_keys,
        format_func=lambda k: algo_display_map.get(k, k.upper()),
        key="eval_selected_algorithms",
    )

    if not selected_algos:
        warning_panel("No Models Selected", "Please select at least one algorithm to proceed with evaluation.")
        return

    # Display selected model cards
    cols_m = st.columns(len(selected_algos))
    for idx, algo_k in enumerate(selected_algos):
        m_rec = next(m for m in eligible_models if m.algorithm == algo_k)
        rep_type = "Unscaled Sparse CSR" if algo_k == "decision_tree" else "Scaled Sparse CSR"
        with cols_m[idx]:
            st.markdown(
                f"""
                <div class="nids-card" style="margin-bottom: 0;">
                    <div class="nids-card-title">{algo_display_map.get(algo_k, algo_k.upper())}</div>
                    <div style="font-size: 0.8125rem; color: var(--color-text-secondary); line-height: 1.7;">
                        <div><strong>Run ID:</strong> <code>{m_rec.training_run_id[:18]}...</code></div>
                        <div><strong>Features:</strong> {m_rec.feature_count} attributes</div>
                        <div><strong>Classes:</strong> {m_rec.class_count} categories</div>
                        <div><strong>Input Rep:</strong> {rep_type}</div>
                        <div><strong>Integrity:</strong> {badge_html('SHA-256 Valid', variant='success')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Section C: Validation Evaluation
    section_header(
        title="Section C — Validation Evaluation (Model Selection & Diagnostics)",
        description="Evaluate models against the validation partition to measure classification accuracy, class-wise sensitivity, and false alarm rates:",
    )

    info_panel(
        title="Zero-Retraining Evaluation Protocol",
        message=(
            "Evaluation executes inference against the frozen weights of selected models on the validation partition. "
            "No fitting, refitting, or weight updates are performed. "
            "Results inform comparative benchmarking and candidate selection before final test gating."
        ),
        icon="⚖️",
    )

    # Session state for active evaluation manifest
    manifest_key = f"active_eval_manifest_{eligible_models[0].training_run_id}"
    active_manifest = st.session_state.get(manifest_key)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Run Validation Evaluation 🚀", key="btn_run_val_eval"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _progress_cb(processed: int, total: int, msg: str) -> None:
                frac = min(1.0, max(0.0, processed / total)) if total > 0 else 0.0
                progress_bar.progress(frac)
                status_text.caption(f"⚙️ {msg}")

            try:
                new_manifest = orchestrator.run_validation_evaluation(
                    training_run_id=eligible_models[0].training_run_id,
                    algorithms=selected_algos,
                    allow_test_fixtures=False,
                    progress_callback=_progress_cb,
                )
                st.session_state[manifest_key] = new_manifest
                active_manifest = new_manifest
                progress_bar.progress(1.0)
                status_text.empty()
                success_panel("Validation Complete", f"Successfully evaluated {len(selected_algos)} models on validation partition.")
                st.rerun()
            except Exception as e:
                error_panel("Evaluation Error", f"Failed to execute validation evaluation: {e}")

    # If no evaluation run yet in session state, check if registry has existing run for this training session
    if not active_manifest:
        past_runs = eval_registry.list_evaluation_runs()
        matching = [r for r in past_runs if r.get("training_session_id") == eligible_models[0].training_run_id]
        if matching:
            loaded_m = eval_registry.get_evaluation_manifest(matching[0]["evaluation_run_id"])
            if loaded_m and loaded_m.validation_results:
                active_manifest = loaded_m
                st.session_state[manifest_key] = loaded_m

    if not active_manifest or not active_manifest.validation_results:
        st.info("Validation evaluation has not been executed yet. Click 'Run Validation Evaluation' above to begin.")
        return

    # Display Validation Results
    val_results = active_manifest.validation_results

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Validation Aggregate Performance Comparison")
    val_df = ModelComparator.build_validation_comparison_dataframe(val_results)
    st.dataframe(val_df, use_container_width=True)

    # Detailed Algorithm Diagnostic Views (Tabs per algorithm)
    algo_tabs = st.tabs([algo_display_map.get(k, k.upper()) for k in val_results.keys()])
    for idx, (algo_k, a_res) in enumerate(val_results.items()):
        with algo_tabs[idx]:
            # Metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Accuracy", f"{a_res.aggregate_metrics.accuracy * 100:.2f}%", subtitle="Overall validation accuracy")
            with c2:
                metric_card("Macro F1-Score", f"{a_res.aggregate_metrics.macro_f1 * 100:.2f}%", subtitle="Unweighted attack mean")
            with c3:
                sec = a_res.security_metrics
                far_text = f"{sec.benign_false_alarm_rate * 100:.2f}%" if sec.is_available and sec.benign_false_alarm_rate is not None else "N/A"
                metric_card("False Alarm Rate", far_text, subtitle="Benign flows flagged as attack")
            with c4:
                rec_text = f"{sec.attack_detection_recall * 100:.2f}%" if sec.is_available and sec.attack_detection_recall is not None else "N/A"
                metric_card("Attack Detection Recall", rec_text, subtitle="Actual intrusions detected")

            # Row: Confusion Matrix + Security Summary
            col_cm, col_sec = st.columns([1.3, 1])
            with col_cm:
                st.markdown("##### Multiclass Confusion Matrix")
                norm_toggle = st.radio(
                    "Display Mode:",
                    options=["Raw Counts", "Row-Normalized (%)"],
                    horizontal=True,
                    key=f"cm_toggle_{algo_k}",
                )
                cm_fig = ConfusionMatrixEngine.create_plotly_heatmap(
                    a_res.confusion_matrix,
                    normalized=(norm_toggle == "Row-Normalized (%)"),
                    title=f"Confusion Matrix — {a_res.algorithm_display_name}",
                )
                st.plotly_chart(cm_fig, use_container_width=True)

            with col_sec:
                st.markdown("##### Security-Oriented Benign vs Attack Summary")
                if a_res.security_metrics.is_available:
                    sec_rec = a_res.security_metrics
                    st.markdown(
                        f"""
                        <div class="nids-card" style="margin-bottom: 0;">
                            <div class="nids-card-title">Intrusion Detection Telemetry</div>
                            <div style="font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.9;">
                                <div><strong>Benign Baseline Label:</strong> <code>{sec_rec.benign_label}</code></div>
                                <div><strong>Total Benign Flows:</strong> {sec_rec.benign_records_count:,}</div>
                                <div><strong>Total Attack Flows:</strong> {sec_rec.attack_records_count:,}</div>
                                <div><strong>Attack Detection Recall:</strong> {sec_rec.attack_detection_recall * 100:.2f}%</div>
                                <div><strong>Attack Precision:</strong> {sec_rec.attack_precision * 100:.2f}%</div>
                                <div><strong>Attack F1-Score:</strong> {sec_rec.attack_f1 * 100:.2f}%</div>
                                <div><strong>Benign False Alarm Rate (FAR):</strong> <span style="color: var(--color-status-warning); font-weight: 600;">{sec_rec.benign_false_alarm_rate * 100:.2f}%</span></div>
                                <div><strong>Benign Specificity:</strong> {sec_rec.benign_specificity * 100:.2f}%</div>
                                <div><strong>Attack Miss Rate (FNR):</strong> {sec_rec.attack_false_negative_rate * 100:.2f}%</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(f"Binary security view unavailable: {a_res.security_metrics.unavailable_reason}")

            # Row: ROC and Precision-Recall Curves
            col_roc, col_pr = st.columns([1, 1])
            with col_roc:
                st.markdown("##### One-vs-Rest ROC Curves")
                if a_res.score_type != "none":
                    roc_fig = ROCAnalyzer.create_plotly_roc_chart(
                        a_res.roc_analysis,
                        a_res.macro_roc_auc,
                        a_res.algorithm_display_name,
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
                    score_desc = "Signed Margin Distances (decision_function)" if a_res.score_type == "decision_function" else "Model Probabilities (predict_proba)"
                    st.caption(f"Score Metric: {score_desc}. Never labeled as calibrated probabilities.")
                else:
                    st.info("Continuous scores unavailable for ROC analysis.")

            with col_pr:
                st.markdown("##### One-vs-Rest Precision-Recall Curves")
                if a_res.score_type != "none":
                    pr_fig = PrecisionRecallAnalyzer.create_plotly_pr_chart(
                        a_res.pr_analysis,
                        a_res.macro_average_precision,
                        a_res.algorithm_display_name,
                    )
                    st.plotly_chart(pr_fig, use_container_width=True)
                    st.caption(f"Macro Average Precision: {a_res.macro_average_precision:.4f}" if a_res.macro_average_precision is not None else "Macro AP: N/A")
                else:
                    st.info("Continuous scores unavailable for PR analysis.")

            # Per-class table
            st.markdown("##### Class-Wise Diagnostic Breakdown")
            class_rows = []
            for m in a_res.per_class_metrics:
                class_rows.append({
                    "Class Index": m.class_index,
                    "Attack Category": m.class_name,
                    "Support": m.support,
                    "Predicted": m.predicted_count,
                    "Precision": f"{m.precision * 100:.2f}%",
                    "Recall": f"{m.recall * 100:.2f}%",
                    "F1-Score": f"{m.f1_score * 100:.2f}%",
                    "Specificity": f"{m.specificity * 100:.2f}%",
                    "False Pos Rate": f"{m.false_positive_rate * 100:.2f}%",
                    "False Neg Rate": f"{m.false_negative_rate * 100:.2f}%",
                    "Diagnostics": m.notes or "OK",
                })
            st.dataframe(pd.DataFrame(class_rows), use_container_width=True)

    # Class-wise multi-model comparison table
    if len(val_results) > 1:
        st.markdown("##### Multi-Model Class-Wise Recall Comparison")
        cw_df = ModelComparator.build_classwise_comparison_dataframe(val_results)
        st.dataframe(cw_df, use_container_width=True)

    # Section D: Candidate Selection & Freezing
    section_header(
        title="Section D — Validation-Based Candidate Selection & Freezing",
        description="Select and freeze candidate models based on validation performance before evaluating on final holdout test data:",
    )

    is_frozen = bool(active_manifest.frozen_candidates)
    test_completed = bool(active_manifest.test_results or active_manifest.test_holdout_locked)

    if is_frozen:
        primary_k = next((k for k, v in active_manifest.frozen_candidates.items() if v.selected_as_primary), "None")
        frozen_info = active_manifest.frozen_candidates.get(primary_k)
        lock_badge = "Permanently Locked (Test Evaluated)" if test_completed else "Frozen (Awaiting Test Gate)"
        success_panel(
            title=f"Candidate Models {lock_badge} 🔒",
            message=(
                f"Candidate models locked at {frozen_info.frozen_at_utc[:19] if frozen_info else 'N/A'} UTC. "
                f"Designated Primary Candidate: '{algo_display_map.get(primary_k, primary_k.upper())}'. "
                f"Selection Criterion: {frozen_info.selection_criterion if frozen_info else 'Standard'}. "
                "Candidate set and cryptographic checksums are immutable."
            ),
        )
    else:
        col_cand1, col_cand2 = st.columns([1, 2])
        with col_cand1:
            chosen_primary = st.selectbox(
                "Designate Primary Candidate Model:",
                options=list(val_results.keys()),
                format_func=lambda k: algo_display_map.get(k, k.upper()),
                key="sb_primary_candidate",
            )
        with col_cand2:
            criterion_text = st.text_input(
                "Documented Selection Criterion:",
                value="Highest validation macro F1-score with balanced recall on rare attacks and lowest false alarm rate",
                key="txt_selection_criterion",
            )

        if st.button("Freeze Candidate Models 🔒", key="btn_freeze_candidates"):
            try:
                updated_m = orchestrator.freeze_candidate_models(
                    evaluation_run_id=active_manifest.evaluation_run_id,
                    primary_candidate_algo=chosen_primary,
                    selection_criterion=criterion_text,
                )
                st.session_state[manifest_key] = updated_m
                active_manifest = updated_m
                st.success(f"Candidate models successfully frozen! Primary candidate: '{chosen_primary.upper()}'.")
                st.rerun()
            except Exception as e:
                error_panel("Freeze Error", f"Failed to freeze candidate models: {e}")

    # Section E: Gated Final Test Evaluation
    section_header(
        title="Section E — Gated Final Test Evaluation (Immutable Holdout)",
        description="Evaluate frozen candidate models strictly against the held-out test partition. Gated by explicit protocol confirmation:",
    )

    warning_panel(
        title="Strict Evaluation Protocol Guardrail",
        message=(
            "The test partition is a final immutable holdout reserved strictly for unbiased generalization assessment. "
            "Under academic ML integrity standards, test metrics must NEVER be used to choose models, tune hyperparameters, "
            "select classification thresholds, or modify feature engineering pipelines."
        ),
    )

    test_completed = bool(active_manifest.test_results)

    if not is_frozen:
        info_panel(
            title="Test Holdout Locked",
            message="Candidate models must be selected and frozen in Section D before final test evaluation is unlocked.",
            icon="🔒",
        )
    else:
        if test_completed:
            success_panel(
                title="Test Holdout Permanently Locked 🔒",
                message=(
                    f"Final test evaluation was executed and locked at "
                    f"{active_manifest.test_holdout_locked_at_utc[:19] if active_manifest.test_holdout_locked_at_utc else 'N/A'} UTC. "
                    "Results are immutable. Rerunning test evaluation or modifying candidate models on this engineering partition is locked."
                ),
            )
        else:
            chk_confirm = st.checkbox(
                "I confirm that candidate models are frozen, and test results will NOT be used for model tuning or selection.",
                key="chk_test_gate_confirm",
            )
            col_t_btn, _ = st.columns([1, 2])
            with col_t_btn:
                if st.button("Run Final Test Evaluation 🛡️", disabled=not chk_confirm, key="btn_run_test_eval"):
                    t_progress = st.progress(0.0)
                    t_status = st.empty()

                    def _t_cb(processed: int, total: int, msg: str) -> None:
                        f_val = min(1.0, max(0.0, processed / total)) if total > 0 else 0.0
                        t_progress.progress(f_val)
                        t_status.caption(f"⚙️ {msg}")

                    try:
                        updated_m = orchestrator.run_gated_test_evaluation(
                            evaluation_run_id=active_manifest.evaluation_run_id,
                            confirmation_token="CONFIRM_TEST_EVALUATION",
                            notes="User confirmed protocol compliance via UI gate.",
                            progress_callback=_t_cb,
                        )
                        st.session_state[manifest_key] = updated_m
                        active_manifest = updated_m
                        t_progress.progress(1.0)
                        t_status.empty()
                        success_panel("Test Evaluation Complete", "Successfully evaluated frozen candidate models on final held-out test partition.")
                        st.rerun()
                    except Exception as e:
                        error_panel("Test Evaluation Error", f"Failed to execute final test evaluation: {e}")

    # Section F: Final Test Results & Generalization Gap Analysis
    if test_completed:
        section_header(
            title="Section F — Final Test Evaluation Results & Generalization Analysis",
            description="Inspect final held-out test metrics and compare against validation performance to assess generalization gaps:",
        )

        st.markdown("##### Validation vs Holdout Test Generalization Gap Analysis")
        gap_df = ModelComparator.build_generalization_gap_dataframe(
            val_results=active_manifest.validation_results,
            test_results=active_manifest.test_results,
        )
        st.dataframe(gap_df, use_container_width=True)

        st.markdown("##### Final Test Aggregate Performance")
        test_df = ModelComparator.build_validation_comparison_dataframe(active_manifest.test_results)
        st.dataframe(test_df, use_container_width=True)

    # Section G: Evaluation History
    section_header(
        title="Section G — Evaluation Run Registry & Audit Trail",
        description="Persisted evaluation sessions stored in models/evaluations/:",
    )

    all_eval_runs = eval_registry.list_evaluation_runs()
    if all_eval_runs:
        eval_history_data = []
        for r in all_eval_runs:
            eval_history_data.append({
                "Evaluation Run ID": r["evaluation_run_id"],
                "Source CP-05 Run": r["engineering_run_id"][:16] + "...",
                "Training Session": r["training_session_id"][:16] + "...",
                "Evaluated Models": ", ".join(r["evaluated_algorithms"]).upper(),
                "Partitions": ", ".join(r["partitions_evaluated"]),
                "Test Gate Confirmed": "Yes" if r.get("test_gate_confirmed") else "No",
                "Created (UTC)": r["created_at_utc"][:19] if r["created_at_utc"] else "N/A",
            })
        st.dataframe(pd.DataFrame(eval_history_data), use_container_width=True)
    else:
        st.info("No past evaluation sessions recorded yet.")


def _render_registry_tab(registry: ModelRegistry) -> None:
    """Render the Model & Evaluation Registry tab."""
    eval_registry = EvaluationRegistry(evaluations_dir=settings.evaluations_dir)

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
    else:
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
    else:
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

    # Persisted Evaluation Runs section
    section_header(
        title="Persisted Evaluation Runs & Manifests",
        description="All evaluation sessions stored in models/evaluations/:",
    )

    eval_runs = eval_registry.list_evaluation_runs()
    if not eval_runs:
        st.info("No evaluation sessions have been saved yet.")
    else:
        eval_table = []
        for er in eval_runs:
            eval_table.append({
                "Evaluation Run ID": er["evaluation_run_id"],
                "Source CP-05 Run": er["engineering_run_id"],
                "Training Run ID": er["training_session_id"],
                "Evaluated Models": ", ".join(er["evaluated_algorithms"]).upper(),
                "Partitions": ", ".join(er["partitions_evaluated"]),
                "Test Gate Confirmed": "Yes" if er.get("test_gate_confirmed") else "No",
                "Created At (UTC)": er["created_at_utc"][:19] if er["created_at_utc"] else "N/A",
            })
        st.dataframe(pd.DataFrame(eval_table), use_container_width=True)

        for er in eval_runs:
            man = eval_registry.get_evaluation_manifest(er["evaluation_run_id"])
            if man:
                with st.expander(f"📊 Evaluation Manifest `{er['evaluation_run_id']}`"):
                    st.json(man.to_dict())


def render() -> None:
    """Render the master Model Laboratory view."""
    page_header(
        title="Model Training & Evaluation Laboratory",
        subtitle="Zero-retraining evaluation protocol, multi-model benchmarking & gated holdout verification",
        icon="🧪",
        badges=[
            ("Governance: Zero Retraining", "success"),
            ("3 Authorized Estimators", "info"),
            ("Validation Selection & Gated Holdout", "neutral"),
        ],
    )

    # Architectural Scope & Stage Guardrail Notice
    info_panel(
        title="Production Evaluation Protocol & Zero-Retraining Invariant",
        message=(
            "The Model Evaluation Laboratory evaluates existing, verified production model artifacts without refitting or updating. "
            "Validation partition results inform model comparison, diagnostic breakdown, and candidate selection. "
            "The held-out test partition is strictly gated, requiring candidate artifact freezing and explicit confirmation before evaluation. "
            "Never tune models or hyperparameters based on test partition performance."
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
    tab_training, tab_evaluation, tab_registry = st.tabs([
        "🚀 Model Training Engine",
        "📊 Model Evaluation Laboratory",
        "📦 Model & Evaluation Registry",
    ])

    with tab_training:
        _render_training_tab(
            artifact_mgr=artifact_mgr,
            resource_mgr=resource_mgr,
            registry=registry,
            runs=completed_runs,
        )

    with tab_evaluation:
        _render_evaluation_tab(
            artifact_mgr=artifact_mgr,
            registry=registry,
            completed_runs=completed_runs,
        )

    with tab_registry:
        _render_registry_tab(registry=registry)
