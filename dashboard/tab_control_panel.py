import os
import time

import streamlit as st

from .process_manager import start_docker_runs, stop_docker_runs


def render_control_panel():  # noqa: PLR0912, PLR0915
    """Render Tab 0: FL Testing Control Panel UI with all parameters and buttons."""
    st.markdown('<div class="section-header">Training Control Panel</div>', unsafe_allow_html=True)
    st.write("Configure model hyperparameters and initiate federated training runs in real-time. This panel deploys training on the Docker Compose stack with dynamic configuration propagation.")

    if st.session_state.docker_active:
        st.success("Docker Compose Stack Status: RUNNING — Check Loki stream for container telemetry.")
    else:
        st.info("Docker Compose Stack Status: IDLE — Launch a training stack to begin.")

    st.markdown("---")

    col_inputs1, col_inputs2 = st.columns(2, gap="large")

    with col_inputs1:
        st.markdown("**Orchestration Parameters (FL Server)**")
        rounds = st.slider("Number of FL Rounds", min_value=1, max_value=50, value=5, help="Number of aggregation iterations between the FL server and clients (FL_NUM_ROUNDS).")

        st.markdown("**Flower Federation Strategy Config**")
        fraction_fit = st.slider("Fraction Fit", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help="Fraction of available clients sampled during training round.")
        fraction_evaluate = st.slider("Fraction Evaluate", min_value=0.0, max_value=1.0, value=1.0, step=0.1, help="Fraction of available clients sampled during evaluation round.")
        min_fit_clients = st.number_input("Min Fit Clients", min_value=1, max_value=10, value=2, step=1, help="Minimum number of clients to sample for training.")
        min_evaluate_clients = st.number_input("Min Evaluate Clients", min_value=1, max_value=10, value=2, step=1, help="Minimum number of clients to sample for evaluation.")

    with col_inputs2:
        st.markdown("**Model Parameters — Logistic Regression**")
        solver = st.selectbox("Solver", ["saga", "lbfgs", "liblinear", "newton-cg"], index=0, help="Optimization algorithm for training the Logistic Regression estimator.")

        if solver == "saga":
            penalty_options = ["l2", "l1", "none"]
        elif solver == "liblinear":
            penalty_options = ["l2", "l1"]
        else:
            penalty_options = ["l2", "none"]

        penalty = st.selectbox("Penalty (Regularization)", penalty_options, index=0, help="Type of regularization norm applied. Options depend on the selected solver.")
        c_val = st.number_input("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, format="%.2f", help="Inverse of regularization strength. Lower values indicate stronger regularization.")
        max_iter = st.slider(
            "Local Epochs (Max Iterations per Round)",
            min_value=1,
            max_value=50,
            value=1,
            step=1,
            help="Local epochs per round (MODEL_MAX_ITER). Use 1 for standard FedAvg with warm_start=True.",
        )

        st.markdown("**Dataset Management (CSV)**")
        uploaded_target = st.selectbox("Target Dataset to Overwrite", ["Server (heart_disease_cleveland_cleaned.csv)", "Client 1 (heart_disease_hungary_cleaned.csv)", "Client 2 (heart_disease_switzerland_cleaned.csv)"])

        uploaded_file = st.file_uploader(
            f"Upload CSV for {uploaded_target.split(' ')[0]}",
            type=["csv"],
            help="Choose a processed heart disease CSV to overwrite the respective local dataset. (Max limit: 1 MB)",
        )

        if uploaded_file is not None:
            max_size_bytes = 1 * 1024 * 1024
            if uploaded_file.size > max_size_bytes:
                st.error(f"Ukuran file melebihi batas maksimal 1 MB ({uploaded_file.size / (1024 * 1024):.2f} MB). Silakan unggah file yang lebih kecil.")
            elif st.button("Save & Overwrite Dataset", width="stretch"):
                target_filename = ""
                if "Server" in uploaded_target:
                    target_filename = "data/processed/heart_disease_cleveland_cleaned.csv"
                elif "Client 1" in uploaded_target:
                    target_filename = "data/processed/heart_disease_hungary_cleaned.csv"
                else:
                    target_filename = "data/processed/heart_disease_switzerland_cleaned.csv"

                try:
                    os.makedirs(os.path.dirname(target_filename), exist_ok=True)
                    with open(target_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Successfully saved and overwrote {target_filename}!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving dataset file: {e}")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2, gap="medium")

    with col_btn1:
        if st.button("Launch Federated Training", width="stretch", type="primary"):
            with st.spinner("Recreating Docker containers with custom environment variables..."):
                start_docker_runs(rounds=rounds, solver=solver, max_iter=max_iter, c=c_val, penalty=penalty, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients)
                st.toast("Docker Compose triggered successfully.")
                time.sleep(2)
                st.rerun()

    with col_btn2:
        if st.button("Stop / Terminate Stack", width="stretch"):
            with st.spinner("Stopping Docker Compose services..."):
                stop_docker_runs()
                st.toast("Docker FL services stopped.")
                time.sleep(1)
                st.rerun()

    st.markdown("---")

    col_log_header, col_log_btn = st.columns([4, 1])
    with col_log_header:
        st.markdown("**Active Execution Logs**")
    with col_log_btn:
        if st.button("Clear Logs", width="stretch"):
            try:
                with open("local_fl_run.log", "w", encoding="utf-8") as f:
                    f.write("=== Logs cleared by User ===\n")
                st.toast("Active execution logs cleared!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear logs: {e}")

    if os.path.exists("local_fl_run.log"):
        with open("local_fl_run.log", encoding="utf-8") as f:
            lines = f.readlines()
            log_content = "".join(lines[-150:])

        st.markdown(
            f'<div style="height: 350px; overflow-y: auto; background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 13px; color: #cbd5e1; white-space: pre-wrap;">{log_content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No active logs found. Launch a training run to generate output logs here.")
