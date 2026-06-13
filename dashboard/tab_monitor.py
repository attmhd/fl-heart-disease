import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# Colour palette — consistent across all charts
# ──────────────────────────────────────────────
_COLORS = [
    "#6366f1",  # indigo
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#84cc16",  # lime
]


def _color(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


def _make_chart(title: str, y_label: str, df_pivot, selected_metrics: list) -> go.Figure:
    """Build a Plotly line chart for the given metric list."""
    fig = go.Figure()
    for i, metric in enumerate(selected_metrics):
        if metric not in df_pivot.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_pivot["Round"],
                y=df_pivot[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color=_color(i), width=2.5),
                marker=dict(size=7),
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f8fafc")),
        xaxis_title="FL Round",
        yaxis_title=y_label,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=20),
        xaxis=dict(dtick=1),
    )
    return fig


def _classify_metrics(all_metrics: list):
    """Split metric names into (train_acc, val_acc, train_loss, val_loss, train_f1, val_f1, duration, other)."""
    train_acc, val_acc = [], []
    train_loss, val_loss = [], []
    train_f1, val_f1 = [], []
    duration, other = [], []

    for m in all_metrics:
        ml = m.lower()
        is_train = "train" in ml or "fedavg train" in ml
        is_val = "val" in ml or "centralized" in ml
        is_acc = "accuracy" in ml
        is_loss = "loss" in ml and "duration" not in ml
        is_f1 = "f1" in ml or "precision" in ml or "recall" in ml
        is_dur = "duration" in ml

        if is_dur:
            duration.append(m)
        elif is_f1 and is_train:
            train_f1.append(m)
        elif is_f1 and (is_val or not is_train):
            val_f1.append(m)
        elif is_acc and is_train:
            train_acc.append(m)
        elif is_acc and (is_val or not is_train):
            val_acc.append(m)
        elif is_loss and is_train:
            train_loss.append(m)
        elif is_loss and (is_val or not is_train):
            val_loss.append(m)
        else:
            other.append(m)

    return train_acc, val_acc, train_loss, val_loss, train_f1, val_f1, duration, other


def render_monitor_tab(df_pivot, best_round, best_accuracy, mlflow_artifacts=None):  # noqa: PLR0915
    """Render Tab 1: Live FL Training Monitor with Plotly charts."""
    st.markdown('<div class="section-header">Federated Model Progression</div>', unsafe_allow_html=True)

    if df_pivot.empty:
        st.warning("No federated training evaluation logs found yet. Start the Flower server and clients to begin streaming metrics.")
        return

    all_metrics = [col for col in df_pivot.columns if col != "Round"]
    train_acc, val_acc, train_loss, val_loss, train_f1, val_f1, duration, _ = _classify_metrics(all_metrics)

    # ──────────────────────────────────────────────
    # Section: Multiselect filters
    # ──────────────────────────────────────────────
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.9rem; margin-top:-0.5rem;'>" "Filter specific server or client-side metrics to trace their progress over the FL rounds." "</p>",
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Metric Filter Settings", expanded=True):
        fc1, fc2 = st.columns(2, gap="medium")

        MAX_DEFAULT_METRICS = 3

        with fc1:
            sel_train_acc = st.multiselect(
                "Training Accuracy Metrics",
                options=train_acc,
                default=train_acc,
                key="sel_train_acc",
            )
            sel_val_acc = st.multiselect(
                "Validation Accuracy Metrics",
                options=val_acc,
                default=val_acc[:MAX_DEFAULT_METRICS] if len(val_acc) > MAX_DEFAULT_METRICS else val_acc,
                key="sel_val_acc",
            )

        with fc2:
            sel_train_loss = st.multiselect(
                "Training Loss Metrics",
                options=train_loss,
                default=train_loss,
                key="sel_train_loss",
            )
            sel_val_loss = st.multiselect(
                "Validation Loss Metrics",
                options=val_loss,
                default=val_loss[:MAX_DEFAULT_METRICS] if len(val_loss) > MAX_DEFAULT_METRICS else val_loss,
                key="sel_val_loss",
            )

    # ──────────────────────────────────────────────
    # Row 1: Training Accuracy | Validation Accuracy
    # ──────────────────────────────────────────────
    st.markdown("---")
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        st.markdown("<h4 style='color:#6366f1; font-weight:600; margin-bottom:0.5rem;'>Training Accuracy</h4>", unsafe_allow_html=True)
        st.plotly_chart(
            _make_chart("Training Accuracy per Round", "Accuracy", df_pivot, sel_train_acc),
            width="stretch",
        )

    with r1c2:
        st.markdown("<h4 style='color:#10b981; font-weight:600; margin-bottom:0.5rem;'>Validation Accuracy</h4>", unsafe_allow_html=True)
        st.plotly_chart(
            _make_chart("Validation Accuracy per Round", "Accuracy", df_pivot, sel_val_acc),
            width="stretch",
        )

    # ──────────────────────────────────────────────
    # Row 2: Training Loss | Validation Loss
    # ──────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        st.markdown("<h4 style='color:#ef4444; font-weight:600; margin-bottom:0.5rem;'>Training Loss</h4>", unsafe_allow_html=True)
        st.plotly_chart(
            _make_chart("Training Loss per Round", "Loss", df_pivot, sel_train_loss),
            width="stretch",
        )

    with r2c2:
        st.markdown("<h4 style='color:#f59e0b; font-weight:600; margin-bottom:0.5rem;'>Validation Loss</h4>", unsafe_allow_html=True)
        st.plotly_chart(
            _make_chart("Validation Loss per Round", "Loss", df_pivot, sel_val_loss),
            width="stretch",
        )

    # ──────────────────────────────────────────────
    # Row 4: Fit Duration (if available)
    # ──────────────────────────────────────────────
    if duration:
        st.markdown("<h4 style='color:#f97316; font-weight:600; margin-bottom:0.5rem;'>Training Duration</h4>", unsafe_allow_html=True)
        sel_duration = st.multiselect(
            "Duration Metrics",
            options=duration,
            default=duration,
            key="sel_duration",
        )
        st.plotly_chart(
            _make_chart("Fit Duration per Round (ms)", "Milliseconds", df_pivot, sel_duration),
            width="stretch",
        )

    # ──────────────────────────────────────────────
    # Section: MLflow Evaluation Artifacts
    # ──────────────────────────────────────────────
    if mlflow_artifacts:
        st.markdown("---")
        st.markdown('<div class="section-header">MLflow Evaluation Artifacts</div>', unsafe_allow_html=True)
        st.write("Classification reports and confusion matrices for each node — retrieved directly from MLflow " "and displayed side-by-side for comparison.")

        # Sort keys: server first, then client_1, client_2, etc.
        node_keys = sorted(
            mlflow_artifacts.keys(),
            key=lambda x: 0 if x == "server" else (int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 99),
        )

        cols_nodes = st.columns(len(node_keys), gap="large")
        for idx, node_key in enumerate(node_keys):
            with cols_nodes[idx]:
                title = "Server Centralized Eval" if node_key == "server" else f"Distributed {node_key.upper().replace('_', ' ')}"
                st.markdown(f"#### {title}")

                node_data = mlflow_artifacts[node_key]

                st.markdown("**Confusion Matrix**")
                if "confusion_matrix" in node_data:
                    st.image(node_data["confusion_matrix"], caption=f"Confusion Matrix — {node_key.upper()}", width="stretch")
                else:
                    st.info("No confusion matrix plot found.")

                st.markdown("**Classification Report**")
                if "report" in node_data:
                    st.code(node_data["report"], language="text")
                else:
                    st.info("No classification report found.")
