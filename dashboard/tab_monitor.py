import plotly.graph_objects as go
import streamlit as st


def render_monitor_tab(df_pivot, best_round, best_accuracy, mlflow_artifacts=None):  # noqa: PLR0915
    """Render Tab 1: Live FL Training Monitor with Plotly accuracy and loss charts."""
    st.markdown('<div class="section-header">Federated Model Progression</div>', unsafe_allow_html=True)

    if not df_pivot.empty:
        # Dynamic filter for metric traces
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem; margin-top:-0.5rem;'>Filter specific server or client-side metrics to trace their progress over the FL rounds.</p>", unsafe_allow_html=True)
        col_sel1, col_sel2 = st.columns(2, gap="medium")

        all_metrics = [col for col in df_pivot.columns if col != "Round"]
        train_acc_metrics = [m for m in all_metrics if "Train" in m and "Accuracy" in m]
        val_acc_metrics = [m for m in all_metrics if ("Val" in m or "Centralized" in m or "Global" in m) and "Accuracy" in m]
        train_loss_metrics = [m for m in all_metrics if "Train" in m and "Loss" in m]
        val_loss_metrics = [m for m in all_metrics if ("Val" in m or "Centralized" in m or "Global" in m) and "Loss" in m]

        with col_sel1:
            selected_train_acc = st.multiselect("Select Training Accuracy Metrics", options=train_acc_metrics, default=train_acc_metrics)
            selected_val_acc = st.multiselect(
                "Select Validation Accuracy Metrics", options=val_acc_metrics, default=[m for m in ["Centralized Accuracy", "Client Val Accuracy", "Global Val Accuracy", "Fedavg Val Accuracy"] if m in val_acc_metrics] or val_acc_metrics[:2]
            )

        with col_sel2:
            selected_train_loss = st.multiselect("Select Training Loss Metrics", options=train_loss_metrics, default=train_loss_metrics)
            selected_val_loss = st.multiselect("Select Validation Loss Metrics", options=val_loss_metrics, default=[m for m in ["Centralized Loss", "Global Val Loss", "Fedavg Val Loss"] if m in val_loss_metrics] or val_loss_metrics[:1])

        # ROW 1: Training Accuracy vs Validation Accuracy
        col_acc_train, col_acc_val = st.columns(2, gap="medium")
        colors = ["#6366f1", "#10b981", "#ef4444", "#f59e0b", "#06b6d4", "#ec4899", "#8b5cf6", "#14b8a6"]

        with col_acc_train:
            st.markdown("<h4 style='color:#6366f1; font-weight:600; margin-bottom:0.5rem;'>Training Accuracy</h4>", unsafe_allow_html=True)
            fig_train_acc = go.Figure()
            for i, metric in enumerate(selected_train_acc):
                color = colors[i % len(colors)]
                fig_train_acc.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot[metric], mode='lines+markers', name=metric, line=dict(color=color, width=2.5), marker=dict(size=7)))
            fig_train_acc.update_layout(
                title=dict(text="Training Accuracy per Round", font=dict(size=14, color="#f8fafc")),
                xaxis_title="FL Round",
                yaxis_title="Accuracy",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_train_acc, width='stretch')

        with col_acc_val:
            st.markdown("<h4 style='color:#10b981; font-weight:600; margin-bottom:0.5rem;'>Validation Accuracy</h4>", unsafe_allow_html=True)
            fig_val_acc = go.Figure()
            for i, metric in enumerate(selected_val_acc):
                color = colors[i % len(colors)]
                fig_val_acc.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot[metric], mode='lines+markers', name=metric, line=dict(color=color, width=2.5), marker=dict(size=7)))
            fig_val_acc.update_layout(
                title=dict(text="Validation Accuracy per Round", font=dict(size=14, color="#f8fafc")),
                xaxis_title="FL Round",
                yaxis_title="Accuracy",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_val_acc, width='stretch')

        # ROW 2: Training Loss vs Validation Loss
        col_loss_train, col_loss_val = st.columns(2, gap="medium")
        colors_loss = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#14b8a6"]

        with col_loss_train:
            st.markdown("<h4 style='color:#ef4444; font-weight:600; margin-bottom:0.5rem;'>Training Loss</h4>", unsafe_allow_html=True)
            fig_train_loss = go.Figure()
            for i, metric in enumerate(selected_train_loss):
                color = colors_loss[i % len(colors_loss)]
                fig_train_loss.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot[metric], mode='lines+markers', name=metric, line=dict(color=color, width=2.5), marker=dict(size=7)))
            fig_train_loss.update_layout(
                title=dict(text="Training Loss per Round", font=dict(size=14, color="#f8fafc")),
                xaxis_title="FL Round",
                yaxis_title="Loss",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_train_loss, width='stretch')

        with col_loss_val:
            st.markdown("<h4 style='color:#f59e0b; font-weight:600; margin-bottom:0.5rem;'>Validation Loss</h4>", unsafe_allow_html=True)
            fig_val_loss = go.Figure()
            for i, metric in enumerate(selected_val_loss):
                color = colors_loss[i % len(colors_loss)]
                fig_val_loss.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot[metric], mode='lines+markers', name=metric, line=dict(color=color, width=2.5), marker=dict(size=7)))
            fig_val_loss.update_layout(
                title=dict(text="Validation Loss per Round", font=dict(size=14, color="#f8fafc")),
                xaxis_title="FL Round",
                yaxis_title="Loss",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_val_loss, width='stretch')

        # MLflow Artifacts Section
        if mlflow_artifacts:
            st.markdown('<div class="section-header">MLflow Evaluation Artifacts</div>', unsafe_allow_html=True)
            st.write("Dynamic centralized and client-side metrics classification reports and confusion matrices retrieved directly from the MLflow Server, displayed side-by-side for comparison.")

            # Sort keys: server first, then client_1, client_2, etc.
            node_keys = list(mlflow_artifacts.keys())
            node_keys.sort(key=lambda x: 0 if x == "server" else int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 99)

            cols_nodes = st.columns(len(node_keys), gap="large")
            for idx, selected_node in enumerate(node_keys):
                with cols_nodes[idx]:
                    title = "Server Centralized Evaluation" if selected_node == "server" else f"Distributed {selected_node.upper()}"
                    st.markdown(f"#### {title}")

                    node_data = mlflow_artifacts[selected_node]

                    st.markdown("**Confusion Matrix Plot**")
                    if "confusion_matrix" in node_data:
                        st.image(node_data["confusion_matrix"], caption=f"Confusion Matrix - {selected_node.upper()}", width='stretch')
                    else:
                        st.info("No confusion matrix plot artifact found.")

                    st.markdown("**Classification Report**")
                    if "report" in node_data:
                        st.code(node_data["report"], language="text")
                    else:
                        st.info("No classification report artifact found.")

    else:
        st.warning("No federated training evaluation logs found yet. Start the Flower server and clients to begin streaming metrics.")
