import plotly.graph_objects as go
import streamlit as st


def render_monitor_tab(df_pivot, best_round, best_accuracy, mlflow_artifacts=None):
    """Render Tab 1: Live FL Training Monitor with Plotly accuracy and loss charts."""
    st.markdown('<div class="section-header">Federated Model Progression</div>', unsafe_allow_html=True)

    if not df_pivot.empty:
        col_chart_acc, col_chart_loss = st.columns(2, gap="medium")

        with col_chart_acc:
            fig_acc = go.Figure()
            if "Centralized Accuracy" in df_pivot.columns:
                fig_acc.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot["Centralized Accuracy"], mode='lines+markers', name='Centralized Accuracy', line=dict(color='#6366f1', width=2.5), marker=dict(size=7)))
            if "Client Val Accuracy" in df_pivot.columns:
                fig_acc.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot["Client Val Accuracy"], mode='lines+markers', name='Client Validation Accuracy', line=dict(color='#10b981', width=2.5), marker=dict(size=7)))

            fig_acc.update_layout(
                title=dict(text="Validation Accuracy per Round", font=dict(size=14, color="#f8fafc")),
                xaxis_title="FL Round",
                yaxis_title="Accuracy",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_acc, width='stretch')

        with col_chart_loss:
            fig_loss = go.Figure()
            if "Centralized Loss" in df_pivot.columns:
                fig_loss.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot["Centralized Loss"], mode='lines+markers', name='Validation Loss', line=dict(color='#ef4444', width=2.5), marker=dict(size=7)))

            fig_loss.update_layout(
                title=dict(text="Model Loss per Round", font=dict(size=14, color="#f8fafc")),
                xaxis_title="FL Round",
                yaxis_title="Loss",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_loss, width='stretch')

        st.info(f"Best global model registered to MLflow Model Registry at **Round {best_round}** " f"with validation performance of **{best_accuracy * 100:.2f}% accuracy**. " f"Registered under alias **@champion** and **@Production**.")

        # MLflow Artifacts Section
        if mlflow_artifacts:
            st.markdown('<div class="section-header">MLflow Evaluation Artifacts</div>', unsafe_allow_html=True)
            st.write("Dynamic centralized and client-side metrics classification reports and confusion matrices retrieved directly from the MLflow Server.")

            # Select target artifact node
            node_options = list(mlflow_artifacts.keys())
            selected_node = st.selectbox("Select Node to View Detailed Evaluation", node_options, format_func=lambda x: "Server Centralized Evaluation" if x == "server" else f"Distributed {x.upper()}")

            node_data = mlflow_artifacts[selected_node]
            col_art1, col_art2 = st.columns([3, 2], gap="large")

            with col_art1:
                st.markdown("**Classification Report**")
                if "report" in node_data:
                    st.code(node_data["report"], language="text")
                else:
                    st.info("No classification report artifact found.")

            with col_art2:
                st.markdown("**Confusion Matrix Plot**")
                if "confusion_matrix" in node_data:
                    st.image(node_data["confusion_matrix"], caption=f"Confusion Matrix - {selected_node.upper()}", width='stretch')
                else:
                    st.info("No confusion matrix plot artifact found.")

    else:
        st.warning("No federated training evaluation logs found yet. Start the Flower server and clients to begin streaming metrics.")
