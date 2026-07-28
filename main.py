import streamlit as st
from dashboard.data_provider import fetch_loki_logs, fetch_metrics_from_mlflow, get_mlflow_evaluation_artifacts, parse_metrics_and_progress
from dashboard.process_manager import get_docker_health_states, init_session_states
from dashboard.styles import apply_custom_styling, render_header
from dashboard.tab_control_panel import render_control_panel
from dashboard.tab_health import render_health_tab
from dashboard.tab_logs import render_logs_tab
from dashboard.tab_monitor import render_monitor_tab

st.set_page_config(page_title="Heart Disease FL Dashboard", page_icon=None, layout="wide", initial_sidebar_state="expanded")

init_session_states()
apply_custom_styling()

with st.sidebar:
    st.markdown("#### Telemetry Settings")
    loki_url = st.text_input("Loki Endpoint URL", value="http://localhost:3100")
    st.markdown("---")
    st.markdown("#### Display Settings")
    auto_refresh = st.checkbox("Auto-refresh logs", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", min_value=2, max_value=15, value=3)
    st.markdown("---")
    st.caption("Heart Disease FL Dashboard · Testing Portal")

CONTAINERS = ["fl_server", "mlflow_server", "client_1", "client_2", "loki", "alloy"]

render_header()

if hasattr(st, "fragment"):
    fragment_decorator = st.fragment
elif hasattr(st, "experimental_fragment"):
    fragment_decorator = st.experimental_fragment
else:

    def fragment_decorator(run_every=None):
        def decorator(func):
            return func

        return decorator


@fragment_decorator(run_every=refresh_interval if auto_refresh else None)
def render_dynamic_dashboard():
    all_logs = fetch_loki_logs(loki_url, '{container=~".+"}', limit=1000)
    stats = parse_metrics_and_progress(all_logs, CONTAINERS)

    df_pivot_mlflow, mlflow_best_round, mlflow_best_acc, mlflow_best_loss = fetch_metrics_from_mlflow()

    if not df_pivot_mlflow.empty:
        active_df = df_pivot_mlflow
        card_best_round = mlflow_best_round
        card_best_acc = mlflow_best_acc
        card_best_loss = mlflow_best_loss
    else:
        active_df = stats["df_pivot"]
        card_best_round = stats["best_round"]
        card_best_acc = stats["best_accuracy"]
        card_best_loss = stats.get("best_loss", 0.0)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Best FL Round</div>
            <div class="metric-value">{f"Round {card_best_round}" if card_best_round > 0 else "—"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value">{f"{card_best_acc * 100:.2f}%" if card_best_acc > 0 else "—"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Best Loss</div>
            <div class="metric-value">{f"{card_best_loss:.4f}" if card_best_loss > 0 else "—"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")

    tab_ctrl, tab1, tab2, tab3 = st.tabs(["Training Control Panel", "Federated Training Monitor", "Infrastructure Health", "System Logs"])

    with tab_ctrl:
        render_control_panel()

    with tab1:
        mlflow_arts = get_mlflow_evaluation_artifacts()
        render_monitor_tab(active_df, card_best_round, card_best_acc, mlflow_artifacts=mlflow_arts)

    with tab2:
        docker_health = get_docker_health_states(CONTAINERS)
        render_health_tab(docker_health, CONTAINERS)

    with tab3:
        render_logs_tab(all_logs, CONTAINERS)


render_dynamic_dashboard()
