import streamlit as st
from dashboard.data_provider import fetch_loki_logs, fetch_metrics_from_mlflow, get_mlflow_evaluation_artifacts, parse_metrics_and_progress
from dashboard.process_manager import get_docker_health_states, init_session_states

# Import refactored dashboard modules
from dashboard.styles import apply_custom_styling, render_header
from dashboard.tab_control_panel import render_control_panel
from dashboard.tab_health import render_health_tab
from dashboard.tab_logs import render_logs_tab
from dashboard.tab_monitor import render_monitor_tab

# Page config — wide layout for dashboard feel
st.set_page_config(page_title="Heart Disease FL Dashboard", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# Initialize session state for background process controls
init_session_states()

# Custom premium styling and Google Fonts
apply_custom_styling()

# Sidebar configurations — no icons, clean labels
with st.sidebar:
    st.markdown("#### Telemetry Settings")
    loki_url = st.text_input("Loki Endpoint URL", value="http://localhost:3100")
    st.markdown("---")
    st.markdown("#### Display Settings")
    auto_refresh = st.checkbox("Auto-refresh logs", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", min_value=2, max_value=15, value=3)
    st.markdown("---")
    st.caption("Heart Disease FL Dashboard · Testing Portal")

# Container list definition
CONTAINERS = ["fl_server", "mlflow_server", "client_1", "client_2", "loki", "alloy"]

# Main UI layout header
render_header()

# Safe fragment decorator resolution
if hasattr(st, "fragment"):
    fragment_decorator = st.fragment
elif hasattr(st, "experimental_fragment"):
    fragment_decorator = st.experimental_fragment
else:

    def fragment_decorator(run_every=None):
        def decorator(func):
            return func

        return decorator


# Wrap dynamic dashboard elements in a Streamlit fragment to prevent full-page reload on auto-refresh
@fragment_decorator(run_every=refresh_interval if auto_refresh else None)
def render_dynamic_dashboard():
    # ── Fetch all telemetry data ───────────────────────────────────────────────────
    all_logs = fetch_loki_logs(loki_url, '{container=~".+"}', limit=1000)
    stats = parse_metrics_and_progress(all_logs, CONTAINERS)

    # Attempt MLflow metric fetch (primary source) — used for metric cards too
    df_pivot_mlflow, mlflow_best_round, mlflow_best_acc = fetch_metrics_from_mlflow()

    # Resolve which source to use for the header metric cards
    if not df_pivot_mlflow.empty:
        active_df = df_pivot_mlflow
        card_best_round = mlflow_best_round
        card_best_acc = mlflow_best_acc
    else:
        active_df = stats["df_pivot"]
        card_best_round = stats["best_round"]
        card_best_acc = stats["best_accuracy"]

    # Derive current round from active_df
    if not active_df.empty and "Round" in active_df.columns:
        card_current_round = int(active_df["Round"].max())
    else:
        card_current_round = stats["current_round"]

    # Derive best FedAvg Val Accuracy from df (if available)
    fedavg_val_col = "FedAvg Val Accuracy"
    card_fedavg_val_acc = None
    if not active_df.empty and fedavg_val_col in active_df.columns:
        card_fedavg_val_acc = active_df[fedavg_val_col].dropna().max()

    # ── Metric Row Cards ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Current FL Round</div>
            <div class="metric-value">{card_current_round if card_current_round > 0 else "—"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Best Centralized Accuracy</div>
            <div class="metric-value">{f"{card_best_acc * 100:.2f}%" if card_best_acc > 0 else "—"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">FedAvg Val Accuracy</div>
            <div class="metric-value">{f"{card_fedavg_val_acc * 100:.2f}%" if card_fedavg_val_acc is not None else "—"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Create Tabs — no emoji icons, clean text labels
    tab_ctrl, tab1, tab2, tab3 = st.tabs(["Training Control Panel", "Federated Training Monitor", "Infrastructure Health", "System Logs"])

    # Render each tab using its respective module function
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


# Run the dynamic dashboard fragment
render_dynamic_dashboard()
