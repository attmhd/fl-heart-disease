import time

import streamlit as st
from dashboard.data_provider import fetch_loki_logs, get_mlflow_evaluation_artifacts, parse_metrics_and_progress
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

# Fetch and parse telemetry data
all_logs = fetch_loki_logs(loki_url, '{container=~".+"}', limit=1000)
stats = parse_metrics_and_progress(all_logs, CONTAINERS)

# Main UI layout header
render_header()

# Metric Row Cards
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Current FL Round</div>
        <div class="metric-value">{stats["current_round"] if stats["current_round"] > 0 else "—"}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Best Global Accuracy</div>
        <div class="metric-value">{f"{stats['best_accuracy'] * 100:.2f}%" if stats['best_accuracy'] > 0 else "—"}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Best Model Achieved At</div>
        <div class="metric-value">{f"Round {stats['best_round']}" if stats['best_round'] > 0 else "—"}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Participating Clients</div>
        <div class="metric-value">{stats['active_clients_count']} Active</div>
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
    render_monitor_tab(stats["df_pivot"], stats["best_round"], stats["best_accuracy"], mlflow_artifacts=mlflow_arts)

with tab2:
    docker_health = get_docker_health_states(CONTAINERS)
    render_health_tab(docker_health, CONTAINERS)

with tab3:
    render_logs_tab(all_logs, CONTAINERS)

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
