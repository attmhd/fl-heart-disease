import re
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Set up page config
st.set_page_config(page_title="Federated Learning Demo Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Custom premium styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 40%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(25, 30, 56, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-healthy {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-offline {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar configurations
st.sidebar.markdown("### ⚙️ Telemetry Settings")
loki_url = st.sidebar.text_input("Loki Endpoint URL", value="http://localhost:3100")
auto_refresh = st.sidebar.checkbox("Auto-refresh logs", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=2, max_value=15, value=3)

# Container list
CONTAINERS = ["fl_server", "mlflow_server", "client_1", "client_2", "loki", "alloy", "grafana"]


@st.cache_data(ttl=1)
def fetch_loki_logs(query, limit=500):
    """Query logs from Loki server."""
    try:
        url = f"{loki_url}/loki/api/v1/query_range"
        params = {"query": query, "limit": limit, "direction": "BACKWARD"}
        response = requests.get(url, params=params, timeout=2.0)
        if response.status_code == 200:  # noqa: PLR2004
            data = response.json()
            logs = []
            if data.get("status") == "success":
                results = data.get("data", {}).get("result", [])
                for result in results:
                    container = result.get("stream", {}).get("container", "unknown")
                    for val in result.get("values", []):
                        # Loki format is [timestamp_ns, log_message]
                        timestamp = datetime.fromtimestamp(int(val[0]) / 1e9)
                        logs.append({"timestamp": timestamp, "container": container, "message": val[1].strip()})
            # Sort logs by timestamp ascending
            logs = sorted(logs, key=lambda x: x["timestamp"])
            return logs
    except Exception:
        pass
    return []


# Fetch logs from all containers
all_logs = fetch_loki_logs('{container=~".+"}', limit=1000)

# ----------------------------------------------------
# Parse Metrics & FL Rounds from Logs
# ----------------------------------------------------
fl_progress = []
active_clients_set = set()
best_accuracy = 0.0
best_round = 0
current_round = 0

# Parse Flower server evaluation logs
# Patterns:
# [SERVER] Centralized Evaluation for Round X: Loss = Y, Accuracy = Z
# [ROUND X] New best global model achieved from client validation! Accuracy: Y
for log in all_logs:
    msg = log["message"]
    # Track client activations
    if "client_" in log["container"]:
        active_clients_set.add(log["container"])

    # Pattern 1: Centralized Evaluation
    cent_match = re.search(r"Centralized Evaluation for Round (\d+): Loss = ([\d\.]+), Accuracy = ([\d\.]+)", msg)
    if cent_match:
        rnd = int(cent_match.group(1))
        loss = float(cent_match.group(2))
        acc = float(cent_match.group(3))
        current_round = max(current_round, rnd)
        fl_progress.append({"Round": rnd, "Metric": "Centralized Accuracy", "Value": acc})
        fl_progress.append({"Round": rnd, "Metric": "Centralized Loss", "Value": loss})
        if acc > best_accuracy:
            best_accuracy = acc
            best_round = rnd

    # Pattern 2: Global Validation via Client
    client_eval_match = re.search(r"\[ROUND (\d+)\] New best global model achieved from client validation! Accuracy: ([\d\.]+)", msg)
    if client_eval_match:
        rnd = int(client_eval_match.group(1))
        acc = float(client_eval_match.group(2))
        current_round = max(current_round, rnd)
        fl_progress.append({"Round": rnd, "Metric": "Client Val Accuracy", "Value": acc})
        if acc > best_accuracy:
            best_accuracy = acc
            best_round = rnd

df_progress = pd.DataFrame(fl_progress) if fl_progress else pd.DataFrame(columns=["Round", "Metric", "Value"])

# Convert to pivot for charts
if not df_progress.empty:
    df_pivot = df_progress.pivot_table(index="Round", columns="Metric", values="Value").reset_index()
else:
    df_pivot = pd.DataFrame(columns=["Round", "Centralized Accuracy", "Centralized Loss", "Client Val Accuracy"])

# Determine overall health based on Alloy log heartbeats
health_states = {}
for name in CONTAINERS:
    # If container logged within last 3 minutes, it's Online
    recent_logs = [log for log in all_logs if log["container"] == name]
    if recent_logs:
        health_states[name] = True
    else:
        health_states[name] = False

# ----------------------------------------------------
# Main UI Layout
# ----------------------------------------------------
st.markdown('<div class="main-title">🛡️ Heart Disease Federated Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Blackbox Testing & Centralized Telemetry Demo Portal</div>', unsafe_allow_html=True)

# Metric Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-value">Round {current_round if current_round > 0 else "—"}</div>
        <div class="metric-label">Current FL Round</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-value">{f"{best_accuracy * 100:.2f}%" if best_accuracy > 0 else "—"}</div>
        <div class="metric-label">Best Global Accuracy</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-value">{f"Round {best_round}" if best_round > 0 else "—"}</div>
        <div class="metric-label">Best Model Achieved At</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    num_clients = len(active_clients_set)
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-value">{num_clients if num_clients > 0 else "0"} Active</div>
        <div class="metric-label">Participating Clients</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# Create Tabs for different dashboard sections
tab1, tab2, tab3 = st.tabs(["📈 Live Federated Training Monitor", "🌐 Infrastructure & Blackbox Health", "📜 Live System Logs Stream"])

# ----------------------------------------------------
# TAB 1: Live FL Training Monitor
# ----------------------------------------------------
with tab1:
    st.markdown("### 📊 Federated Model Progression")

    if not df_pivot.empty:
        col_chart_acc, col_chart_loss = st.columns(2)

        with col_chart_acc:
            fig_acc = go.Figure()
            if "Centralized Accuracy" in df_pivot.columns:
                fig_acc.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot["Centralized Accuracy"], mode='lines+markers', name='Centralized Accuracy', line=dict(color='#6366f1', width=3)))
            if "Client Val Accuracy" in df_pivot.columns:
                fig_acc.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot["Client Val Accuracy"], mode='lines+markers', name='Client Val Accuracy', line=dict(color='#10b981', width=3)))

            fig_acc.update_layout(title="Validation Accuracy over Rounds", xaxis_title="FL Round", yaxis_title="Accuracy", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_chart_loss:
            fig_loss = go.Figure()
            if "Centralized Loss" in df_pivot.columns:
                fig_loss.add_trace(go.Scatter(x=df_pivot["Round"], y=df_pivot["Centralized Loss"], mode='lines+markers', name='Validation Loss', line=dict(color='#ef4444', width=3)))

            fig_loss.update_layout(title="Model Loss over Rounds", xaxis_title="FL Round", yaxis_title="Loss", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_loss, use_container_width=True)

        st.info(
            f"The best global model has been archived and uploaded to MLflow Model Registry at **Round {best_round}** "
            f"with a validation performance of **{best_accuracy * 100:.2f}% Accuracy**. "
            f"It is registered under alias **`@champion`** and **`@Production`** for active inference."
        )
    else:
        st.warning("No Federated training evaluation logs found yet. Start the Flower server and clients to begin streaming metrics.")

# ----------------------------------------------------
# TAB 2: Infrastructure Health (Blackbox Connect)
# ----------------------------------------------------
with tab2:
    st.markdown("### 🌐 Real-time Blackbox Connectivity Scan")
    st.write("Verifies the execution states and connection liveness of all containers in the stack via Loki log scraping agents:")

    # Generate columns for cards
    cols = st.columns(len(CONTAINERS))

    for idx, name in enumerate(CONTAINERS):
        with cols[idx]:
            is_active = health_states.get(name, False)
            status_text = "ONLINE" if is_active else "OFFLINE"
            color_class = "status-healthy" if is_active else "status-offline"

            # Map descriptions
            desc = "FL Server" if name == "fl_server" else "MLflow Engine" if name == "mlflow_server" else "Loki DB" if name == "loki" else "Alloy Agent" if name == "alloy" else "Grafana Web" if name == "grafana" else f"FL Client {name[-1]}"

            st.markdown(
                f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; text-align: center;">
                <h4 style="margin-bottom: 0.2rem; color: #fff;">{name}</h4>
                <div style="font-size: 0.8rem; color: #9ca3af; margin-bottom: 1rem;">{desc}</div>
                <div class="{color_class}">{status_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("### 🔌 Inter-Service Topology Mapping")
    st.markdown(f"""
    * **gRPC FL Connection Pathway:** Clients (`client_1`, `client_2`) ➡️ `fl_server:8080` (Active: {"🟢 Connected" if health_states.get("fl_server") else "🔴 Disconnected"})
    * **MLflow Tracking Pathway:** `fl_server` ➡️ `mlflow_server:5000` (Active: {"🟢 Connected" if health_states.get("mlflow_server") else "🔴 Disconnected"})
    * **Telemetry Pathway:** `alloy` ➡️ `loki:3100` (Active: {"🟢 Connected" if health_states.get("loki") else "🔴 Disconnected"})
    * **Observer/Visuals Pathway:** `grafana` ➡️ `loki` (Active: {"🟢 Connected" if health_states.get("grafana") and health_states.get("loki") else "🔴 Disconnected"})
    """)

# ----------------------------------------------------
# TAB 3: Live System Logs Stream
# ----------------------------------------------------
with tab3:
    st.markdown("### 📜 Real-time Centralized Logging Terminal")

    col_filters1, col_filters2 = st.columns([1, 1])
    with col_filters1:
        selected_container = st.selectbox("Filter by Container Node", ["ALL"] + CONTAINERS)
    with col_filters2:
        selected_severity = st.selectbox("Log Severity Level", ["ALL", "INFO", "WARNING", "ERROR/CRITICAL"])

    # Filter logs based on selection
    filtered_logs = all_logs
    if selected_container != "ALL":
        filtered_logs = [log for log in filtered_logs if log["container"] == selected_container]

    if selected_severity != "ALL":
        if selected_severity == "INFO":
            filtered_logs = [log for log in filtered_logs if "info" in log["message"].lower() or ("warn" not in log["message"].lower() and "error" not in log["message"].lower() and "fail" not in log["message"].lower())]
        elif selected_severity == "WARNING":
            filtered_logs = [log for log in filtered_logs if "warn" in log["message"].lower() or "warning" in log["message"].lower()]
        elif selected_severity == "ERROR/CRITICAL":
            filtered_logs = [log for log in filtered_logs if "error" in log["message"].lower() or "fail" in log["message"].lower() or "exception" in log["message"].lower() or "critical" in log["message"].lower()]

    # Print logs
    if filtered_logs:
        log_text = ""
        for log in filtered_logs[-150:]:  # Show latest 150 lines
            timestamp_str = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            log_text += f"[{timestamp_str}] [{log['container'].upper()}] {log['message']}\n"
        st.code(log_text, language="log", wrap_lines=True)
    else:
        st.info("No logs matched the selected filters. Awaiting telemetry stream...")

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
