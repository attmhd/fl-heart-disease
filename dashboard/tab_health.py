"""
Infrastructure Health connectivity scanner and dashboard visualizer.

This module renders real-time connectivity scan cards for Flower Server, MLflow,
Loki, and Alloy agents alongside their network topology states.
"""

import streamlit as st

CONTAINER_DESC = {
    "fl_server": "FL Server",
    "mlflow_server": "MLflow Engine",
    "loki": "Loki DB",
    "alloy": "Alloy Agent",
}


def render_health_tab(health_states, containers):
    """Render Tab 2: Infrastructure & Blackbox Health connectivity scan cards and topography.

    Args:
        health_states (dict): Liveness state dictionary map of container names.
        containers (list): Target list of active containers to render status.
    """
    st.markdown('<div class="section-header">Real-time Connectivity Scan</div>', unsafe_allow_html=True)
    st.write("Verifies execution states and connection liveness of all services in the stack via Loki log scraping.")

    # Generate columns for cards
    cols = st.columns(len(containers), gap="small")

    for idx, name in enumerate(containers):
        with cols[idx]:
            is_active = health_states.get(name, False)
            status_text = "ONLINE" if is_active else "OFFLINE"
            color_class = "status-healthy" if is_active else "status-offline"
            desc = CONTAINER_DESC.get(name, f"FL Client {name[-1]}")

            st.markdown(
                f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:1.4rem 1rem; text-align:center;">
                    <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">{desc}</div>
                    <div style="font-size:1rem; font-weight:700; color:#f8fafc; margin-bottom:0.8rem;">{name}</div>
                    <div class="{color_class}">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-header">Inter-Service Topology</div>', unsafe_allow_html=True)

    def conn(key):
        return "Connected" if health_states.get(key) else "Disconnected"

    st.markdown(rf"""
    | Connection Pathway | Status |
    |---|---|
    | FL Clients (client\_1, client\_2) → fl\_server:8080 | **{conn("fl_server")}** |
    | fl\_server → mlflow\_server:5000 | **{conn("mlflow_server")}** |
    | alloy → loki:3100 | **{conn("loki")}** |
    """)
