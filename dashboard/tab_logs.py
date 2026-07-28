"""
Centralized Log Terminal visualizer and severity filter component.

This module processes, filters, and renders formatted logs scraped from the
centralized Loki telemetry server inside a scrollable web console.
"""

import streamlit as st


def render_logs_tab(all_logs, containers):
    """Render Tab 3: Live System Logs Stream with filters by node and severity.

    Args:
        all_logs (list): Complete list of logs fetched from Loki.
        containers (list): Container names list for dropdown filtering.
    """
    st.markdown('<div class="section-header">Centralized Log Terminal</div>', unsafe_allow_html=True)

    col_filters1, col_filters2 = st.columns([1, 1], gap="medium")
    with col_filters1:
        selected_container = st.selectbox("Filter by Container", ["ALL"] + containers)
    with col_filters2:
        selected_severity = st.selectbox("Severity Level", ["ALL", "INFO", "WARNING", "ERROR / CRITICAL"])

    filtered_logs = all_logs
    if selected_container != "ALL":
        filtered_logs = [log for log in filtered_logs if log["container"] == selected_container]

    if selected_severity != "ALL":
        if selected_severity == "INFO":
            filtered_logs = [log for log in filtered_logs if "info" in log["message"].lower() or ("warn" not in log["message"].lower() and "error" not in log["message"].lower() and "fail" not in log["message"].lower())]
        elif selected_severity == "WARNING":
            filtered_logs = [log for log in filtered_logs if "warn" in log["message"].lower() or "warning" in log["message"].lower()]
        elif selected_severity == "ERROR / CRITICAL":
            filtered_logs = [log for log in filtered_logs if "error" in log["message"].lower() or "fail" in log["message"].lower() or "exception" in log["message"].lower() or "critical" in log["message"].lower()]

    if filtered_logs:
        log_text = ""
        for log in filtered_logs[-150:]:  # Slice latest 150 lines from list
            timestamp_str = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            log_text += f"[{timestamp_str}] [{log['container'].upper()}] {log['message']}\n"

        st.markdown(
            f'<div style="height: 450px; overflow-y: auto; background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 13px; color: #cbd5e1; white-space: pre-wrap;">{log_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No logs matched the selected filters. Awaiting telemetry stream from Loki.")
