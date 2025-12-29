import streamlit as st
from datetime import datetime
import time

def render_timeline(container, logs):
    container.empty()

    with container.container():
        for i, log in enumerate(logs):
            status_icon = {
                "OK": "🟢",
                "ERROR": "🔴",
                "WARNING": "🟡"
            }.get(log["status"], "⚪")

            col_icon, col_content = st.columns([1, 12])

            with col_icon:
                st.markdown(status_icon)
                if i < len(logs) - 1:
                    st.markdown("│")

            with col_content:
                st.markdown(
                    f"""
**{log['node']}**  
{log['action']}  
Estado: `{log['status']}`  
Hora: `{log['timestamp']}`  
{f"Detalle: {log['detail']}" if log.get("detail") else ""}
"""
                )

def log_event(timeline_placeholder, logs, step, node, action, status="OK", detail=None):
    logs.append({
        "step": step,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "node": node,
        "action": action,
        "status": status,
        "detail": detail
    })

    render_timeline(timeline_placeholder, logs)
    time.sleep(0.4)
