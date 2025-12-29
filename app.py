import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from timeline import log_event

# ------------------------------
# CONFIGURACIÓN INICIAL
# ------------------------------
st.set_page_config(page_title="Agente de Ventas", layout="wide")
st.title("🧠 Agente Inteligente de Análisis de Ventas")

# ------------------------------
# ESTADO GLOBAL
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------
# UTILIDAD DE LOGGING VISUAL
# ------------------------------
def log_node(log_placeholder, logs, node, action, status="OK", detail=None):
    icon = "✅" if status == "OK" else "❌"
    entry = {
        "node": node,
        "action": action,
        "status": status,
        "detail": detail
    }
    logs.append(entry)

    rendered_logs = []
    for l in logs:
        rendered_logs.append(
            f"""🔹 **Nodo:** {l['node']}
- **Acción:** {l['action']}
- **Estado:** {icon if l['status']=="OK" else "❌"} {l['status']}
{f"- **Detalle:** {l['detail']}" if l['detail'] else ""}
"""
        )

    log_placeholder.markdown("\n".join(rendered_logs))
    time.sleep(0.4)

# ------------------------------
# AGENTE (SIMULADO)
# ------------------------------
def run_agent(user_input, timeline_placeholder):
    logs = []
    step = 1

    log_event(timeline_placeholder, logs, step, "Entrada",
              "Recibir mensaje del usuario")
    step += 1

    log_event(timeline_placeholder, logs, step, "Interpretación",
              "Analizar intención y entidades")
    step += 1

    log_event(timeline_placeholder, logs, step, "Decisión",
              "Detectar salidas solicitadas",
              detail="tabla + gráfico + archivo")
    step += 1

    log_event(timeline_placeholder, logs, step, "SQL",
              "Generar consulta SQL")
    step += 1

    log_event(timeline_placeholder, logs, step, "SQL",
              "Ejecutar consulta en MySQL")
    step += 1

    # Simulación de resultado
    df = pd.DataFrame({
        "producto": ["Laptop", "Monitor", "Teclado"],
        "total_vendido": [120, 95, 80]
    })

    log_event(timeline_placeholder, logs, step, "Visualización",
              "Generar gráfico de barras")
    step += 1

    log_event(timeline_placeholder, logs, step, "Persistencia",
              "Guardar resultados en CSV",
              detail="outputs/ventas_top.csv")
    step += 1

    log_event(timeline_placeholder, logs, step, "Final",
              "Respuesta lista para el usuario")

    return {
        "text": "Aquí tienes el resultado solicitado.",
        "table": df,
        "logs": logs
    }


# ------------------------------
# HISTORIAL DE CHAT
# ------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("table") is not None:
            st.dataframe(msg["table"])

        if msg.get("chart") is not None:
            st.pyplot(msg["chart"])

        if msg.get("file") is not None:
            st.success(f"Archivo generado: `{msg['file']}`")

        if msg.get("logs") is not None:
            with st.expander("Logs del agente"):
                for l in msg["logs"]:
                    st.markdown(
                        f"""🔹 **Nodo:** {l['node']}
- **Acción:** {l['action']}
- **Estado:** {'✅ OK' if l['status']=='OK' else '❌ ERROR'}
{f"- **Detalle:** {l['detail']}" if l['detail'] else ""}
"""
                    )

# ------------------------------
# INPUT DE CHAT
# ------------------------------
user_input = st.chat_input("Haz una pregunta sobre ventas...")

if user_input:
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(user_input)

    # Guardar en el estado
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Respuesta del agente
    with st.chat_message("assistant"):
        with st.expander("Ejecución del agente (línea de tiempo)", expanded=False):
            timeline_placeholder = st.empty()
            response = run_agent(user_input, timeline_placeholder)

        st.markdown(response["text"])
        st.dataframe(response["table"])


    # Guardar respuesta del agente
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["text"],
        "table": response["table"],
        "logs": response["logs"]
    })
