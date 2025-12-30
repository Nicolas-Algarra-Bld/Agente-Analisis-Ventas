import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from timeline import log_event, render_timeline

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
# AGENTE
# ------------------------------
from agent_graph import create_graph
from streamlit.runtime.scriptrunner import get_script_run_ctx
def run_agent(user_input, timeline_placeholder):
    graph = create_graph()
    
    # Initialize state
    initial_state = {
        "user_input": user_input,
        "logs": [],
        "messages": [],
        "peticiones": [],
        "nombre_archivo": "",
        "consulta_sql": "",
        "error_sql": "",
        "tabla_consulta": []
    }
    
    # Stream the graph execution
    # We pass the timeline_placeholder via config so nodes can log in real-time
    ctx = get_script_run_ctx()
    config = {"configurable": {"timeline": timeline_placeholder, "streamlit_ctx": ctx}}
    
    final_state = initial_state
    
    full_logs = []
    
    for event in graph.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            # Update local tracking of state
            final_state.update(state_update)
            
            # Note: Logging is now handled inside the nodes via log_event
            if "logs" in state_update:
                full_logs.extend(state_update["logs"])
            
            # Identify final response
            if "messages" in state_update and state_update["messages"]:
                 final_response = state_update["messages"][-1]

    # Extract results
    df = None
    if final_state.get("tabla_consulta"):
        df = pd.DataFrame(final_state["tabla_consulta"])

    table_res = df if "tabla" in final_state.get("peticiones", []) else None
    
    # Check for file
    file_res = None
    if "archivo" in final_state.get("peticiones", []) and final_state.get("nombre_archivo"):
         file_res = final_state["nombre_archivo"]

    # Check for graphics - we don't have a chart object, but if we had code to gen it, we'd pass it. 
    # For now, we rely on the generic 'generador_graficos' logic. 
    # Does app.py expect a plot figure? 
    # "if msg.get("chart") is not None: st.pyplot(msg["chart"])"
    # My agent doesn't produce a matplotlib figure object in the state yet.
    # I should update 'generador_graficos' to produce one OR handle it here.
    
    chart_res = None
    if "grafica" in final_state.get("peticiones", []) and df is not None:
         fig, ax = plt.subplots()
         # Simple bar chart based on first numerical column vs first string column?
         # Heuristic:
         try:
             num_cols = df.select_dtypes(include=['number']).columns
             cat_cols = df.select_dtypes(include=['object', 'string']).columns
             if len(num_cols) > 0 and len(cat_cols) > 0:
                 ax.bar(df[cat_cols[0]], df[num_cols[0]])
                 ax.set_xlabel(cat_cols[0])
                 ax.set_ylabel(num_cols[0])
                 chart_res = fig
         except:
             pass

    text_res = final_response if final_response else "Proceso completado."

    return {
        "text": text_res,
        "table": table_res,
        "chart": chart_res,
        "file": file_res, 
        "logs": full_logs
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
