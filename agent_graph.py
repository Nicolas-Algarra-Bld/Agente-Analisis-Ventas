import os
import mysql.connector
import pandas as pd
from typing import TypedDict, List, Annotated
import operator
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
import json
from timeline import log_event
import socket
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Helper for Context
def ensure_streamlit_context(config):
    ctx = config.get("configurable", {}).get("streamlit_ctx")
    if ctx:
        add_script_run_ctx(threading.current_thread(), ctx)

# Define the State
class AgentState(TypedDict):
    messages: List[str]
    peticiones: List[str]  # ["grafica", "tabla", "archivo"]
    nombre_archivo: str
    consulta_sql: str
    error_sql: str
    tabla_consulta: dict  # Check serialization, or store as json-compatible dict
    logs: Annotated[List[dict], operator.add]
    user_input: str

# Helper for Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "172.17.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def get_db_schema():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        schema = ""
        for (table_name,) in tables:
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            schema += f"Table: {table_name}\nColumns:\n"
            for col in columns:
                schema += f"  - {col[0]} ({col[1]})\n"
        conn.close()
        return schema
    except Exception as e:
        return f"Error getting schema: {str(e)}"

# Helper for LLM
def get_llm():
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

# --- Nodes ---

def analisis_intencion(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    # Este nodo analisa que resultados desea obtener el usuario
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    # Pre-log or Post-log? User wants to see it "al pasar por cada nodo".
    # I'll log successful execution at the end of the node as done before, 
    # but since I am calling it here, it renders immediately.
    
    llm = get_llm()
    prompt = f"""Analiza la siguiente petición del usuario: "{state['user_input']}"
    
    Identifica:
    1. Si el usuario quiere una "grafica", "tabla" y/o "archivo".
    2. Si quiere un archivo, extrae el nombre deseado. Si no lo especifica, genera uno descriptivo (ej. reporte_ventas.csv). Este archivo sera UNICAMENTE un archivo .csv o .xlsx.
    
    Responde ÚNICAMENTE con un JSON con este formato:
    {{
        "peticiones": ["grafica", "archivo"],
        "nombre_archivo": "ventas_2024.csv"
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    
    try:
        data = json.loads(content)
        status = "OK"
    except:
        data = {"peticiones": [], "nombre_archivo": "output.csv"}
        status = "WARNING"
        
    if timeline:
        log_event(timeline, render_logs, step, "Análisis de intención", f"Se identificaron la(s) petición(es): {data.get('peticiones', [])}", status=status)
        
    return {
        "peticiones": data.get("peticiones", []),
        "nombre_archivo": data.get("nombre_archivo", ""),
        "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
    }

def generador_sql(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    llm = get_llm()
    schema = get_db_schema()
    context_error = ""
    if state.get("error_sql"):
        context_error = f"La consulta anterior falló con: {state['error_sql']}. Corrige el error."

    prompt = f"""
    Eres un experto en SQL. Genera UNA sola consulta SQL SOLAMENTE SELECT para MySQL basada en:
    Solicitud: "{state['user_input']}"
    Schema:
    {schema}
    
    {context_error}
    
    Responde ÚNICAMENTE con la consulta SQL SELECT pura, sin markdown, sin explicaciones, 
    y en caso de que el usuario solicite guardar en algun archivo información IGNORALO y 
    solo genera la consulta que da la información deseada por el usuario, NO intentes guardar ni crear archivos.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
    
    if timeline:
        log_event(timeline, render_logs, step, "Generador SQL", f"Consulta SQL generada: {sql}", status="OK")
    
    return {
        "consulta_sql": sql,
        "error_sql": "",
        "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
    }

def ejecutador_sql(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    sql = state["consulta_sql"]
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        df_data = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        
        if timeline:
            log_event(timeline, render_logs, step, "Ejecutador SQL", "Consulta SQL ejecutada", status="OK")

        return {
            "tabla_consulta": df_data,
            "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
        }
    except Exception as e:
        if timeline:
            log_event(timeline, render_logs, step, "Ejecutador SQL", "Error ejecutando consulta SQL", status="ERROR", detail=str(e))
            
        return {
            "error_sql": str(e),
            "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
        }

def generador_graficos(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    if timeline:
        log_event(timeline, render_logs, step, "Generador Gráficos", "Preparado gráfico", status="OK")
        
    return {
        "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
    }

def generador_archivos(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    filename = state.get("nombre_archivo", "output.csv")
    data = state.get("tabla_consulta", [])
    
    if not data:
        if timeline:
            log_event(timeline, render_logs, step, "Generador Archivos", "No hay datos en la consulta", status="WARNING")
        return {"logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []}
    
    df = pd.DataFrame(data)
    output_path = os.path.join(os.getcwd(), f"outputs/{filename}")
    if filename.endswith(".xlsx"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
        
    if timeline:
        log_event(timeline, render_logs, step, "Generador Archivos", f"Archivo guardado: {output_path}", status="OK", detail=filename)
        
    return {
        "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
    }

def generador_tablas(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    if timeline:
        log_event(timeline, render_logs, step, "Generador Tablas", "Preparada tabla", status="OK")
        
    return {
        "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
    }

def generador_respuesta(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    llm = get_llm()
    data = state.get("tabla_consulta", [])
    
    prompt = f"""
    Genera una respuesta natural y breve para el usuario basada en su petición: "{state['user_input']}"
    y el hecho de que se han generado los resultados solicitados ({', '.join(state['peticiones'])}).
    
    Resumen de datos obtenidos: {str(data)}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    if timeline:
        log_event(timeline, render_logs, step, "Generador Respuesta", "Respuesta final", status="OK")
    
    return {
        "messages": [response.content],
        "logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []
    }

# Connection Router
def route_sql_result(state: AgentState):
    if state.get("error_sql"):
        return "generador_sql"
    return "router"

def router_split(state: AgentState):
    reqs = state.get("peticiones", [])
    routes = []
    if "grafica" in reqs:
        routes.append("generador_graficos")
    if "archivo" in reqs:
        routes.append("generador_archivos")
    if "tabla" in reqs:
        routes.append("generador_tablas")
    
    if not routes:
        return ["generador_respuesta"]
        
    return routes

def router_node(state: AgentState, config: RunnableConfig):
    ensure_streamlit_context(config)
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    render_logs = list(logs)
    start_len = len(logs)
    step = len(render_logs) + 1
    
    if timeline:
        log_event(timeline, render_logs, step, "Router", "Distribuyendo tareas", status="OK")
        
    return {"logs": [render_logs[-1]] if timeline and len(render_logs) > start_len else []}

    
# --- Graph Construction ---
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analisis_intencion", analisis_intencion)
    workflow.add_node("generador_sql", generador_sql)
    workflow.add_node("ejecutador_sql", ejecutador_sql)
    workflow.add_node("generador_graficos", generador_graficos)
    workflow.add_node("generador_archivos", generador_archivos)
    workflow.add_node("generador_tablas", generador_tablas)
    workflow.add_node("generador_respuesta", generador_respuesta)
    workflow.add_node("router", router_node)
    
    workflow.set_entry_point("analisis_intencion")
    workflow.add_edge("analisis_intencion", "generador_sql")
    workflow.add_edge("generador_sql", "ejecutador_sql")
    
    workflow.add_conditional_edges(
        "ejecutador_sql",
        route_sql_result,
        {
            "generador_sql": "generador_sql",
            "router": "router"
        }
    )
    
    workflow.add_conditional_edges(
        "router",
        router_split,
        [
            "generador_graficos", 
            "generador_archivos", 
            "generador_tablas",
            "generador_respuesta"
        ] # Note: if split returns multiple, they run in parallel? LangGraph supports list return for map-reduce essentially?
          # Yes, if map-reduce is set up. But "router_split" returning a list works for branching.
          # But we need them to join back at "generador_respuesta"?
          # The diagram/text lists "Generador respuesta" as separate or implied final?
          # "En caso de estar vacia, llama al nodo Generador respuesta." - implies if list is empty, go there.
          # If list is NOT empty, we go to graph/file/table. And THEN what?
          # Usually they should converge to "generador_respuesta" to give the final text.
    )
    
    # Converge edges
    workflow.add_edge("generador_graficos", "generador_respuesta")
    workflow.add_edge("generador_archivos", "generador_respuesta")
    workflow.add_edge("generador_tablas", "generador_respuesta")

    # From generador_respuesta to END
    workflow.add_edge("generador_respuesta", END)
    
    return workflow.compile()
