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

# Define the State
class AgentState(TypedDict):
    messages: List[str]
    peticiones: List[str]  # ["grafica", "tabla", "archivo"]
    nombre_archivo: str
    consulta_sql: str
    error_sql: str
    tabla_consulta: dict  # Check serialization, or store as json-compatible dict
    logs: List[dict]
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
    # Este nodo analisa que resultados desea obtener el usuario
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    # Pre-log or Post-log? User wants to see it "al pasar por cada nodo".
    # I'll log successful execution at the end of the node as done before, 
    # but since I am calling it here, it renders immediately.
    
    llm = get_llm()
    prompt = f"""Analiza la siguiente petición del usuario: "{state['user_input']}"
    
    Identifica:
    1. Si el usuario quiere una "grafica", "tabla" y/o "archivo".
    2. Si quiere un archivo, extrae el nombre deseado. Si no lo especifica, genera uno descriptivo (ej. reporte_ventas.csv).
    
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
        log_event(timeline, logs, step, "Análisis de intención", "Analizado petición", status=status)
        
    return {
        "peticiones": data.get("peticiones", []),
        "nombre_archivo": data.get("nombre_archivo", ""),
        "logs": logs
    }

def generador_sql(state: AgentState, config: RunnableConfig):
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    llm = get_llm()
    schema = get_db_schema()
    context_error = ""
    if state.get("error_sql"):
        context_error = f"La consulta anterior falló con: {state['error_sql']}. Corrige el error."

    prompt = f"""
    Eres un experto en SQL. Genera UNA sola consulta SQL SELECT para MySQL basada en:
    Solicitud: "{state['user_input']}"
    Schema:
    {schema}
    
    {context_error}
    
    Responde ÚNICAMENTE con la consulta SQL pura, sin markdown, sin explicaciones.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
    
    if timeline:
        log_event(timeline, logs, step, "Generador SQL", "Generado SQL", status="OK")
    
    return {
        "consulta_sql": sql,
        "error_sql": "",
        "logs": logs
    }

def ejecutador_sql(state: AgentState, config: RunnableConfig):
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
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
            log_event(timeline, logs, step, "Ejecutador SQL", f"Ejecutado consulta SQL: {sql}", status="OK")

        return {
            "tabla_consulta": df_data,
            "logs": logs
        }
    except Exception as e:
        if timeline:
            log_event(timeline, logs, step, "Ejecutador SQL", f"Error ejecutando consulta SQL: {sql}", status="ERROR", detail=str(e))
            
        return {
            "error_sql": str(e),
            "logs": logs
        }

def generador_graficos(state: AgentState, config: RunnableConfig):
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    if timeline:
        log_event(timeline, logs, step, "Generador Gráficos", "Preparado gráfico", status="OK")
        
    return {
        "logs": logs
    }

def generador_archivos(state: AgentState, config: RunnableConfig):
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    filename = state.get("nombre_archivo", "output.csv")
    data = state.get("tabla_consulta", [])
    
    if not data:
        if timeline:
            log_event(timeline, logs, step, "Generador Archivos", "No hay datos", status="WARNING")
        return {"logs": logs}
    
    df = pd.DataFrame(data)
    output_path = os.path.join(os.getcwd(), filename)
    if filename.endswith(".xlsx"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
        
    if timeline:
        log_event(timeline, logs, step, "Generador Archivos", f"Archivo guardado: {filename}", status="OK", detail=filename)
        
    return {
        "logs": logs
    }

def generador_tablas(state: AgentState, config: RunnableConfig):
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    if timeline:
        log_event(timeline, logs, step, "Generador Tablas", "Preparada tabla", status="OK")
        
    return {
        "logs": logs
    }

def generador_respuesta(state: AgentState, config: RunnableConfig):
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    llm = get_llm()
    data = state.get("tabla_consulta", [])
    
    prompt = f"""
    Genera una respuesta natural y breve para el usuario basada en su petición: "{state['user_input']}"
    y el hecho de que se han generado los resultados solicitados ({', '.join(state['peticiones'])}).
    
    Resumen de datos obtenidos (primeras 3 filas): {str(data[:3])}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    if timeline:
        log_event(timeline, logs, step, "Generador Respuesta", "Respuesta final", status="OK")
    
    return {
        "messages": [response.content],
        "logs": logs
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
    timeline = config.get("configurable", {}).get("timeline")
    logs = state.get("logs", [])
    step = len(logs) + 1
    
    if timeline:
        log_event(timeline, logs, step, "Router", "Distribuyendo tareas", status="OK")
        
    return {"logs": logs}

    
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
