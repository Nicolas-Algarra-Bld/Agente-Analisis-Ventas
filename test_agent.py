import sys
from unittest.mock import MagicMock

# Mock mysql.connector
mock_mysql = MagicMock()
sys.modules["mysql"] = mock_mysql
sys.modules["mysql.connector"] = mock_mysql

# Restore Pandas Mock since it is not installed in this environment
mock_pandas = MagicMock()
sys.modules["pandas"] = mock_pandas

# Mock matplotlib
mock_matplotlib = MagicMock()
mock_pyplot = MagicMock()
mock_matplotlib.pyplot = mock_pyplot
sys.modules["matplotlib"] = mock_matplotlib
sys.modules["matplotlib.pyplot"] = mock_pyplot

# ... (rest of imports) ...



import unittest
from unittest.mock import patch
# We need to mock imports BEFORE usage in agent_graph
# But verify if langgraph is installed. Since user said "build using LangGraph", 
# and requirements.txt was updated, maybe they are working or maybe not.
# If I can't install, I must mock everything to prove logic correctness.

# Let's try to import agent_graph and catch errors to mock more.
try:
    import agent_graph
except ImportError as e:
    # Aggressively mock missing modules
    missing_module = str(e).split("'")[-2]
    sys.modules[missing_module] = MagicMock()
    # Retry import? No, python caches imports.
    # I need to know what's missing.
    pass

# Force mock of langchain deps if we suspect they are missing
sys.modules["langchain_aws"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.messages"] = MagicMock()
sys.modules["langchain_core.runnables"] = MagicMock()
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["timeline"] = MagicMock()
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit.runtime"] = MagicMock()
sys.modules["streamlit.runtime.scriptrunner"] = MagicMock()


# Now define what agent_graph needs from those mocks that are used at MODULE LEVEL
# agent_graph.py: "from langgraph.graph import StateGraph, END"
sys.modules["langgraph.graph"].StateGraph = MagicMock()
sys.modules["langgraph.graph"].END = "END"

# agent_graph.py: "class AgentState(TypedDict):" -> TypedDict is standard.
# agent_graph.py: "from langchain_aws import ChatBedrock"

import agent_graph
from agent_graph import create_graph

class TestAgentGraph(unittest.TestCase):
    
    @patch('agent_graph.get_llm')
    @patch('agent_graph.get_db_connection')
    def test_graph_flow(self, mock_db_conn, mock_get_llm):
        # Setup log_event side effect to simulate appending
        def mock_log_append(timeline, logs, *args, **kwargs):
            logs.append({"step": "mock"})
            
        agent_graph.log_event.side_effect = mock_log_append

        # Since we mocked the whole graph engine, create_graph returns a Mock.
        # So we can't test actual flow execution nicely unless we partially mock or use real langgraph.
        # Assuming langgraph MIGHT be present but pandas/mysql are implementation details.
        
        # If I mocked StateGraph, I can't run the graph. 
        # I only want to test the NODE functions logic in isolation?
        
        # Let's test the nodes individually!
        
        state = {
            "user_input": "Quiero una gráfica",
            "logs": [],
            "peticiones": [],
            "messages": [],
            "nombre_archivo": "",
            "consulta_sql": "",
            "error_sql": "",
            "tabla_consulta": []
        }
        config = {"configurable": {"timeline": MagicMock()}}
        
        # Test 1: Analisis Intencion
        # Mock LLM inside node
        mock_llm_instance = MagicMock()
        mock_get_llm.return_value = mock_llm_instance
        mock_llm_instance.invoke.return_value.content = '{"peticiones": ["grafica"]}'
        
        res = agent_graph.analisis_intencion(state, config)
        self.assertIn("grafica", res["peticiones"])
        
        # Test 2: Generador SQL
        state["peticiones"] = ["grafica"]
        mock_llm_instance.invoke.return_value.content = "SELECT * FROM t"
        res_sql = agent_graph.generador_sql(state, config)
        self.assertEqual(res_sql["consulta_sql"], "SELECT * FROM t")
        
        # Test 3: Ejecutador SQL -> need to mock DB
        state["consulta_sql"] = "SELECT * FROM t"
        mock_cursor = MagicMock()
        mock_db_conn.return_value.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [(10,)]
        mock_cursor.description = [("col1",)]
        
        res_exec = agent_graph.ejecutador_sql(state, config)
        self.assertEqual(res_exec["tabla_consulta"], [{"col1": 10}])
        
        # Test 4: Log Accumulation (Use the new returns)
        logs = []
        state["logs"] = logs
        render_logs = list(logs)
        start_len = len(logs)
        
        # Simulate node 1
        res1 = agent_graph.analisis_intencion(state, config)
        logs.extend(res1["logs"])
        
        # Simulate node 2
        state["logs"] = logs # Update state for next node
        res2 = agent_graph.generador_sql(state, config)
        logs.extend(res2["logs"])
        
        self.assertEqual(len(logs), 2) # Should have 2 log entries now
        
        # Test 5: Graph Generation with String Numbers
        state["tabla_consulta"] = [{"producto": "A", "ventas": "100"}, {"producto": "B", "ventas": "200"}]
        # Mock matplotlib savefig to just do nothing or verify call
        mock_pyplot.savefig.return_value = None
        
        # Configure pd mock to simulate finding numeric columns
        mock_df = MagicMock()
        mock_pandas.DataFrame.return_value = mock_df
        
        def side_effect_select_dtypes(*args, **kwargs):
            include = kwargs.get("include")
            m = MagicMock()
            if include == ['number']:
                m.columns = ['ventas']
            elif include == ['object', 'string']:
                m.columns = ['producto']
            else:
                m.columns = []
            return m
            
        mock_df.select_dtypes.side_effect = side_effect_select_dtypes
        
        res_graph = agent_graph.generador_graficos(state, config)
        self.assertTrue(res_graph["grafica_path"].endswith(".png"))
        self.assertIn("outputs", res_graph["grafica_path"])
        
        print("Unit tests of nodes passed.")

if __name__ == '__main__':
    unittest.main()
