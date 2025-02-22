import json
import re
from typing import Dict, Any, TypedDict, Annotated, Union
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class SQLAgentOrchestrator:
    def __init__(self, openai_api_key: str, server: str = None, database: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # gpt-4-turbo
            openai_api_key=openai_api_key
        )
        self.server = server
        self.database = database
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the Langgraph workflow for SQL query generation."""
        # Define the schema for our state
        class AgentState(TypedDict):
            user_input: str
            metadata: Dict
            parsed_intent: Annotated[str, "The parsed user intent"]
            generated_query: Annotated[str, "The generated SQL query"]
            is_valid: Annotated[bool, "Whether the query is valid"]
            error: Union[str, None]  # Error message if query is invalid

        # Initialize the workflow with the schema
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("parse_intent", self._parse_user_intent)
        workflow.add_node("generate_query", self._generate_sql_query)
        workflow.add_node("validate_query", self._validate_sql)
        
        # Define edges
        workflow.add_edge("parse_intent", "generate_query")
        workflow.add_edge("generate_query", "validate_query")
        
        # Set the entry point
        workflow.set_entry_point("parse_intent")
        
        # Compile the workflow
        return workflow.compile()
    
    def _parse_user_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse user's natural language query intent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the key elements from this database query request."),
            ("user", "{query}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(query=state["user_input"]))
        state["parsed_intent"] = response.content
        return state
    
    def _generate_sql_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query based on parsed intent and metadata."""
        # Check metadata state
        if not state.get("metadata"):
            state["generated_query"] = "ERROR: No metadata provided"
            return state
            
        if "error" in state["metadata"]:
            state["generated_query"] = f"ERROR: {state['metadata']['error']}"
            return state

        if not any(state["metadata"].get(key) for key in ["tables", "views", "procedures"]):
            state["generated_query"] = "ERROR: No database objects found in the provided SQL files"
            return state

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL query generator. Generate queries using ONLY the available database objects.
            
Available database objects:
{metadata}

Available Stored Procedures:
{procedures}

IMPORTANT: If there are stored procedures that match the user's intent, ALWAYS prefer using them over writing new queries.
Use EXEC or EXECUTE to call procedures with appropriate parameters.

Rules:
1. ONLY use tables, views, and procedures that exist in the schema
2. If the required database objects don't exist, respond with 'ERROR: Required database objects not found'
3. Do not invent or assume the existence of any database objects
4. For complex operations, ALWAYS check and use existing stored procedures first
5. Return only the SQL query or procedure call, no explanations
6. When using stored procedures, follow their exact parameter requirements"""),
            ("user", "{intent}")
        ])
        
        # Format procedure information for the prompt
        procedures_info = ""
        if "procedure_info" in state["metadata"]:
            for proc_name, info in state["metadata"]["procedure_info"].items():
                procedures_info += f"\n{proc_name}:\n"
                procedures_info += f"  Description: {info['description']}\n"
                procedures_info += "  Parameters:\n"
                for param in info['parameters']:
                    procedures_info += f"    - @{param['name']} ({param['type']}) {param['direction']}\n"

        response = self.llm.invoke(prompt.format_messages(
            metadata=json.dumps({k:v for k,v in state["metadata"].items() if k != "procedure_info"}, indent=2),
            procedures=procedures_info,
            intent=state["parsed_intent"]
        ))
        state["generated_query"] = response.content
        return state
    
    def _validate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated SQL query."""
        query = state["generated_query"].upper()
        
        # Check if it's an error message
        if query.startswith('ERROR:'):
            state["is_valid"] = False
            state["error"] = query.replace('ERROR:', '').strip()
            return state
            
        # Enhanced validation
        if not query.strip():
            state["is_valid"] = False
            state["error"] = "Empty query generated"
            return state
            
        # Allow both SELECT queries and stored procedure calls
        if not ("SELECT" in query or "EXEC" in query or "EXECUTE" in query):
            state["is_valid"] = False
            state["error"] = "Query must be either a SELECT statement or a stored procedure call"
            return state

        # Validate stored procedure calls
        if "EXEC" in query or "EXECUTE" in query:
            proc_match = re.search(r'(?:EXEC|EXECUTE)\s+(\w+)', query)
            if proc_match:
                proc_name = proc_match.group(1)
                if proc_name not in state["metadata"].get("procedures", []):
                    state["is_valid"] = False
                    state["error"] = f"Procedure {proc_name} does not exist"
                    return state
            
        if "DROP" in query or "DELETE" in query or "TRUNCATE" in query:
            state["is_valid"] = False
            state["error"] = "Destructive operations not allowed"
            return state
            
        # Extract table names from query
        # Simple regex to match table names after FROM and JOIN
        import re
        tables_in_query = set(re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', query))
        tables_in_query = {t[0] or t[1] for t in tables_in_query}  # Flatten tuple matches
        
        # Check if all tables exist in metadata
        available_tables = set(state["metadata"].get("tables", []))
        unknown_tables = tables_in_query - available_tables
        
        if unknown_tables:
            state["is_valid"] = False
            state["error"] = f"Query references non-existent tables: {', '.join(unknown_tables)}"
            return state
            
        state["is_valid"] = True
        state["error"] = None
        return state
    
    def process_query(self, user_input: str, metadata: Dict) -> Dict[str, Any]:
        """Process a user query through the workflow."""
        initial_state = {
            "user_input": user_input,
            "metadata": metadata,
            "parsed_intent": "",
            "generated_query": "",
            "is_valid": False,
            "error": None
        }
        result = self.workflow.invoke(initial_state)
        # Ensure parsed intent is included in the output
        if not result.get("parsed_intent"):
            result["parsed_intent"] = "Intent parsing failed"
        return result
