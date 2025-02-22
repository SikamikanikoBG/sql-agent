import json
from typing import Dict, Any, TypedDict, Annotated, Union
from langgraph.graph import Graph, StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class SQLAgentOrchestrator:
    def __init__(self, openai_api_key: str, server: str = None, database: str = None):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key)
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
        
        response = self.llm(prompt.format_messages(query=state["user_input"]))
        state["parsed_intent"] = response.content
        return state
    
    def _generate_sql_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query based on parsed intent and metadata."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL query generator. Generate queries using ONLY the tables and columns available in the schema.
            
Available tables and their schemas:
{metadata}

Rules:
1. ONLY use tables and columns that exist in the schema above
2. If the requested tables or columns don't exist, respond with 'ERROR: Required tables/columns not found in schema'
3. Do not invent or assume the existence of any tables or columns
4. Return only the SQL query, no explanations"""),
            ("user", "{intent}")
        ])
        
        response = self.llm(prompt.format_messages(
            metadata=json.dumps(state["metadata"], indent=2),
            intent=state["parsed_intent"]
        ))
        state["generated_query"] = response.content
        return state
    
    def _validate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated SQL query."""
        query = state["generated_query"].upper()
        
        # Check if it's an error message from the LLM
        if query.startswith('ERROR:'):
            state["is_valid"] = False
            state["error"] = query
            return state
            
        # Enhanced validation
        if not query.strip():
            state["is_valid"] = False
            state["error"] = "Empty query generated"
            return state
            
        if "SELECT" not in query:
            state["is_valid"] = False
            state["error"] = "Query must include SELECT statement"
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
        return self.workflow.invoke(initial_state)
