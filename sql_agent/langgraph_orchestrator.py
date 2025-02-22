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
        
        return workflow
    
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
            ("system", "Generate a SQL query based on this intent and schema:\n{metadata}"),
            ("user", "{intent}")
        ])
        
        response = self.llm(prompt.format_messages(
            metadata=state["metadata"],
            intent=state["parsed_intent"]
        ))
        state["generated_query"] = response.content
        return state
    
    def _validate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated SQL query."""
        query = state["generated_query"].upper()
        
        # Enhanced validation
        if not query.strip():
            state["is_valid"] = False
            state["error"] = "Empty query generated"
        elif "SELECT" not in query:
            state["is_valid"] = False
            state["error"] = "Query must include SELECT statement"
        elif "DROP" in query or "DELETE" in query or "TRUNCATE" in query:
            state["is_valid"] = False
            state["error"] = "Destructive operations not allowed"
        else:
            state["is_valid"] = True
            state["error"] = None
        return state
    
    def process_query(self, user_input: str, metadata: Dict) -> Dict[str, Any]:
        """Process a user query through the workflow."""
        initial_state = {
            "user_input": user_input,
            "metadata": metadata
        }
        return self.workflow.run(initial_state)
