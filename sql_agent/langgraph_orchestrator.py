import re
from typing import List, Dict, Any

class SQLAgentOrchestrator:
    def __init__(self):
        self.metadata = None
    
    def _create_workflow(self) -> StateGraph:
        """Create the Langgraph workflow for SQL query generation."""
        # Define the schema for our state
        class AgentState(TypedDict):
            user_input: str
            metadata: Dict
            parsed_intent: Annotated[str, "The parsed user intent"]
            relevant_content: Annotated[List[Dict], "Relevant SQL content from vector search"]
            generated_query: Annotated[str, "The generated SQL query"]
            is_valid: Annotated[bool, "Whether the query is valid"]
    
    def process_query(self, user_input: str, metadata: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process a user's natural language query and generate an SQL query."""
        # Placeholder for actual processing logic
        return {
            "agent_interactions": {},
            "similarity_search": [],
            "generated_query": "SELECT * FROM users WHERE id = 1",
            "usage_stats": {
                "tokens": {"prompt": 0, "completion": 0},
                "cost": 0.0
            }
        }

    def extract_metadata(self, sql_content: str) -> List[str]:
        """Extract metadata from SQL content including tables"""
        
        try:
            match = re.search(r'CREATE TABLE (.*)', sql_content)
            while match:
                yield match.group(1)
                match = re.search(r'CREATE TABLE (.*)', sql_content, startposition=len(match.laststart))
            
            return []
        
    def extract_metadata_from_sql_files(self, files: List[str]) -> Dict[str, Any]:
        """Extract metadata from a list of SQL files."""
        if not self.metadata:
            self.metadata = {}
            for file in files:
                with open(file, 'r') as f:
                    sql_content = f.read()
                    tables = list(self.extract_metadata(sql_content))
                    if tables:
                        self.metadata[file] = {"tables": tables}
        return self.metadata
