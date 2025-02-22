from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class MetadataExtractionAgent:
    def __init__(self, llm):
        self.llm = llm

    def extract_tables(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract table definitions using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL metadata extraction specialist. 
Analyze the SQL content and extract table definitions.
Return ONLY a JSON array of table objects with these properties:
- name: table name
- columns: array of column objects with name, type, and constraints
- relationships: array of foreign key relationships

Example output:
[{
    "name": "users",
    "columns": [
        {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY"},
        {"name": "email", "type": "VARCHAR(255)", "constraints": "UNIQUE NOT NULL"}
    ],
    "relationships": [
        {"from_column": "role_id", "to_table": "roles", "to_column": "id"}
    ]
}]"""),
            ("user", "SQL Content: {sql_content}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(sql_content=sql_content))
        try:
            import json
            # Clean the response content by stripping whitespace and ensuring it starts with [
            content = response.content.strip()
            if not content.startswith('['):
                return []
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during table extraction: {e}")
            return []

    def extract_views(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract view definitions using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL metadata extraction specialist.
Analyze the SQL content and extract view definitions.
Return ONLY a JSON array of view objects with these properties:
- name: view name
- definition: the SQL definition

Example output:
[{
    "name": "active_users",
    "definition": "SELECT * FROM users WHERE status = 'active'"
}]"""),
            ("user", "SQL Content: {sql_content}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(sql_content=sql_content))
        try:
            import json
            content = response.content.strip()
            if not content.startswith('['):
                return []
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during view extraction: {e}")
            return []

    def extract_procedures(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract stored procedure definitions using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL metadata extraction specialist.
Analyze the SQL content and extract stored procedure definitions.
Return ONLY a JSON array of procedure objects with these properties:
- name: procedure name
- parameters: array of parameter objects with name, type, and direction
- description: extracted from comments
- body: the procedure body

Example output:
[{
    "name": "create_user",
    "parameters": [
        {"name": "email", "type": "VARCHAR(255)", "direction": "IN"},
        {"name": "user_id", "type": "INTEGER", "direction": "OUT"}
    ],
    "description": "Creates a new user account",
    "body": "INSERT INTO users (email) VALUES (@email)..."
}]"""),
            ("user", "SQL Content: {sql_content}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(sql_content=sql_content))
        try:
            import json
            content = response.content.strip()
            if not content.startswith('['):
                return []
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during procedure extraction: {e}")
            return []
