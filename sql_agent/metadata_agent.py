from typing import Dict, Any, List
import json
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class MetadataExtractionAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def _extract_basic_metadata(self, sql_content: str) -> Dict[str, List[str]]:
        """Extract basic metadata using regex patterns."""
        metadata = {
            'tables': [],
            'views': [],
            'procedures': []
        }
        
        # Table pattern - include both CREATE TABLE and ALTER TABLE
        table_pattern = r"(?:CREATE|ALTER)\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)"
        for match in re.finditer(table_pattern, sql_content, re.IGNORECASE):
            table_name = match.group(1).strip('[]"').strip()
            if table_name and table_name not in metadata['tables']:
                metadata['tables'].append(table_name)
            
        # View pattern    
        view_pattern = r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([^\s(]+)"
        for match in re.finditer(view_pattern, sql_content, re.IGNORECASE):
            view_name = match.group(1).strip('[]"')
            metadata['views'].append(view_name)
            
        # Procedure pattern
        proc_pattern = r"CREATE\s+PROCEDURE\s+([^\s(]+)"
        for match in re.finditer(proc_pattern, sql_content, re.IGNORECASE):
            proc_name = match.group(1).strip('[]"')
            metadata['procedures'].append(proc_name)
            
        return metadata

    def extract_tables(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract table definitions using regex first, then enhance with LLM."""
        basic_metadata = self._extract_basic_metadata(sql_content)
        # If no tables found by regex, try LLM
        if not basic_metadata['tables']:
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
        
        try:
            response = self.llm.invoke(prompt.format_messages(sql_content=sql_content))
            if not response or not response.content:
                print("Empty response from LLM")
                return []
                
            # Clean and normalize the response content
            content = response.content.strip()
            print(f"Raw LLM response: {content}")  # Debug logging
            
            # Handle common JSON formatting issues
            content = content.replace('\n', ' ').replace('\r', '')
            content = content.replace('```json', '').replace('```', '')
            
            # Ensure content is a valid JSON array
            if not content.strip().startswith('['):
                content = f"[{content}]"
                
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                content = content.replace("'", '"')  # Replace single quotes with double quotes
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON after cleanup: {e}")
                    return []
            
            if not isinstance(parsed, list):
                parsed = [parsed]
                
            # Validate each table object
            valid_tables = []
            for table in parsed:
                if not isinstance(table, dict):
                    continue
                    
                # Normalize table object
                table_obj = {
                    'name': table.get('name', '').strip(),
                    'columns': table.get('columns', []),
                    'relationships': table.get('relationships', [])
                }
                
                if table_obj['name']:  # Only add if name is not empty
                    valid_tables.append(table_obj)
                
            return valid_tables
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}\nContent: {response.content if response else 'No response'}")
            return []
        except Exception as e:
            print(f"Unexpected error during table extraction: {type(e).__name__}: {e}")
            return []

    def extract_views(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract view definitions using regex first, then enhance with LLM."""
        basic_metadata = self._extract_basic_metadata(sql_content)
        
        # If no views found by regex, try LLM
        if not basic_metadata['views']:
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
        
        try:
            response = self.llm.invoke(prompt.format_messages(sql_content=sql_content))
            if not response or not response.content:
                return []
                
            content = response.content.strip()
            content = content.replace('\n', ' ').replace('\r', '')
            content = content.replace('```json', '').replace('```', '')
            
            if not content.strip().startswith('['):
                content = f"[{content}]"
                
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                content = content.replace("'", '"')
                try:
                    parsed = json.loads(content)
                except:
                    return []
                    
            if not isinstance(parsed, list):
                parsed = [parsed]
                
            valid_views = []
            for view in parsed:
                if isinstance(view, dict) and 'name' in view:
                    view_obj = {
                        'name': view.get('name', '').strip(),
                        'definition': view.get('definition', '')
                    }
                    if view_obj['name']:
                        valid_views.append(view_obj)
                        
            return valid_views
        except Exception as e:
            print(f"Error extracting views: {e}")
            return []

    def extract_procedures(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract stored procedure definitions using regex first, then enhance with LLM."""
        basic_metadata = self._extract_basic_metadata(sql_content)
        
        # If no procedures found by regex, try LLM
        if not basic_metadata['procedures']:
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
        
        try:
            response = self.llm.invoke(prompt.format_messages(sql_content=sql_content))
            if not response or not response.content:
                return []
                
            content = response.content.strip()
            content = content.replace('\n', ' ').replace('\r', '')
            content = content.replace('```json', '').replace('```', '')
            
            if not content.strip().startswith('['):
                content = f"[{content}]"
                
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                content = content.replace("'", '"')
                try:
                    parsed = json.loads(content)
                except:
                    return []
                    
            if not isinstance(parsed, list):
                parsed = [parsed]
                
            valid_procs = []
            for proc in parsed:
                if isinstance(proc, dict) and 'name' in proc:
                    proc_obj = {
                        'name': proc.get('name', '').strip(),
                        'parameters': proc.get('parameters', []),
                        'description': proc.get('description', ''),
                        'body': proc.get('body', '')
                    }
                    if proc_obj['name']:
                        valid_procs.append(proc_obj)
                        
            return valid_procs
        except Exception as e:
            print(f"Error extracting procedures: {e}")
            return []
