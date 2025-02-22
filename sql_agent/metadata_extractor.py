import re
from typing import List, Dict, Any
from sql_agent.metadata_agent import MetadataExtractionAgent
from langchain_openai import ChatOpenAI

def extract_metadata_from_sql_files(files: List[str], openai_api_key: str = None) -> Dict[str, Any]:
    """Extract metadata from SQL files including tables, views, procedures and their schemas."""
    if not files:
        return {
            "tables": [],
            "views": [],
            "procedures": [],
            "schemas": {},
            "view_definitions": {},
            "procedure_info": {},
            "relationships": [],
            "raw": [],
            "error": "No SQL files provided"
        }

    # Initialize the LLM-based metadata extraction agent
    llm = ChatOpenAI(
        model="gpt-4-0125-preview",
        openai_api_key=openai_api_key,
        temperature=0
    )
    agent = MetadataExtractionAgent(llm)
    
    metadata = []
    relationships = []
    
    for file in files:
        with open(file, 'r') as f:
            sql_content = f.read()
            
            # Extract tables using LLM
            tables = agent.extract_tables(sql_content)
            for table in tables:
                metadata.append({
                    'type': 'table',
                    'name': table['name'],
                    'schema': table['columns'],
                    'source_file': file
                })
                relationships.extend([
                    {
                        'from_table': table['name'],
                        'from_column': rel['from_column'],
                        'to_table': rel['to_table'],
                        'to_column': rel['to_column']
                    }
                    for rel in table.get('relationships', [])
                ])
            
            # Extract views using LLM
            views = agent.extract_views(sql_content)
            for view in views:
                metadata.append({
                    'type': 'view',
                    'name': view['name'],
                    'definition': view['definition'],
                    'source_file': file
                })
            
            # Extract procedures using LLM
            procedures = agent.extract_procedures(sql_content)
            for proc in procedures:
                metadata.append({
                    'type': 'procedure',
                    'name': proc['name'],
                    'parameters': proc['parameters'],
                    'body': proc['body'],
                    'source_file': file,
                    'description': proc['description']
                })
    
    # Organize metadata into a more structured format
    return {
        "tables": [item["name"] for item in metadata if item["type"] == "table"],
        "views": [item["name"] for item in metadata if item["type"] == "view"],
        "procedures": [item["name"] for item in metadata if item["type"] == "procedure"],
        "schemas": {
            item["name"]: item["schema"] for item in metadata if item["type"] == "table"
        },
        "view_definitions": {
            item["name"]: item["definition"] for item in metadata if item["type"] == "view"
        },
        "procedure_info": {
            item["name"]: {
                "parameters": item["parameters"],
                "description": item["description"],
                "body": item["body"]
            } for item in metadata if item["type"] == "procedure"
        },
        "relationships": relationships,
        "raw": metadata
    }

def _parse_schema(schema_text: str) -> List[Dict]:
    """Parse column definitions from schema text."""
    columns = []
    current_column = []
    paren_count = 0
    
    # Split on commas, but respect parentheses in type definitions
    for char in schema_text:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        
        if char == ',' and paren_count == 0:
            if current_column:
                col_def = ''.join(current_column).strip()
                if col_def:
                    parts = col_def.split(None, 2)  # Split into max 3 parts
                    if parts:
                        columns.append({
                            'name': parts[0],
                            'type': parts[1] if len(parts) > 1 else 'UNKNOWN',
                            'constraints': parts[2] if len(parts) > 2 else ''
                        })
            current_column = []
        else:
            current_column.append(char)
    
    # Don't forget the last column
    if current_column:
        col_def = ''.join(current_column).strip()
        if col_def:
            parts = col_def.split(None, 2)
            if parts:
                columns.append({
                    'name': parts[0],
                    'type': parts[1] if len(parts) > 1 else 'UNKNOWN',
                    'constraints': parts[2] if len(parts) > 2 else ''
                })
    
    return columns

def _parse_parameters(params_text: str) -> List[Dict]:
    """Parse stored procedure parameters."""
    if not params_text.strip():
        return []
        
    parameters = []
    # Split on commas but not within parentheses
    params = re.findall(r'@\w+\s+[^,]+(?:,|$)', params_text)
    
    for param in params:
        param = param.strip().strip(',')
        if param:
            # Extract parameter components
            param_match = re.match(
                r'@(\w+)\s+([\w\(\)\d,\s]+)(?:\s+(OUTPUT|OUT|INPUT|IN))?(?:\s+=\s+[^,]+)?',
                param,
                re.IGNORECASE
            )
            if param_match:
                name, type_info, direction = param_match.groups()
                parameters.append({
                    'name': name,
                    'type': type_info.strip(),
                    'direction': (direction or 'IN').upper()
                })
    return parameters

def _extract_procedure_description(body: str) -> str:
    """Extract procedure description from comments in the body."""
    # Look for block comments at the start
    block_comment = re.search(r'^\s*/\*\s*(.*?)\s*\*/', body, re.DOTALL)
    if block_comment:
        desc = block_comment.group(1).strip()
        # Clean up common documentation markers
        desc = re.sub(r'[@=\-_]+\s*', ' ', desc)
        return ' '.join(line.strip() for line in desc.split('\n'))
    
    # Look for consecutive single line comments at the start
    lines = body.lstrip().split('\n')
    comments = []
    for line in lines:
        line = line.strip()
        if line.startswith('--'):
            comments.append(line[2:].strip())
        elif not line:
            continue
        else:
            break
    
    if comments:
        return ' '.join(comments)
    
    return ""
