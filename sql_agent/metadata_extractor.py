import re
import json
from typing import List, Dict, Any

def extract_metadata_from_sql_files(files: List[str]) -> Dict[str, Any]:
    """Extract metadata from SQL files including tables, views, procedures and their schemas."""
    if not files:
        return {
            "tables": [],
            "views": [],
            "procedures": [],
            "schemas": {},
            "view_definitions": {},
            "procedure_info": {},
            "raw": [],
            "error": "No SQL files provided"
        }

    metadata = []
    
    for file in files:
        with open(file, 'r') as f:
            sql_content = f.read()
            
            # Extract table definitions
            tables = re.findall(r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);', 
                              sql_content, re.DOTALL | re.IGNORECASE)
            
            # Extract view definitions
            views = re.findall(r'CREATE\s+VIEW\s+(\w+)\s+AS\s+(.*?);',
                             sql_content, re.DOTALL | re.IGNORECASE)
            
            # Extract stored procedure definitions with more flexible T-SQL syntax
            procedures = re.findall(
                r'CREATE\s+(?:OR\s+ALTER\s+)?PROC(?:EDURE)?\s+(\w+)(?:\s*\((.*?)\))?\s*(?:WITH\s+[^;]+)?(?:AS|IS)\s*(?:BEGIN)?\s*(.*?)(?:END;?|GO)',
                sql_content, re.DOTALL | re.IGNORECASE
            )
            
            # Process tables
            for table_name, schema in tables:
                metadata.append({
                    'type': 'table',
                    'name': table_name.strip(),
                    'schema': _parse_schema(schema),
                    'source_file': file
                })
            
            # Process views
            for view_name, definition in views:
                metadata.append({
                    'type': 'view',
                    'name': view_name.strip(),
                    'definition': definition.strip(),
                    'source_file': file
                })
            
            # Process stored procedures
            for proc_name, params, body in procedures:
                metadata.append({
                    'type': 'procedure',
                    'name': proc_name.strip(),
                    'parameters': _parse_parameters(params),
                    'body': body.strip(),
                    'source_file': file,
                    'description': _extract_procedure_description(body)
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
        "raw": metadata
    }

def _parse_schema(schema_text: str) -> List[Dict]:
    """Parse column definitions from schema text."""
    columns = []
    for column in schema_text.split(','):
        if column.strip():
            parts = column.strip().split()
            if parts:
                columns.append({
                    'name': parts[0],
                    'type': parts[1] if len(parts) > 1 else 'UNKNOWN',
                    'constraints': ' '.join(parts[2:]) if len(parts) > 2 else ''
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
