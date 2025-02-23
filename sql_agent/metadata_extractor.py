import re
from typing import List, Dict, Any

def extract_metadata_from_sql_files(files: List[str]) -> Dict[str, Any]:
    """Extract metadata from SQL files including tables."""
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

    metadata = []
    relationships = []
    
    for file in files:
        with open(file, 'r') as f:
            sql_content = f.read()
            
            # Extract tables using regex
            tables = re.findall(r'CREATE TABLE (\w+)', sql_content)
            for table in tables:
                metadata.append({
                    'type': 'table',
                    'name': table,
                    'schema': [],
                    'source_file': file
                })
                relationships.extend([
                    {
                        'from_table': table,
                        'from_column': None,
                        'to_table': None,
                        'to_column': None
                    }
                ])
            
            # Extract views using regex
            views = re.findall(r'CREATE VIEW (\w+) AS', sql_content)
            for view in views:
                metadata.append({
                    'type': 'view',
                    'name': view,
                    'definition': '',
                    'source_file': file
                })
            
            # Extract procedures using regex
            procedures = re.findall(r'CREATE PROCEDURE (\w+)', sql_content)
            for proc in procedures:
                metadata.append({
                    'type': 'procedure',
                    'name': proc,
                    'parameters': [],
                    'body': '',
                    'source_file': file,
                    'description': ''
                })
    
    # Organize metadata into a more structured format
    return {
        "tables": [item["name"] for item in metadata if item["type"] == "table"],
        "views": [item["name"] for item in metadata if item["type"] == "view"],
        "procedures": [item["name"] for item in metadata if item["type"] == "procedure"],
        "schemas": {
            item["name"]: [] for item in metadata if item["type"] == "table"
        },
        "view_definitions": {
            item["name"]: '' for item in metadata if item["type"] == "view"
        },
        "procedure_info": {
            item["name"]: {
                "parameters": [],
                "description": '',
                "body": ''
            } for item in metadata if item["type"] == "procedure"
        },
        "relationships": relationships,
        "raw": metadata
    }
