import re
import json
from typing import List, Dict, Any

from .mssql_utils import MSSQLConnection

def extract_metadata_from_server(server: str, database: str, 
                               trusted_connection: bool = True,
                               username: str = None, 
                               password: str = None) -> Dict[str, Any]:
    """Extract metadata from SQL Server including tables, views, and their schemas."""
    conn = MSSQLConnection(
        server=server,
        database=database,
        username=username,
        password=password,
        trusted_connection=trusted_connection
    )
    
    return conn.extract_metadata()
            
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
    
    # Organize metadata into a more structured format
    return {
        "tables": [item["name"] for item in metadata if item["type"] == "table"],
        "views": [item["name"] for item in metadata if item["type"] == "view"],
        "schemas": {
            item["name"]: item["schema"] for item in metadata if item["type"] == "table"
        },
        "view_definitions": {
            item["name"]: item["definition"] for item in metadata if item["type"] == "view"
        },
        "raw": metadata
    }

def _parse_schema(schema_text: str) -> List[Dict]:
    """Parse column definitions from schema text."""
    columns = []
    for column in schema_text.split(','):
        if column.strip():
            parts = column.strip().split()
            columns.append({
                'name': parts[0],
                'type': parts[1],
                'constraints': ' '.join(parts[2:]) if len(parts) > 2 else ''
            })
    return columns
