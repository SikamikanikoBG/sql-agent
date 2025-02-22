import re
import json
from typing import List, Dict

def extract_metadata_from_sql_files(files: List[str]) -> List[Dict]:
    """Extract metadata from SQL files including tables, views, and their schemas."""
    metadata = []
    
    for file in files:
        with open(file, 'r') as f:
            sql_content = f.read()
            
            # Extract table and view definitions
            tables = re.findall(r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);', 
                              sql_content, re.DOTALL | re.IGNORECASE)
            views = re.findall(r'CREATE\s+VIEW\s+(\w+)\s+AS\s+(.*?);',
                             sql_content, re.DOTALL | re.IGNORECASE)
            
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
    
    return metadata

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
