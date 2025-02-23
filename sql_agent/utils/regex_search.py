import re
from typing import Dict, List, Optional

def search_sql_content(content: str) -> Dict[str, List[str]]:
    """Search SQL content using regex patterns for various database objects.
    
    Args:
        content: The SQL query or schema definition to analyze
        
    Returns:
        Dictionary with found objects by type
    """
    # Common pattern components
    identifier_pattern = r'\b(?:[A-Za-z_]\w*)+\b'  # Matches valid SQL identifiers
    
    # Tables
    table_create_pattern = re.compile(r'\bCREATE\s+TABLE\b', re.IGNORECASE)
    tables = re.findall(identifier_pattern, content[table_create_pattern.search(content):])
    
    # Views
    view_create_pattern = re.compile(r'\bCREATE\s+VIEW\b', re.IGNORECASE)
    views = re.findall(identifier_pattern, content[view_create_pattern.search(content):])
    
    # Procedures
    proc_create_pattern = re.compile(r'\bCREATE\s+PROCEDURE\b', re.IGNORECASE)
    procedures = re.findall(identifier_pattern, content[proc_create_pattern.search(content):])
    
    # Columns
    column_patterns = [
        r'\bALTER\s+TABLE\b',  # Existing columns through ALTER TABLE
        r'\bCREATE\s+INDEX\b'   # Indexed columns
    ]
    columns = []
    for pattern in column_patterns:
        idx = content.find(pattern)
        if idx != -1:
            cols = re.findall(identifier_pattern, content[idx:])
            columns.extend(cols)
    
    # Primary Keys
    pk_pattern = r'\bPRIMARY\s+KEY\b'
    pks = re.findall(identifier_pattern, content[pk_pattern.search(content):])
    
    # Foreign Keys
    fk_pattern = r'\bFOREIGN\s+KEY\b'
    fks = re.findall(identifier_pattern, content[fk_pattern.search(content):])
    
    # Indexes
    index_patterns = [
        r'\bCREATE\s+INDEX\b',
        r'\bUNIQUE\s+CONSTRAINT\b'  # Unique indexes through constraints
    ]
    indexes = []
    for pattern in index_patterns:
        idx = content.find(pattern)
        if idx != -1:
            index_cols = re.findall(identifier_pattern, content[idx:])
            indexes.extend(index_cols)
    
    return {
        'tables': list(set(tables)),
        'views': list(set(views)),
        'procedures': list(set(procedures)),
        'columns': list(set(columns)),
        'primary_keys': list(set(pks)),
        'foreign_keys': list(set(fks)),
        'indexes': list(set(indexes))
    }
