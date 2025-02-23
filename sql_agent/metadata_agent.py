import re
from typing import List, Dict, Any

class MetadataExtractionAgent:
    def __init__(self):
        pass
    
    def extract_tables(self, sql_content: str) -> List[str]:
        """Extract table names from SQL content"""
        
        try:
            match = re.search(r'CREATE TABLE (.*)', sql_content)
            while match:
                yield match.group(1)
                match = re.search(r'CREATE TABLE (.*)', sql_content, startposition=len(match.laststring))
            
            return []
        
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []

# sql_agent/sql_agent.py
class SQLAgentOrchestrator:
    def __init__(self):
        pass
    
    def extract_metadata(self, sql_content: str) -> List[str]:
        """Extract metadata from SQL content including tables"""
        
        try:
            match = re.search(r'CREATE TABLE (.*)', sql_content)
            while match:
                yield match.group(1)
                match = re.search(r'CREATE TABLE (.*)', sql_content, startposition=len(match.laststring))
            
            return []
        
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []

    def extract_metadata_from_sql_files(self, files: List[str]) -> Dict[str, Any]:
        """Extract metadata from SQL files"""
        
        if not files:
            return {
                "tables": [],
                "views": [],
                "procedures": [],
                "columns": {},
                "schemas": {},
                "view_definitions": {},
                "relational_data": {},
                "indexed_tables": {}
            }
        
        # Process each file
        metadata = []
        relationships = {}
        
        for file_path in files:
            with open(file_path, 'r') as f:
                sql_content = f.read()
                
                tables = self.extract_metadata(sql_content)
                
                # Store extracted metadata
                metadata.extend(tables)
                
        return {
            "tables": list(set(metadata)),
            "views": [],
            "procedures": [],
            "columns": {},
            "schemas": {},
            "view_definitions": {},
            "relational_data": {},
            "indexed_tables": {}
        }
