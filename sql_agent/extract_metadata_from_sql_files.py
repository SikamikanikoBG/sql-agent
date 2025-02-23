import re
from typing import List, Dict, Any
import logging

class SQLAgentOrchestrator:
    def extract_metadata(self, sql_content: str) -> Dict[str, Any]:
        """Extract metadata from SQL content including tables"""
        metadata = {
            "tables": [],
            "views": [],
            "procedures": [],
            "temp_tables": [],
            "table_variables": []
        }
        
        # Regular table pattern
        table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)\s*\((.*?)\);'
        for match in re.finditer(table_pattern, sql_content, re.IGNORECASE | re.DOTALL):
            table_name = match.group(1).strip('[]" ')
            if not table_name.startswith('#'):  # Skip temp tables
                metadata["tables"].append({
                    "name": table_name,
                    "definition": match.group(2).strip()
                })

        # Temporary table pattern
        temp_pattern = r'CREATE\s+TABLE\s+(#[\w\d_]+)\s*\((.*?)\);'
        for match in re.finditer(temp_pattern, sql_content, re.IGNORECASE | re.DOTALL):
            metadata["temp_tables"].append({
                "name": match.group(1),
                "definition": match.group(2).strip()
            })

        # View pattern
        view_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([^\s(]+)\s+AS\s+(.*?);'
        for match in re.finditer(view_pattern, sql_content, re.IGNORECASE | re.DOTALL):
            metadata["views"].append({
                "name": match.group(1).strip('[]" '),
                "definition": match.group(2).strip()
            })

        # Stored procedure pattern
        proc_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([^\s(]+)\s*\((.*?)\)\s*AS\s*BEGIN(.*?)END;'
        for match in re.finditer(proc_pattern, sql_content, re.IGNORECASE | re.DOTALL):
            metadata["procedures"].append({
                "name": match.group(1).strip('[]" '),
                "parameters": match.group(2).strip(),
                "body": match.group(3).strip()
            })

        return metadata

    def extract_metadata_from_sql_files(self, files: List[str]) -> Dict[str, Any]:
        """Extract metadata from SQL files"""
        if not files:
            return {
                "tables": [],
                "views": [],
                "procedures": [],
                "temp_tables": [],
                "table_variables": []
            }

        combined_metadata = {
            "tables": [],
            "views": [],
            "procedures": [],
            "temp_tables": [],
            "table_variables": []
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                    file_metadata = self.extract_metadata(sql_content)
                    
                    # Merge metadata
                    for key in combined_metadata:
                        combined_metadata[key].extend(file_metadata[key])
                        
                    # Add source file information
                    for items in file_metadata.values():
                        for item in items:
                            if isinstance(item, dict):
                                item["source_file"] = file_path

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")

        return combined_metadata