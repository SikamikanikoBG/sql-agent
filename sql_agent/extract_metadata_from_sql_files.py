import re
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

class SQLAgentOrchestrator:
    def extract_metadata(self, sql_content: str) -> Dict[str, Any]:
        """Extract metadata from SQL content including tables and their dependencies"""
        metadata = {
            "tables": [],
            "views": [],
            "procedures": [],
            "temp_tables": [],
            "table_variables": [],
            "dependencies": {}
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

        # Temporary table pattern with full statement capture
        temp_pattern = r'(CREATE\s+TABLE\s+#[\w\d_]+\s*\(.*?;\s*(?:SELECT|INSERT).*?(?:;\s*|$))'
        for match in re.finditer(temp_pattern, sql_content, re.IGNORECASE | re.DOTALL):
            full_stmt = match.group(1)
            table_match = re.search(r'CREATE\s+TABLE\s+(#[\w\d_]+)', full_stmt, re.IGNORECASE)
            if table_match:
                temp_name = table_match.group(1)
                # Find dependencies in the full statement
                deps = set()
                dep_pattern = r'(?:FROM|JOIN)\s+(#?[\w\d_]+)'
                for dep_match in re.finditer(dep_pattern, full_stmt, re.IGNORECASE):
                    dep_table = dep_match.group(1)
                    if dep_table != temp_name:  # Avoid self-reference
                        deps.add(dep_table)
                
                metadata["temp_tables"].append({
                    "name": temp_name,
                    "definition": full_stmt.strip(),
                    "dependencies": list(deps)
                })
                metadata["dependencies"][temp_name] = list(deps)

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

    def _get_all_sql_files(self, directory: str) -> List[str]:
        """Recursively find all SQL files in directory and subdirectories."""
        sql_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.sql'):
                    sql_files.append(os.path.join(root, file))
        return sql_files

    def extract_metadata_from_sql_files(self, paths: List[str]) -> Dict[str, Any]:
        """Extract metadata from SQL files and directories"""
        if not paths:
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

        # Collect all SQL files, including those in subdirectories
        sql_files = []
        for path in paths:
            if os.path.isdir(path):
                sql_files.extend(self._get_all_sql_files(path))
            elif path.lower().endswith('.sql'):
                sql_files.append(path)

        for file_path in sql_files:
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
