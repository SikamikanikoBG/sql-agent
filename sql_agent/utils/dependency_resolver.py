import re
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)

class TempTableDependencyResolver:
    """Resolves dependencies between temporary tables in SQL queries."""
    
    def __init__(self):
        self.temp_tables: Dict[str, Dict] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        
    def add_temp_table(self, name: str, definition: str, source_file: str) -> None:
        """Add a temp table and its definition."""
        self.temp_tables[name] = {
            'definition': definition,
            'source_file': source_file
        }
        # Extract dependencies from definition
        deps = self._extract_dependencies(definition)
        self.dependencies[name] = deps
        
    def _extract_dependencies(self, sql: str) -> Set[str]:
        """Extract temp table dependencies from SQL."""
        deps = set()
        # Match temp tables (#TableName) in FROM and JOIN clauses
        pattern = r'(?:FROM|JOIN)\s+(#[\w]+)'
        matches = re.finditer(pattern, sql, re.IGNORECASE)
        for match in matches:
            deps.add(match.group(1))
        return deps
        
    def get_creation_sequence(self, target_table: str) -> List[Dict[str, str]]:
        """Get ordered sequence of temp table creation statements."""
        visited = set()
        sequence = []
        
        def visit(table: str) -> None:
            if table in visited:
                return
            if table not in self.temp_tables:
                logger.warning(f"Missing definition for temp table: {table}")
                return
                
            # First process dependencies
            for dep in self.dependencies.get(table, set()):
                visit(dep)
                
            visited.add(table)
            sequence.append({
                'table': table,
                'definition': self.temp_tables[table]['definition'],
                'source_file': self.temp_tables[table]['source_file']
            })
            
        visit(target_table)
        return sequence
        
    def get_all_dependencies(self, query: str) -> List[Dict[str, str]]:
        """Get all temp table dependencies for a query."""
        # Find all temp tables referenced in the query
        pattern = r'(?:FROM|JOIN)\s+(#[\w]+)'
        temp_tables = set(re.findall(pattern, query, re.IGNORECASE))
        
        # Get creation sequence for each temp table
        all_sequences = []
        visited = set()
        
        for table in temp_tables:
            sequence = self.get_creation_sequence(table)
            for item in sequence:
                if item['table'] not in visited:
                    visited.add(item['table'])
                    all_sequences.append(item)
                    
        return all_sequences
