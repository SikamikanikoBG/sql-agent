import graphviz
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DependencyVisualizer:
    """Visualizes SQL temporary table dependencies."""
    
    def __init__(self):
        self.dot = graphviz.Digraph(comment='Temporary Table Dependencies')
        self.dot.attr(rankdir='LR')
        
    def create_dependency_graph(self, dependencies: List[Dict[str, str]]) -> str:
        """Creates an HTML representation of the dependency graph."""
        try:
            # Clear previous graph
            self.dot.clear()
            
            # Add nodes and edges
            added_nodes = set()
            added_edges = set()
            
            for dep in dependencies:
                table_name = dep['table']
                if table_name not in added_nodes:
                    self.dot.node(table_name, table_name)
                    added_nodes.add(table_name)
                
                # Extract dependencies from definition
                definition = dep['definition']
                dep_pattern = r'(?:FROM|JOIN)\s+(#[\w]+)'
                import re
                for match in re.finditer(dep_pattern, definition, re.IGNORECASE):
                    dep_table = match.group(1)
                    if dep_table != table_name:  # Avoid self-reference
                        if dep_table not in added_nodes:
                            self.dot.node(dep_table, dep_table)
                            added_nodes.add(dep_table)
                        edge = (dep_table, table_name)
                        if edge not in added_edges:
                            self.dot.edge(*edge)
                            added_edges.add(edge)
            
            # Return SVG as string
            return self.dot.pipe(format='svg').decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error creating dependency graph: {str(e)}")
            return f"<p>Error creating dependency graph: {str(e)}</p>"
