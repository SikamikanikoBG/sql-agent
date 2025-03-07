from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class DependencyVisualizer:
    """Visualizes SQL temporary table dependencies."""
    
    def create_dependency_graph(self, dependencies: List[Dict[str, str]]) -> str:
        """Creates a text-based representation of the dependency graph."""
        try:
            if not dependencies:
                return "<pre>No temporary table dependencies found.</pre>"
                
            # Build dependency tree
            tree = {}
            for dep in dependencies:
                table_name = dep['table']
                definition = dep['definition']
                deps = set()
                
                # Extract dependencies
                dep_pattern = r'(?:FROM|JOIN)\s+(#[\w]+)'
                for match in re.finditer(dep_pattern, definition, re.IGNORECASE):
                    dep_table = match.group(1)
                    if dep_table != table_name:  # Avoid self-reference
                        deps.add(dep_table)
                        
                tree[table_name] = list(deps)
            
            # Generate ASCII tree
            lines = ['<pre>Temporary Table Dependencies:', '']
            
            def print_tree(table: str, prefix: str = '', is_last: bool = True) -> List[str]:
                result = []
                branch = '└── ' if is_last else '├── '
                result.append(prefix + branch + table)
                
                if table in tree:
                    new_prefix = prefix + ('    ' if is_last else '│   ')
                    deps = tree[table]
                    for i, dep in enumerate(deps):
                        result.extend(print_tree(dep, new_prefix, i == len(deps) - 1))
                        
                return result
            
            # Print each root node
            root_tables = set(tree.keys()) - {d for deps in tree.values() for d in deps}
            for i, root in enumerate(sorted(root_tables)):
                lines.extend(print_tree(root, '', i == len(root_tables) - 1))
            
            lines.append('</pre>')
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Error creating dependency graph: {str(e)}")
            return f"<pre>Error creating dependency graph: {str(e)}</pre>"
