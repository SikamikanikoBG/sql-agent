import re
import logging
from typing import Dict, List, Optional, Pattern
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SQLPattern:
    """Represents a regex pattern for SQL object detection."""
    name: str
    pattern: str
    flags: int = re.IGNORECASE | re.MULTILINE
    description: str = ""
    _compiled: Optional[Pattern] = field(default=None, init=False)
    
    def compile(self) -> Pattern:
        """Compile the regex pattern."""
        if not self._compiled:
            self._compiled = re.compile(self.pattern, self.flags)
        return self._compiled
    
    def search(self, content: str) -> List[str]:
        """Search content using the pattern."""
        pattern = self.compile()
        return [match.group(1).strip() for match in pattern.finditer(content) if match.group(1)]

class SQLRegexSearcher:
    """Class for searching SQL content using regex patterns."""
    
    def __init__(self):
        """Initialize the searcher with predefined patterns."""
        self.patterns = {
            'tables': SQLPattern(
                name='tables',
                pattern=r'CREATE\s+TABLE\s+([^\s(]+)',
                description="Matches CREATE TABLE statements"
            ),
            'views': SQLPattern(
                name='views',
                pattern=r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([^\s(]+)',
                description="Matches CREATE VIEW statements"
            ),
            'procedures': SQLPattern(
                name='procedures',
                pattern=r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([^\s(]+)',
                description="Matches CREATE PROCEDURE statements"
            ),
            'functions': SQLPattern(
                name='functions',
                pattern=r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([^\s(]+)',
                description="Matches CREATE FUNCTION statements"
            ),
            'triggers': SQLPattern(
                name='triggers',
                pattern=r'CREATE\s+TRIGGER\s+([^\s(]+)',
                description="Matches CREATE TRIGGER statements"
            ),
            'indexes': SQLPattern(
                name='indexes',
                pattern=r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+([^\s(]+)',
                description="Matches CREATE INDEX statements"
            ),
            'columns': SQLPattern(
                name='columns',
                pattern=r'(?:CREATE\s+TABLE\s+\w+\s*\(|\bALTER\s+TABLE\s+\w+\s+ADD\s+)(?:\s*\w+\s+[^,)]+,)*\s*(\w+)\s+[^,)]+',
                description="Matches column definitions"
            ),
            'foreign_keys': SQLPattern(
                name='foreign_keys',
                pattern=r'FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([^\s(]+)',
                description="Matches foreign key constraints"
            ),
            'constraints': SQLPattern(
                name='constraints',
                pattern=r'CONSTRAINT\s+([^\s(]+)',
                description="Matches named constraints"
            ),
            'sequences': SQLPattern(
                name='sequences',
                pattern=r'CREATE\s+SEQUENCE\s+([^\s(]+)',
                description="Matches CREATE SEQUENCE statements"
            )
        }
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the searcher."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def add_pattern(self, name: str, pattern: str, flags: int = re.IGNORECASE | re.MULTILINE,
                   description: str = "") -> None:
        """Add a new search pattern."""
        self.patterns[name] = SQLPattern(name, pattern, flags, description)
        logger.info(f"Added new pattern: {name}")

    def search_sql_content(self, content: str, pattern_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Search SQL content using specified patterns."""
        try:
            results = {}
            patterns_to_use = (
                {name: pattern for name, pattern in self.patterns.items() if name in pattern_names}
                if pattern_names
                else self.patterns
            )

            for name, pattern in patterns_to_use.items():
                try:
                    matches = pattern.search(content)
                    if matches:
                        results[name] = matches
                except re.error as e:
                    logger.error(f"Regex error in pattern {name}: {str(e)}")
                    results[name] = []
                except Exception as e:
                    logger.error(f"Error processing pattern {name}: {str(e)}")
                    results[name] = []

            return results

        except Exception as e:
            logger.error(f"Error searching SQL content: {str(e)}")
            return {name: [] for name in (pattern_names or self.patterns.keys())}

    def batch_search(self, contents: List[str], pattern_names: Optional[List[str]] = None) -> List[Dict[str, List[str]]]:
        """Search multiple SQL contents using specified patterns."""
        return [self.search_sql_content(content, pattern_names) for content in contents]

    def extract_relationships(self, content: str) -> List[Dict[str, str]]:
        """Extract table relationships from SQL content."""
        relationships = []
        fk_matches = self.patterns['foreign_keys'].search(content)
        
        for match in self.patterns['foreign_keys'].compile().finditer(content):
            source_cols = match.group(1).strip()
            target_table = match.group(2).strip()
            relationships.append({
                'source_columns': [col.strip() for col in source_cols.split(',')],
                'target_table': target_table,
                'type': 'foreign_key'
            })
            
        return relationships

def get_default_searcher() -> SQLRegexSearcher:
    """Get a preconfigured instance of SQLRegexSearcher."""
    return SQLRegexSearcher()