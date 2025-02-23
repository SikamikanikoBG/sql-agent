import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SQLObject:
    """Represents a SQL database object."""
    type: str
    name: str
    definition: str
    source_file: str
    database: Optional[str] = None
    schema: Optional[List[Dict[str, str]]] = None
    parameters: Optional[List[Dict[str, str]]] = None
    description: Optional[str] = None

class MetadataExtractor:
    """Extracts metadata from SQL files."""
    
    def __init__(self):
        self._setup_logging()
        self.patterns = {
            'table': re.compile(r'CREATE\s+TABLE\s+(\[?[^\s(]+\]?)\s*\((.*?)\);', re.IGNORECASE | re.DOTALL | re.MULTILINE),
            'view': re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\[?[^\s(]+\]?)\s+AS\s+(.*?);', re.IGNORECASE | re.DOTALL | re.MULTILINE),
            'procedure': re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(\[?[^\s(]+\]?)(?:\s*\((.*?)\))?\s*(.*?)(?:\$\$|;)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
            'column': re.compile(r'\s*([^\s,()]+)\s+([^\s,()]+(?:\([^)]*\))?)', re.IGNORECASE),
            'foreign_key': re.compile(r'FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([^\s(]+)\s*\(([^)]+)\)', re.IGNORECASE)
        }
    
    def _setup_logging(self) -> None:
        """Configure logging for the metadata extractor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _extract_database_name(self, sql_content: str, table_name: str) -> Optional[str]:
        """Extract database name from USE statement or fully qualified name.
        
        Args:
            sql_content: Full SQL content
            table_name: Table name to check for database prefix
            
        Returns:
            Database name if found
        """
        # Check for USE DATABASE statement before CREATE TABLE
        use_db_pattern = re.compile(r'USE\s+(\[?[\w]+\]?)\s*;', re.IGNORECASE)
        matches = use_db_pattern.finditer(sql_content)
        last_use_db = None
        for match in matches:
            last_use_db = match.group(1).strip('[]')
            
        # Check if table name is fully qualified with database
        if '.' in table_name:
            parts = table_name.split('.')
            if len(parts) >= 3:  # [database].[schema].[table]
                return parts[0].strip('[]')
            
        return last_use_db

    def _extract_table_columns(self, create_statement: str) -> List[Dict[str, str]]:
        """Extract column definitions from CREATE TABLE statement.
        
        Args:
            create_statement: The CREATE TABLE SQL statement
            
        Returns:
            List of column definitions
        """
        columns = []
        # Split on commas but ignore commas inside parentheses
        parts = re.split(r',(?![^(]*\))', create_statement)
        
        for part in parts:
            part = part.strip()
            if part and not part.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT')):
                match = self.patterns['column'].search(part)
                if match:
                    name, data_type = match.groups()
                    columns.append({
                        'name': name.strip(),
                        'type': data_type.strip()
                    })
                    logger.debug(f"Extracted column: {name.strip()} ({data_type.strip()})")
        
        return columns
    
    def _extract_foreign_keys(self, create_statement: str) -> List[Dict[str, str]]:
        """Extract foreign key relationships from CREATE TABLE statement.
        
        Args:
            create_statement: The CREATE TABLE SQL statement
            
        Returns:
            List of foreign key relationships
        """
        relationships = []
        matches = self.patterns['foreign_key'].finditer(create_statement)
        
        for match in matches:
            source_cols, target_table, target_cols = match.groups()
            relationships.append({
                'source_columns': [col.strip() for col in source_cols.split(',')],
                'target_table': target_table.strip(),
                'target_columns': [col.strip() for col in target_cols.split(',')]
            })
            
        return relationships

    def extract_metadata_from_sql_files(self, files: List[str]) -> Dict[str, Any]:
        """Extract metadata from SQL files including tables, views, and procedures.
        
        Args:
            files: List of SQL file paths to analyze
            
        Returns:
            Dictionary containing extracted metadata
        """
        if not files:
            logger.warning("No SQL files provided for metadata extraction")
            return self._empty_metadata()
            
        metadata = {
            "objects": [],
            "relationships": [],
            "schemas": {},
            "errors": []
        }
        
        for file_path in files:
            try:
                file_metadata = self._process_file(file_path)
                metadata["objects"].extend(file_metadata.get("objects", []))
                metadata["relationships"].extend(file_metadata.get("relationships", []))
                metadata["schemas"].update(file_metadata.get("schemas", {}))
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
                metadata["errors"].append({
                    "file": file_path,
                    "error": str(e)
                })
        
        return self._organize_metadata(metadata)
    
    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single SQL file and extract its metadata.
        
        Args:
            file_path: Path to the SQL file
            
        Returns:
            Dictionary containing file-specific metadata
        """
        file_metadata = {
            "objects": [],
            "relationships": [],
            "schemas": {}
        }
        
        logger.info(f"Processing SQL file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split content into statements
            statements = re.split(r';(?=[^\)]*(?:\(|$))', content)
            logger.info(f"Split content into {len(statements)} statements")
            
            # Extract tables
            for statement in statements:
                if 'CREATE TABLE' in statement.upper():
                    match = self.patterns['table'].search(statement + ';')  # Add back the semicolon
                    if match:
                        name, definition = match.groups()
                        clean_name = name.strip('[] \n\t')
                        logger.info(f"Processing table: {clean_name}")
                
                try:
                    columns = self._extract_table_columns(definition)
                    logger.info(f"Extracted {len(columns)} columns from {clean_name}")
                    
                    relationships = self._extract_foreign_keys(definition)
                    logger.info(f"Extracted {len(relationships)} relationships from {clean_name}")
                    
                    database_name = self._extract_database_name(content, clean_name)
                    
                    sql_object = SQLObject(
                        type="table",
                        name=clean_name,
                        definition=definition.strip(),
                        source_file=file_path,
                        database=database_name,
                        schema=columns
                    )
                    
                    file_metadata["objects"].append(vars(sql_object))
                    file_metadata["relationships"].extend(relationships)
                    file_metadata["schemas"][clean_name] = columns
                    
                except Exception as e:
                    logger.error(f"Error processing table {clean_name}: {str(e)}", exc_info=True)
            
            # Extract views
            for statement in statements:
                if 'CREATE VIEW' in statement.upper():
                    match = self.patterns['view'].search(statement + ';')
                    if match:
                        name, definition = match.groups()
                        clean_name = name.strip('[] \n\t')
                        logger.info(f"Processing view: {clean_name}")
                        sql_object = SQLObject(
                            type="view",
                            name=clean_name,
                            definition=definition.strip(),
                            source_file=file_path
                        )
                        file_metadata["objects"].append(vars(sql_object))
            
            # Extract procedures
            for statement in statements:
                if 'CREATE PROCEDURE' in statement.upper():
                    match = self.patterns['procedure'].search(statement + ';')
                    if not match:
                        continue
                    name, params, body = match.groups()
                    clean_name = name.strip('[] \n\t')
                    logger.info(f"Processing procedure: {clean_name}")
                parameters = self._parse_procedure_parameters(params)
                description = self._extract_procedure_description(body)
                
                sql_object = SQLObject(
                    type="procedure",
                    name=name.strip(),
                    definition=body.strip(),
                    source_file=file_path,
                    parameters=parameters,
                    description=description
                )
                file_metadata["objects"].append(vars(sql_object))
        
        return file_metadata

    def _parse_procedure_parameters(self, params_str: str) -> List[Dict[str, str]]:
        """Parse procedure parameters into structured format.
        
        Args:
            params_str: String containing procedure parameters
            
        Returns:
            List of parameter definitions
        """
        parameters = []
        if not params_str.strip():
            return parameters
            
        param_list = params_str.split(',')
        for param in param_list:
            parts = param.strip().split()
            if len(parts) >= 2:
                param_info = {
                    'name': parts[0].strip(),
                    'type': parts[1].strip(),
                    'mode': 'IN'  # Default mode
                }
                
                # Check for parameter mode (IN/OUT/INOUT)
                if any(mode in param.upper() for mode in ['IN', 'OUT', 'INOUT']):
                    for mode in ['IN', 'OUT', 'INOUT']:
                        if mode in param.upper():
                            param_info['mode'] = mode
                            break
                
                parameters.append(param_info)
                
        return parameters

    def _extract_procedure_description(self, body: str) -> Optional[str]:
        """Extract procedure description from comments in the body.
        
        Args:
            body: Procedure body text
            
        Returns:
            Extracted description if found
        """
        # Look for block comments
        block_comment_pattern = re.compile(r'/\*(.*?)\*/', re.DOTALL)
        block_match = block_comment_pattern.search(body)
        if block_match:
            return block_match.group(1).strip()
        
        # Look for line comments
        line_comment_pattern = re.compile(r'--\s*(.*?)(?:\n|$)')
        line_matches = line_comment_pattern.findall(body)
        if line_matches:
            return ' '.join(line.strip() for line in line_matches)
        
        return None

    def _organize_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Organize raw metadata into a structured format.
        
        Args:
            raw_metadata: Raw metadata extracted from files
            
        Returns:
            Organized metadata dictionary
        """
        organized = {
            "tables": [],
            "views": [],
            "procedures": [],
            "schemas": raw_metadata["schemas"],
            "relationships": raw_metadata["relationships"],
            "errors": raw_metadata.get("errors", []),
            "statistics": self._calculate_statistics(raw_metadata)
        }
        
        # Organize objects by type
        for obj in raw_metadata["objects"]:
            if obj["type"] == "table":
                organized["tables"].append(obj)
            elif obj["type"] == "view":
                organized["views"].append(obj)
            elif obj["type"] == "procedure":
                organized["procedures"].append(obj)
        
        return organized

    def _calculate_statistics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various statistics about the extracted metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Dictionary of calculated statistics
        """
        stats = {
            "total_objects": len(metadata["objects"]),
            "object_counts": {
                "tables": len([obj for obj in metadata["objects"] if obj["type"] == "table"]),
                "views": len([obj for obj in metadata["objects"] if obj["type"] == "view"]),
                "procedures": len([obj for obj in metadata["objects"] if obj["type"] == "procedure"])
            },
            "relationship_count": len(metadata["relationships"]),
            "error_count": len(metadata.get("errors", [])),
            "schema_coverage": len(metadata["schemas"])
        }
        
        return stats

    def _empty_metadata(self) -> Dict[str, Any]:
        """Create an empty metadata structure.
        
        Returns:
            Empty metadata dictionary
        """
        return {
            "tables": [],
            "views": [],
            "procedures": [],
            "schemas": {},
            "relationships": [],
            "errors": [],
            "statistics": {
                "total_objects": 0,
                "object_counts": {
                    "tables": 0,
                    "views": 0,
                    "procedures": 0
                },
                "relationship_count": 0,
                "error_count": 0,
                "schema_coverage": 0
            }
        }
