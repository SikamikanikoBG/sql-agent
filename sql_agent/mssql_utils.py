import pyodbc
from typing import Dict, Any, Optional

class MSSQLConnection:
    def __init__(self, server: str, database: str, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 trusted_connection: bool = True):
        """Initialize SQL Server connection parameters."""
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.trusted_connection = trusted_connection

    def get_connection(self) -> pyodbc.Connection:
        """Create and return a SQL Server connection."""
        conn_str = [
            f"DRIVER={{SQL Server}}",
            f"SERVER={self.server}",
            f"DATABASE={self.database}"
        ]

        if self.trusted_connection:
            conn_str.append("Trusted_Connection=yes")
        else:
            conn_str.extend([
                f"UID={self.username}",
                f"PWD={self.password}"
            ])

        return pyodbc.connect(";".join(conn_str))

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract database metadata including tables and their schemas."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            tables = cursor.execute("""
                SELECT TABLE_NAME, TABLE_TYPE 
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
            """).fetchall()
            
            metadata = {
                "tables": [],
                "schemas": {},
                "views": [],
                "view_definitions": {}
            }
            
            # Get columns for each table
            for table_name, _ in tables:
                columns = cursor.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, table_name).fetchall()
                
                metadata["tables"].append(table_name)
                metadata["schemas"][table_name] = [
                    {
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == 'YES'
                    } for col in columns
                ]
            
            # Get views
            views = cursor.execute("""
                SELECT TABLE_NAME, VIEW_DEFINITION
                FROM INFORMATION_SCHEMA.VIEWS
            """).fetchall()
            
            for view_name, definition in views:
                metadata["views"].append(view_name)
                metadata["view_definitions"][view_name] = definition
                
            return metadata
