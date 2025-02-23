import pytest
import sqlite3
import logging
from pathlib import Path
from typing import Generator, Tuple
from unittest.mock import Mock
from _pytest.fixtures import FixtureRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_sql_files(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary SQL files for testing.
    
    Args:
        tmp_path: Pytest temporary directory fixture
        
    Yields:
        Path to temporary directory containing SQL files
    """
    data_dir = tmp_path / "sql_files"
    data_dir.mkdir()
    
    # Create sample SQL files
    files = {
        "table1.sql": """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(255) UNIQUE
        );
        """,
        "table2.sql": """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            total DECIMAL(10,2),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """,
        "view1.sql": """
        CREATE VIEW user_orders AS
        SELECT u.name, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.id, u.name;
        """
    }
    
    for filename, content in files.items():
        (data_dir / filename).write_text(content)
        
    yield data_dir

@pytest.fixture
def mock_db_connection() -> Tuple[Mock, Mock]:
    """Create mock database connection for testing.
    
    Returns:
        Tuple containing mock connection and cursor
    """
    conn = Mock()
    cur = Mock()
    conn.cursor.return_value = cur
    return conn, cur

@pytest.fixture
def in_memory_db() -> Generator[Tuple[sqlite3.Connection, sqlite3.Cursor], None, None]:
    """Create in-memory SQLite database for testing.
    
    Yields:
        Tuple containing database connection and cursor
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    
    # Create sample tables
    cur.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        );
        
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            total DECIMAL(10,2),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        
        INSERT INTO users (id, name, email) VALUES
            (1, 'John Doe', 'john@example.com'),
            (2, 'Jane Smith', 'jane@example.com');
            
        INSERT INTO orders (id, user_id, total) VALUES
            (1, 1, 99.99),
            (2, 1, 149.99),
            (3, 2, 199.99);
    """)
    
    yield conn, cur
    conn.close()

@pytest.fixture
def mock_openai_response() -> Mock:
    """Create mock OpenAI API response.
    
    Returns:
        Mock object simulating OpenAI API response
    """
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "SELECT * FROM users"
    return response

@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Create sample metadata for testing.
    
    Returns:
        Dictionary containing sample metadata
    """
    return {
        "tables": ["users", "orders"],
        "views": ["user_orders"],
        "procedures": [],
        "schemas": {
            "users": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR(100)"},
                {"name": "email", "type": "VARCHAR(255)"}
            ],
            "orders": [
                {"name": "id", "type": "INTEGER"},
                {"name": "user_id", "type": "INTEGER"},
                {"name": "total", "type": "DECIMAL(10,2)"}
            ]
        },
        "relationships": [
            {
                "source_columns": ["user_id"],
                "target_table": "users",
                "target_columns": ["id"],
                "type": "foreign_key"
            }
        ]
    }