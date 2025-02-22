import pytest
import sqlite3
import logging
from unittest.mock import Mock

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@pytest.fixture
def mock_db_connection():
    """Create mock database connection for testing"""
    conn = Mock()
    cur = Mock()
    conn.cursor.return_value = cur
    return conn, cur

@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing"""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    yield conn, cur
    conn.close()
