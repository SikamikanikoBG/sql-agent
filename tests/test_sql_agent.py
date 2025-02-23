import pytest
from unittest.mock import Mock

def test_metadata_extraction(mock_db_connection):
    """Test metadata extraction from SQL files"""
    
    conn, cur = mock_db_connection
    
    # Mock SQL file content
    sql_content = """
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        date DATE,
        amount DECIMAL(10,2)
    );
    """
    
    # Extract tables using regex search
    agent = SQLAgentOrchestrator()
    extracted_tables = agent.extract_metadata(sql_content)
    
    assert len(extracted_tables) == 1
    
def test_query_generation():
    """Test query generation using regex search"""
    
    assert True  # Implement later with regex-based query generation
