import streamlit as st
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import openai
from sql_agent.langgraph_orchestrator import SQLAgentOrchestrator
from sql_agent.metadata_extractor import MetadataExtractor
from sql_agent.visualization import SimilaritySearchResultPlot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLAgentApp:
    """Streamlit application for SQL query generation and analysis."""
    
    def __init__(self):
        """Initialize the application components."""
        self.agent = SQLAgentOrchestrator()
        self.metadata_extractor = MetadataExtractor()
        self._init_session_state()
        
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'metadata' not in st.session_state:
            st.session_state.metadata = None
        if 'api_key_configured' not in st.session_state:
            st.session_state.api_key_configured = False
            
    def setup_sidebar(self) -> bool:
        """Configure the sidebar and check prerequisites.
        
        Returns:
            Boolean indicating if setup was successful
        """
        st.sidebar.title("SQL Agent Configuration")
        
        # API Key Configuration
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=os.getenv('OPENAI_API_KEY', ''),
            type="password",
            help="Enter your OpenAI API key to enable query generation"
        )
        
        if api_key:
            try:
                openai.api_key = api_key
                models = openai.models.list()
                available_models = [model.id for model in models if "gpt" in model.id.lower()]
                
                st.sidebar.success("✅ OpenAI API connection successful")
                st.sidebar.info(f"Available models: {', '.join(available_models)}")
                st.session_state.api_key_configured = True
                
            except Exception as e:
                st.sidebar.error(f"❌ OpenAI API Error: {str(e)}")
                st.session_state.api_key_configured = False
                return False
        else:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key")
            return False
            
        st.sidebar.markdown("---")
        return True
        
    def load_metadata(self, data_folder: str = "./sql_agent/data") -> Optional[Dict]:
        """Load and process SQL metadata from the data folder.
        
        Args:
            data_folder: Path to the SQL files directory
            
        Returns:
            Processed metadata if successful, None otherwise
        """
        try:
            data_path = Path(data_folder)
            if not data_path.exists() or not data_path.is_dir():
                st.error(f"❌ Data folder not found: {data_folder}")
                return None
                
            sql_files = list(data_path.glob("*.sql"))
            if not sql_files:
                st.error("❌ No SQL files found in data folder")
                return None
                
            with st.spinner("Loading SQL metadata..."):
                metadata = self.metadata_extractor.extract_metadata_from_sql_files(
                    [str(f) for f in sql_files]
                )
                
                if not metadata:
                    st.warning("⚠️ No metadata extracted from SQL files")
                    return None
                    
                st.sidebar.success(f"📁 Loaded {len(sql_files)} SQL files")
                self._display_metadata_stats(metadata)
                return metadata
                
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}", exc_info=True)
            st.error(f"❌ Error loading metadata: {str(e)}")
            return None
            
    def _display_metadata_stats(self, metadata: Dict):
        """Display metadata statistics in the sidebar."""
        with st.sidebar.expander("📊 Knowledge Base Stats", expanded=True):
            stats = metadata.get("statistics", {})
            object_counts = stats.get("object_counts", {})
            
            st.markdown(f"""
            - 📝 Tables: {object_counts.get('tables', 0)}
            - 📊 Views: {object_counts.get('views', 0)}
            - 🔧 Procedures: {object_counts.get('procedures', 0)}
            - 🔗 Relationships: {stats.get('relationship_count', 0)}
            - ⚠️ Errors: {stats.get('error_count', 0)}
            """)
            
    def process_query(self, query: str, metadata: Dict) -> None:
        """Process a natural language query and display results.
        
        Args:
            query: User's natural language query
            metadata: Database metadata
        """
        try:
            with st.spinner("Generating SQL query..."):
                # Use asyncio to run the async function
                import asyncio
                results, usage_stats = asyncio.run(self.agent.process_query(query, metadata))
                
            if results.error:
                st.error(f"❌ Error: {results.error}")
                return
                
            self._display_query_results(results, usage_stats)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.error(f"❌ Error processing query: {str(e)}")
            
    def _display_query_results(self, results, usage_stats):
        """Display query processing results and visualizations."""
        # Display agent steps
        st.markdown("### 🤖 Processing Steps")
        for step_name, step_data in results.get("agent_interactions", {}).items():
            with st.expander(f"Step: {step_name}", expanded=False):
                st.code(step_data.get("system_prompt", ""), language="text")
                st.code(step_data.get("user_prompt", ""), language="text")
                st.code(step_data.get("result", ""), language="text")
        
        # Display similarity search results
        if results.get("similarity_search"):
            st.markdown("### 🔍 Similar SQL Patterns")
            with st.expander("View Similar Patterns", expanded=False):
                for score, content in results["similarity_search"]:
                    st.markdown(f"**Similarity Score:** {score:.3f}")
                    st.code(content, language="sql")
        
        # Display generated query
        st.markdown("### 📝 Generated SQL Query")
        if results.get("generated_query", "").startswith("ERROR:"):
            st.error(results["generated_query"])
        else:
            st.code(results["generated_query"], language="sql")
            
            # Add copy button
            if st.button("📋 Copy Query"):
                st.code(results["generated_query"], language="sql")
                st.success("Query copied to clipboard!")
        
        # Display usage statistics
        with st.expander("📊 Usage Statistics", expanded=False):
            tokens = usage_stats.get("tokens", {})
            st.markdown(f"""
            - Prompt Tokens: {tokens.get('prompt', 0):,}
            - Completion Tokens: {tokens.get('completion', 0):,}
            - Estimated Cost: ${usage_stats.get('cost', 0):.4f}
            """)
        
        # Display any relevant file contents
        if results.get("relevant_files"):
            st.markdown("### 📑 Related SQL Files")
            for file_path in results["relevant_files"]:
                with st.expander(f"📄 {Path(file_path).name}", expanded=False):
                    try:
                        with open(file_path, 'r') as f:
                            st.code(f.read(), language="sql")
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SQL Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 SQL Agent - Query Generation and Analysis")
    st.markdown("""
    Enter your query in natural language, and I'll help you generate the appropriate SQL query.
    Upload your SQL files to the data folder to enable context-aware query generation.
    """)
    
    # Initialize application
    app = SQLAgentApp()
    
    # Check prerequisites
    if not app.setup_sidebar():
        return
        
    # Load metadata
    metadata = app.load_metadata()
    if not metadata:
        return
        
    # Query input
    query = st.text_area(
        "Enter your query in natural language:",
        height=100,
        help="Describe what you want to query from the database"
    )
    
    if st.button("🚀 Generate Query", disabled=not query):
        if not query.strip():
            st.warning("⚠️ Please enter a valid query")
            return
            
        app.process_query(query, metadata)

if __name__ == "__main__":
    main()
