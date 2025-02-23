import streamlit as st
import os
import logging
import asyncio
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
                openai.models.list()  # Test the connection
                st.sidebar.success("✅ OpenAI API connection successful")
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
            
                # Enhanced metadata display
                st.markdown("""
                    <div class="metadata-card">
                        <h3>📚 Database Schema Overview</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                stats = metadata.get("statistics", {})
                object_counts = stats.get("object_counts", {})
                
                with col1:
                    st.markdown("""
                        <div class="stat-card">
                            <h4>📝 Tables</h4>
                            <h2>{}</h2>
                        </div>
                    """.format(object_counts.get('tables', 0)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="stat-card">
                            <h4>📊 Views</h4>
                            <h2>{}</h2>
                        </div>
                    """.format(object_counts.get('views', 0)), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class="stat-card">
                            <h4>🔧 Procedures</h4>
                            <h2>{}</h2>
                        </div>
                    """.format(object_counts.get('procedures', 0)), unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                        <div class="stat-card">
                            <h4>🔗 Relations</h4>
                            <h2>{}</h2>
                        </div>
                    """.format(stats.get('relationship_count', 0)), unsafe_allow_html=True)
                
                with col5:
                    st.markdown("""
                        <div class="stat-card">
                            <h4>⚠️ Errors</h4>
                            <h2>{}</h2>
                        </div>
                    """.format(stats.get('error_count', 0)), unsafe_allow_html=True)
                
                with st.expander("🔍 Detailed Schema Information", expanded=False):
                    st.json(metadata)
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}", exc_info=True)
            st.error(f"❌ Error loading metadata: {str(e)}")
            return None
            
            
    def process_query(self, query: str, metadata: Dict) -> None:
        """Process a natural language query and display results.
        
        Args:
            query: User's natural language query
            metadata: Database metadata
        """
        # Initialize vector store with examples if needed
        if not hasattr(self.agent, 'vector_store') or self.agent.vector_store is None:
            with st.spinner("Initializing knowledge base..."):
                data_path = Path("./sql_agent/data")
                sql_files = list(data_path.glob("*.sql"))
                asyncio.run(self.agent.initialize_vector_store([str(f) for f in sql_files]))
        try:
            with st.spinner("Generating SQL query..."):
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
        for step_name, step_data in results.agent_interactions.items():
            with st.expander(f"Step: {step_name}", expanded=False):
                st.markdown(f"**🤖 Agent Model:** {self.agent.model_name}")
                st.markdown("**📝 System Prompt:**")
                st.code(step_data["system_prompt"], language="text")
                st.markdown("**📥 User Input:**")
                st.code(step_data["user_prompt"], language="text")
                st.markdown("**📤 Agent Response:**")
                st.code(step_data["result"], language="text")
                st.markdown(f"*🎯 Tokens used: {step_data['tokens_used']}*")
        
        # Display relevant context and similarity search results
        if results.similarity_search or results.relevant_files:
            st.markdown("### 🔍 Context and Similar Patterns")
            
            # Display relevant files
            if results.relevant_files:
                with st.expander("📁 Relevant Context Files", expanded=True):
                    st.markdown("**Files used for context:**")
                    for file in results.relevant_files:
                        base_name = os.path.basename(file)
                        with st.expander(f"📄 {base_name}", expanded=False):
                            try:
                                with open(file, 'r', encoding='utf-8') as f:
                                    st.code(f.read(), language="sql")
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
            
            # Display vector store and similar patterns
            if results.similarity_search:
                with st.expander("🔍 Vector Store Results", expanded=False):
                    st.markdown("#### 🧠 Knowledge Base")
                    st.markdown(f"""
                    - 🔤 Embedding Model: {type(self.agent.embeddings).__name__}
                    - 📚 Vector Store: {type(self.agent.vector_store).__name__}
                    - 🎯 Similarity Threshold: {self.agent.similarity_threshold}
                    """)
                    
                    st.markdown("#### 📊 Similar Examples Used in Prompts")
                    st.code(self.agent._format_examples(results.similarity_search), language="text")
                    
                    st.markdown("#### 🔍 Raw Vector Search Results")
                    for score, content in results.similarity_search:
                        with st.expander(f"Example (Score: {score:.3f})", expanded=False):
                            if isinstance(content, dict):
                                st.markdown("**Source:**")
                                st.markdown(f"`{content.get('source', 'Unknown')}`")
                                st.markdown("**Content:**")
                                st.code(content.get('content', ''), language="sql")
                            else:
                                st.code(str(content), language="sql")
                    
                    st.markdown("#### 📈 Similarity Distribution")
                    scores = [score for score, _ in results.similarity_search]
                    st.bar_chart(scores)
        
        # Enhanced query results display
        st.markdown("""
            <div class="results-section">
                <h3>📝 Generated SQL Query</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if results.error:
            st.error(f"Error: {results.error}")
        else:
            col1, col2 = st.columns([4,1])
            with col1:
                st.markdown('<div class="generated-query">', unsafe_allow_html=True)
                st.code(results.generated_query, language="sql")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if st.button("📋 Copy to Clipboard", type="secondary"):
                    st.code(results.generated_query, language="sql")
                    st.success("✅ Query copied!")
                if st.button("💾 Save Query", type="secondary"):
                    st.download_button(
                        label="Download SQL",
                        data=results.generated_query,
                        file_name="generated_query.sql",
                        mime="text/plain"
                    )
        
        # Display usage statistics
        with st.expander("📊 Usage Statistics", expanded=False):
            st.markdown(f"""
            - Prompt Tokens: {usage_stats.prompt_tokens:,}
            - Completion Tokens: {usage_stats.completion_tokens:,}
            - Total Tokens: {usage_stats.total_tokens:,}
            - Estimated Cost: ${usage_stats.cost:.4f}
            """)

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SQL Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    with open("sql_agent/static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Main header section
    st.markdown("""
        <div class="main-header">
            <h1>🤖 SQL Agent</h1>
            <p>Your AI-powered SQL assistant for natural language query generation</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize application
    app = SQLAgentApp()
    
    # Check prerequisites
    if not app.setup_sidebar():
        return
        
    # Load metadata
    metadata = app.load_metadata()
    if not metadata:
        return
        
    # Enhanced query input section
    st.markdown("""
        <div class="query-input">
            <h3>🔍 What would you like to query?</h3>
            <p>Describe your query in natural language, and I'll help you generate the SQL.</p>
        </div>
    """, unsafe_allow_html=True)
    
    query = st.text_area(
        "",
        placeholder="Example: Show me all customers who made purchases last month...",
        height=100,
        help="Be as specific as possible for better results"
    )
    
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("🚀 Generate SQL Query", type="primary", disabled=not query):
        if not query.strip():
            st.warning("⚠️ Please enter a valid query")
            return
            
        app.process_query(query, metadata)

if __name__ == "__main__":
    main()
