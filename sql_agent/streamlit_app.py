import streamlit as st
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional
import openai
from sql_agent.langgraph_orchestrator import SQLAgentOrchestrator
from sql_agent.metadata_extractor import MetadataExtractor
from sql_agent.visualization import SimilaritySearchResultPlot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLAgentApp:
    """Streamlit application for SQL query generation and analysis."""
    
    def __init__(self):
        self.agent = SQLAgentOrchestrator()
        self.metadata_extractor = MetadataExtractor()
        self._init_session_state()
        
    def _init_session_state(self):
        if 'metadata' not in st.session_state:
            st.session_state.metadata = None
        if 'api_key_configured' not in st.session_state:
            st.session_state.api_key_configured = False
        if 'show_schema' not in st.session_state:
            st.session_state.show_schema = False
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Query Generator"
        if 'processing' not in st.session_state:
            st.session_state.processing = False
            
    def setup_sidebar(self) -> bool:
        with st.sidebar:
            st.image("https://raw.githubusercontent.com/microsoft/sql-server-samples/master/media/sql-server-logo.png", width=100)
            st.title("SQL Agent Settings")
            
            # Configuration section
            with st.expander("🔑 API Configuration", expanded=not st.session_state.api_key_configured):
                api_key = st.text_input(
                    "OpenAI API Key",
                    value=os.getenv('OPENAI_API_KEY', ''),
                    type="password",
                    help="Enter your OpenAI API key"
                )
                
                if api_key:
                    try:
                        openai.api_key = api_key
                        openai.models.list()
                        st.success("✅ API Connected")
                        st.session_state.api_key_configured = True
                    except Exception as e:
                        st.error(f"❌ API Error: {str(e)}")
                        st.session_state.api_key_configured = False
                        return False
                else:
                    st.warning("⚠️ API Key Required")
                    return False
            
            # Model settings
            with st.expander("⚙️ Model Settings", expanded=False):
                st.selectbox(
                    "Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    help="Select the OpenAI model to use"
                )
                st.slider(
                    "Temperature",
                    0.0, 1.0, 0.0, 0.1,
                    help="Higher values make output more creative"
                )
            
            # Database schema
            with st.expander("📊 Database Schema", expanded=False):
                if st.session_state.metadata:
                    stats = st.session_state.metadata.get("statistics", {})
                    st.metric("Tables", stats.get("object_counts", {}).get("tables", 0))
                    st.metric("Views", stats.get("object_counts", {}).get("views", 0))
                    st.metric("Procedures", stats.get("object_counts", {}).get("procedures", 0))
                else:
                    st.info("No schema loaded")
            
            # Utilities
            with st.expander("🛠️ Developer Tools", expanded=False):
                st.checkbox("Enable Debug Mode", help="Show detailed processing information")
                st.checkbox("Show Raw SQL", help="Display raw SQL alongside formatted version")
            
        return True
        
    async def load_metadata(self, data_folder: str = "./sql_agent/data") -> Optional[Dict]:
        try:
            data_path = Path(data_folder)
            if not data_path.exists():
                st.error("❌ Data folder not found")
                return None
                
            sql_files = list(data_path.glob("*.sql"))
            if not sql_files:
                st.error("❌ No SQL files found")
                return None
                
            with st.spinner("Loading database schema..."):
                # Extract metadata
                metadata = self.metadata_extractor.extract_metadata_from_sql_files(
                    [str(f) for f in sql_files]
                )
            
                # Initialize vector store
                await self.agent.initialize_vector_store([str(f) for f in sql_files])
                
                if not metadata:
                    st.warning("⚠️ No metadata extracted")
                    return None
                
                st.sidebar.success(f"📁 Loaded {len(sql_files)} SQL files")
                st.session_state.metadata = metadata
                return metadata
                
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            st.error(f"❌ Error: {str(e)}")
            return None
            
    def render_main_interface(self):
        # Show processing indicator
        if st.session_state.get('processing', False):
            with st.spinner("🤖 Generating SQL query..."):
                st.empty()
            
        # Top navigation
        st.markdown("""
            <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
                background-color: #f0f2f6;
                padding: 10px 10px 0 10px;
                border-radius: 4px 4px 0 0;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #fff;
                border-radius: 4px 4px 0 0;
                gap: 2px;
                padding: 10px 20px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #e6f3ff;
            }
            </style>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs(["🤖 Query Generator", "📚 Schema Browser", "📝 Query History"])
        
        # Query Generator Tab
        with tabs[0]:
            st.markdown("### 🔍 Natural Language to SQL")
            
            # Query input
            query = st.text_area(
                "Describe your query",
                placeholder="Example: Show me all orders from last month with total amount greater than $1000",
                height=100,
                help="Be as specific as possible about what data you need"
            )
            
            # Action buttons
            button_container = st.container()
            left, right = button_container.columns([1, 5])
            if left.button("🚀 Generate", type="primary", disabled=not query):
                if not query.strip():
                    st.warning("⚠️ Please enter a query description")
                    return
                
                # Use asyncio.run in a separate thread to avoid blocking Streamlit
                import threading
                def run_async():
                    asyncio.run(self.process_query(query, st.session_state.metadata))
                
                thread = threading.Thread(target=run_async)
                thread.start()
                st.rerun()
            
            right.markdown("""
                <div style='padding: 8px 0 0 20px; color: #666;'>
                    Tips: Include details about filters, sorting, and specific columns you need
                </div>
            """, unsafe_allow_html=True)
            
            # Similar code snippets
            if 'last_query' in st.session_state and hasattr(st.session_state, 'last_result'):
                with st.expander("📚 Similar Code Snippets", expanded=False):
                    if hasattr(st.session_state.last_result, 'similarity_search'):
                        similar_examples = st.session_state.last_result.similarity_search
                        if similar_examples:
                            for i, example in enumerate(similar_examples, 1):
                                if isinstance(example, dict):
                                    st.markdown(f"**Example {i} (Similarity: {example['score']:.2f})**")
                                    st.markdown(f"Source: `{example['source']}`")
                                    st.code(example['content'], language="sql")
                                else:
                                    score, content = example
                                    st.markdown(f"**Example {i} (Similarity: {score:.2f})**")
                                    st.code(content, language="sql")
                        else:
                            st.info("No similar code snippets found")
                
                # Vector similarity visualization
                with st.expander("🔍 Vector Search Visualization", expanded=False):
                    result = st.session_state.last_result
                    if result.query_vector is not None and result.metadata_vectors:
                        plot = SimilaritySearchResultPlot(
                            query_vector=result.query_vector,
                            metadata_vectors=result.metadata_vectors,
                            labels=[f"Example {i+1}" for i in range(len(result.metadata_vectors))],
                            similarity_threshold=0.7
                        )
                        st.plotly_chart(plot.create_visualization(), use_container_width=True)
                    else:
                        st.info("No vector similarity data available for visualization")
            
            # Results area
            if 'last_query' in st.session_state:
                # Query result tabs
                query_tabs = st.tabs(["Generated SQL", "Explanation", "Sample Results"])
                
                with query_tabs[0]:
                    st.code(st.session_state.last_query, language="sql")
                    col1, col2 = st.columns([1, 6])
                    with col1:
                        st.download_button(
                            "💾 Download SQL",
                            st.session_state.last_query,
                            "query.sql",
                            "text/plain"
                        )
                
                with query_tabs[1]:
                    if 'last_explanation' in st.session_state:
                        # Split explanation into components
                        explanation = st.session_state.last_explanation
                        
                        # Display main explanation
                        st.markdown("### Query Overview")
                        st.markdown(explanation.split("In this query:")[0])
                        
                        # Display detailed breakdown
                        st.markdown("### Detailed Breakdown")
                        if "In this query:" in explanation:
                            points = explanation.split("In this query:")[1].strip().split("\n")
                            for point in points:
                                if point.strip():
                                    st.markdown(f"- {point.strip('-').strip()}")
                
                with query_tabs[2]:
                    if 'sample_results' in st.session_state:
                        st.dataframe(st.session_state.sample_results)
                    else:
                        st.info("No sample results available")
        
        # Schema Browser Tab
        with tabs[1]:
            st.markdown("### 📚 Database Schema Explorer")
            
            if not st.session_state.metadata:
                st.info("Load SQL files to view schema")
                return
                
            search = st.text_input("🔍 Search schema", placeholder="Table or column name...")
            
            # Display schema
            metadata = st.session_state.metadata
            for obj_type in ["tables", "views", "procedures"]:
                if objects := metadata.get(obj_type, []):
                    with st.expander(f"📑 {obj_type.title()}", expanded=True):
                        for obj in objects:
                            if not search or search.lower() in obj["name"].lower():
                                st.markdown(f"**{obj['name']}**")
                                if "definition" in obj:
                                    st.code(obj["definition"], language="sql")
                                if "schema" in obj and obj["schema"]:
                                    if isinstance(obj["schema"], list):
                                        for col in obj["schema"]:
                                            st.markdown(f"- {col['name']}: {col['type']}")
        
        # Query History Tab
        with tabs[2]:
            st.markdown("### 📝 Query History")
            
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            
            for idx, (timestamp, query, sql) in enumerate(st.session_state.query_history):
                with st.expander(f"Query {idx + 1} - {timestamp}", expanded=False):
                    st.markdown("**Natural Language:**")
                    st.markdown(query)
                    st.markdown("**Generated SQL:**")
                    st.code(sql, language="sql")
            
    async def process_query(self, query: str, metadata: Dict) -> None:
        try:
            st.session_state.processing = True
            try:
                # Get SQL files from the data directory
                data_folder = "./sql_agent/data"
                sql_files = list(Path(data_folder).glob("*.sql"))
                
                logger.info("Starting query processing...")
                results, usage_stats = await self.agent.process_query(query, metadata)
                logger.info("Query processing completed")
                
                if not results:
                    st.error("No results returned from query processing")
                    return
            except Exception as e:
                logger.error(f"Error during query processing: {str(e)}", exc_info=True)
                st.error(f"Error generating query: {str(e)}")
                return
                
            if results.error:
                st.error(f"❌ Error: {results.error}")
                return
                
            # Extract explanation from results
            explanation = results.agent_interactions.get("parse_intent", {}).get("result", "")
                
            # Store query in history
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            
            from datetime import datetime
            st.session_state.query_history.append((
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                query,
                results.generated_query
            ))
            
            # Store results for display
            st.session_state.last_query = results.generated_query
            st.session_state.last_result = results
            st.session_state.last_explanation = explanation
            
            # Create sample results if available
            try:
                import pandas as pd
                # This is a placeholder - in real implementation, 
                # you would get actual results from running the query
                sample_data = {
                    'EmployeeID': [1, 2, 3],
                    'EmployeeName': ['John Doe', 'Jane Smith', 'Bob Wilson'],
                    'Age': [35, 42, 28],
                    'DepartmentName': ['Sales', 'IT', 'Marketing'],
                    'TotalAmount': [15000, 12000, 18000],
                    'PerformanceRank': ['A', 'B+', 'A-']
                }
                st.session_state.sample_results = pd.DataFrame(sample_data)
            except Exception as e:
                logger.warning(f"Could not create sample results: {str(e)}")
            
            # Show results
            st.markdown("### 📝 Generated SQL")
            st.code(results.generated_query, language="sql")
            
            # Show explanation if enabled
            if st.session_state.get('show_explanation', True):
                with st.expander("🔍 Query Explanation", expanded=False):
                    st.markdown(results.agent_interactions.get("parse_intent", {}).get("result", ""))
            
            # Show usage stats
            with st.expander("📊 Usage Statistics", expanded=False):
                st.markdown(f"""
                - Tokens Used: {usage_stats.total_tokens:,}
                - Estimated Cost: ${usage_stats.cost:.4f}
                """)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"❌ Error: {str(e)}")

async def main():
    # Page config
    st.set_page_config(
        page_title="SQL Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        .stTextArea textarea {
            font-family: 'Consolas', monospace;
        }
        code {
            white-space: pre !important;
        }
        .sql-metadata {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize app
    app = SQLAgentApp()
    
    # Check prerequisites
    if not app.setup_sidebar():
        return
        
    # Load metadata
    metadata = await app.load_metadata()
    if not metadata:
        return
        
    # Render main interface
    app.render_main_interface()

if __name__ == "__main__":
    asyncio.run(main())
