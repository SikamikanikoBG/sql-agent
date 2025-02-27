import os
import logging
import gradio as gr
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import openai
from sql_agent.langgraph_orchestrator import SQLAgentOrchestrator
from sql_agent.metadata_extractor import MetadataExtractor
from sql_agent.visualization import SimilaritySearchResultPlot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLAgentGradioApp:
    """Gradio application for SQL query generation and analysis."""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.metadata = None
        # Initialize agent and vector store once at startup
        self.agent = SQLAgentOrchestrator()
        self._initialize_data()
        # Initialize column cache
        self.column_cache = self._build_column_cache()

    def _build_column_cache(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Build a cache of all columns from metadata for quick lookup"""
        column_cache = {}
        if not self.metadata:
            return column_cache

        # Extract columns from permanent tables
        for table in self.metadata.get("permanent_tables", []):
            if isinstance(table, dict):
                table_name = table.get("name", "unknown")
                if table_name in self.metadata.get("schemas", {}):
                    for column in self.metadata["schemas"][table_name]:
                        col_name = column["name"].lower()
                        if col_name not in column_cache:
                            column_cache[col_name] = []
                        column_cache[col_name].append((
                            column["name"],
                            table_name,
                            column["type"]
                        ))

        return column_cache

    def _get_column_suggestions(self, query: str) -> Optional[List[List[str]]]:
        """Get column suggestions based on the current query text"""
        if not query or '/' not in query:
            return None

        # Find the last slash and get the search term
        last_slash_idx = query.rindex('/')
        search_term = query[last_slash_idx + 1:].lower()

        # Filter columns based on search term
        suggestions = []
        for col_name, columns in self.column_cache.items():
            if col_name.startswith(search_term):
                for col in columns:
                    suggestions.append(list(col))

        # Sort suggestions by column name
        suggestions.sort(key=lambda x: x[0])
        
        return suggestions if suggestions else None

    def _insert_column(self, query: str, selected_row: List[str]) -> str:
        """Insert the selected column into the query text"""
        if not query or not selected_row:
            return query

        # Find the last slash
        last_slash_idx = query.rindex('/')
        
        # Replace the partial term with the selected column
        return query[:last_slash_idx] + '/' + selected_row[0] + query[query.find(' ', last_slash_idx) if ' ' in query[last_slash_idx:] else len(query):]
        
    def _initialize_data(self, data_folder: str = "./sql_agent/data") -> None:
        """Initialize data and vector store once at startup"""
        try:
            data_path = Path(data_folder)
            # Create data folder if it doesn't exist
            data_path.mkdir(parents=True, exist_ok=True)
            
            sql_files = list(data_path.glob("*.sql"))
            if not sql_files:
                logger.warning(f"No SQL files found in {data_folder}")
                return "⚠️ No SQL files found in data folder. Please add .sql files to continue."
                
            # Extract metadata and initialize vector store once
            self.metadata = self.metadata_extractor.extract_metadata_from_sql_files(
                [str(f) for f in sql_files]
            )
            if self.metadata:
                # Initialize vector store once
                self.agent.initialize_vector_store([str(f) for f in sql_files])
                return "✅ Data initialized successfully"
            return "⚠️ No metadata extracted"
            
        except Exception as e:
            logger.error(f"Error initializing data: {str(e)}")
            return f"❌ Error initializing data: {str(e)}"

    def process_query(self, api_key: str, query: str, model: str, temperature: float, similarity_threshold: float) -> Tuple[str, str, str, str, str]:
        """Process a query and return results"""
        if not api_key.strip():
            return "⚠️ API Key Required", "", "", "", ""
            
        if not query.strip():
            return "⚠️ Please enter a query", "", "", "", ""
            
        try:
            # Set API key
            openai.api_key = api_key
            
            # Update agent settings
            self.agent.model_name = model
            self.agent.temperature = temperature
            self.agent.similarity_threshold = similarity_threshold
            
            # Process query
            results, usage_stats = self.agent.process_query(query, self.metadata)
            
            if results.error:
                return f"❌ Error: {results.error}", "", "", "", ""
                
            # Format agent interactions
            agent_interactions = "## 🤖 Agent Interactions\n\n"
            for step, interaction in results.agent_interactions.items():
                agent_interactions += f"### Step: {step}\n\n"
                
                if "system_prompt" in interaction:
                    agent_interactions += "<div class='system-message'>\n\n"
                    agent_interactions += "**System Prompt:**\n\n"
                    agent_interactions += f"```\n{interaction['system_prompt']}\n```\n\n"
                    agent_interactions += "</div>\n\n"
                
                if "user_prompt" in interaction:
                    agent_interactions += "<div class='user-message'>\n\n"
                    agent_interactions += "**User Prompt:**\n\n"
                    agent_interactions += f"```\n{interaction['user_prompt']}\n```\n\n"
                    agent_interactions += "</div>\n\n"
                
                if "result" in interaction:
                    agent_interactions += "<div class='assistant-message'>\n\n"
                    agent_interactions += "**Assistant Response:**\n\n"
                    agent_interactions += f"```\n{interaction['result']}\n```\n\n"
                    agent_interactions += "</div>\n\n"
                
                agent_interactions += "---\n\n"
            
            # Format similar examples
            similar_examples = "## 📚 Similar Examples\n\n"
            if results.similarity_search:
                for i, (score, example) in enumerate(results.similarity_search, 1):
                    similar_examples += f"### Example {i} (Similarity: {score:.2f})\n\n"
                    if isinstance(example, dict):
                        similar_examples += f"**Source:** {example.get('source', 'Unknown')}\n\n"
                        similar_examples += f"```sql\n{example.get('content', '')}\n```\n\n"
                    else:
                        similar_examples += f"```sql\n{str(example)}\n```\n\n"
            
            # Format explanation
            explanation = "## 🎯 Query Analysis\n\n"
            explanation += results.agent_interactions.get("parse_intent", {}).get("result", "")
            
            # Format usage stats
            usage_info = "## 📊 Usage Statistics\n\n"
            usage_info += f"- **Total Tokens:** {usage_stats.total_tokens:,}\n"
            usage_info += f"- **Cost:** ${usage_stats.cost:.4f}\n"
            
            return results.generated_query, explanation, similar_examples, usage_info, agent_interactions
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"❌ Error: {str(e)}", "", "", ""

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    app = SQLAgentGradioApp()
    
    # Custom CSS
    custom_css = """
    .container { 
        max-width: 1400px; 
        margin: auto;
        padding: 20px;
    }
    .output-panel { 
        min-height: 300px;
        border-radius: 8px;
        background: #f8f9fa;
        padding: 15px;
        margin: 10px 0;
    }
    .markdown-text {
        font-size: 16px;
        line-height: 1.6;
    }
    .agent-interaction {
        border-left: 4px solid #007bff;
        padding-left: 15px;
        margin: 10px 0;
    }
    .system-message {
        background: #e9ecef;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    .user-message {
        background: #f1f3f5;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    .assistant-message {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css) as interface:
        gr.Markdown("# SQL Agent")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("### ⚙️ Settings")
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Enter your OpenAI API key",
                    type="password",
                    value=os.getenv('OPENAI_API_KEY', '')
                )
                model = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-4"],
                    value="gpt-3.5-turbo",
                    label="Model"
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Temperature"
                )
                similarity_threshold = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Similarity Threshold"
                )
            
            with gr.Column(scale=2):
                # Query input with column suggestions
                gr.Markdown("### 🔍 Query Input")
                with gr.Row():
                    query = gr.Textbox(
                        label="Describe your query",
                        placeholder="Example: Show customer profile based on /age /salary /education",
                        lines=4,
                        interactive=True
                    )
                    column_suggestions = gr.Dataframe(
                        headers=["Column Name", "Table", "Type"],
                        visible=False,
                        interactive=True
                    )
                generate_btn = gr.Button("🚀 Generate SQL", variant="primary")

                # Add event handlers for column suggestions
                query.change(
                    fn=app._get_column_suggestions,
                    inputs=[query],
                    outputs=[column_suggestions],
                    show_progress=False
                )

                column_suggestions.select(
                    fn=app._insert_column,
                    inputs=[query, column_suggestions],
                    outputs=[query],
                    show_progress=False
                )
        
        with gr.Tabs() as tabs:
            with gr.TabItem("📝 Generated SQL"):
                sql_output = gr.Code(language="sql", label="Generated SQL Query")
            
            with gr.TabItem("🎯 Query Analysis"):
                explanation_output = gr.Markdown(label="Analysis")
            
            with gr.TabItem("📚 Similar Examples"):
                examples_output = gr.Markdown(label="Examples")
            
            with gr.TabItem("📊 Usage Statistics"):
                usage_output = gr.Markdown(label="Usage")
            
            with gr.TabItem("🤖 Agent Interactions"):
                agent_interactions_output = gr.Markdown(label="Interactions")
        
        # Set up event handler
        generate_btn.click(
            fn=app.process_query,
            inputs=[api_key, query, model, temperature, similarity_threshold],
            outputs=[sql_output, explanation_output, examples_output, usage_output, agent_interactions_output]
        )
        
        # Initialize app
        init_status = app._initialize_data()
        gr.Markdown(f"### Status: {init_status}")
    
    return interface

def main():
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
