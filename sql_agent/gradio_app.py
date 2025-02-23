import os
import logging
import gradio as gr
from pathlib import Path
from typing import Dict, Optional, Tuple
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
        
    def _initialize_data(self, data_folder: str = "./sql_agent/data") -> None:
        """Initialize data and vector store once at startup"""
        try:
            data_path = Path(data_folder)
            if not data_path.exists():
                return "❌ Data folder not found"
                
            sql_files = list(data_path.glob("*.sql"))
            if not sql_files:
                return "❌ No SQL files found"
                
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

    def process_query(self, api_key: str, query: str, model: str, temperature: float) -> Tuple[str, str, str, str]:
        """Process a query and return results"""
        if not api_key.strip():
            return "⚠️ API Key Required", "", "", ""
            
        if not query.strip():
            return "⚠️ Please enter a query", "", "", ""
            
        try:
            # Set API key
            openai.api_key = api_key
            
            # Update agent settings
            self.agent.model_name = model
            self.agent.temperature = temperature
            
            # Process query
            results, usage_stats = self.agent.process_query(query, self.metadata)
            
            if results.error:
                return f"❌ Error: {results.error}", "", "", ""
                
            # Format similar examples
            similar_examples = ""
            if results.similarity_search:
                similar_examples = "Similar SQL Examples:\n\n"
                for i, (score, example) in enumerate(results.similarity_search, 1):
                    if isinstance(example, dict):
                        similar_examples += f"Example {i} (Score: {score:.2f})\n"
                        similar_examples += f"Source: {example.get('source', 'Unknown')}\n"
                        similar_examples += f"```sql\n{example.get('content', '')}\n```\n\n"
                    else:
                        similar_examples += f"Example {i} (Score: {score:.2f})\n"
                        similar_examples += f"```sql\n{str(example)}\n```\n\n"
            
            # Format explanation
            explanation = results.agent_interactions.get("parse_intent", {}).get("result", "")
            
            # Format usage stats
            usage_info = f"""
            Usage Statistics:
            - Total Tokens: {usage_stats.total_tokens:,}
            - Cost: ${usage_stats.cost:.4f}
            """
            
            return results.generated_query, explanation, similar_examples, usage_info
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"❌ Error: {str(e)}", "", "", ""

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    app = SQLAgentGradioApp()
    
    # Custom CSS
    custom_css = """
    .container { max-width: 1200px; margin: auto; }
    .output-panel { min-height: 300px; }
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
            
            with gr.Column(scale=2):
                # Query input
                gr.Markdown("### 🔍 Query Input")
                query = gr.Textbox(
                    label="Describe your query",
                    placeholder="Example: Show me all orders from last month with total amount greater than $1000",
                    lines=4
                )
                generate_btn = gr.Button("🚀 Generate SQL", variant="primary")
        
        # Output panels
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Generated SQL")
                sql_output = gr.Code(language="sql")
            
            with gr.Column():
                gr.Markdown("### 🎯 Query Analysis")
                explanation_output = gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📚 Similar Examples")
                examples_output = gr.Markdown()
            
            with gr.Column():
                gr.Markdown("### 📊 Usage Statistics")
                usage_output = gr.Markdown()
        
        # Set up event handler
        generate_btn.click(
            fn=app.process_query,
            inputs=[api_key, query, model, temperature],
            outputs=[sql_output, explanation_output, examples_output, usage_output]
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
