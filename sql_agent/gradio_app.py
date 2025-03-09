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

    def _get_available_columns(self) -> List[str]:
        """Get list of all available columns from SQL examples"""
        columns = set()
        if not self.metadata:
            return []

        # Extract columns from SQL examples in vector store
        if self.agent.vector_store:
            for doc in self.agent.vector_store.docstore._dict.values():
                if hasattr(doc, 'metadata') and doc.metadata.get('type') == 'column_list':
                    # Split column list and clean up each column name
                    col_list = doc.page_content.split(',')
                    for col in col_list:
                        clean_col = col.strip('[] \n\t')
                        if clean_col:
                            columns.add(clean_col)

        return sorted(list(columns))
        
    def _format_sql(self, sql: str) -> str:
        """Format SQL code for better readability."""
        # Basic SQL formatting
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 
                   'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
                   'CREATE', 'ALTER', 'DROP', 'INSERT', 'UPDATE', 'DELETE',
                   'AND', 'OR', 'UNION', 'UNION ALL', 'INTO']
                   
        # Add newlines before keywords
        formatted = sql
        for keyword in keywords:
            formatted = re.sub(f'\\s+{keyword}\\s+', f'\n{keyword} ', formatted, flags=re.IGNORECASE)
            
        # Indent subqueries and parenthetical expressions
        lines = formatted.split('\n')
        indent = 0
        result = []
        for line in lines:
            # Count opening and closing parentheses
            indent += line.count('(') - line.count(')')
            # Add appropriate indentation
            if line.strip():
                result.append('    ' * max(0, indent) + line.strip())
            
        return '\n'.join(result)
        
    def _initialize_data(self) -> None:
        """Initialize data and vector store once at startup"""
        try:
            # Get the path relative to the module
            module_dir = Path(__file__).parent
            data_path = module_dir / "data"
            # Create data folder if it doesn't exist
            data_path.mkdir(parents=True, exist_ok=True)
            
            sql_files = list(data_path.rglob("*.sql"))
            if not sql_files:
                logger.warning(f"No SQL files found in {data_path} or its subdirectories")
                return "⚠️ No SQL files found in data folder or subdirectories. Please add .sql files to continue."
                
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

    def process_query(self, api_key: str, query: str, columns: List[str], model: str, temperature: float, similarity_threshold: float) -> Tuple[str, str, str, str, str]:
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
            
            # Format similar examples with clickable links and improved SQL formatting
            similar_examples = """## 📚 Similar Examples

<style>
.sql-example {
    margin: 10px 0;
    border: 1px solid #ddd;
    border-radius: 4px;
    overflow: hidden;
}
.sql-header {
    background: #f8f9fa;
    padding: 8px 15px;
    border-bottom: 1px solid #ddd;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sql-content {
    padding: 15px;
    background: white;
    max-height: 500px;
    overflow: auto;
}
.sql-section {
    margin-bottom: 15px;
}
.sql-section-title {
    font-weight: bold;
    margin-bottom: 5px;
    color: #666;
}
.sql-code {
    white-space: pre;
    font-family: 'Consolas', monospace;
    tab-size: 4;
}
.similarity-score {
    background: #e9ecef;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.9em;
}
</style>
"""
            if results.similarity_search:
                for i, (score, example) in enumerate(results.similarity_search, 1):
                    if isinstance(example, dict):
                        source = example.get('source', 'Unknown')
                        # Format SQL for better readability
                        matching_content = self._format_sql(example.get('matching_content', ''))
                        full_content = self._format_sql(example.get('full_content', ''))
                        
                        similar_examples += f"""
<div class="sql-example">
    <div class="sql-header">
        <span>Example {i}</span>
        <span class="similarity-score">Similarity: {score:.2f}</span>
    </div>
    <div class="sql-content">
        <div class="sql-section">
            <div class="sql-section-title">📁 Source: <a href='file://{source}' target='_blank'>{source}</a></div>
        </div>
        <div class="sql-section">
            <div class="sql-section-title">🎯 Matching Content:</div>
            <div class="sql-code">```sql
{matching_content}
```</div>
        </div>
        <details>
            <summary>📑 View Complete File</summary>
            <div class="sql-section">
                <div class="sql-code">```sql
{full_content}
```</div>
            </div>
        </details>
    </div>
</div>
"""
                    else:
                        similar_examples += f"""
<div class="sql-example">
    <div class="sql-header">
        <span>Example {i}</span>
        <span class="similarity-score">Similarity: {score:.2f}</span>
    </div>
    <div class="sql-content">
        <div class="sql-code">```sql
{self._format_sql(str(example))}
```</div>
    </div>
</div>
"""
            
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
                    choices=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
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
                    value=0.06,
                    step=0.05,
                    label="Similarity Threshold"
                )
            
            with gr.Column(scale=2):
                # Query input and column selection
                gr.Markdown("### 🔍 Query Input")
                query = gr.Textbox(
                    label="Describe your query",
                    placeholder="Example: Show customer profile based on selected columns",
                    lines=4
                )
                columns = gr.Dropdown(
                    choices=app._get_available_columns(),
                    multiselect=True,
                    label="Select Columns",
                    info="Choose columns to include in your query"
                )
                generate_btn = gr.Button("🚀 Generate SQL", variant="primary")
        
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
            inputs=[api_key, query, columns, model, temperature, similarity_threshold],
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
