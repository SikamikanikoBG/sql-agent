import streamlit as st
import os
import tempfile
from typing import Dict, List
import re
import openai  # Import the openai module

from sql_agent.langgraph_orchestrator import SQLAgentOrchestrator

def main():
    st.title("SQL Agent - Query Generation and Execution")
    st.write("Enter your SQL-related prompt below to generate queries and analyze data.")

    st.sidebar.markdown("### Configuration")
    
    # Get OpenAI API key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv('OPENAI_API_KEY', ''),
        type="password"
    )

    # Check OpenAI API connectivity
    if api_key:
        try:
            openai.api_key = api_key
            models = openai.models.list()
            st.sidebar.success("✅ OpenAI API connection successful")
            available_models = [model.id for model in models if "gpt" in model.id.lower()]
            st.sidebar.info(f"Available GPT models: {', '.join(available_models)}")
        except Exception as e:
            st.sidebar.error(f"❌ OpenAI API Error: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Operation Mode")
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Initialize agent
    agent = SQLAgentOrchestrator()

    # Fixed data folder path
    data_folder = "./sql_agent/data"

    # Scan data folder and show stats in sidebar
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        sql_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                    if f.endswith('.sql')]
        
        st.sidebar.markdown("### Data Files Status")
        if sql_files:
            try:
                metadata = extract_metadata_from_sql_files(sql_files)
                if not metadata:
                    st.sidebar.warning("⚠️ No metadata extracted from SQL files")
                    return
                st.sidebar.success(f"📁 Found {len(sql_files)} SQL files in knowledge base")
                
                st.sidebar.markdown("### Knowledge Base Stats")
                st.sidebar.text(f"📊 Tables: {len(metadata.get('tables', []))}")
                st.sidebar.text(f"📊 Views: {len(metadata.get('views', []))}")
                st.sidebar.text(f"📊 Procedures: {len(metadata.get('procedures', []))}")
                if 'procedure_info' in metadata:
                    st.sidebar.text(f"📊 Detailed Procedures: {len(metadata['procedure_info'])}")
            except Exception as e:
                st.sidebar.error(f"Error extracting metadata: {str(e)}")
                return
            
        else:
            st.sidebar.error("No SQL files found in knowledge base")
            st.error("No SQL files found in the knowledge base")
            return
    else:
        st.sidebar.error("Knowledge base folder not found")
        st.error("Knowledge base folder not found")
        return

    # Query input section
    user_query = st.text_area(
        "Enter your natural language query:",
        height=100
    )

    if st.button("Generate Query"):
            if not user_query.strip():
                st.warning("Please enter a valid query or prompt.")
                return

            try:
                # Use the already extracted metadata
                sql_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                           if f.endswith('.sql')]
                metadata = extract_metadata_from_sql_files(sql_files)

                # Initialize agent steps display
                st.markdown("### 🤖 SQL Agent Workflow")
                steps = {
                    "parse_intent": "1️⃣ Parse User Intent",
                    "find_relevant_content": "2️⃣ Find Relevant Content",
                    "generate_query": "3️⃣ Generate SQL Query",
                    "validate_query": "4️⃣ Validate Query"
                }
                
                # Process query and show each step
                result, usage_stats = agent.process_query(user_query, metadata)
                
                # Display each agent's interaction
                for step_name, step_data in result["agent_interactions"].items():
                    with st.expander(f"🤖 {steps[step_name]}", expanded=True):
                        st.markdown("**System Prompt:**")
                        st.code(step_data["system_prompt"])
                        st.markdown("**User Prompt:**")
                        st.code(step_data["user_prompt"])
                        st.markdown("**Result:**")
                        st.code(step_data["result"])
                
                # Display final results
                if result.get("similarity_search"):
                    with st.expander("📊 Similarity Search Results"):
                        st.markdown("### Top Matching Chunks:")
                        for idx, (score, chunk) in enumerate(result["similarity_search"], 1):
                            st.markdown(f"**Match {idx}** (Score: {score:.3f})")
                            st.code(chunk, language="sql")
                
                # Display final query with formatting
                st.markdown("### 📝 Final Generated Query")
                if result["generated_query"].startswith('ERROR:'):
                    error_msg = result["generated_query"].split('\n')
                    st.error(error_msg[0])
                    if metadata:
                        with st.expander("Available Database Objects"):
                            if metadata.get('tables'):
                                st.markdown("**Tables:**")
                                for table in metadata['tables']:
                                    st.markdown(f"- `{table}`")
                            if metadata.get('views'):
                                st.markdown("**Views:**")
                                for view in metadata['views']:
                                    st.markdown(f"- `{view}`")
                            if metadata.get('procedures'):
                                st.markdown("**Procedures:**")
                                for proc in metadata['procedures']:
                                    st.markdown(f"- `{proc}`")
                else:
                    st.code(result["generated_query"], language="sql")
                
                # Show usage statistics
                with st.expander("📊 Usage Statistics"):
                    tokens = usage_stats["tokens"]
                    st.info(
                        f"Tokens: {tokens['prompt']:,} sent, {tokens['completion']:,} received\n\n"
                        f"Cost: ${usage_stats['cost']:.2f} for this query"
                    )

                # Display error details if any
                if result.get("available_objects"):
                    with st.expander("Show available database objects"):
                        st.text(result["available_objects"])

                # Display relevant files content
                if result.get("relevant_files"):
                    st.markdown("### 📑 File Contents")
                    relevant_files_content = []
                    for file_path in result["relevant_files"]:
                        try:
                            with open(file_path, 'r') as f:
                                relevant_files_content.append((os.path.basename(file_path), f.read()))
                        except Exception as e:
                            relevant_files_content.append((os.path.basename(file_path), f"Error reading file: {str(e)}"))
                            
                    for filename, content in relevant_files_content:
                        with st.expander(f"📄 {filename}"):
                            if content.startswith("Error"):
                                st.error(content)
                            else:
                                st.code(content, language="sql")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.exception(e)  # Show detailed error for debugging

if __name__ == "__main__":
    main()
