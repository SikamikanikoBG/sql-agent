import streamlit as st
import os
import tempfile
from typing import Dict
import openai
from sql_agent.langgraph_orchestrator import SQLAgentOrchestrator
from sql_agent.metadata_extractor import extract_metadata_from_sql_files

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
    agent = SQLAgentOrchestrator(openai_api_key=api_key)

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
                metadata = extract_metadata_from_sql_files(sql_files, openai_api_key=api_key)

                with st.status("🤖 SQL Agent Workflow", expanded=True) as status:
                    status.update(label="🧠 Processing query through LLM agents...")
                    with st.spinner("Parsing user intent..."):
                        result, usage_stats = agent.process_query(user_query, metadata)
                        
                    # Show relevant files found
                    st.markdown("### 📑 Relevant Files")
                    if result.get("relevant_files"):
                        st.success(f"Found {len(result['relevant_files'])} relevant files:")
                        # Move file display outside the status block
                        relevant_files_content = []
                        for file in result["relevant_files"]:
                            try:
                                with open(file, 'r') as f:
                                    relevant_files_content.append((os.path.basename(file), f.read()))
                            except Exception as e:
                                relevant_files_content.append((os.path.basename(file), f"Error reading file: {str(e)}"))
                    else:
                        st.warning("No relevant files found")

                    # Create columns for workflow visualization
                    col1, col2, col3 = st.columns(3)
                    
                    # Show each step's result
                    with col1:
                        st.markdown("### 1️⃣ Parsed Intent")
                        st.info(result.get("parsed_intent", "No intent parsed"))
                        
                    with col2:
                        st.markdown("### 2️⃣ Generated Query")
                        if result["generated_query"].startswith('ERROR:'):
                            error_msg = result["generated_query"].split('\n')
                            st.error(error_msg[0])  # Main error
                            # Store additional info for later display
                            if len(error_msg) > 1:
                                result['available_objects'] = '\n'.join(error_msg[1:])
                        else:
                            st.code(result["generated_query"], language="sql")
                            
                    with col3:
                        st.markdown("### 3️⃣ Validation")
                        if result.get("is_valid", False):
                            st.success("✅ Query validation passed!")
                        else:
                            st.error(f"❌ Validation failed:\n{result.get('error', 'Unknown error')}")
                    
                    # Show usage statistics
                    st.markdown("### 📊 Usage Statistics")
                    tokens = usage_stats["tokens"]
                    st.info(
                        f"Tokens: {tokens['prompt']:,} sent, {tokens['completion']:,} received\n\n"
                        f"Cost: ${usage_stats['cost']:.2f} for this query"
                    )
                    
                    status.update(label="✅ Processing complete!", state="complete")

                # Display error details if any
                if result.get("available_objects"):
                    with st.expander("Show available database objects"):
                        st.text(result["available_objects"])

                # Display relevant files content outside the status block
                if result.get("relevant_files"):
                    st.markdown("### 📑 File Contents")
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
