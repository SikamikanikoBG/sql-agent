import streamlit as st
import os
import tempfile
from typing import Dict
from sql_agent.langgraph_orchestrator import SQLAgentOrchestrator
from sql_agent.metadata_extractor import extract_metadata_from_sql_files

def main():
    st.title("SQL Agent - Query Generation and Execution")
    st.write("Enter your SQL-related prompt below to generate queries and analyze data.")

    # Get OpenAI API key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv('OPENAI_API_KEY', ''),
        type="password"
    )

    # Mode selection
    mode = st.sidebar.selectbox(
        "Operation Mode",
        ["Local SQL Files", "Remote SQL Server (Coming Soon)"]
    )

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Initialize agent
    agent = SQLAgentOrchestrator(openai_api_key=api_key)

    # Query input section
    user_query = st.text_area(
        "Enter your natural language query:",
        height=100
    )

    if mode == "Local SQL Files":
        # File uploader for SQL files
        uploaded_files = st.file_uploader(
            "Upload SQL files",
            accept_multiple_files=True,
            type=['sql']
        )
        
        # Data folder path option
        data_folder = st.text_input(
            "Or enter path to folder with SQL files:",
            value="./sql_agent/data"
        )

    if st.button("Generate Query"):
            if not user_query.strip():
                st.warning("Please enter a valid query or prompt.")
                return

            try:
                with st.status("🤖 SQL Agent Workflow", expanded=True) as status:
                    # Extract metadata based on mode
                    metadata = {}
                    
                    status.update(label="📂 Processing SQL files...")
                    if mode == "Local SQL Files":
                        # Process uploaded files if any
                        temp_files = []
                        if uploaded_files:
                            for file in uploaded_files:
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.sql') as tmp:
                                    tmp.write(file.getvalue())
                                    temp_files.append(tmp.name)
                            
                            st.info(f"Processing {len(temp_files)} uploaded SQL files...")
                            metadata = extract_metadata_from_sql_files(temp_files)
                            
                            # Cleanup temp files
                            for file in temp_files:
                                os.remove(file)
                    
                    # Process data folder if specified and exists
                    elif os.path.exists(data_folder) and os.path.isdir(data_folder):
                        sql_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                                     if f.endswith('.sql')]
                        
                        if sql_files:
                            st.info(f"Processing {len(sql_files)} SQL files from {data_folder}...")
                            metadata = extract_metadata_from_sql_files(sql_files)
                        else:
                            st.warning(f"No SQL files found in {data_folder}")
                    else:
                        st.error(f"Data folder {data_folder} not found or not a directory")
                        return

                    # Display metadata found
                    if metadata:
                        with st.expander("📊 Extracted SQL Metadata", expanded=False):
                            st.json(metadata)

                    status.update(label="🧠 Processing query through LLM agents...")
                    with st.spinner("Parsing user intent..."):
                        result = agent.process_query(user_query, metadata)
                        
                    # Create columns for workflow visualization
                    col1, col2, col3 = st.columns(3)
                    
                    # Show each step's result
                    with col1:
                        st.markdown("### 1️⃣ Parsed Intent")
                        st.info(result.get("parsed_intent", "No intent parsed"))
                        
                    with col2:
                        st.markdown("### 2️⃣ Generated Query")
                        if result["generated_query"].startswith('ERROR:'):
                            st.error(result["generated_query"])
                        else:
                            st.code(result["generated_query"], language="sql")
                            
                    with col3:
                        st.markdown("### 3️⃣ Validation")
                        if result.get("is_valid", False):
                            st.success("✅ Query validation passed!")
                        else:
                            st.error(f"❌ Validation failed:\n{result.get('error', 'Unknown error')}")
                    
                    status.update(label="✅ Processing complete!", state="complete")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.exception(e)  # Show detailed error for debugging

if __name__ == "__main__":
    main()
