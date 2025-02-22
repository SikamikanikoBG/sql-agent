import streamlit as st
import os
from typing import Dict
from .langgraph_orchestrator import SQLAgentOrchestrator
from .metadata_extractor import extract_metadata_from_sql_files

def main():
    st.title("SQL Agent - Query Generation and Execution")
    st.write("Enter your SQL-related prompt below to generate queries and analyze data.")

    # Get OpenAI API key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv('OPENAI_API_KEY', ''),
        type="password"
    )

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Initialize agent
    agent = SQLAgentOrchestrator(openai_api_key=api_key)

    # Query input section
    with st.expander("Enter your query", expanded=True):
        user_query = st.text_area(
            "Enter your natural language query:",
            height=100
        )

        # File uploader for SQL files
        uploaded_files = st.file_uploader(
            "Upload SQL files (optional)",
            accept_multiple_files=True,
            type=['sql']
        )

        if st.button("Generate Query"):
            if not user_query.strip():
                st.warning("Please enter a valid query or prompt.")
                return

            try:
                # Extract metadata from uploaded files if any
                metadata = {}
                if uploaded_files:
                    # Save uploaded files temporarily
                    temp_files = []
                    for file in uploaded_files:
                        with open(f"temp_{file.name}", "w") as f:
                            f.write(file.getvalue().decode())
                        temp_files.append(f"temp_{file.name}")
                    
                    metadata = extract_metadata_from_sql_files(temp_files)
                    
                    # Cleanup temp files
                    for file in temp_files:
                        os.remove(file)

                # Process the query
                result = agent.process_query(user_query, metadata)

                # Display results
                st.subheader("Generated SQL Query:")
                st.code(result["generated_query"], language="sql")

                if result.get("is_valid", False):
                    st.success("Query validation passed!")
                else:
                    st.error(f"Query validation failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
