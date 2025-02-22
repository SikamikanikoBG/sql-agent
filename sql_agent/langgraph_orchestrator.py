import json
import os
import re
from typing import Dict, Any, TypedDict, Annotated, Union, List, Tuple
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class SQLAgentOrchestrator:
    def _create_workflow(self) -> StateGraph:
        """Create the Langgraph workflow for SQL query generation."""
        # Define the schema for our state
        class AgentState(TypedDict):
            user_input: str
            metadata: Dict
            parsed_intent: Annotated[str, "The parsed user intent"]
            relevant_content: Annotated[List[Dict], "Relevant SQL content from vector search"]
            generated_query: Annotated[str, "The generated SQL query"]
            is_valid: Annotated[bool, "Whether the query is valid"]
            error: Union[str, None]  # Error message if query is invalid
            agent_interactions: Dict[str, Dict]  # Store interactions for each agent

        # Initialize the workflow with the schema
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("parse_intent", self._parse_user_intent)
        workflow.add_node("find_relevant_content", self._find_relevant_content)
        workflow.add_node("generate_query", self._generate_sql_query)
        workflow.add_node("validate_query", self._validate_sql)
        
        # Define edges
        workflow.add_edge("parse_intent", "find_relevant_content")
        workflow.add_edge("find_relevant_content", "generate_query")
        workflow.add_edge("generate_query", "validate_query")
        
        # Set the entry point
        workflow.set_entry_point("parse_intent")
        
        # Compile the workflow
        return workflow.compile()
    
    def __init__(self, openai_api_key: str, server: str = None, database: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            openai_api_key=openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.server = server
        self.database = database
        self.workflow = self._create_workflow()
        self.total_tokens = {"prompt": 0, "completion": 0}
        self.total_cost = 0.0

    def _find_relevant_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Find relevant SQL content using RAG with vector similarity search."""
        system_prompt = """You are a RAG-based SQL content retrieval system.
Find relevant SQL code, table definitions, and stored procedures using vector similarity search."""
        
        user_prompt = f"Finding relevant SQL content for intent: {state['parsed_intent']}"

        # Prepare content chunks for embedding
        chunks = []
        chunk_sources = []
        
        # Process stored procedures
        if "procedure_info" in state["metadata"]:
            for proc_name, info in state["metadata"]["procedure_info"].items():
                chunk = f"PROCEDURE: {proc_name}\n"
                chunk += f"Description: {info['description']}\n"
                chunk += "Parameters:\n"
                for param in info.get('parameters', []):
                    chunk += f"  @{param['name']} ({param['type']}) {param['direction']}\n"
                chunk += f"Body:\n{info.get('body', '')}"
                chunks.append(chunk)
                chunk_sources.append(("procedure", proc_name))

        # Process table definitions
        if "tables" in state["metadata"]:
            for table in state["metadata"]["tables"]:
                chunk = f"TABLE: {table}\n"
                # Add table structure if available
                chunks.append(chunk)
                chunk_sources.append(("table", table))

        # Create FAISS index
        if not chunks:  # Handle empty case
            state["relevant_content"] = []
            state["agent_interactions"]["find_relevant_content"] = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "result": "No content available for vector search"
            }
            return state

        search_index = FAISS.from_texts(chunks, self.embeddings)
        
        # Perform similarity search
        results = search_index.similarity_search_with_score(
            state["parsed_intent"],
            k=min(5, len(chunks))  # Get top 5 or all if less
        )
        
        # Format results
        relevant_content = []
        for doc, score in results:
            idx = chunks.index(doc.page_content)
            source_type, source_name = chunk_sources[idx]
            relevant_content.append({
                "content": doc.page_content,
                "score": float(score),
                "type": source_type,
                "name": source_name
            })
        
        state["relevant_content"] = relevant_content
        
        # Store interaction details
        state["agent_interactions"]["find_relevant_content"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "result": f"Found {len(relevant_content)} relevant items:\n" + 
                     "\n".join(f"{item['type'].upper()}: {item['name']} (Score: {item['score']:.3f})"
                              for item in relevant_content)
        }
        
        return state

    def _analyze_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema relationships and semantic meanings."""
        system_prompt = """You are a database schema analyst. Analyze the provided SQL content to:
1. Identify table relationships (foreign keys, references)
2. Understand semantic meanings (e.g., which tables represent transactions, users, etc.)
3. Map business concepts to database structures
4. Identify relevant tables and columns for common query patterns

Return a concise analysis focusing on relationships and meanings relevant to the user's intent."""

        user_prompt = f"""User Intent: {state["parsed_intent"]}

Available Schema:
{state["knowledge_base"]}

Analyze the schema and explain how it relates to the user's intent."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            intent=state["parsed_intent"],
            knowledge_base=state["knowledge_base"]
        ))
        
        state["schema_analysis"] = response.content
        
        # Store interaction details
        state["agent_interactions"]["analyze_schema"].update({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "result": response.content
        })
        
        return state

    def _build_knowledge_base(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build knowledge base from relevant files."""
        system_prompt = """Analyze if this SQL content is relevant to the user's intent.
Focus on finding table definitions, queries, or stored procedures that could help
answer the user's query. Return YES or NO."""

        knowledge = []
        analysis_results = []
        
        for file_path in state["relevant_files"]:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Ask LLM if this content is relevant
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "Intent: {intent}\nSQL Content: {content}")
            ])
            
            response = self.llm.invoke(prompt.format_messages(
                intent=state["parsed_intent"],
                content=content
            ))
            
            is_relevant = "YES" in response.content.upper()
            analysis_results.append(f"File {file_path}: {'Relevant' if is_relevant else 'Not relevant'}")
            
            if is_relevant:
                knowledge.append(content)
        
        state["knowledge_base"] = "\n\n".join(knowledge)
        
        # Store interaction details
        state["agent_interactions"]["build_knowledge_base"].update({
            "system_prompt": system_prompt,
            "user_prompt": f"Analyzing relevance of {len(state['relevant_files'])} files for intent: {state['parsed_intent']}",
            "result": "\n".join(analysis_results)
        })
        
        return state

    def _parse_user_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse user's natural language query intent."""
        system_prompt = "Extract the key elements from this database query request."
        user_prompt = state["user_input"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(query=user_prompt))
        state["parsed_intent"] = response.content
        
        # Store interaction details
        state["agent_interactions"]["parse_intent"].update({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "result": response.content
        })
        
        return state
    
    def _generate_sql_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query based on parsed intent and metadata."""
        # Check metadata state
        if not state.get("metadata"):
            state["generated_query"] = "ERROR: No metadata provided"
            return state
            
        if "error" in state["metadata"]:
            state["generated_query"] = f"ERROR: {state['metadata']['error']}"
            return state

        if not any(state["metadata"].get(key) for key in ["tables", "views", "procedures"]):
            available_objects = []
            if state["metadata"].get("tables"):
                available_objects.append(f"Tables: {', '.join(state['metadata']['tables'])}")
            if state["metadata"].get("views"):
                available_objects.append(f"Views: {', '.join(state['metadata']['views'])}")
            if state["metadata"].get("procedures"):
                available_objects.append(f"Procedures: {', '.join(state['metadata']['procedures'])}")
            
            state["generated_query"] = "ERROR: Required database objects not found.\nAvailable objects:\n" + "\n".join(available_objects)
            return state

        system_prompt = """You are a SQL query generator. Generate queries using the provided knowledge base and available database objects.
            
Rules:
1. ONLY use tables, views, and procedures that exist in the schema with EXACT case sensitivity
2. If the required database objects don't exist, respond with 'ERROR: Required database objects not found'
3. Do not invent or assume the existence of any database objects
4. For complex operations, ALWAYS check and use existing stored procedures first
5. Use the knowledge base content as reference for similar queries and table relationships
6. Return only the SQL query or procedure call, no explanations
7. When using stored procedures, follow their exact parameter requirements
8. Use the schema analysis to understand table relationships and semantic meanings"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """Intent: {intent}

Available Objects:
{metadata}

Available Stored Procedures:
{procedures}

Knowledge Base:
{knowledge_base}

Schema Analysis:
{schema_analysis}""")
        ])
        
        # Format procedure information for the prompt
        procedures_info = ""
        if "procedure_info" in state["metadata"]:
            for proc_name, info in state["metadata"]["procedure_info"].items():
                procedures_info += f"\n{proc_name}:\n"
                procedures_info += f"  Description: {info['description']}\n"
                procedures_info += "  Parameters:\n"
                for param in info['parameters']:
                    procedures_info += f"    - @{param['name']} ({param['type']}) {param['direction']}\n"

        # Format user prompt with all context
        user_prompt = f"""Intent: {state['parsed_intent']}

Available Objects:
{json.dumps({k:v for k,v in state["metadata"].items() if k != "procedure_info"}, indent=2)}

Procedures:
{procedures_info}

Knowledge Base:
{state.get("knowledge_base", "No relevant SQL examples found.")}

Schema Analysis:
{state.get("schema_analysis", "No schema analysis available.")}"""

        response = self.llm.invoke(prompt.format_messages(
            metadata=json.dumps({k:v for k,v in state["metadata"].items() if k != "procedure_info"}, indent=2),
            procedures=procedures_info,
            knowledge_base=state.get("knowledge_base", "No relevant SQL examples found."),
            schema_analysis=state.get("schema_analysis", "No schema analysis available."),
            intent=state["parsed_intent"]
        ))
        
        state["generated_query"] = response.content
        
        # Format the prompts first
        formatted_messages = prompt.format_messages(
            metadata=json.dumps({k:v for k,v in state["metadata"].items() if k != "procedure_info"}, indent=2),
            procedures=procedures_info,
            knowledge_base=state.get("knowledge_base", "No relevant SQL examples found."),
            schema_analysis=state.get("schema_analysis", "No schema analysis available."),
            intent=state["parsed_intent"]
        )
        
        # Store interaction details with formatted prompts
        state["agent_interactions"]["generate_query"].update({
            "system_prompt": system_prompt,
            "user_prompt": formatted_messages[1].content,  # Get the formatted user message content
            "result": response.content
        })
        
        return state
    
    def _validate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated SQL query."""
        system_prompt = """Validate the generated SQL query against these rules:
1. Must be either a SELECT statement or a stored procedure call
2. All referenced tables must exist in the schema
3. No destructive operations allowed (DROP, DELETE, TRUNCATE)
4. Stored procedures must exist and be called with correct parameters"""

        user_prompt = f"""Validating query:
{state["generated_query"]}

Against available objects:
{json.dumps(state["metadata"], indent=2)}"""

        query = state["generated_query"].upper()
        
        # Check if it's an error message
        if query.startswith('ERROR:'):
            state["is_valid"] = False
            state["error"] = query.replace('ERROR:', '').strip()
            return state
            
        # Enhanced validation
        if not query.strip():
            state["is_valid"] = False
            state["error"] = "Empty query generated"
            return state
            
        # Allow both SELECT queries and stored procedure calls
        if not ("SELECT" in query or "EXEC" in query or "EXECUTE" in query):
            state["is_valid"] = False
            state["error"] = "Query must be either a SELECT statement or a stored procedure call"
            return state

        # Validate stored procedure calls
        if "EXEC" in query or "EXECUTE" in query:
            proc_match = re.search(r'(?:EXEC|EXECUTE)\s+(\w+)', query)
            if proc_match:
                proc_name = proc_match.group(1)
                if proc_name not in state["metadata"].get("procedures", []):
                    state["is_valid"] = False
                    state["error"] = f"Procedure {proc_name} does not exist"
                    return state
            
        if "DROP" in query or "DELETE" in query or "TRUNCATE" in query:
            state["is_valid"] = False
            state["error"] = "Destructive operations not allowed"
            return state
            
        # Extract table names from query
        # Simple regex to match table names after FROM and JOIN
        import re
        tables_in_query = set(re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', query))
        tables_in_query = {(t[0] or t[1]).lower() for t in tables_in_query}  # Flatten tuple matches and convert to lowercase
        
        # Check if all tables exist in metadata (case-insensitive)
        available_tables = {t.lower() for t in state["metadata"].get("tables", [])}
        unknown_tables = tables_in_query - available_tables
        
        if unknown_tables:
            state["is_valid"] = False
            state["error"] = f"Query references non-existent tables: {', '.join(unknown_tables)}"
            return state
            
        state["is_valid"] = True
        state["error"] = None
        
        # Store interaction details
        state["agent_interactions"]["validate_query"].update({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "result": "Query validation passed" if state["is_valid"] else f"Validation failed: {state['error']}"
        })
        
        return state
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on GPT-4 Turbo pricing."""
        return (prompt_tokens * 0.01 + completion_tokens * 0.03) / 1000  # $0.01/1K prompt, $0.03/1K completion

    def process_query(self, user_input: str, metadata: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process a user query through the workflow."""
        initial_state = {
            "user_input": user_input,
            "metadata": metadata,
            "parsed_intent": "",
            "relevant_content": [],
            "generated_query": "",
            "is_valid": False,
            "error": None,
            "agent_interactions": {}  # Store each agent's prompts and results
        }
        
        # Initialize agent_interactions with empty dictionaries
        initial_state["agent_interactions"] = {
            "parse_intent": {
                "system_prompt": "",
                "user_prompt": "",
                "result": ""
            },
            "find_relevant_content": {
                "system_prompt": "",
                "user_prompt": "",
                "result": ""
            },
            "generate_query": {
                "system_prompt": "",
                "user_prompt": "",
                "result": ""
            },
            "validate_query": {
                "system_prompt": "",
                "user_prompt": "",
                "result": ""
            }
        }
        # Use callback handler to track token usage
        with get_openai_callback() as cb:
            result = self.workflow.invoke(initial_state)
            
            self.total_tokens = {
                "prompt": cb.prompt_tokens,
                "completion": cb.completion_tokens
            }
            self.total_cost = cb.total_cost

        # Ensure parsed intent is included in the output
        if not result.get("parsed_intent"):
            result["parsed_intent"] = "Intent parsing failed"

        usage_stats = {
            "tokens": self.total_tokens,
            "cost": self.total_cost
        }
        
        return result, usage_stats
