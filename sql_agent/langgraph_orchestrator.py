import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import streamlit as st
from pathlib import Path
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Represents the result of a query processing operation."""
    generated_query: str
    agent_interactions: Dict[str, Any]
    similarity_search: List[Tuple[float, str]]
    validation_result: Dict[str, Any]
    relevant_files: List[str]
    error: Optional[str] = None
    query_vector: Optional[List[float]] = None
    metadata_vectors: Optional[List[List[float]]] = None

@dataclass
class UsageStats:
    """Tracks token usage and cost statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

class SQLAgentOrchestrator:
    """Orchestrates SQL query generation and processing using LangChain."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        similarity_threshold: float = 0.3,  # Lower threshold to catch more examples
        max_examples: int = 5  # Increase max examples
    ):
        """Initialize the SQL Agent Orchestrator.
        
        Args:
            model_name: The OpenAI model to use for query generation
            temperature: Model temperature (0-1)
            similarity_threshold: Threshold for similarity search results
            max_examples: Maximum number of similar examples to retrieve
        """
        self.model_name = model_name
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.max_examples = max_examples
        
        self.metadata = None
        self.vector_store = None
        self.usage_stats = UsageStats()
        
        self._setup_components()
        self._setup_logging()
        
    def _setup_components(self) -> None:
        """Initialize LangChain components."""
        # Initialize language model
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize text splitter for SQL
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Setup processing chains
        self._setup_chains()
        
    def _setup_logging(self) -> None:
        """Configure logging for the orchestrator."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _setup_chains(self) -> None:
        """Set up LangChain processing chains."""
        # Intent parsing chain
        self.intent_prompt = PromptTemplate(
            input_variables=["query", "metadata", "similar_examples"],
            template="""Analyze the user's query intent using the context below:

{similar_examples}

User Query:
{query}

Identify:
1. Required tables and their relationships
2. Desired columns/calculations
3. Any filters or conditions
4. Sorting or grouping requirements

Intent Analysis:"""
        )
        self.intent_chain = self.intent_prompt | self.llm
        
        # Query generation chain for MS SQL
        self.query_prompt = PromptTemplate(
            input_variables=["intent", "metadata", "similar_examples"],
            template="""Generate a MS SQL query based on the analyzed intent and similar examples:

{similar_examples}

Analyzed Intent:
{intent}

Consider MS SQL best practices:
1. Use appropriate JOIN types with proper NOLOCK hints where needed
2. Include schema names and use square brackets for identifiers
3. Add proper WHERE clauses with parameter sniffing consideration
4. Include GROUP BY/ORDER BY with appropriate indexing hints
5. Use appropriate MS SQL features:
   - CTEs for complex queries
   - OFFSET/FETCH for paging
   - OUTPUT clause for DML operations
   - Appropriate data type functions
   - Proper NULL handling with ISNULL/COALESCE
   - Table variables vs temp tables based on size
   - Proper transaction isolation levels

Important Notes for Temporary Tables:
1. If you need data from a temporary table (#temp), look at its source tables in the metadata
2. You can either:
   a) Use the source tables directly with appropriate JOINs and conditions
   b) Recreate the temporary table using its definition from metadata
3. Choose the approach that best matches the original query's intent and performance needs

Generated SQL Query:"""
        )
        self.query_chain = self.query_prompt | self.llm
        
        # Query validation chain for MS SQL
        self.validation_prompt = PromptTemplate(
            input_variables=["query", "metadata"],
            template="""Validate the following MS SQL query against the database metadata:

Database Metadata:
{metadata}

SQL Query:
{query}

Check for:
1. Table existence and relationships (including schema names)
2. Column validity and data types
3. MS SQL syntax correctness
4. Performance considerations:
   - Proper indexing hints
   - JOIN optimization
   - NOLOCK usage where appropriate
   - Execution plan hints
5. SQL injection risks
6. MS SQL specific features:
   - Proper use of square brackets for identifiers
   - Schema qualification
   - Appropriate collation settings
   - Table hints and locking hints

Validation Results:"""
        )
        self.validation_chain = self.validation_prompt | self.llm

    def process_query(self, query: str, metadata: Dict, sql_files: Optional[List[str]] = None) -> Tuple[QueryResult, UsageStats]:
        """Process a user's natural language query and generate an SQL query.
        
        Args:
            query: The natural language query from the user
            metadata: Database metadata including tables, views, etc.
            
        Returns:
            Tuple containing QueryResult and UsageStats
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Reset usage stats
            self.usage_stats = UsageStats()
            
            # Initialize vector store if needed
            if sql_files and not self.vector_store:
                self.initialize_vector_store(sql_files)
            
            # Find similar examples and get vectors
            with st.spinner("🔍 Searching vector store..."):
                similar_examples, query_vector, metadata_vectors = self._find_similar_examples(query)
            
            # Format similar examples for display
            with st.spinner("📝 Formatting examples..."):
                if not similar_examples:
                    st.warning("⚠️ No similar information found in the SQL codebase")
                formatted_examples = []
            for score, content in similar_examples:
                if isinstance(content, dict):
                    formatted_examples.append({
                        "score": float(score),
                        "content": content.get("content", ""),
                        "source": content.get("source", "Unknown")
                    })
                else:
                    formatted_examples.append({
                        "score": float(score),
                        "content": str(content),
                        "source": "Unknown"
                    })
            
            # Parse intent
            with st.spinner("🎯 Analyzing query intent..."):
                formatted_examples = self._format_examples(similar_examples)
                context = formatted_examples if similar_examples else self._format_metadata(metadata)
                
                logger.info(f"Using {len(similar_examples)} similar examples in prompt")
                for i, (score, example) in enumerate(similar_examples):
                    logger.info(f"Example {i+1} (score: {score:.2f}):")
                    logger.info(f"Content: {example.get('content', '')[:200]}...")
                
                logger.info("Starting intent parsing...")
            try:
                intent_result = self.intent_chain.invoke({
                    "query": query,
                    "metadata": "",  # Empty metadata when we have examples
                    "similar_examples": context
                })
                logger.info("Intent parsing completed")
                self._update_usage_stats(intent_result)
            except Exception as e:
                logger.error(f"Error during intent parsing: {str(e)}", exc_info=True)
                raise
            
            # Generate query
            with st.spinner("✍️ Generating SQL query..."):
                logger.info("Starting query generation...")
            try:
                query_result = self.query_chain.invoke({
                    "intent": intent_result.content,
                    "metadata": "",  # Empty metadata when we have examples
                    "similar_examples": context
                })
                logger.info("Query generation completed")
                self._update_usage_stats(query_result)
            except Exception as e:
                logger.error(f"Error during query generation: {str(e)}", exc_info=True)
                raise
            
            # Validate generated query
            with st.spinner("✅ Validating generated query..."):
                logger.info("Starting query validation...")
            try:
                # For validation, we need minimal schema info
                minimal_metadata = {
                    k: v for k, v in metadata.items() 
                    if k in ['tables', 'views', 'columns', 'relationships']
                }
                validation_result = self.validation_chain.invoke({
                    "query": query_result.content,
                    "metadata": self._format_metadata(minimal_metadata)
                })
                logger.info("Query validation completed")
                self._update_usage_stats(validation_result)
            except Exception as e:
                logger.error(f"Error during query validation: {str(e)}", exc_info=True)
                raise

            # Track relevant context
            relevant_files = []
            for score, content in similar_examples:
                if isinstance(content, dict) and "source" in content:
                    if score > self.similarity_threshold:
                        relevant_files.append(content["source"])
                elif isinstance(content, str) and hasattr(content, "metadata"):
                    metadata = getattr(content, "metadata", {})
                    if metadata.get("source") and score > self.similarity_threshold:
                        relevant_files.append(metadata["source"])
            
            # Prepare result
            result = QueryResult(
                generated_query=query_result.content,
                agent_interactions={
                    "parse_intent": {
                        "system_prompt": self.intent_prompt.template,
                        "user_prompt": query,
                        "result": intent_result.content,
                        "tokens_used": intent_result.generation_info.get('token_usage', {}).get('total_tokens', 0) if hasattr(intent_result, 'generation_info') else 0
                    },
                    "generate_query": {
                        "system_prompt": self.query_prompt.template,
                        "user_prompt": intent_result.content,
                        "result": query_result.content,
                        "tokens_used": query_result.generation_info.get('token_usage', {}).get('total_tokens', 0) if hasattr(query_result, 'generation_info') else 0
                    },
                    "validate_query": {
                        "system_prompt": self.validation_prompt.template,
                        "user_prompt": query_result.content,
                        "result": validation_result.content,
                        "tokens_used": validation_result.generation_info.get('token_usage', {}).get('total_tokens', 0) if hasattr(validation_result, 'generation_info') else 0
                    }
                },
                similarity_search=similar_examples,
                query_vector=query_vector,
                metadata_vectors=metadata_vectors,
                validation_result=self._parse_validation_result(validation_result.content),
                relevant_files=relevant_files
            )
            
            return result, self.usage_stats
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return QueryResult(
                generated_query=f"ERROR: {str(e)}",
                agent_interactions={},
                similarity_search=[],
                validation_result={},
                relevant_files=[],
                error=str(e)
            ), self.usage_stats
            
    def _find_similar_examples(self, query: str) -> Tuple[List[Tuple[float, str]], List[float], List[List[float]]]:
        """Find similar SQL examples from the vector store.
        
        Args:
            query: The user's query to find similar examples for
            
        Returns:
            Tuple containing:
            - List of (score, content) tuples
            - Query vector
            - List of metadata vectors
        """
        if not self.vector_store:
            return [], [], []
            
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=self.max_examples
            )
            logger.info(f"Found {len(results)} similar examples")
            
            # Log the actual results for debugging
            for doc, score in results:
                logger.info(f"Similarity score: {score}")
                logger.info(f"Content: {doc.page_content[:100]}...")  # First 100 chars
                logger.info(f"Metadata: {doc.metadata}")
                
            # Convert L2 distance to cosine similarity (0-1 range)
            # FAISS returns (Document, score) tuples
            max_distance = max(score for doc, score in results) if results else 1
            normalized_results = []
            for doc, score in results:
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 - (score / max_distance)
                normalized_results.append((similarity, doc))
                logger.info(f"Original L2 distance: {score}, Normalized similarity: {similarity}")
            
            logger.info("Normalized similarity scores:")
            for score, doc in normalized_results:
                logger.info(f"Normalized score: {score}")
            
            # Get embeddings
            query_vector = self.embeddings.embed_query(query)
            metadata_vectors = [self.embeddings.embed_query(doc.page_content) for doc, _ in results]
            
            similar_examples = []
            for similarity, doc in normalized_results:
                logger.info(f"Checking similarity {similarity} against threshold {self.similarity_threshold}")
                # Always include top examples even if below threshold
                if len(similar_examples) < 2 or similarity >= self.similarity_threshold:
                    try:
                        example = {
                            'content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                            'source': doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                        }
                        similar_examples.append((similarity, example))
                        logger.info(f"Added example with similarity {similarity}")
                    except Exception as e:
                        logger.error(f"Error processing document: {str(e)}")
                else:
                    logger.info(f"Skipping example with similarity {similarity} below threshold {self.similarity_threshold}")
            
            return similar_examples, query_vector, metadata_vectors
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return [], [], []
                          
        return similar_examples, query_vector, metadata_vectors
    
    def _format_metadata(self, metadata: Dict) -> str:
        """Format metadata for prompt templates.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Formatted metadata string
        """
        sections = []
        
        # Format permanent tables
        if metadata.get("permanent_tables"):
            sections.append("Permanent Tables:")
            for table in metadata["permanent_tables"]:
                if isinstance(table, dict):
                    table_name = table.get("name", "unknown")
                    database = table.get("database", "")
                    prefix = f"[{database}]." if database else ""
                    sections.append(f"- {prefix}{table_name}")
                    if metadata.get("schemas", {}).get(table_name):
                        schema = metadata["schemas"][table_name]
                        if isinstance(schema, list):
                            for column in schema:
                                sections.append(f"  * {column['name']}: {column['type']}")
                else:
                    sections.append(f"- {table}")

        # Format temporary tables with their source tables
        if metadata.get("temp_tables"):
            sections.append("\nTemporary Tables:")
            for temp_name, temp_info in metadata["temp_tables"].items():
                sections.append(f"- {temp_name}")
                sections.append("  Definition:")
                sections.append(f"  {temp_info['definition']}")
                if temp_info.get("source_tables"):
                    sections.append("  Source Tables:")
                    for source in temp_info["source_tables"]:
                        sections.append(f"  * {source}")
        
        if metadata.get("views"):
            sections.append("\nViews:")
            for view in metadata["views"]:
                sections.append(f"- {view}")
        
        if metadata.get("relationships"):
            sections.append("\nRelationships:")
            for rel in metadata.get("relationships", []):
                source_table = rel.get("source_table", "unknown")
                target_table = rel.get("target_table", "unknown")
                source_cols = rel.get("source_columns", ["unknown"])
                target_cols = rel.get("target_columns", ["unknown"])
                sections.append(
                    f"- {source_table}.{','.join(source_cols)} -> "
                    f"{target_table}.{','.join(target_cols)}"
                )
        
        return "\n".join(sections)
    
    def _format_examples(self, examples: List[Tuple[float, str]]) -> str:
        """Format similar examples for prompt templates.
        
        Args:
            examples: List of (score, content) tuples
            
        Returns:
            Formatted examples string
        """
        if not examples:
            return "No similar examples found."
            
        sections = ["Similar SQL examples found:"]
        for i, (score, content) in enumerate(examples, 1):
            if isinstance(content, dict):
                sections.extend([
                    f"\nExample {i} (Similarity: {score:.2f}):",
                    f"Source: {content.get('source', 'Unknown')}",
                    "SQL:",
                    content.get('content', ''),
                    "-" * 80  # Separator
                ])
            else:
                sections.extend([
                    f"\nExample {i} (Similarity: {score:.2f}):",
                    "SQL:",
                    str(content),
                    "-" * 80  # Separator
                ])
        
        return "\n".join(sections)
    
    def _parse_validation_result(self, validation_result: str) -> Dict[str, Any]:
        """Parse validation result into structured format.
        
        Args:
            validation_result: Raw validation result string
            
        Returns:
            Structured validation result
        """
        try:
            # Try to parse as JSON first
            return json.loads(validation_result)
        except json.JSONDecodeError:
            # Fall back to basic parsing
            lines = validation_result.split("\n")
            result = {
                "valid": any("valid" in line.lower() for line in lines),
                "issues": [line for line in lines if "issue" in line.lower()],
                "warnings": [line for line in lines if "warning" in line.lower()],
                "suggestions": [line for line in lines if "suggest" in line.lower()]
            }
            return result
    
    def _update_usage_stats(self, response) -> None:
        """Update usage statistics from LLM interactions."""
        try:
            # Try different ways to get token usage
            usage = None
            
            # Try getting from generation_info
            if hasattr(response, 'generation_info'):
                usage = response.generation_info.get('token_usage', {})
            
            # Try getting from response.usage directly
            elif hasattr(response, 'usage'):
                usage = response.usage
            
            # Try getting from AIMessage additional_kwargs
            elif hasattr(response, 'additional_kwargs'):
                usage = response.additional_kwargs.get('token_usage', {})
            
            if usage:
                # Update token counts
                self.usage_stats.prompt_tokens += int(usage.get('prompt_tokens', 0))
                self.usage_stats.completion_tokens += int(usage.get('completion_tokens', 0))
                self.usage_stats.total_tokens = (
                    self.usage_stats.prompt_tokens + self.usage_stats.completion_tokens
                )
                
                # Calculate approximate cost
                prompt_rate = 0.0015 if "gpt-4" in self.model_name else 0.0005
                completion_rate = 0.002 if "gpt-4" in self.model_name else 0.0005
                
                self.usage_stats.cost = (
                    self.usage_stats.prompt_tokens * prompt_rate / 1000 +
                    self.usage_stats.completion_tokens * completion_rate / 1000
                )
                
                logger.info(f"Updated usage stats - Prompt: {self.usage_stats.prompt_tokens}, "
                          f"Completion: {self.usage_stats.completion_tokens}, "
                          f"Total: {self.usage_stats.total_tokens}")
        except Exception as e:
            logger.error(f"Error updating usage stats: {str(e)}")
    
    def initialize_vector_store(self, sql_files: List[str]) -> None:
        """Initialize the vector store with SQL examples.
        
        Args:
            sql_files: List of SQL file paths
        """
        texts = []
        metadatas = []
        
        for file_path in sql_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Split SQL into meaningful chunks
                    sql_statements = [s.strip() for s in content.split(';') if s.strip()]
                    
                    for stmt in sql_statements:
                        # Only store non-trivial SQL statements
                        if len(stmt.split()) > 5:  # More than 5 words
                            texts.append(stmt)
                            metadatas.append({
                                "source": file_path,
                                "content": stmt,
                                "type": "sql_statement"
                            })
                            
                            # Also store the statement with its surrounding context
                            start_idx = content.find(stmt)
                            if start_idx >= 0:
                                context_start = max(0, start_idx - 200)
                                context_end = min(len(content), start_idx + len(stmt) + 200)
                                context = content[context_start:context_end]
                                
                                texts.append(context)
                                metadatas.append({
                                    "source": file_path,
                                    "content": context,
                                    "type": "sql_context"
                                })
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        if texts:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            logger.info(f"Initialized vector store with {len(texts)} chunks")
    
    def extract_metadata(self, sql_content: str) -> Dict[str, Any]:
        """Extract metadata from MS SQL content including tables and their relationships.
        
        Args:
            sql_content: The SQL content to analyze
            
        Returns:
            Dictionary containing tables and their relationships
        """
        try:
            metadata = {
                "permanent_tables": [],
                "temp_tables": {},
                "table_variables": {},
                "ctes": {}
            }

            # Extract permanent tables from SELECT/JOIN/etc
            permanent_pattern = r'(?:FROM|JOIN)\s+(\[?[\w\.\[\]]+\]?)(?:\s|$)'
            for match in re.finditer(permanent_pattern, sql_content, re.IGNORECASE):
                table_name = match.group(1).strip('[]')
                if table_name and not table_name.startswith(('#', '@')):
                    metadata["permanent_tables"].append(table_name)

            # Extract temp tables and their source tables
            temp_pattern = r'CREATE\s+TABLE\s+(#[\w\.\[\]]+)\s+(?:AS\s+)?(.*?)(;|\s*CREATE|\s*$)'
            for match in re.finditer(temp_pattern, sql_content, re.IGNORECASE | re.DOTALL):
                temp_name = match.group(1)
                definition = match.group(2)
                source_tables = re.findall(permanent_pattern, definition, re.IGNORECASE)
                metadata["temp_tables"][temp_name] = {
                    "definition": definition.strip(),
                    "source_tables": [t.strip('[]') for t in source_tables if not t.startswith(('#', '@'))]
                }

            # Extract table variables and their source tables
            var_pattern = r'DECLARE\s+(@[\w]+)\s+TABLE\s*\((.*?)\)'
            for match in re.finditer(var_pattern, sql_content, re.IGNORECASE | re.DOTALL):
                var_name = match.group(1)
                definition = match.group(2)
                metadata["table_variables"][var_name] = {
                    "definition": definition.strip()
                }

            # Extract CTEs and their source tables
            cte_pattern = r'WITH\s+(\[?[\w]+\]?)\s+AS\s*\((.*?)\)\s*(?:,|SELECT|INSERT|UPDATE|DELETE)'
            for match in re.finditer(cte_pattern, sql_content, re.IGNORECASE | re.DOTALL):
                cte_name = match.group(1).strip('[]')
                definition = match.group(2)
                source_tables = re.findall(permanent_pattern, definition, re.IGNORECASE)
                metadata["ctes"][cte_name] = {
                    "definition": definition.strip(),
                    "source_tables": [t.strip('[]') for t in source_tables if not t.startswith(('#', '@'))]
                }

            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}", exc_info=True)
            return []
    
    def extract_metadata_from_sql_files(self, files: List[str]) -> Dict[str, Any]:
        """Extract metadata from a list of SQL files.
        
        Args:
            files: List of SQL file paths to analyze
            
        Returns:
            Dictionary containing extracted metadata
        """
        if not self.metadata:
            self.metadata = {
                "permanent_tables": [],
                "temp_tables": {},
                "table_variables": {},
                "ctes": {},
                "views": [],
                "procedures": [],
                "columns": {},
                "schemas": {},
                "relationships": []
            }
            
            for file in files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        sql_content = f.read()
                        file_metadata = self.extract_metadata(sql_content)
                        
                        # Merge permanent tables
                        self.metadata["permanent_tables"].extend(file_metadata["permanent_tables"])
                        
                        # Merge temp tables with their source tables
                        for temp_name, temp_info in file_metadata["temp_tables"].items():
                            if temp_name not in self.metadata["temp_tables"]:
                                self.metadata["temp_tables"][temp_name] = temp_info
                            else:
                                # If temp table already exists, merge source tables
                                existing_sources = set(self.metadata["temp_tables"][temp_name]["source_tables"])
                                new_sources = set(temp_info["source_tables"])
                                self.metadata["temp_tables"][temp_name]["source_tables"] = list(existing_sources | new_sources)
                        
                        # Merge table variables
                        self.metadata["table_variables"].update(file_metadata["table_variables"])
                        
                        # Merge CTEs
                        self.metadata["ctes"].update(file_metadata["ctes"])
                        
                        # Store file-specific metadata
                        self.metadata[file] = file_metadata
                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}", exc_info=True)
                    
            # Remove duplicates from tables list
            self.metadata["tables"] = list(set(self.metadata["tables"]))
            
        return self.metadata
