import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import streamlit as st
from pathlib import Path
import json
import tiktoken
import os
import hashlib
from datetime import datetime
from sql_agent.utils.decorators import prevent_rerun

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
        similarity_threshold: float = 0.01,  # Very low threshold to catch more potential matches
        max_examples: int = 25,  # Significantly more examples for richer context
        max_tokens: int = 14000,  # Safe limit below model's context length
        index_path: str = "data/faiss_index",
        metadata_path: str = "data/index_metadata.json"
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
        self.max_tokens = max_tokens
        
        # Add index paths and metadata tracking
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_metadata = {
            "last_updated": None,
            "files_hash": None,
            "indexed_files": [],
            "total_chunks": 0
        }
        
        self.metadata = None
        self.vector_store = None
        self.usage_stats = UsageStats()
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
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
            chunk_size=8000,  # Much larger chunks to maintain full query context
            chunk_overlap=2000,  # Larger overlap to ensure we don't miss context
            separators=["\nGO\n", ";\n\n", ";\n", ";", "\nBEGIN\n", "\nEND\n", "\n\n"],  # Prioritize keeping statements together
            length_function=len
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
            template="""Analyze the user's query intent using ONLY the tables and columns found in the similar examples below:

{similar_examples}

User Query:
{query}

Important Rules:
1. You MUST ONLY use tables and columns that appear in the example queries above
2. Do not invent or assume any tables or columns that are not shown in the examples
3. If required tables/columns are not found in examples, state this explicitly

Please identify:
1. Required tables (ONLY from examples above) and their relationships
2. Available columns from these tables that match the requirements
3. Filters or conditions using only existing columns
4. Sorting or grouping requirements using only existing columns

If any part of the query cannot be satisfied with the available tables/columns, explain what is missing.

Intent Analysis:"""
        )
        self.intent_chain = self.intent_prompt | self.llm
        
        # Query generation chain for MS SQL
        self.query_prompt = PromptTemplate(
            input_variables=["intent", "metadata", "similar_examples"],
            template="""Generate a complete MS SQL query solution based on the analyzed intent and similar examples.
Pay special attention to temporary tables and their dependencies.

{similar_examples}

Analyzed Intent:
{intent}

Follow these steps to generate the query:

1. Analyze Required Temporary Tables:
   - Identify ALL temporary tables needed from the examples
   - Find their complete creation logic in the example files
   - Understand dependencies between temporary tables
   - Copy the EXACT creation sequence from examples

2. Pattern Matching:
   - Find the most similar example query patterns
   - Note how temporary tables are created and used
   - Understand the complete workflow from table creation to final query

3. Solution Construction:
   - Start with ALL necessary temporary table creation statements
   - Copy the EXACT creation logic from examples
   - Maintain the correct order of temporary table creation
   - Include ALL required INSERT/SELECT INTO statements
   - Finally add the main query that uses these temp tables

4. Table and Column Selection:
   - Use ONLY tables and columns from examples
   - Include ALL necessary temporary tables
   - Maintain exact column names and data types
   - Follow example patterns for calculations

5. Query Structure:
   - Replicate the complete query structure from examples
   - Include ALL temporary table creation steps
   - Maintain transaction patterns if present
   - Keep all NOLOCK hints and other optimization hints

6. Validation:
   - Verify ALL temporary tables are created in correct order
   - Check ALL dependencies are satisfied
   - Ensure the complete solution matches example patterns
   - Validate final query uses correct temp table columns

CRITICAL:
- Generate a COMPLETE solution including ALL temporary table creation
- Follow the EXACT sequence of operations from examples
- Include ALL necessary steps to make the query work
- If anything is missing, explain what's not available

Generated SQL Query:"""
        )
        self.query_chain = self.query_prompt | self.llm
        
        # Query validation chain for MS SQL
        self.validation_prompt = PromptTemplate(
            input_variables=["query", "metadata", "similar_examples"],
            template="""Review the following MS SQL query and provide validation feedback:

Similar Examples (Reference patterns):
{similar_examples}

SQL Query to Review:
{query}

Review Guidelines:
1. Table Usage:
   - Note which tables match example patterns exactly
   - Identify tables with similar patterns but different names
   - List any tables without clear precedent

2. Column Usage:
   - Note columns that match examples exactly
   - Identify columns with similar patterns/purposes
   - List any columns without clear precedent

3. Query Structure:
   - Compare JOIN patterns with examples
   - Check function usage against examples
   - Review WHERE/GROUP BY/ORDER BY patterns

4. Confidence Assessment:
   - High: Query follows example patterns closely
   - Medium: Query uses similar patterns with some variations
   - Low: Query deviates significantly from examples

Provide:
1. Confidence level (High/Medium/Low)
2. List of validated elements
3. List of potential risks or uncertainties
4. Suggestions for improvement

Note: Generate warnings for uncertainties but allow query execution unless critical issues found.

Review Results:"""
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
            

            # Check if we have any relevant examples
            max_similarity = max([score for score, _ in similar_examples]) if similar_examples else 0
            if not similar_examples or (max_similarity < self.similarity_threshold and len(similar_examples) < 3):
                return QueryResult(
                    generated_query="",
                    agent_interactions={},
                    similarity_search=similar_examples,
                    validation_result={},
                    relevant_files=[],
                    error="⚠️ No sufficiently similar SQL patterns found (threshold: {:.2f}). Showing all search results above for inspection.".format(self.similarity_threshold),
                    query_vector=query_vector,
                    metadata_vectors=metadata_vectors
                ), self.usage_stats
            
            # Format similar examples for display
            with st.spinner("📝 Formatting examples..."):
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
                    "metadata": self._format_metadata(minimal_metadata),
                    "similar_examples": context
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
            
    def _find_similar_examples(self, query: str) -> Tuple[List[Tuple[float, Dict]], List[float], List[List[float]]]:
        """Find similar SQL examples and their complete source files.
        
        Args:
            query: The user's query to find similar examples for
            
        Returns:
            Tuple containing:
            - List of (score, content) tuples with full file contents
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
            max_distance = max(score for doc, score in results) if results else 1
            normalized_results = []
            for doc, score in results:
                similarity = 1 - (score / max_distance)
                normalized_results.append((similarity, doc))
            
            # Get embeddings
            query_vector = self.embeddings.embed_query(query)
            metadata_vectors = [self.embeddings.embed_query(doc.page_content) for doc, _ in results]
            
            # Track processed files to avoid duplicates
            processed_files = set()
            similar_examples = []
            
            for similarity, doc in normalized_results:
                if similarity >= self.similarity_threshold:
                    try:
                        source_file = doc.metadata.get('source', 'Unknown')
                        
                        # Only process each source file once
                        if source_file not in processed_files and source_file != 'Unknown':
                            processed_files.add(source_file)
                            
                            # Read the complete source file
                            try:
                                with open(source_file, 'r', encoding='utf-8') as f:
                                    full_content = f.read()
                            except UnicodeDecodeError:
                                with open(source_file, 'r', encoding='latin1') as f:
                                    full_content = f.read()
                            
                            # Create example with both the matching part and full file
                            example = {
                                'matching_content': doc.page_content,
                                'full_content': full_content,
                                'source': source_file,
                                'matching_score': similarity
                            }
                            similar_examples.append((similarity, example))
                            logger.info(f"Added full file content from {source_file}")
                            
                    except Exception as e:
                        logger.error(f"Error processing document: {str(e)}")
            
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
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
        
    def _format_examples(self, examples: List[Tuple[float, Dict]]) -> str:
        """Format similar examples and their complete source files for prompt templates.
        
        Args:
            examples: List of (score, content) tuples with full file contents
            
        Returns:
            Formatted examples string truncated to fit token limit
        """
        if not examples:
            return "No similar examples found."
            
        sections = ["Complete SQL context from relevant files:"]
        total_tokens = self._count_tokens("\n".join(sections))
        
        for i, (score, content) in enumerate(examples, 1):
            if isinstance(content, dict):
                # Format the current example
                new_sections = [
                    f"\nFile {i}: {content.get('source', 'Unknown')}",
                    f"Most relevant section (Similarity: {score:.2f}):",
                    "```sql",
                    content.get('matching_content', ''),
                    "```",
                    "\nComplete file content:",
                    "```sql",
                    content.get('full_content', ''),
                    "```",
                    "-" * 80  # Separator
                ]
                
                # Check if adding this example would exceed token limit
                example_tokens = self._count_tokens("\n".join(new_sections))
                if total_tokens + example_tokens > self.max_tokens:
                    logger.warning(f"Truncating examples at {i} to stay within token limit")
                    break
                    
                sections.extend(new_sections)
                total_tokens += example_tokens
        
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
            # Get usage from LangChain AIMessage
            if hasattr(response, 'llm_output') and isinstance(response.llm_output, dict):
                usage = response.llm_output.get('token_usage', {})
            else:
                # Fallback to additional_kwargs for older LangChain versions
                usage = getattr(response, 'additional_kwargs', {}).get('token_usage', {})

            if usage:
                # Update token counts
                self.usage_stats.prompt_tokens += int(usage.get('prompt_tokens', 0))
                self.usage_stats.completion_tokens += int(usage.get('completion_tokens', 0))
                self.usage_stats.total_tokens = (
                    self.usage_stats.prompt_tokens + self.usage_stats.completion_tokens
                )

                # Calculate cost based on current OpenAI pricing
                model_pricing = {
                    "gpt-4": {"input": 0.03, "output": 0.06},
                    "gpt-4-32k": {"input": 0.06, "output": 0.12},
                    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
                }

                # Get pricing for the current model
                pricing = model_pricing.get(self.model_name, model_pricing["gpt-3.5-turbo"])

                # Calculate cost per 1000 tokens
                prompt_cost = (self.usage_stats.prompt_tokens * pricing["input"]) / 1000
                completion_cost = (self.usage_stats.completion_tokens * pricing["output"]) / 1000
                self.usage_stats.cost += prompt_cost + completion_cost

                logger.info(f"Updated usage stats - Prompt: {self.usage_stats.prompt_tokens}, "
                          f"Completion: {self.usage_stats.completion_tokens}, "
                          f"Total: {self.usage_stats.total_tokens}, "
                          f"Cost: ${self.usage_stats.cost:.4f}")
        except Exception as e:
            logger.error(f"Error updating usage stats: {str(e)}")
    
    def _get_files_hash(self, sql_files: List[str]) -> str:
        """Generate a hash of the SQL files content and metadata to detect changes"""
        content = []
        for file in sorted(sql_files):
            try:
                file_path = Path(file)
                if not file_path.exists():
                    continue
                    
                # Include file metadata in hash
                stats = file_path.stat()
                file_meta = f"{file}:{stats.st_mtime}:{stats.st_size}"
                content.append(file_meta)
                
            except Exception as e:
                logger.warning(f"Error reading file {file}: {e}")
                continue
                
        return hashlib.md5("|".join(content).encode()).hexdigest()

    def _save_index_metadata(self) -> None:
        """Save index metadata to disk"""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.index_metadata, f, default=str)
            logger.info(f"Saved index metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving index metadata: {e}")

    def _load_index_metadata(self) -> bool:
        """Load index metadata from disk"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.index_metadata = json.load(f)
                logger.info(f"Loaded index metadata from {self.metadata_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading index metadata: {e}")
            return False

    @prevent_rerun(timeout=60)
    def initialize_vector_store(self, sql_files: List[str]) -> None:
        """Initialize or load the vector store with SQL examples."""
        try:
            # Create directories if they don't exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            current_hash = self._get_files_hash(sql_files)
            
            # Try to load existing index
            if self.index_path.exists() and self._load_index_metadata():
                if current_hash == self.index_metadata["files_hash"]:
                    try:
                        self.vector_store = FAISS.load_local(str(self.index_path), self.embeddings)
                        logger.info(f"Loaded existing index with {self.index_metadata['total_chunks']} chunks")
                        return
                    except Exception as e:
                        logger.error(f"Error loading existing index: {e}")
                else:
                    logger.info("Files changed, rebuilding index...")

            # If we reach here, we need to create new index
            texts = []
            metadatas = []
            
            for file_path in sql_files:
                try:
                    # Process file content as before
                    content = None
                    for encoding in ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        logger.warning(f"Could not read file {file_path} with any supported encoding")
                        continue

                    # Process SQL content chunks as before
                    statements = re.split(r'(?i)(CREATE|ALTER|DROP|SELECT|INSERT|UPDATE|DELETE|MERGE)\s+', content)
                    
                    for i in range(1, len(statements), 2):
                        if i+1 < len(statements):
                            operation_type = statements[i].strip().upper()
                            full_stmt = operation_type + " " + statements[i+1]
                            
                            if len(full_stmt.split()) > 10:
                                # Get context as before
                                start_idx = max(0, i-2)
                                context_before = ' '.join(statements[start_idx:i]).strip()
                                
                                end_idx = min(len(statements), i+3)
                                context_after = ' '.join(statements[i+2:end_idx]).strip()
                                
                                full_context = f"{context_before}\n{full_stmt}\n{context_after}".strip()
                                cleaned_stmt = ' '.join(full_context.split())
                                
                                texts.append(cleaned_stmt)
                                metadatas.append({
                                    "source": file_path,
                                    "content": cleaned_stmt,
                                    "type": "sql_statement",
                                    "operation": operation_type
                                })

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

            if not texts:
                raise ValueError("No valid SQL content found to index")

            # Create new index
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Update metadata
            self.index_metadata.update({
                "last_updated": datetime.now().isoformat(),
                "files_hash": current_hash,
                "indexed_files": sql_files,
                "total_chunks": len(texts)
            })
            
            # Save index and metadata
            self.vector_store.save_local(str(self.index_path))
            self._save_index_metadata()
            
            logger.info(f"Created new index with {len(texts)} chunks from {len(sql_files)} files")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}", exc_info=True)
            raise
    
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
                    # Try different encodings
                    encodings = ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']
                    sql_content = None
            
                    for encoding in encodings:
                        try:
                            with open(file, 'r', encoding=encoding) as f:
                                sql_content = f.read()
                                logger.debug(f"Successfully read file with {encoding} encoding")
                                break
                        except UnicodeDecodeError:
                            logger.debug(f"Failed to read with {encoding} encoding")
                            continue
            
                    if sql_content is None:
                        raise ValueError(f"Could not read file {file} with any supported encoding")
                
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
    def update_vector_store(self, new_sql_files: List[str]) -> None:
        """Add new SQL files to existing index"""
        try:
            if not self.vector_store:
                logger.info("No existing index, creating new one...")
                self.initialize_vector_store(new_sql_files)
                return

            # Filter out already indexed files
            new_files = [f for f in new_sql_files if f not in self.index_metadata["indexed_files"]]
            if not new_files:
                logger.info("No new files to index")
                return

            texts = []
            metadatas = []
            
            # Process new files using the same logic as initialize_vector_store
            for file_path in new_files:
                try:
                    content = None
                    for encoding in ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        logger.warning(f"Could not read file {file_path} with any supported encoding")
                        continue

                    statements = re.split(r'(?i)(CREATE|ALTER|DROP|SELECT|INSERT|UPDATE|DELETE|MERGE)\s+', content)
                    
                    for i in range(1, len(statements), 2):
                        if i+1 < len(statements):
                            operation_type = statements[i].strip().upper()
                            full_stmt = operation_type + " " + statements[i+1]
                            
                            if len(full_stmt.split()) > 10:
                                start_idx = max(0, i-2)
                                context_before = ' '.join(statements[start_idx:i]).strip()
                                
                                end_idx = min(len(statements), i+3)
                                context_after = ' '.join(statements[i+2:end_idx]).strip()
                                
                                full_context = f"{context_before}\n{full_stmt}\n{context_after}".strip()
                                cleaned_stmt = ' '.join(full_context.split())
                                
                                texts.append(cleaned_stmt)
                                metadatas.append({
                                    "source": file_path,
                                    "content": cleaned_stmt,
                                    "type": "sql_statement",
                                    "operation": operation_type
                                })

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

            if texts:
                # Add new texts to existing index
                self.vector_store.add_texts(texts, metadatas=metadatas)
                
                # Update metadata
                self.index_metadata["indexed_files"].extend(new_files)
                self.index_metadata["total_chunks"] += len(texts)
                self.index_metadata["last_updated"] = datetime.now().isoformat()
                self.index_metadata["files_hash"] = self._get_files_hash(
                    self.index_metadata["indexed_files"]
                )
                
                # Save updated index and metadata
                self.vector_store.save_local(str(self.index_path))
                self._save_index_metadata()
                
                logger.info(f"Added {len(texts)} new chunks from {len(new_files)} files to index")
            
        except Exception as e:
            logger.error(f"Error updating vector store: {e}", exc_info=True)
            raise
