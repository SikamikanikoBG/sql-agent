import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
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
    error: Optional[str] = None

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
        similarity_threshold: float = 0.7,
        max_examples: int = 3
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
        intent_prompt = PromptTemplate(
            input_variables=["query", "metadata", "similar_examples"],
            template="""Based on the database metadata and similar examples below, analyze the user's query intent:

Database Metadata:
{metadata}

Similar Examples:
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
        self.intent_chain = intent_prompt | self.llm
        
        # Query generation chain
        query_prompt = PromptTemplate(
            input_variables=["intent", "metadata", "similar_examples"],
            template="""Generate a SQL query based on the analyzed intent and database metadata:

Database Metadata:
{metadata}

Similar Examples:
{similar_examples}

Analyzed Intent:
{intent}

Consider:
1. Use appropriate JOIN types based on requirements
2. Include necessary table aliases
3. Add proper WHERE clauses for filtering
4. Include GROUP BY/ORDER BY if needed
5. Follow SQL best practices

Generated SQL Query:"""
        )
        self.query_chain = query_prompt | self.llm
        
        # Query validation chain
        validation_prompt = PromptTemplate(
            input_variables=["query", "metadata"],
            template="""Validate the following SQL query against the database metadata:

Database Metadata:
{metadata}

SQL Query:
{query}

Check for:
1. Table existence and relationships
2. Column validity
3. Syntax correctness
4. Potential performance issues
5. SQL injection risks

Validation Results:"""
        )
        self.validation_chain = validation_prompt | self.llm

    async def process_query(self, query: str, metadata: Dict) -> Tuple[QueryResult, UsageStats]:
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
            
            # Find similar examples
            similar_examples = await self._find_similar_examples(query)
            
            # Parse intent
            intent_result = await self.intent_chain.ainvoke({
                "query": query,
                "metadata": self._format_metadata(metadata),
                "similar_examples": self._format_examples(similar_examples)
            })
            
            # Generate query
            query_result = await self.query_chain.ainvoke({
                "intent": intent_result.content,
                "metadata": self._format_metadata(metadata),
                "similar_examples": self._format_examples(similar_examples)
            })
            
            # Validate generated query
            validation_result = await self.validation_chain.ainvoke({
                "query": query_result.content,
                "metadata": self._format_metadata(metadata)
            })
            
            # Update usage stats
            self._update_usage_stats()
            
            # Prepare result
            result = QueryResult(
                generated_query=query_result,
                agent_interactions={
                    "parse_intent": {
                        "system_prompt": self.intent_chain.prompt.template,
                        "user_prompt": query,
                        "result": intent_result
                    },
                    "generate_query": {
                        "system_prompt": self.query_chain.prompt.template,
                        "user_prompt": intent_result,
                        "result": query_result
                    },
                    "validate_query": {
                        "system_prompt": self.validation_chain.prompt.template,
                        "user_prompt": query_result,
                        "result": validation_result
                    }
                },
                similarity_search=similar_examples,
                validation_result=self._parse_validation_result(validation_result)
            )
            
            return result, self.usage_stats
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return QueryResult(
                generated_query=f"ERROR: {str(e)}",
                agent_interactions={},
                similarity_search=[],
                validation_result={},
                error=str(e)
            ), self.usage_stats
            
    async def _find_similar_examples(self, query: str) -> List[Tuple[float, str]]:
        """Find similar SQL examples from the vector store.
        
        Args:
            query: The user's query to find similar examples for
            
        Returns:
            List of (score, content) tuples
        """
        if not self.vector_store:
            return []
            
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.max_examples
        )
        
        return [(score, content) for score, content in results 
                if score >= self.similarity_threshold]
    
    def _format_metadata(self, metadata: Dict) -> str:
        """Format metadata for prompt templates.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Formatted metadata string
        """
        sections = []
        
        if metadata.get("tables"):
            sections.append("Tables:")
            for table in metadata["tables"]:
                sections.append(f"- {table}")
                if isinstance(table, str) and metadata.get("schemas", {}).get(table):
                    schema = metadata["schemas"][table]
                    if isinstance(schema, list):
                        for column in schema:
                            sections.append(f"  * {column['name']}: {column['type']}")
        
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
            
        sections = []
        for i, (score, content) in enumerate(examples, 1):
            sections.extend([
                f"Example {i} (Similarity: {score:.2f}):",
                content,
                ""
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
    
    def _update_usage_stats(self) -> None:
        """Update usage statistics from LLM interactions."""
        # Get usage from ChatOpenAI if available
        if hasattr(self.llm, "last_token_usage"):
            usage = self.llm.last_token_usage
            self.usage_stats.prompt_tokens += usage.get("prompt_tokens", 0)
            self.usage_stats.completion_tokens += usage.get("completion_tokens", 0)
            self.usage_stats.total_tokens = (
                self.usage_stats.prompt_tokens + self.usage_stats.completion_tokens
            )
            
            # Calculate approximate cost
            # Adjust rates based on your model
            prompt_rate = 0.0015 if "gpt-4" in self.model_name else 0.0005
            completion_rate = 0.002 if "gpt-4" in self.model_name else 0.0005
            
            self.usage_stats.cost = (
                self.usage_stats.prompt_tokens * prompt_rate / 1000 +
                self.usage_stats.completion_tokens * completion_rate / 1000
            )
    
    async def initialize_vector_store(self, sql_files: List[str]) -> None:
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
                    chunks = self.text_splitter.split_text(content)
                    
                    texts.extend(chunks)
                    metadatas.extend([{
                        "source": file_path,
                        "chunk_size": len(chunk)
                    } for chunk in chunks])
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        if texts:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            logger.info(f"Initialized vector store with {len(texts)} chunks")
    
    def extract_metadata(self, sql_content: str) -> List[str]:
        """Extract metadata from SQL content including tables.
        
        Args:
            sql_content: The SQL content to analyze
            
        Returns:
            List of table names found in the content
        """
        try:
            tables = []
            pattern = re.compile(r'CREATE\s+TABLE\s+([^\s(]+)', re.IGNORECASE)
            
            for match in pattern.finditer(sql_content):
                table_name = match.group(1).strip()
                if table_name:
                    tables.append(table_name)
            
            return tables
            
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
                "tables": [],
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
                        tables = self.extract_metadata(sql_content)
                        if tables:
                            self.metadata["tables"].extend(tables)
                            # Store file-specific metadata
                            self.metadata[file] = {
                                "tables": tables,
                                "content": sql_content
                            }
                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}", exc_info=True)
                    
            # Remove duplicates from tables list
            self.metadata["tables"] = list(set(self.metadata["tables"]))
            
        return self.metadata
