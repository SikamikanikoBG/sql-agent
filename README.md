# SQL Agent

An AI-powered SQL query generation and analysis tool that helps you write MS SQL Server queries using natural language.

## Features

- 🤖 Natural language to SQL query conversion
- 📊 Multi-database support with proper schema handling
- 🔍 Context-aware query generation using similar examples
- 📝 Detailed query validation and optimization
- 📈 Token usage and cost tracking
- 🎯 Vector similarity search for relevant examples
- 🧠 Advanced MS SQL Server features support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arsensokolov/sql-agent.git
cd sql-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_key_here`

## Usage

1. Add your SQL files to the `sql_agent/data` directory

2. Run the Streamlit app:
```bash
streamlit run sql_agent/streamlit_app.py
```

3. Enter your natural language query and get the generated SQL

## Example

Input:
```
Show me all transactions of our employees
```

Output:
```sql
SELECT 
    e.first_name,
    e.last_name,
    s.amount,
    s.created_date
FROM employees e
LEFT JOIN sales s ON e.id = s.customer_id
ORDER BY s.created_date DESC;
```

## Architecture

- 🔄 Three-stage processing:
  1. Intent Analysis
  2. Query Generation
  3. Query Validation
- 📚 Vector store for similar example retrieval
- 🎯 MS SQL Server specific optimizations
- 📊 Comprehensive metadata extraction

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black sql_agent/
```

Lint code:
```bash
flake8 sql_agent/
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
