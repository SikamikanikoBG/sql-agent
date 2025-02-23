# SQL Agent

An AI-powered SQL query generation and analysis tool that helps you write MS SQL Server queries using natural language. Features both Streamlit and Gradio interfaces for maximum flexibility.

## Features

- 🤖 Natural language to SQL query conversion
- 📊 Multi-database support with proper schema handling
- 🔍 Context-aware query generation using similar examples
- 📝 Detailed query validation and optimization
- 📈 Token usage and cost tracking
- 🎯 Vector similarity search for relevant examples
- 🧠 Advanced MS SQL Server features support
- 🎨 Choice of user interfaces:
  - Streamlit: Full-featured dashboard
  - Gradio: Simple, elegant interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SikamikanikoBG/sql-agent.git
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

2. Launch the Gradio interface:
   ```bash
   python sql_agent/gradio_app.py
   ```

3. Enter your natural language query and get the generated SQL

## Example

Input:
```
Show me all orders from last month with total amount greater than $1000
```

Output:
```sql
-- Generated SQL based on your database schema
SELECT 
    o.OrderID,
    o.OrderDate,
    o.CustomerID,
    SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalAmount
FROM 
    Orders o
    JOIN [Order Details] od ON o.OrderID = od.OrderID
WHERE 
    o.OrderDate >= DATEADD(month, -1, GETDATE())
    AND o.OrderDate < GETDATE()
GROUP BY 
    o.OrderID,
    o.OrderDate,
    o.CustomerID
HAVING 
    SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) > 1000
ORDER BY 
    o.OrderDate DESC;
```

## Architecture

- 🔄 Three-stage processing:
  1. Intent Analysis
  2. Query Generation
  3. Query Validation
- 📚 Vector store for similar example retrieval
- 🎯 MS SQL Server specific optimizations
- 📊 Comprehensive metadata extraction
- 🎨 Dual interface support:
  - Streamlit for rich dashboard experience
  - Gradio for simple, focused interaction

## Interface Features

The Gradio interface provides an intuitive way to interact with the SQL Agent:

![Main Interface](media/tab1.png)

### Generated SQL
View the generated SQL query with syntax highlighting and explanations:
![Generated SQL](media/tab2.png)

### Query Analysis
Get detailed analysis of how the query was constructed:
![Query Analysis](media/tab3.png)

### Similar Examples
See related SQL examples from your codebase:
![Similar Examples](media/tab4.png)

Key Features:
- 🚀 Quick query generation
- 🎯 Simple, intuitive design
- 📱 Mobile-friendly layout
- 🔄 Real-time query processing
- 📊 Usage statistics tracking
- 🔍 Detailed query analysis

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

## Support

- 📚 Documentation: Check the `docs/` directory
- 🐛 Issues: Submit via GitHub Issues
- 💬 Discussions: Use GitHub Discussions for questions
