import click
from typing import List
import json
import os
from .agent import SQLAgent

@click.group()
def cli():
    """SQL Agent CLI interface."""
    pass

@cli.command()
@click.option('--db-path', type=str, default=":memory:",
              help='Path to SQLite database (default: in-memory)')
@click.option('--api-key', '-k', envvar='OPENAI_API_KEY',
              help='OpenAI API key (can also be set via OPENAI_API_KEY env var)')
def interactive(db_path: str, api_key: str):
    """Start interactive SQL query session."""
    if not api_key:
        click.echo("Error: OpenAI API key not provided")
        return

    agent = SQLAgent(db_path=db_path)
    agent.setup_database()  # Initialize test database
    
    click.echo("SQL Agent Interactive Mode")
    click.echo("Enter your questions in natural language (or 'quit' to exit)")
    
    while True:
        question = click.prompt("\nYour question")
        
        if question.lower() == 'quit':
            break
            
        try:
            query = agent.generate_query(question)
            if query:
                click.echo("\nGenerated SQL:")
                click.echo(query)
                
                results = agent.execute_query(query)
                click.echo("\nResults:")
                click.echo(json.dumps(results, indent=2))
                
                # Create visualization if applicable
                fig = agent.visualize_results(results)
                if fig:
                    fig.show()
            else:
                click.echo("\nCould not generate query for this input")
                
        except Exception as e:
            click.echo(f"\nError: {str(e)}")

if __name__ == '__main__':
    cli()
