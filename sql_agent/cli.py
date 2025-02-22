import click
from typing import List
import json
from .metadata_extractor import extract_metadata_from_sql_files
from .langgraph_orchestrator import SQLAgentOrchestrator

@click.group()
def cli():
    """SQL Agent CLI interface."""
    pass

@cli.command()
@click.argument('sql_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='metadata.json',
              help='Output file for extracted metadata')
def extract_metadata(sql_files: List[str], output: str):
    """Extract metadata from SQL files."""
    metadata = extract_metadata_from_sql_files(sql_files)
    
    with open(output, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    click.echo(f"Metadata extracted and saved to {output}")

@cli.command()
@click.option('--metadata', '-m', type=click.Path(exists=True), required=True,
              help='Path to metadata JSON file')
@click.option('--api-key', '-k', required=True, help='OpenAI API key')
def interactive(metadata: str, api_key: str):
    """Start interactive SQL query session."""
    with open(metadata, 'r') as f:
        metadata_content = json.load(f)
    
    orchestrator = SQLAgentOrchestrator(api_key)
    
    click.echo("SQL Agent Interactive Mode")
    click.echo("Enter your questions in natural language (or 'quit' to exit)")
    
    while True:
        question = click.prompt("\nYour question")
        
        if question.lower() == 'quit':
            break
        
        result = orchestrator.process_query(question, metadata_content)
        
        if result["is_valid"]:
            click.echo("\nGenerated SQL:")
            click.echo(result["generated_query"])
        else:
            click.echo("\nError:")
            click.echo(result["error"])

if __name__ == '__main__':
    cli()
