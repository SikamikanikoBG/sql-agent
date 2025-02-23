from setuptools import setup, find_packages

setup(
    name="sql_agent",
    version="0.1.0",
    description="An AI-powered SQL query generation and analysis tool",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.29.0",
        "openai>=1.0.0",
        "langgraph>=0.0.1",
        "langchain>=0.1.0",
        "langchain-community>=0.0.1",
        "pyodbc>=4.0.39",
        "plotly>=5.13.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.0.0",
        "click>=8.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)