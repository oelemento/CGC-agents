# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run main script: `python crew_main.py`
- Run mini script: `python crew_main_mini.py`
- Environment setup: `pip install -r requirements.txt` (requirements.txt should be created if missing)

## Code Style Guidelines
- Imports: Organized by standard library, third-party, local imports with line breaks between groups
- Formatting: Use 4 spaces for indentation, 100 character line length
- Types: Use type hints when defining functions and methods
- Naming: 
  - Classes: CamelCase (e.g., `KnowledgeReaderTool`)
  - Functions/variables: snake_case (e.g., `parse_json_from_llm_output`)
  - Constants: UPPER_CASE (e.g., `SKIP_FACT_EXTRACTION`)
- Error handling: Use try/except blocks with specific exception types when possible
- Python version: Compatible with Python 3.8+
- Comments: Use descriptive docstrings and inline comments for complex logic
- Structure: Maintain agent/task separation pattern used in existing code