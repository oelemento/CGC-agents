# crew_main.py (Minimal Hierarchical Test: Manager + Reader)

import os
import traceback
import json
import re
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
#from crewai.tools.agent_tools import AskQuestionTool
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool

import PyPDF2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import sys
import time

# --- Configuration Flag ---
SKIP_FACT_EXTRACTION = True # Keep True for this test
# ---

# Load environment variables
load_dotenv()

# --- Feedback Tracker (Not needed for this test) ---

# --- Custom Tools ---
# Only KnowledgeReaderTool is strictly needed for this test
# PDFExtractionTool and KnowledgeAppendTool definitions can remain but won't be used.

class KnowledgeReaderTool(BaseTool):
    name: str = "Knowledge Reader Tool"
    description: str = ("Reads and returns the entire content from knowledge_base.txt.")
    input_file: str = "knowledge_base.txt"
    def _run(self) -> str:
        print(f"\n--- TOOL: KnowledgeReaderTool on: {self.input_file} ---")
        try:
            if not os.path.exists(self.input_file): return f"ERROR: KB file not found: {self.input_file}."
            if os.path.getsize(self.input_file) == 0: return "KB file is empty."
            with open(self.input_file, "r", encoding='utf-8') as f: content = f.read()
            print(f"--- TOOL: Read ~{len(content)} chars from KB. ---")
            max_kb_chars = 40000
            if len(content) > max_kb_chars: print(f"--- WARNING: KB content truncated to {max_kb_chars}. ---"); return content[:max_kb_chars]
            return content
        except Exception as e: print(f"--- ERROR reading KB: {e} ---"); return f"Error reading KB: {e}"

# --- Instantiate Tools ---
kb_reader_tool = KnowledgeReaderTool()

# --- Configure LLM ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2) # Use capable LLM

# --- Define WORKER Agent ---
knowledge_reader = Agent(
    role='Knowledge_Reader', # Clear role name
    goal=f"Read the entire content of the file '{kb_reader_tool.input_file}' using the KnowledgeReaderTool and return the full content.",
    backstory="A specialist agent focused solely on retrieving text content from the knowledge base file.",
    tools=[kb_reader_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# --- Define MANAGER Agent ---
ask_tool = AskQuestionTool(
    description="Delegate a task to another agent",
    agents=[knowledge_reader]  # Pass the list of agents this tool can delegate to
)
research_manager = Agent(
    role='Research_Manager',
    goal="Coordinate specialist agents to achieve a specific task.",
    backstory="""You are the central coordinator. Your only available specialist agent is `Knowledge_Reader`.
    You MUST achieve the goal by delegating the specific task to the `Knowledge_Reader` agent by its role name.
    You do not perform tasks yourself.""",
    tools=[ask_tool],  # Add the ask tool here
    llm=llm,
    verbose=True
)

# --- Define Task for the Manager ---
# **** MINIMAL TASK FOR TESTING DELEGATION ****
manage_kb_reading = Task(
  description="""Your goal is to get the content of the knowledge base file ('knowledge_base.txt').
  To do this, you MUST delegate the task to the `Knowledge_Reader` agent using the AskQuestionTool. 
  Delegate with a clear instruction like: "Read the knowledge base file using the KnowledgeReaderTool".
  Return the exact content provided by the `Knowledge_Reader` agent as your final output.""",
  expected_output="The full text content of the knowledge base file.",
  agent=research_manager
)

# --- Define Minimal Hierarchical Crew ---
minimal_crew = Crew(
    agents=[ research_manager, knowledge_reader ], # Manager first, then the worker
    tasks=[manage_kb_reading], # Only manager's task
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True # Keep verbose to see delegation attempts
)

# --- Helper Functions (Not needed for this minimal test) ---
# def parse_json_from_llm_output(text): ...
# def collect_direct_feedback(): ...

# --- Main Workflow Logic ---
def main():
    global minimal_crew

    kb_file = kb_reader_tool.input_file
    print(f"Attempting to use KB: {kb_file}")
    if not os.path.exists(kb_file) or os.path.getsize(kb_file) == 0:
        print(f"FATAL ERROR: Knowledge base file '{kb_file}' is missing or empty! Please ensure it exists and has content.")
        sys.exit(1)

    print("\n--- Starting Minimal Hierarchical Crew (Manager + Reader) ---")
    # No specific inputs needed for this simple task description
    initial_inputs = {} # Empty dictionary

    final_result = None
    try:
        # KICK OFF THE HIERARCHICAL CREW
        result_obj = minimal_crew.kickoff(inputs=initial_inputs)
        final_result = getattr(result_obj, 'raw', str(result_obj)) # Manager's final output

        print("\n--- Minimal Crew Execution Finished ---")

    except Exception as e:
        print(f"\n--- ERROR during Minimal Hierarchical Crew execution: {e} ---")
        traceback.print_exc()

    # --- Final Output ---
    print("\n--- Workflow Complete ---")
    print("========================================")
    print("   Final Output from Research Manager   ")
    print("========================================")
    if final_result:
        # Check if the output actually contains the expected KB content markers
        if "KimGGO2025.pdf" in final_result and "YoffeGGO2025.pdf" in final_result:
             print("SUCCESS: Manager appears to have returned KB content.")
             # Optionally print a snippet
             print(final_result[:500] + "...")
        else:
             print("WARNING: Final output does not appear to be the KB content.")
             print(final_result)
    else:
        print("Workflow did not complete successfully or produced no final output.")
    print("========================================")

# Remove print_final_output if not used

if __name__ == "__main__":
    main()