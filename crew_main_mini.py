# crew_main_hierarchical_asktool_full.py (Manager + All Workers - Using AskQuestionTool + Fixes)

import os
import traceback
import json
import re
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
# Use the correct import path for the tool
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool

import PyPDF2 # Keep for potential future use
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import sys
import time

# --- Configuration Flag ---
SKIP_FACT_EXTRACTION = True
# ---

# Load environment variables
load_dotenv()

# --- Custom Tools ---
class KnowledgeReaderTool(BaseTool):
    name: str = "Knowledge Reader Tool"
    description: str = ("Reads and returns the entire content from a specified knowledge base file.")
    # Tool takes file_path argument
    def _run(self, file_path: str) -> str:
        print(f"\n--- TOOL: KnowledgeReaderTool on: {file_path} ---")
        if not file_path or not isinstance(file_path, str): return f"ERROR: file_path argument missing or invalid."
        try:
            if not os.path.exists(file_path): return f"ERROR: KB file not found: {file_path}."
            if os.path.getsize(file_path) == 0: return f"KB file is empty: {file_path}."
            with open(file_path, "r", encoding='utf-8') as f: content = f.read()
            print(f"--- TOOL: Read ~{len(content)} chars from {file_path}. ---")
            max_kb_chars = 40000
            if len(content) > max_kb_chars: print(f"--- WARNING: Content truncated from {file_path}. ---"); return content[:max_kb_chars]
            return content
        except Exception as e: print(f"--- ERROR reading file {file_path}: {e} ---"); return f"Error reading file {file_path}: {e}"

# --- Instantiate Tools ---
kb_reader_tool = KnowledgeReaderTool()
# PDFExtractionTool and KnowledgeAppendTool would be defined here if needed for extraction

# --- Configure LLM ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# --- Define WORKER Agents ---
original_kb_reader = Agent(
    role='Original_KB_Reader',
    goal="Read 'knowledge_base.txt' using KnowledgeReaderTool and return content.",
    backstory="Specialist in retrieving data from 'knowledge_base.txt'.",
    tools=[kb_reader_tool], llm=llm, verbose=True, allow_delegation=False
)

extended_kb_reader = Agent(
    role='Extended_KB_Reader',
    goal="Read 'extended_knowledge_base.txt' using KnowledgeReaderTool and return content.",
    backstory="Specialist in retrieving data from 'extended_knowledge_base.txt'.",
    tools=[kb_reader_tool], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_generator = Agent(
    role='Hypothesis_Generator',
    goal="Generate up to 10 diverse, specific, testable, mechanistic hypotheses based *only* on the provided Original Knowledge Base Content context. Cite supporting facts/sources.",
    backstory="Generates creative hypotheses from primary KB data.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# **** MODIFIED GOAL (Fix 1) ****
novelty_scorer = Agent(
    role='Novelty_Scorer',
    goal="Assess novelty (score 1-5) of each hypothesis provided relative to the provided Extended Knowledge Base Content context. Output a valid JSON list, where each item is an object containing keys 'hypothesis', 'novelty_score', and 'novelty_reasoning'.",
    backstory="Evaluates originality against general knowledge context.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# **** MODIFIED GOAL (Fix 1) ****
feasibility_scorer = Agent(
    role='Feasibility_Scorer',
    goal="Assess experimental feasibility (score 1-5) of each hypothesis provided based on standard lab resources. Output a valid JSON list, where each item is an object containing keys 'hypothesis', 'feasibility_score', and 'feasibility_reasoning'.",
    backstory="Provides quick feasibility scores.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_ranker_selector = Agent(
    role='Hypothesis_Ranker_Selector',
    goal="Rank hypotheses based on provided novelty and feasibility scores (JSON lists). Balance both scores. Select TOP 3 hypotheses. Output *only* the text of the selected top 3.",
    backstory="Prioritizes promising research directions.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_presenter = Agent(
    role='Hypothesis_Presenter',
    goal="Clearly present the provided final list/text of hypotheses.",
    backstory="Formats final results for presentation.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# --- Define MANAGER Agent ---
# Define the AskQuestionTool with ALL worker agents it needs to contact
delegation_tool = AskQuestionTool(
    name="AskQuestionTool",
    description="Use this tool to ask a specific question or delegate a task instruction to a specific coworker agent.",
    agents=[ # List all potential workers
        original_kb_reader,
        extended_kb_reader,
        hypothesis_generator,
        novelty_scorer,
        feasibility_scorer,
        hypothesis_ranker_selector,
        hypothesis_presenter
        ]
)

research_manager = Agent(
    role='Research_Manager',
    goal="Coordinate specialist agents via AskQuestionTool to generate, score, rank, and select the top 3 hypotheses using dual knowledge bases.",
    backstory=f"""You are the central coordinator managing a hypothesis pipeline. Use the AskQuestionTool ONLY to delegate tasks sequentially to the correct specialist agent. Your available specialists are:
    - `Original_KB_Reader`: Reads 'knowledge_base.txt'.
    - `Extended_KB_Reader`: Reads 'extended_knowledge_base.txt'.
    - `Hypothesis_Generator`: Generates hypotheses from ORIGINAL KB content.
    - `Novelty_Scorer`: Scores novelty using EXTENDED KB content.
    - `Feasibility_Scorer`: Scores feasibility.
    - `Hypothesis_Ranker_Selector`: Ranks/Selects top 3 based on scores.
    - `Hypothesis_Presenter`: Presents final list.
    Follow the workflow precisely, passing necessary context in each delegation.""",
    tools=[delegation_tool], # Give manager ONLY this tool
    llm=llm,
    verbose=True
)

# --- Define Task for the Manager ---
manage_full_pipeline_asktool = Task(
  description=f"""Orchestrate the full pipeline to get the top 3 hypotheses, generated from 'knowledge_base.txt' and scored for novelty against 'extended_knowledge_base.txt'.

  **Workflow steps using AskQuestionTool:**
  1. Ask `Original_KB_Reader`: "Read 'knowledge_base.txt' using KnowledgeReaderTool". Get original_kb_content.
  2. Ask `Extended_KB_Reader`: "Read 'extended_knowledge_base.txt' using KnowledgeReaderTool". Get extended_kb_content.
  3. Ask `Hypothesis_Generator`: "Generate up to 10 hypotheses based *only* on the provided Original KB content". Pass original_kb_content as context. Get hypotheses_list.
  4. Ask `Novelty_Scorer`: "Score novelty of hypotheses based on Extended KB Content". Pass hypotheses_list AND extended_kb_content as context. Get novelty_scores (JSON list).
  5. Ask `Feasibility_Scorer`: "Score feasibility of hypotheses". Pass hypotheses_list as context. Get feasibility_scores (JSON list).
  6. Ask `Hypothesis_Ranker_Selector`: "Rank hypotheses and select TOP 3". Pass hypotheses_list, novelty_scores, and feasibility_scores as context. Get selected_hypotheses_text.
  7. Ask `Hypothesis_Presenter`: "Present this final list". Pass selected_hypotheses_text as context.
  8. Return the exact output from `Hypothesis_Presenter`.""",
  expected_output="The final formatted presentation of the top 3 selected hypotheses.",
  agent=research_manager
)

# --- Define Hierarchical Crew ---
research_crew = Crew(
    agents=[ # Manager first, then all workers
        research_manager,
        original_kb_reader,
        extended_kb_reader,
        hypothesis_generator,
        novelty_scorer,
        feasibility_scorer,
        hypothesis_ranker_selector,
        hypothesis_presenter
        ],
    tasks=[manage_full_pipeline_asktool], # Only manager's task
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True
)

# --- Helper Functions ---
def parse_json_from_llm_output(text):
    # Keep robust JSON parsing
    if not isinstance(text, str): return None
    patterns = [ r'```json\s*(\[.*?\]|\{.*?\})\s*```', r'```\s*(\[.*?\]|\{.*?\})\s*```', ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1);
            try: return json.loads(json_str)
            except json.JSONDecodeError as e: print(f"Warn: Failed JSON block parse: {e}")
    try:
        start_obj = text.find('{'); end_obj = text.rfind('}') + 1
        start_list = text.find('['); end_list = text.rfind(']') + 1
        json_str = None
        if 0 <= start_list < end_list : json_str = text[start_list:end_list]
        elif 0 <= start_obj < end_obj: json_str = text[start_obj:end_obj]
        if json_str:
             if json_str.count('{') >= json_str.count('}') and json_str.count('[') >= json_str.count(']'):
                  try: return json.loads(json_str)
                  except json.JSONDecodeError as e: print(f"Warn: Failed raw parse: {e}")
    except Exception as e: print(f"Warn: Error during raw search: {e}")
    print(f"ERROR: Failed to parse JSON/List: {text}")
    return None

# --- Main Workflow Logic ---
def main():
    global research_crew

    kb_file_original = 'knowledge_base.txt'
    kb_file_extended = 'extended_knowledge_base.txt'
    print(f"Attempting to use KBs: {kb_file_original}, {kb_file_extended}")

    # Basic checks for file existence
    if not os.path.exists(kb_file_original) or os.path.getsize(kb_file_original) == 0:
        print(f"FATAL ERROR: Original KB file '{kb_file_original}' missing or empty!")
        sys.exit(1)
    if not os.path.exists(kb_file_extended) or os.path.getsize(kb_file_extended) == 0:
        print(f"FATAL ERROR: Extended KB file '{kb_file_extended}' missing or empty!")
        sys.exit(1)

    print("\n--- Starting Hierarchical Crew (Full Pipeline - Dual KB / AskTool) ---")
    # **** FIX 2: Comprehensive initial_inputs to prevent KeyErrors ****
    initial_inputs = {
        'skip_extraction_flag': str(SKIP_FACT_EXTRACTION),
        'pdf_directory': '',
        'user_feedback': '', # Required by hypothesis_generator's original goal string
        'hypothesis': '', # Required by scorer goals potentially (dummy)
        'hypotheses_context': '',
        'human_feedback': '',
        'feedback_evaluation_context': '{}',
        'knowledge_base_content': '',
        'feasibility_context': '',
        # Add any other keys referenced ANYWHERE in ANY agent goal/task description
    }

    final_result = None
    try:
        result_obj = research_crew.kickoff(inputs=initial_inputs)
        final_result = getattr(result_obj, 'raw', str(result_obj))
        print("\n--- Crew Execution Finished ---")

    except Exception as e:
        print(f"\n--- ERROR during Crew execution: {e} ---")
        traceback.print_exc()

    # --- Final Output ---
    print("\n--- Workflow Complete ---")
    print("========================================")
    print("   Final Output from Research Manager   ")
    print("========================================")
    if final_result:
        if "hypothesis" in final_result.lower() and ("1." in final_result or "2." in final_result or "3." in final_result):
             print("SUCCESS: Manager appears to have returned the final presented hypotheses.")
             print(final_result)
        else:
             print("WARNING: Final output format unexpected.")
             print(final_result)
    else:
        print("Workflow did not complete successfully or produced no final output.")
    print("========================================")


if __name__ == "__main__":
    main()