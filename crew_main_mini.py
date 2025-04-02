# crew_main_hierarchical_full_w_reasoning.py (Manager + All Workers + AskTool + Reasoning)

import os
import traceback
import json
import re
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
# Correct import path for the tool
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
    # Tool now takes file_path argument
    def _run(self, file_path: str) -> str:
        print(f"\n--- TOOL: KnowledgeReaderTool on: {file_path} ---")
        if not file_path or not isinstance(file_path, str):
             return f"ERROR: file_path argument missing or invalid."
        try:
            if not os.path.exists(file_path): return f"ERROR: KB file not found: {file_path}."
            if os.path.getsize(file_path) == 0: return f"KB file is empty: {file_path}."
            with open(file_path, "r", encoding='utf-8') as f: content = f.read()
            print(f"--- TOOL: Read ~{len(content)} chars from {file_path}. ---")
            max_kb_chars = 40000 # Apply truncation if needed
            if len(content) > max_kb_chars: print(f"--- WARNING: Content truncated from {file_path}. ---"); return content[:max_kb_chars]
            return content
        except Exception as e: print(f"--- ERROR reading file {file_path}: {e} ---"); return f"Error reading file {file_path}: {e}"

# --- Instantiate Tools ---
kb_reader_tool = KnowledgeReaderTool()
# Assume PDFExtractionTool, KnowledgeAppendTool are defined if needed for extraction

# --- Configure LLM ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# --- Define WORKER Agents ---
original_kb_reader = Agent(
    role='Original_KB_Reader',
    goal="Read 'knowledge_base.txt' using KnowledgeReaderTool and return full content.",
    backstory="Specialist in retrieving data from 'knowledge_base.txt'.",
    tools=[kb_reader_tool], llm=llm, verbose=True, allow_delegation=False
)

extended_kb_reader = Agent(
    role='Extended_KB_Reader',
    goal="Read 'extended_knowledge_base.txt' using KnowledgeReaderTool and return full content.",
    backstory="Specialist in retrieving data from 'extended_knowledge_base.txt'.",
    tools=[kb_reader_tool], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_generator = Agent(
    role='Hypothesis_Generator',
    # **** MODIFY THIS GOAL ****
    goal="""Generate up to 10 diverse, specific, testable, mechanistic hypotheses based *only* on the provided Original Knowledge Base Content context. 
    **Prioritize hypotheses focusing on TUMOR-INTRINSIC mechanisms (e.g., genetic alterations, pathway dysregulation within tumor cells, cell plasticity) over TME-focused mechanisms (e.g., immune interactions, fibroblasts, angiogenesis). Aim for roughly 6-7 intrinsic vs 3-4 TME if possible.** Cite supporting facts/sources from the provided context.""",
    backstory="Generates creative hypotheses grounded in provided textual evidence, focusing on requested mechanistic themes.", # Minor backstory update
    tools=[],
    llm=llm, verbose=True, allow_delegation=False
)

novelty_scorer = Agent(
    role='Novelty_Scorer',
    goal="Assess novelty (score 1-5) of each hypothesis provided relative to the provided Extended Knowledge Base Content context. Output a valid JSON list, where each item is an object containing keys 'hypothesis', 'novelty_score', and 'novelty_reasoning'.",
    backstory="Evaluates originality against general knowledge context.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

feasibility_scorer = Agent(
    role='Feasibility_Scorer',
    goal="Assess experimental feasibility (score 1-5) of each hypothesis provided based on standard lab resources. Output a valid JSON list, where each item is an object containing keys 'hypothesis', 'feasibility_score', and 'feasibility_reasoning'.",
    backstory="Provides quick feasibility scores.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# **** MODIFIED GOAL (Fix 1 from prev response) ****
hypothesis_ranker_selector = Agent(
    role='Hypothesis_Ranker_Selector',
    # **** MODIFY THIS GOAL (Add explicit mention of including reasoning strings) ****
    goal="""Rank hypotheses based on provided novelty and feasibility scores (JSON lists). Balance both scores. Select the TOP 3 hypotheses. 
    Output ONLY a valid JSON list for the top 3, where each object MUST contain these exact keys: 'hypothesis' (string), 'novelty_score' (int), 'novelty_reasoning' (string), 'feasibility_score' (int), and 'feasibility_reasoning' (string).""",
    backstory="Prioritizes promising research directions and prepares detailed structured JSON output for presentation.", # Updated backstory
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# **** MODIFIED GOAL (Fix 2 from prev response) ****
hypothesis_presenter = Agent(
    role='Hypothesis_Presenter',
    goal="""Receive a list of selected hypotheses (likely as a JSON list/string containing text, scores, and reasoning). 
    Format and present all details for each selected hypothesis clearly and readably.""",
    backstory="Formats final complex results for clear presentation.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# --- Define MANAGER Agent ---
# Define the AskQuestionTool with ALL worker agents it needs to contact
delegation_tool = AskQuestionTool(
    name="AskQuestionTool",
    description="Use this tool to ask a specific question or delegate a task instruction to a specific coworker agent ('Original_KB_Reader', 'Extended_KB_Reader', 'Hypothesis_Generator', 'Novelty_Scorer', 'Feasibility_Scorer', 'Hypothesis_Ranker_Selector', 'Hypothesis_Presenter'). Provide the question/task instruction and necessary context.",
    agents=[ # List all worker agents manager can delegate to
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
    goal="Coordinate specialist agents via AskQuestionTool to generate, score, rank, and select the top 3 hypotheses, including reasoning.",
    backstory=f"""You are the central coordinator managing a hypothesis pipeline. Use the AskQuestionTool ONLY to delegate tasks sequentially to the correct specialist agent. Your available specialists are:
    - `Original_KB_Reader`: Reads 'knowledge_base.txt'.
    - `Extended_KB_Reader`: Reads 'extended_knowledge_base.txt'.
    - `Hypothesis_Generator`: Generates hypotheses from ORIGINAL KB content.
    - `Novelty_Scorer`: Scores novelty using EXTENDED KB content. Outputs JSON with reasoning.
    - `Feasibility_Scorer`: Scores feasibility. Outputs JSON with reasoning.
    - `Hypothesis_Ranker_Selector`: Ranks/Selects top 3 based on scores. Outputs JSON with full details (text, scores, reasoning).
    - `Hypothesis_Presenter`: Presents final list with all details.
    Follow the workflow precisely, passing necessary context in each delegation.""",
    tools=[delegation_tool],
    llm=llm,
    verbose=True
)

# --- Define Task for the Manager ---
# **** MODIFIED: Task description updated for reasoning pass-through ****
manage_full_pipeline_dual_kb = Task(
  description=f"""Your goal is to orchestrate the full pipeline to get the top 3 most promising mechanistic hypotheses...

  **Workflow to coordinate using AskQuestionTool:**
  1. Ask `Original_KB_Reader`: "Read 'knowledge_base.txt'..." Get original_kb_content.
  2. Ask `Extended_KB_Reader`: "Read 'extended_knowledge_base.txt'..." Get extended_kb_content.
  # **** MODIFY THIS STEP ****
  3. Ask `Hypothesis_Generator`: "Generate up to 10 diverse hypotheses based *only* on the provided Original KB content, **prioritizing tumor-intrinsic mechanisms (genetics, pathways) over TME mechanisms.**" Pass the original_kb_content from Step 1 as 'context'. Get hypotheses_list.
  4. Ask `Novelty_Scorer`: "Score novelty..." Pass hypotheses_list AND extended_kb_content. Get novelty_scores (JSON list).
  5. Ask `Feasibility_Scorer`: "Score feasibility..." Pass hypotheses_list. Get feasibility_scores (JSON list).
  6. Ask `Hypothesis_Ranker_Selector`: "Rank hypotheses based on novelty and feasibility scores and select TOP 3. Output ONLY a valid JSON list containing full details (hypothesis text, scores, AND reasoning) for the selected 3." Pass hypotheses_list, novelty_scores, and feasibility_scores as context. 
  7. **Receive Selected Hypotheses Data:** Get the JSON list/string **containing full details including reasoning** for the top 3 back. 
  8. Ask `Hypothesis_Presenter`: "Present all details (hypothesis, scores, reasoning) for this final list of selected hypotheses clearly." Pass the structured data (JSON list/string) from Step 7 as 'context'.
  9. **Return Final Presentation:** Return the exact formatted output received from the `Hypothesis_Presenter` as your final answer.""",
  expected_output="The final formatted presentation of the top 3 selected hypotheses, including novelty and feasibility scores and reasoning for each.",
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
    tasks=[manage_full_pipeline_dual_kb], # Use the updated task
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

    if not os.path.exists(kb_file_original) or os.path.getsize(kb_file_original) == 0:
        print(f"FATAL ERROR: Original KB file '{kb_file_original}' missing or empty!")
        sys.exit(1)
    if not os.path.exists(kb_file_extended) or os.path.getsize(kb_file_extended) == 0:
        print(f"FATAL ERROR: Extended KB file '{kb_file_extended}' missing or empty!")
        sys.exit(1)

    print("\n--- Starting Hierarchical Crew (Full Pipeline - Dual KB / AskTool + Reasoning) ---")
    # **** Comprehensive initial_inputs to prevent KeyErrors ****
    initial_inputs = {
        'skip_extraction_flag': str(SKIP_FACT_EXTRACTION), # Not used in task, but good practice
        'pdf_directory': '', # Not used in task
        'user_feedback': '', # Not used in task
        'hypothesis': '', # Dummy key
        'hypotheses_context': '', # Placeholder
        'human_feedback': '', # Placeholder
        'feedback_evaluation_context': '{}', # Placeholder
        'knowledge_base_content': '', # Placeholder
        'feasibility_context': '', # Placeholder
        # Add any other keys if new placeholders were added to goals/descriptions
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
        # **** MODIFIED: Check for reasoning ****
        if "hypothesis" in final_result.lower() and "reasoning" in final_result.lower() and ("1." in final_result or "2." in final_result):
             print("SUCCESS: Manager appears to have returned the final presented hypotheses with reasoning.")
             print(final_result)
        # Check if it accidentally returned KB content again
        elif "KimGGO2025.pdf" in final_result and "YoffeGGO2025.pdf" in final_result:
             print("WARNING: Final output appears to be KB content, task may not have fully completed.")
             print(final_result[:500] + "...")
        else:
             print("WARNING: Final output format unexpected (missing reasoning?).")
             print(final_result)
    else:
        print("Workflow did not complete successfully or produced no final output.")
    print("========================================")

if __name__ == "__main__":
    main()