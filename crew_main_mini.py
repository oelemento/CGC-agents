# crew_main_hierarchical_full.py (Manager + All Workers - Using AskQuestionTool)

import os
import traceback
import json
import re
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
# Explicitly import the tool being used
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool

import PyPDF2
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
            if len(content) > max_kb_chars: print(f"--- WARNING: KB truncated to {max_kb_chars}. ---"); return content[:max_kb_chars]
            return content
        except Exception as e: print(f"--- ERROR reading KB: {e} ---"); return f"Error reading KB: {e}"

# --- Instantiate Tools ---
kb_reader_tool = KnowledgeReaderTool()

# --- Configure LLM ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# --- Define WORKER Agents ---
knowledge_reader = Agent(
    role='Knowledge_Reader',
    goal=f"Read '{kb_reader_tool.input_file}' using KnowledgeReaderTool and return full content.",
    backstory="Specialist in retrieving data from the knowledge base file.",
    tools=[kb_reader_tool],
    llm=llm, verbose=True, allow_delegation=False
)

hypothesis_generator = Agent(
    role='Hypothesis_Generator',
    goal="Generate up to 10 diverse, specific, testable, mechanistic hypotheses based *only* on the provided Knowledge Base Content context. Cite supporting facts/sources.",
    backstory="Creative idea generator focused on mechanism and evidence from provided text.",
    tools=[], # Relies on context
    llm=llm, verbose=True, allow_delegation=False
)

# **** ADDED BACK: Scoring/Ranking Agents ****
novelty_scorer = Agent(
    role='Novelty_Scorer',
    goal="Assess novelty (score 1-5) of each hypothesis provided relative to the provided Knowledge Base Content context. Output a valid JSON list: [{'hypothesis': text, 'novelty_score': N, 'novelty_reasoning': text}]",
    backstory="Evaluates originality against source material.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

feasibility_scorer = Agent(
    role='Feasibility_Scorer',
    goal="Assess experimental feasibility (score 1-5) of each hypothesis provided based on standard lab resources. Output a valid JSON list: [{'hypothesis': text, 'feasibility_score': N, 'feasibility_reasoning': text}]",
    backstory="Provides quick feasibility scores based on common lab capabilities.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_ranker_selector = Agent(
    role='Hypothesis_Ranker_Selector',
    goal="Rank hypotheses based on provided novelty and feasibility scores (JSON lists). Balance both scores (e.g., sum or average). Select the TOP 3 hypotheses. Output *only* the text of the selected top 3 hypotheses, numbered 1, 2, 3.",
    backstory="Prioritizes promising research directions based on combined novelty and practicality scores.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_presenter = Agent(
    role='Hypothesis_Presenter',
    goal="Clearly present the provided final list/text of hypotheses.",
    backstory="Formats final results for presentation.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# --- Define MANAGER Agent ---
# **** MODIFIED: Update tool with all worker agents ****
delegation_tool = AskQuestionTool(
    name="AskQuestionTool",
    description="Use this tool to ask a specific question or delegate a task instruction to a specific coworker agent ('Knowledge_Reader', 'Hypothesis_Generator', 'Novelty_Scorer', 'Feasibility_Scorer', 'Hypothesis_Ranker_Selector', 'Hypothesis_Presenter'). Provide the question/task instruction and necessary context.",
    agents=[ # List all worker agents manager can delegate to
        knowledge_reader,
        hypothesis_generator,
        novelty_scorer,
        feasibility_scorer,
        hypothesis_ranker_selector,
        hypothesis_presenter
        ]
)

research_manager = Agent(
    role='Research_Manager',
    goal="Coordinate specialist agents to generate, score, rank, and select the top 3 hypotheses.",
    # **** MODIFIED: Update backstory with all agents ****
    backstory=f"""You are the central coordinator. Your available specialist agents are:
    - `Knowledge_Reader`: Reads the knowledge base ('{kb_reader_tool.input_file}').
    - `Hypothesis_Generator`: Generates up to 10 diverse hypotheses from provided text content.
    - `Novelty_Scorer`: Scores novelty (1-5). Outputs JSON list.
    - `Feasibility_Scorer`: Scores feasibility (1-5). Outputs JSON list.
    - `Hypothesis_Ranker_Selector`: Ranks based on scores, selects top 3. Outputs text list.
    - `Hypothesis_Presenter`: Presents final list.

    You MUST achieve your goal by orchestrating these agents using the AskQuestionTool. 
    Plan the steps, delegate tasks using the tool, pass necessary context (KB content, hypotheses, scores), and return the final presented list.""",
    tools=[delegation_tool], # Give manager the tool
    llm=llm,
    verbose=True
)

# --- Define Task for the Manager ---
# **** MODIFIED: Task now involves the full pipeline ****
manage_full_pipeline = Task(
  description=f"""Your goal is to orchestrate the full pipeline to get the top 3 most promising mechanistic hypotheses based on the knowledge base '{kb_reader_tool.input_file}'.

  **Workflow to coordinate using AskQuestionTool:**
  1.  **Delegate Reading:** Ask `Knowledge_Reader`: "Read the knowledge base file ('{kb_reader_tool.input_file}') and return the full content."
  2.  **Receive KB Content:** Get the full text content back.
  3.  **Delegate Generation:** Ask `Hypothesis_Generator`: "Generate up to 10 diverse hypotheses based *only* on the provided knowledge base content." Pass the KB content from Step 2 as 'context'.
  4.  **Receive Hypotheses:** Get the list of 10 hypotheses back.
  5.  **Delegate Novelty Scoring:** Ask `Novelty_Scorer`: "Score the novelty of the provided hypotheses based on the provided Knowledge Base Content." Pass the hypotheses list AND the KB content as 'context'.
  6.  **Receive Novelty Scores:** Get the JSON list of novelty scores back.
  7.  **Delegate Feasibility Scoring:** Ask `Feasibility_Scorer`: "Score the experimental feasibility of the provided hypotheses." Pass the hypotheses list as 'context'.
  8.  **Receive Feasibility Scores:** Get the JSON list of feasibility scores back.
  9.  **Delegate Ranking/Selection:** Ask `Hypothesis_Ranker_Selector`: "Rank the hypotheses based on novelty and feasibility scores and select the TOP 3." Pass the original hypotheses list, the novelty score JSON, and the feasibility score JSON as 'context'.
  10. **Receive Selected Hypotheses:** Get the text of the top 3 selected hypotheses back.
  11. **Delegate Presentation:** Ask `Hypothesis_Presenter`: "Present this final list of selected hypotheses." Pass the selected hypotheses text from Step 10 as 'context'.
  12. **Return Final Presentation:** Return the exact output received from the `Hypothesis_Presenter` as your final answer.""",
  expected_output="The final formatted presentation of the top 3 selected hypotheses.",
  agent=research_manager
)

# --- Define Hierarchical Crew ---
# **** MODIFIED: Include all agents ****
research_crew = Crew(
    agents=[ # Manager first, then all workers
        research_manager,
        knowledge_reader,
        hypothesis_generator,
        novelty_scorer,
        feasibility_scorer,
        hypothesis_ranker_selector,
        hypothesis_presenter
        ],
    tasks=[manage_full_pipeline], # Only manager's task
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True
)

# --- Main Workflow Logic ---
def main():
    global research_crew

    kb_file = kb_reader_tool.input_file
    print(f"Attempting to use KB: {kb_file}")
    if not os.path.exists(kb_file) or os.path.getsize(kb_file) == 0:
        print(f"FATAL ERROR: KB file '{kb_file}' missing or empty!"); sys.exit(1)

    print("\n--- Starting Hierarchical Crew (Full Pipeline) ---")
    initial_inputs = {} # No external inputs needed for this task description

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
        # Check if the output looks like the final presented hypotheses
        if "hypothesis" in final_result.lower() and ("1." in final_result or "2." in final_result):
             print("SUCCESS: Manager appears to have returned the final presented hypotheses.")
             print(final_result)
        # Check if it accidentally returned KB content again
        elif "KimGGO2025.pdf" in final_result and "YoffeGGO2025.pdf" in final_result:
             print("WARNING: Final output appears to be KB content, not hypotheses.")
             print(final_result[:500] + "...")
        else:
             print("WARNING: Final output format unexpected.")
             print(final_result)
    else:
        print("Workflow did not complete successfully or produced no final output.")
    print("========================================")


if __name__ == "__main__":
    main()