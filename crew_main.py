# crew_main.py (Hierarchical Redesign - Attempt 3)

import os
import traceback
import json
import re
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
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

# --- Feedback Tracker (Simplified) ---
# In this model, feedback might be passed into the manager's initial context
# for a subsequent run if refinement is needed.
class FeedbackTracker:
    def __init__(self):
        self.last_feedback = ""
    def add_feedback(self, feedback):
        if feedback is not None and isinstance(feedback, str) and feedback.strip():
            self.last_feedback = feedback
        else:
             self.last_feedback = "OK" # Treat empty as OK for simplicity now
    def get_last_feedback(self): return self.last_feedback
    def clear(self): self.last_feedback = ""

feedback_tracker = FeedbackTracker()

# --- Custom Tools ---
# Assume PDFExtractionTool, KnowledgeAppendTool, KnowledgeReaderTool robust definitions are here
class PDFExtractionTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts text content from a given local PDF file path."
    def _run(self, pdf_path: str) -> str:
        print(f"\n--- TOOL: Running PDFExtractionTool on: {pdf_path} ---")
        if not pdf_path or not isinstance(pdf_path, str): return f"Error: Invalid PDF path: {pdf_path}"
        if not os.path.exists(pdf_path): return f"Error: PDF file not found: {pdf_path}"
        if not pdf_path.lower().endswith(".pdf"): return f"Error: Invalid file type: {pdf_path}"
        try:
            text_content = [];
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file); num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    try: page = reader.pages[page_num]; text = page.extract_text();
                    except Exception as page_error: print(f"Warn: Skipping page {page_num+1}: {page_error}"); continue
                    if text: text_content.append(text.strip())
                extracted_text = "\n".join(filter(None, text_content))
                if not extracted_text.strip(): return f"Warning: No text extracted."
                max_chars = 25000; print(f"Extracted ~{len(extracted_text)} chars.")
                return extracted_text[:max_chars] if len(extracted_text) > max_chars else extracted_text
        except Exception as e: print(f"--- ERROR PDF extraction: {e} ---"); return f"Unexpected PDF error: {e}"

class KnowledgeAppendTool(BaseTool):
    name: str = "Knowledge Append Tool"
    description: str = "Appends facts to knowledge_base.txt. Args: source_filename (str), facts_to_store (str)."
    output_file: str = "knowledge_base.txt"
    def _run(self, source_filename: str, facts_to_store: str) -> str:
        print(f"\n--- TOOL: KnowledgeAppendTool for: {source_filename} ---")
        if not isinstance(source_filename, str) or not source_filename.strip(): return "Error: Invalid source_filename."
        if not isinstance(facts_to_store, str): facts_to_store = str(facts_to_store)
        if not facts_to_store.strip(): return f"Warning: No facts provided for {source_filename}."
        try:
            os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
            with open(self.output_file, "a", encoding='utf-8') as f: f.write(f"--- Facts from: {os.path.basename(source_filename)} ---\n{facts_to_store.strip()}\n--- End Facts ---\n\n")
            return f"Appended facts for {os.path.basename(source_filename)}."
        except Exception as e: print(f"--- ERROR storing knowledge: {e} ---"); return f"Error storing knowledge: {e}"

class KnowledgeReaderTool(BaseTool):
    name: str = "Knowledge Reader Tool"
    description: str = "Reads and returns the entire content from knowledge_base.txt."
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
pdf_tool = PDFExtractionTool()
kb_append_tool = KnowledgeAppendTool()
kb_reader_tool = KnowledgeReaderTool()

# --- Configure LLM ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5) # More capable model, moderate temp for manager

# --- Define WORKER Agents ---
# Use clear, distinct roles for the manager to reference

knowledge_reader = Agent( # Renamed for clarity
    role='Knowledge_Reader',
    goal=f"Read '{kb_reader_tool.input_file}' using KnowledgeReaderTool and return content.",
    backstory="Specialist in retrieving data from the knowledge base.",
    tools=[kb_reader_tool], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_generator = Agent( # Renamed for clarity
    role='Hypothesis_Generator',
    goal="Generate a diverse set of up to 10 specific, testable, mechanistic hypotheses based *only* on the provided Knowledge Base Content context. Cite supporting facts/sources.",
    backstory="Creative idea generator focused on mechanism and evidence.",
    tools=[], llm=llm, verbose=True, allow_delegation=True # Allow delegation for potential feasibility checks?
)

novelty_scorer = Agent(
    role='Novelty_Scorer',
    # **** MODIFIED GOAL: Describe output format instead of showing literal JSON ****
    goal="""Assess the novelty of each hypothesis provided in the context, considering the main themes and facts presented in the accompanying Knowledge Base Content context. 
    Output a valid JSON list, where each item is an object containing keys 'hypothesis' (with the original hypothesis text), 'novelty_score' (integer 1-5, 5=high), and 'novelty_reasoning' (string explanation).""",
    backstory="Expert in identifying unique angles and less obvious connections within scientific data. Compares hypotheses against the source material.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

feasibility_scorer = Agent(
    role='Feasibility_Scorer',
     # **** MODIFIED GOAL: Describe output format instead of showing literal JSON ****
    goal="""Assess the experimental feasibility of each hypothesis provided in the context, considering standard lab resources (cells, organoids, mice, basic assays). 
    Output a valid JSON list, where each item is an object containing keys 'hypothesis' (with the original hypothesis text), 'feasibility_score' (integer 1-5, 5=high), and 'feasibility_reasoning' (string explanation).""",
    backstory="Pragmatic experimentalist providing a quick feasibility score based on common lab capabilities.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_ranker_selector = Agent(
    role='Hypothesis_Ranker_Selector',
    goal="Rank hypotheses based on provided novelty and feasibility scores, balancing both. Select the TOP 3 hypotheses. Output *only* the text of the selected top 3.",
    backstory="Prioritizes promising research directions.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# Detailed Feasibility Assessor (can be used instead of scorer if manager chooses)
detailed_feasibility_assessor = Agent(
    role='Detailed_Feasibility_Assessor',
    goal='Provide a *detailed* experimental feasibility assessment for provided hypotheses, suggesting concrete approaches and challenges using standard lab resources.',
    backstory="Pragmatic experimental biologist providing in-depth feasibility analysis.",
    tools=[], llm=llm, verbose=True, allow_delegation=True # Could delegate queries back if needed
)

# Hypothesis Refiner (for feedback/feasibility)
hypothesis_refiner = Agent(
    role='Hypothesis_Refiner',
    goal="Refine hypotheses based on provided feedback (user or feasibility report) and Knowledge Base Content context. Ensure refinements are evidence-based.",
    backstory="Improves hypotheses based on critique and data.",
    tools=[], # Needs KB content passed in context
    llm=llm, verbose=True, allow_delegation=False
)

hypothesis_presenter = Agent(
    role='Hypothesis_Presenter',
    goal="Clearly present the provided list/text of hypotheses.",
    backstory="Formats results for presentation.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# Fact Extraction Agents
# Fact Extraction Agents
pdf_reader = Agent( 
    role='PDF_Reader', 
    goal='Extract text from PDF.', 
    backstory="Expert in reading PDF files and extracting text content accurately.", # **** ADD THIS LINE ****
    tools=[pdf_tool], 
    llm=llm, 
    verbose=True, 
    allow_delegation=False 
)
fact_summarizer = Agent( 
    role='Fact_Summarizer', 
    goal='List key facts from text.', 
    backstory="Skilled at identifying and extracting key factual statements from scientific text.", # **** ADD THIS LINE ****
    tools=[], 
    llm=llm, 
    verbose=True, 
    allow_delegation=False 
)
knowledge_manager = Agent( 
    role='Knowledge_Manager', 
    goal='Store facts in KB.', 
    backstory="Efficiently organizes and saves information into the knowledge base.", # **** ADD THIS LINE ****
    tools=[kb_append_tool], 
    llm=llm, 
    verbose=True, 
    allow_delegation=False 
)

# --- Define MANAGER Agent ---
research_manager = Agent(
    role='Research_Manager',
    goal="""Orchestrate the generation, evaluation, and selection of high-quality, novel, and feasible mechanistic hypotheses about early lung cancer/GGOs. 
    Coordinate specialist agents, manage workflow, synthesize results, and deliver the final selected hypotheses.""",
    backstory=f"""You are the central coordinator for a hypothesis generation project. 
    Your available specialist agents are:
    - `Knowledge_Reader`: Reads the knowledge base ('{kb_reader_tool.input_file}').
    - `Hypothesis_Generator`: Generates up to 10 diverse hypotheses from KB content.
    - `Novelty_Scorer`: Scores novelty of hypotheses against KB content (1-5).
    - `Feasibility_Scorer`: Scores feasibility of hypotheses (1-5).
    - `Hypothesis_Ranker_Selector`: Ranks hypotheses based on scores and selects the top 3.
    - `Detailed_Feasibility_Assessor`: Provides in-depth feasibility analysis (alternative to scorer).
    - `Hypothesis_Refiner`: Refines hypotheses based on feedback or feasibility reports.
    - `Hypothesis_Presenter`: Presents lists of hypotheses.
    - `PDF_Reader`, `Fact_Summarizer`, `Knowledge_Manager`: For optional fact extraction prep.

    You MUST achieve the goal by delegating tasks to these agents by their specific role names. 
    Plan the workflow steps, pass necessary context between tasks, and make decisions based on agent outputs. 
    You can choose between different workflows, e.g., generate->score->rank->select OR generate->assess_detailed->refine.""",
    tools=[], # Manager delegates, doesn't perform tasks itself
    llm=llm,
    verbose=True
    # Delegation is implicit in hierarchical process
)

# --- Define Task for the Manager ---
# This task tells the manager the overall objective and inputs.
# The manager's LLM + prompt should figure out the specific delegation steps.

manage_hypothesis_pipeline = Task(
  description="""Develop a final list of the top 3 most promising (novel and feasible) mechanistic hypotheses about early lung cancer/GGOs.

  **Workflow:**
  1.  **Prepare Knowledge:** Ensure the knowledge base is ready. If fact extraction is required ('{skip_extraction_flag}' is False, PDF dir: '{pdf_directory}'), coordinate `PDF_Reader`, `Fact_Summarizer`, and `Knowledge_Manager`. If skipping, ensure KB exists by delegating to `Knowledge_Reader`.
  2.  **Generate Ideas:** Delegate hypothesis generation (up to 10) to `Hypothesis_Generator`, providing the KB content.
  3.  **Evaluate & Select:** Decide how to evaluate the generated hypotheses. EITHER:
      * **Option A (Scoring & Ranking):** Delegate novelty scoring to `Novelty_Scorer`, feasibility scoring to `Feasibility_Scorer`, then ranking and selection of top 3 to `Hypothesis_Ranker_Selector`.
      * **Option B (Detailed Assessment First - Less preferred unless needed):** Delegate detailed feasibility assessment to `Detailed_Feasibility_Assessor`. You might then refine or select based on this.
      * *(Choose the most efficient path, likely Option A)*.
  4.  **(Optional) Refinement:** Based on scores or assessment, you *could* choose to delegate refinement of selected hypotheses to `Hypothesis_Refiner` before final presentation, passing the relevant context (scores/assessment, KB content).
  5.  **Present Final List:** Delegate the presentation of the *final selected (and possibly refined)* top 3 hypotheses to `Hypothesis_Presenter`.

  **Inputs:**
  - Skip Fact Extraction Flag: {skip_extraction_flag}
  - PDF Directory (if extracting): {pdf_directory}
  - Initial User Feedback (for context): {user_feedback}

  **Output:** The final, selected (and possibly refined) top 3 hypotheses.""",
  expected_output="The final selected top 3 mechanistic hypotheses, clearly presented.",
  agent=research_manager # Assign task to the manager
)


# --- Define Hierarchical Crew ---
research_crew = Crew(
    agents=[ # Manager first, then all workers
        research_manager,
        knowledge_reader,
        hypothesis_generator,
        novelty_scorer,
        feasibility_scorer,
        hypothesis_ranker_selector,
        detailed_feasibility_assessor, # Include all potential workers
        hypothesis_refiner,
        hypothesis_presenter,
        pdf_reader,
        fact_summarizer,
        knowledge_manager
    ],
    tasks=[manage_hypothesis_pipeline], # Only manager's task
    process=Process.hierarchical,
    manager_llm=llm, # Specify LLM for manager
    verbose=True # Use boolean True
)

# --- Helper Functions ---
# Keep parse_json_from_llm_output
def parse_json_from_llm_output(text):
    # ... (robust implementation) ...
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
        if 0 <= start_list < end_list : json_str = text[start_list:end_list] # Prioritize list for scorers
        elif 0 <= start_obj < end_obj: json_str = text[start_obj:end_obj]
        if json_str:
             if json_str.count('{') >= json_str.count('}') and json_str.count('[') >= json_str.count(']'): # Relaxed check for lists
                  try: return json.loads(json_str)
                  except json.JSONDecodeError as e: print(f"Warn: Failed raw parse: {e}")
    except Exception as e: print(f"Warn: Error during raw search: {e}")
    print(f"ERROR: Failed to parse JSON/List: {text}")
    return None

# Remove collect_direct_feedback for now

# --- Main Workflow Logic ---
def main():
    global SKIP_FACT_EXTRACTION, feedback_tracker, extraction_crew, research_crew

    kb_file = kb_reader_tool.input_file

    # --- Step 1: Fact Extraction (Optional, could also be delegated by manager) ---
    # For simplicity, we'll still do this outside the main crew if needed.
    if not SKIP_FACT_EXTRACTION:
        # ... (fact extraction logic - same as previous script) ...
        if os.path.exists(kb_file): print(f"Clearing KB: {kb_file}"); os.remove(kb_file)
        feedback_tracker.clear()
        pdf_directory = os.getenv("PAPERS_PATH") or input("Enter PDF directory: ").strip()
        if not pdf_directory or not os.path.isdir(pdf_directory): print(f"Invalid dir: {pdf_directory}"); sys.exit(1)
        pdf_files = sorted([f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")])
        if not pdf_files: print("No PDFs found."); sys.exit(1)
        print(f"--- Starting Fact Extraction for {len(pdf_files)} files ---")
        all_tasks = [];
        for pdf_filename in pdf_files:
             tasks = create_fact_extraction_tasks(os.path.join(pdf_directory, pdf_filename), pdf_filename)
             all_tasks.extend(tasks)
        if all_tasks:
             extraction_crew.tasks = all_tasks;
             try: extraction_crew.kickoff(); print("--- Extraction Finished ---")
             except Exception as e: print(f"\n--- ERROR Extraction: {e} ---"); sys.exit(1)
        else: print("--- No extraction tasks. ---"); sys.exit(1)
    else:
        print(f"Skipping fact extraction. Using existing KB: {kb_file}")
        if not os.path.exists(kb_file) or os.path.getsize(kb_file) == 0: print(f"Warning: KB {kb_file} missing or empty!")

    # --- Step 2: Run the Main Hierarchical Crew ---
    print("\n\n--- Starting Hierarchical Hypothesis Pipeline ---")
    # Pass initial context to the manager's task
    initial_inputs = {
        'skip_extraction_flag': str(SKIP_FACT_EXTRACTION), # Pass as string
        'pdf_directory': pdf_directory if not SKIP_FACT_EXTRACTION else '',
        'user_feedback': feedback_tracker.get_last_feedback(), # Pass any carry-over feedback
        'hypothesis': '' # **** ADD THIS DUMMY KEY AS A WORKAROUND ****
    }

    final_result = None
    try:
        # KICK OFF THE HIERARCHICAL CREW
        result_obj = research_crew.kickoff(inputs=initial_inputs)
        final_result = getattr(result_obj, 'raw', str(result_obj)) # Manager's final output

        print("\n\n--- Hierarchical Crew Execution Finished ---")

    except Exception as e:
        print(f"\n--- ERROR during Hierarchical Crew execution: {e} ---")
        traceback.print_exc()

    # --- Final Output ---
    print("\n\n\n--- Workflow Complete ---")
    print("========================================")
    print("   Final Output from Research Manager   ")
    print("========================================")
    if final_result:
        print(final_result)
    else:
        print("Workflow did not complete successfully or produced no final output.")
    print("========================================")


if __name__ == "__main__":
    main()