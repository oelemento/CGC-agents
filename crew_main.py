# crew_main.py (Complex Collaboration: Scoring, Ranking, Selection)

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

# --- Feedback Tracker (Simplified for this version) ---
# We'll remove the interactive loop for now to focus on scoring/selection
# Feedback could be reintroduced later for the selected hypotheses.

# --- Custom Tools ---
# Assume PDFExtractionTool, KnowledgeAppendTool, KnowledgeReaderTool definitions are here
class PDFExtractionTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts text content from a given local PDF file path. Input must be the file path string."
    def _run(self, pdf_path: str) -> str:
        print(f"\n--- TOOL: Running PDFExtractionTool on: {pdf_path} ---")
        # ... (robust implementation from previous versions) ...
        if not pdf_path or not isinstance(pdf_path, str): return f"Error: Invalid PDF path: {pdf_path}"
        if not os.path.exists(pdf_path): return f"Error: PDF file not found: {pdf_path}"
        if not pdf_path.lower().endswith(".pdf"): return f"Error: Invalid file type: {pdf_path}"
        try:
            text_content = [];
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file); num_pages = len(reader.pages)
                print(f"Reading {num_pages} pages from '{os.path.basename(pdf_path)}'...")
                for page_num in range(num_pages):
                    try: page = reader.pages[page_num]; text = page.extract_text();
                    except Exception as page_error: print(f"Warn: Skipping page {page_num+1} due to error: {page_error}"); continue
                    if text: text_content.append(text.strip())
                extracted_text = "\n".join(filter(None, text_content))
                if not extracted_text.strip(): return f"Warning: No text extracted from {pdf_path}."
                print(f"Extracted approx {len(extracted_text)} chars.")
                max_chars = 25000
                if len(extracted_text) > max_chars: print(f"Truncating to {max_chars} chars."); return extracted_text[:max_chars]
                else: return extracted_text
        except PyPDF2.errors.PdfReadError as e: return f"Error reading PDF structure {pdf_path}: {e}."
        except Exception as e: print(f"--- ERROR PDF extraction {pdf_path}: {e} ---"); traceback.print_exc(); return f"Unexpected PDF error: {e}"


class KnowledgeAppendTool(BaseTool):
    name: str = "Knowledge Append Tool"
    description: str = ("Appends extracted facts for a source file to knowledge_base.txt. "
                        "Requires 'source_filename' (string) and 'facts_to_store' (string).")
    output_file: str = "knowledge_base.txt"
    def _run(self, source_filename: str, facts_to_store: str) -> str:
        print(f"\n--- TOOL: KnowledgeAppendTool for: {source_filename} ---")
        # ... (robust implementation from previous versions) ...
        if not isinstance(source_filename, str) or not source_filename.strip(): return "Error: Invalid source_filename."
        if not isinstance(facts_to_store, str): facts_to_store = str(facts_to_store)
        if not facts_to_store.strip(): return f"Warning: No facts provided for {source_filename}."
        try:
            os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
            with open(self.output_file, "a", encoding='utf-8') as f:
                f.write(f"--- Facts from: {os.path.basename(source_filename)} ---\n{facts_to_store.strip()}\n--- End Facts ---\n\n")
            return f"Successfully appended facts for {os.path.basename(source_filename)}."
        except Exception as e: print(f"--- ERROR storing knowledge: {e} ---"); traceback.print_exc(); return f"Error storing knowledge: {e}"

class KnowledgeReaderTool(BaseTool):
    name: str = "Knowledge Reader Tool"
    description: str = ("Reads and returns the entire content from knowledge_base.txt.")
    input_file: str = "knowledge_base.txt"
    def _run(self) -> str:
        print(f"\n--- TOOL: KnowledgeReaderTool on: {self.input_file} ---")
        # ... (robust implementation from previous versions) ...
        try:
            if not os.path.exists(self.input_file): return f"ERROR: Knowledge base file not found: {self.input_file}."
            if os.path.getsize(self.input_file) == 0: return "Knowledge base file is empty."
            with open(self.input_file, "r", encoding='utf-8') as f: content = f.read()
            print(f"--- TOOL: Read approx {len(content)} chars from KB. ---")
            max_kb_chars = 40000
            if len(content) > max_kb_chars: print(f"--- WARNING: KB content truncated to {max_kb_chars}. ---"); return content[:max_kb_chars]
            if len(content.strip()) < 50: print(f"--- WARNING: KB content seems short ({len(content)} chars). ---")
            return content
        except Exception as e: print(f"--- ERROR reading KB: {e} ---"); traceback.print_exc(); return f"Error reading KB: {e}"

# --- Instantiate Tools ---
pdf_tool = PDFExtractionTool()
kb_append_tool = KnowledgeAppendTool()
kb_reader_tool = KnowledgeReaderTool()

# --- Configure LLM ---
# Use a capable model, especially for scoring/ranking agents
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# --- Define Agents ---

knowledge_reader_agent = Agent(
    role='Knowledge Base Reader',
    goal=f"Accurately read the entire content of the file '{kb_reader_tool.input_file}' using the KnowledgeReaderTool and return it.",
    backstory="Reliable KB content retriever.",
    tools=[kb_reader_tool], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_synthesizer_agent = Agent(
    role='Hypothesis Synthesizer',
    # **** MODIFIED GOAL ****
    goal="""Generate up to 10 diverse, specific, testable, mechanistic hypotheses based *only* on the provided Knowledge Base Content context. 
    Cite supporting facts/sources for each. Ensure variety in the mechanisms proposed (e.g., intrinsic, TME, immune, metabolic).""",
    backstory="Creative scientific thinker generating a broad set of initial ideas from data.",
    tools=[], llm=llm, verbose=True, allow_delegation=False # Keep simple for now
)

# **** NEW AGENT ****
novelty_scorer_agent = Agent(
    role='Novelty Scorer',
    goal="""Assess the novelty of each hypothesis provided in the context, considering the main themes and facts presented in the accompanying Knowledge Base Content context. 
    Output a list of JSON objects, one for each hypothesis, containing {'hypothesis': <original hypothesis text>, 'novelty_score': <score 1-5, 5=high>, 'novelty_reasoning': '<brief explanation>'}.""",
    backstory="Expert in identifying unique angles and less obvious connections within scientific data. Compares hypotheses against the source material.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# **** NEW AGENT ****
feasibility_scorer_agent = Agent(
    role='Feasibility Scorer',
    goal="""Assess the experimental feasibility of each hypothesis provided in the context, considering standard lab resources (cells, organoids, mice, basic assays). 
    Output a list of JSON objects, one for each hypothesis, containing {'hypothesis': <original hypothesis text>, 'feasibility_score': <score 1-5, 5=high>, 'feasibility_reasoning': '<brief explanation>'}.""",
    backstory="Pragmatic experimentalist providing a quick feasibility score based on common lab capabilities.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# **** NEW AGENT ****
hypothesis_ranker_selector_agent = Agent(
    role='Hypothesis Ranker and Selector',
    goal="""Receive a list of hypotheses with novelty scores and feasibility scores. Rank them based on a balance of high novelty and high feasibility (e.g., simple average or weighted score). 
    Select the TOP 3 ranked hypotheses. Output *only* the text of the selected top 3 hypotheses, clearly numbered.""",
    backstory="Strategic thinker prioritizing promising research directions based on novelty and practicality.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# --- Kept Agents ---
hypothesis_presenter_agent = Agent(
    role='Hypothesis Presenter',
    goal="Clearly present the provided final list of selected hypotheses text to the user.",
    backstory="Formats and displays final results clearly.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)
# Fact Extraction Agents remain the same
pdf_reader_agent = Agent( role='PDF_Reader', goal='Extract text from PDF.', backstory="PDF expert.", tools=[pdf_tool], llm=llm, verbose=True, allow_delegation=False )
fact_summarizer_agent = Agent( role='Fact_Summarizer', goal='List key facts from text.', backstory="Fact expert.", tools=[], llm=llm, verbose=True, allow_delegation=False )
knowledge_manager_agent = Agent( role='Knowledge_Manager', goal='Store facts in KB.', backstory="Storage expert.", tools=[kb_append_tool], llm=llm, verbose=True, allow_delegation=False )


# --- Define Tasks for the New Workflow ---

task_read_knowledge_base = Task(
    description=f"Read the content of '{kb_reader_tool.input_file}' using KnowledgeReaderTool.",
    expected_output="String containing the full KB content.",
    agent=knowledge_reader_agent
)

# **** MODIFIED TASK ****
task_generate_initial_hypotheses = Task(
    description="""**Using the Knowledge Base Content provided in the context**, generate up to 10 diverse, specific, testable mechanistic hypotheses about early lung cancer/GGOs. 
    Cover different mechanism types (intrinsic, TME, immune, metabolic etc.). Cite supporting facts/sources for each.""",
    expected_output="A list/text containing up to 10 diverse hypotheses with rationale.",
    agent=hypothesis_synthesizer_agent,
    context=[task_read_knowledge_base] # Depends on KB content
)

# **** NEW TASK ****
task_score_novelty = Task(
    description="""Score the novelty of the generated hypotheses (provided in context) based on the Knowledge Base Content (also in context). 
    Compare each hypothesis against the main facts/themes in the KB. Output a list of JSON objects: {'hypothesis': <text>, 'novelty_score': <1-5>, 'novelty_reasoning': '<text>'}.""",
    expected_output="List of JSON objects with novelty scores and reasoning.",
    agent=novelty_scorer_agent,
    context=[task_read_knowledge_base, task_generate_initial_hypotheses] # Needs KB and Hypotheses
)

# **** NEW TASK ****
task_score_feasibility = Task(
    description="""Score the experimental feasibility of the generated hypotheses (provided in context) using standard lab resources. 
    Output a list of JSON objects: {'hypothesis': <text>, 'feasibility_score': <1-5>, 'feasibility_reasoning': '<text>'}.""",
    expected_output="List of JSON objects with feasibility scores and reasoning.",
    agent=feasibility_scorer_agent,
    context=[task_generate_initial_hypotheses] # Needs Hypotheses
)

# **** NEW TASK ****
task_rank_select_hypotheses = Task(
    description="""Rank the hypotheses based on the novelty scores and feasibility scores provided in context. Prioritize hypotheses with a good balance of high novelty and high feasibility. 
    Select the TOP 3 hypotheses from the ranking. Output *only* the text of these selected top 3 hypotheses, numbered 1, 2, 3.""",
    expected_output="The text of the top 3 selected hypotheses, numbered.",
    agent=hypothesis_ranker_selector_agent,
    context=[task_generate_initial_hypotheses, task_score_novelty, task_score_feasibility] # Needs hypotheses and both scores
)

# **** MODIFIED TASK ****
task_present_selected_hypotheses = Task(
    description="""Present the final list of TOP 3 selected hypotheses (provided in context) clearly to the user.""",
    expected_output="Formatted display of the top 3 selected hypotheses.",
    agent=hypothesis_presenter_agent,
    context=[task_rank_select_hypotheses] # Depends on the selection task
)


# Fact Extraction Task Definition (remains the same)
def create_fact_extraction_tasks(pdf_full_path, pdf_filename):
    task_extract_text = Task( description=f"Extract text from '{pdf_filename}' at path: '{pdf_full_path}'.", expected_output="Extracted text.", agent=pdf_reader_agent )
    task_extract_facts = Task( description=f"Extract key facts related to early lung cancer/GGOs from text of '{pdf_filename}'.", expected_output="Bulleted list of facts.", agent=fact_summarizer_agent, context=[task_extract_text] )
    task_store_facts = Task( description=f"Store facts for '{pdf_filename}'.", expected_output="Confirmation message.", agent=knowledge_manager_agent, context=[task_extract_facts] )
    return [task_extract_text, task_extract_facts, task_store_facts]


# --- Define Crews ---
# Crew for the analysis workflow
# **** MODIFIED: Include new agents and tasks in sequence ****
analysis_crew = Crew(
    agents=[
        knowledge_reader_agent,
        hypothesis_synthesizer_agent,
        novelty_scorer_agent,         # New
        feasibility_scorer_agent,     # New
        hypothesis_ranker_selector_agent, # New
        hypothesis_presenter_agent
    ],
    tasks=[ # Define the new sequence
        task_read_knowledge_base,
        task_generate_initial_hypotheses,
        task_score_novelty,
        task_score_feasibility,
        task_rank_select_hypotheses,
        task_present_selected_hypotheses
        ],
    process=Process.sequential,
    verbose=True
)

# Fact extraction crew remains the same
extraction_crew = Crew(
    agents=[pdf_reader_agent, fact_summarizer_agent, knowledge_manager_agent],
    tasks=[], # Added dynamically
    process=Process.sequential,
    verbose=True
)

# --- Helper Functions ---
# Keep robust parse_json_from_llm_output
def parse_json_from_llm_output(text):
    # ... (use robust implementation from previous version) ...
    if not isinstance(text, str): return None # Return None if not string
    patterns = [ r'```json\s*(\[.*?\]|\{.*?\})\s*```', r'```\s*(\[.*?\]|\{.*?\})\s*```', ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            try: return json.loads(json_str)
            except json.JSONDecodeError as e: print(f"Warn: Failed JSON block parse ({pattern}): {e}")
    try: # Try finding raw list or object
        start_obj = text.find('{'); end_obj = text.rfind('}') + 1
        start_list = text.find('['); end_list = text.rfind(']') + 1
        json_str = None
        if 0 <= start_obj < end_obj: json_str = text[start_obj:end_obj]
        elif 0 <= start_list < end_list: json_str = text[start_list:end_list]
        if json_str:
             if json_str.count('{') == json_str.count('}') and json_str.count('[') == json_str.count(']'):
                  try: return json.loads(json_str)
                  except json.JSONDecodeError as e: print(f"Warn: Failed raw parse: {e}")
    except Exception as e: print(f"Warn: Error during raw object search: {e}")
    print(f"ERROR: Failed to parse JSON/List from LLM output: {text}")
    return None # Indicate failure clearly

# Remove collect_direct_feedback as the loop is removed for now

# --- Main Workflow Logic ---
def main():
    global SKIP_FACT_EXTRACTION, extraction_crew, analysis_crew

    kb_file = kb_reader_tool.input_file

    # --- Step 1: Fact Extraction (Optional) ---
    if not SKIP_FACT_EXTRACTION:
        # ... (fact extraction logic remains the same) ...
        if os.path.exists(kb_file):
            print(f"Clearing existing KB file: {kb_file}")
            try: os.remove(kb_file)
            except OSError as e: print(f"Error removing KB: {e}"); sys.exit(1)
        # Get PDF directory logic...
        pdf_directory = os.getenv("PAPERS_PATH") # Add fallback or input as before
        if not pdf_directory: pdf_directory = input("Enter PDF directory path: ").strip()
        if not pdf_directory or not os.path.isdir(pdf_directory): print(f"Invalid dir: {pdf_directory}"); sys.exit(1)
        pdf_files = sorted([f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")])
        if not pdf_files: print("No PDFs found."); sys.exit(1)
        print(f"--- Starting Fact Extraction for {len(pdf_files)} files ---")
        all_extraction_tasks = []
        for pdf_filename in pdf_files:
             pdf_full_path = os.path.join(pdf_directory, pdf_filename)
             tasks = create_fact_extraction_tasks(pdf_full_path, pdf_filename)
             all_extraction_tasks.extend(tasks)
        if all_extraction_tasks:
            extraction_crew.tasks = all_extraction_tasks
            try: extraction_crew.kickoff(); print("--- Fact Extraction Finished ---")
            except Exception as e: print(f"\n--- ERROR during Extraction: {e} ---"); traceback.print_exc(); sys.exit(1)
        else: print("--- No extraction tasks. ---"); sys.exit(1)
    else:
        print(f"Skipping fact extraction. Using existing KB: {kb_file}")
        if not os.path.exists(kb_file) or os.path.getsize(kb_file) == 0:
             print(f"Warning: KB {kb_file} missing or empty! Analysis might fail.")

    # --- Step 2: Run the Analysis Crew (Read KB -> Generate -> Score -> Rank/Select -> Present) ---
    print("\n\n--- Starting Analysis, Scoring, and Selection Crew ---")
    # No specific inputs needed here unless a task description requires them globally
    # Context flows between tasks defined in the analysis_crew.tasks list
    final_result = None
    try:
        # Kickoff the full analysis sequence
        result_obj = analysis_crew.kickoff()
        final_result = getattr(result_obj, 'raw', str(result_obj)) # Get final output (presentation)

        print("\n\n--- Analysis Crew Execution Finished ---")

    except Exception as e:
        print(f"\n--- ERROR during Analysis Crew run: {e} ---")
        traceback.print_exc()

    # --- Final Output ---
    # The final result should be the output of the last task (presentation)
    print("\n\n\n--- Workflow Complete ---")
    print("========================================")
    print(" Final Output (Selected Hypotheses) ")
    print("========================================")
    if final_result:
        print(final_result)
    else:
        print("Workflow did not complete successfully or produced no final output.")
    print("========================================")


# Remove the old print_final_output function if not needed elsewhere

if __name__ == "__main__":
    main()