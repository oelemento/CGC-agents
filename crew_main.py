# crew_main.py (Sequential Process + Peer Delegation + Input Fixes v2)

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

# --- Feedback Tracker ---
class FeedbackTracker:
    # Using the simplified version from the previous attempt
    def __init__(self):
        self.last_feedback = ""
        self.feedback_provided_since_last_run = False

    def add_feedback(self, feedback):
        if feedback is not None and isinstance(feedback, str) and feedback.strip():
            self.last_feedback = feedback
            self.feedback_provided_since_last_run = True
        else:
            # Treat empty input after requesting feedback as potential approval only if no feedback was given this cycle
            if not self.feedback_provided_since_last_run:
                 self.last_feedback = "OK" # Implicit approval if nothing entered
            # If feedback *was* given and then empty entered, keep last real feedback

    def get_last_feedback(self):
        """Returns the most recent feedback entry."""
        # Let's not reset the flag here, but when feedback is actually used/checked
        return self.last_feedback

    def reset_feedback_flag(self):
         """Resets the flag indicating if new feedback was provided."""
         self.feedback_provided_since_last_run = False

    def is_approval(self, feedback_text):
        """ Checks if feedback text clearly indicates approval. """
        if not isinstance(feedback_text, str): # Handle None or other types
             return False
        approval_terms = ['ok', 'yes', 'approve', 'good', 'proceed', 'looks good', 'fine', 'accept']
        # Check if feedback is exactly "OK" (case-insensitive) or contains other approval terms
        return feedback_text.strip().lower() == "ok" or any(term in feedback_text.lower() for term in approval_terms)

    def clear(self):
        self.last_feedback = ""
        self.feedback_provided_since_last_run = False

feedback_tracker = FeedbackTracker()

# --- Custom Tools ---
class PDFExtractionTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts text content from a given local PDF file path. Input must be the file path string."
    def _run(self, pdf_path: str) -> str:
        print(f"\n--- TOOL: Running PDFExtractionTool on: {pdf_path} ---")
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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4)

# --- Define Agents ---
knowledge_reader_agent = Agent(
    role='Knowledge Base Reader',
    goal=f"Accurately read the entire content of the file '{kb_reader_tool.input_file}' using the KnowledgeReaderTool and return it.",
    backstory="A reliable agent focused solely on retrieving the full text content from the knowledge base file.",
    tools=[kb_reader_tool],
    llm=llm, verbose=True, allow_delegation=False
)

hypothesis_synthesizer_agent = Agent(
    role='Hypothesis Synthesizer',
    # **** REMOVED {user_feedback} from goal to simplify interpolation ****
    goal="""Generate 2-3 specific, testable, mechanistic hypotheses based *only* on the provided Knowledge Base Content context. 
    Incorporate user feedback provided in the task description if it requires refinement. 
    Cite supporting facts/sources. If feasibility is uncertain, clearly state it.""",
    backstory="""Creative scientific thinker specializing in hypothesis generation from provided text data. 
    Focuses on evidence. **Can delegate specific feasibility check questions to the Feasibility_Assessor if a hypothesis seems borderline or requires specific experimental insight it lacks.**""",
    tools=[],
    llm=llm, verbose=True,
    allow_delegation=True
)

feasibility_assessor_agent = Agent(
    role='Feasibility Assessor',
    # **** REMOVED {hypotheses_context} from goal to simplify interpolation ****
    goal="""Evaluate experimental testability of provided hypotheses using standard lab resources. 
    Suggest concrete approaches and challenges.""",
    backstory="""Pragmatic experimental biologist. Assesses feasibility based on provided hypothesis text from task description. 
    **If assessment requires deeper synthesis of original facts not present in the hypothesis text itself, can delegate a query back to the Hypothesis_Synthesizer (providing necessary context).**""",
    tools=[],
    llm=llm, verbose=True,
    allow_delegation=True
)

feedback_evaluator_agent = Agent(
    role='Feedback Evaluator',
     # **** REMOVED {human_feedback} and {hypotheses_context} from goal to simplify interpolation ****
    goal="""Analyze user feedback on hypotheses provided in the task description to determine if it signifies approval (proceed) or requires revision. 
    Output a structured JSON evaluation.""",
    backstory="Interprets user feedback accurately.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

hypothesis_presenter_agent = Agent(
    role='Hypothesis Presenter',
     # **** REMOVED {hypotheses_context} from goal to simplify interpolation ****
    goal="""Clearly present provided hypotheses text from task description to the user for review.""",
    backstory="Formats and displays information clearly.",
    tools=[], llm=llm, verbose=True, allow_delegation=False
)

# Fact Extraction Agents
pdf_reader_agent = Agent( role='PDF_Reader', goal='Extract text from PDF.', backstory="PDF expert.", tools=[pdf_tool], llm=llm, verbose=True, allow_delegation=False )
fact_summarizer_agent = Agent( role='Fact_Summarizer', goal='List key facts from text.', backstory="Fact expert.", tools=[], llm=llm, verbose=True, allow_delegation=False )
knowledge_manager_agent = Agent( role='Knowledge_Manager', goal='Store facts in KB.', backstory="Storage expert.", tools=[kb_append_tool], llm=llm, verbose=True, allow_delegation=False )

# --- Define Tasks in Sequence ---

task_read_knowledge_base = Task(
    description=f"Read the entire content of the knowledge base file '{kb_reader_tool.input_file}' using the KnowledgeReaderTool. Ensure you return the full content.",
    expected_output="A string containing the full text content of the knowledge base file.",
    agent=knowledge_reader_agent
)

# **** MODIFIED ****: Pass needed variables explicitly in description
task_synthesize_hypotheses = Task(
    description="""**Using the Knowledge Base Content provided in the context**, generate 2-3 novel, specific, testable mechanistic hypotheses about early lung cancer/GGOs mechanisms (intrinsic/TME). 
    Suggest potential therapeutic targets/strategies. Provide rationale citing supporting facts/sources from the provided context. 
    Consider prior user feedback if relevant: '{user_feedback}'.
    **Collaboration Hint:** If you create a hypothesis but are unsure about its experimental feasibility using standard lab techniques (cell lines, basic assays), consider delegating a specific question to the `Feasibility_Assessor` agent.""",
    expected_output="2-3 clear, mechanistic hypotheses based *only* on the provided knowledge base context, with rationale linked to specific facts/sources. May include delegation attempts.",
    agent=hypothesis_synthesizer_agent,
    context=[task_read_knowledge_base]
)

# **** MODIFIED ****: Pass needed variables explicitly in description
task_present_hypotheses = Task(
    description="""Present the generated hypotheses (provided as '{hypotheses_context}') clearly to the user for review. 
    Add the message: 'Please review these hypotheses. Feedback will be collected directly.'""",
    expected_output="Formatted display of hypotheses for the user.",
    agent=hypothesis_presenter_agent,
    # Context dependency managed in main loop
)

# **** MODIFIED ****: Pass needed variables explicitly in description
task_evaluate_feedback = Task(
    description="""Analyze the provided human feedback ('{human_feedback}') on the presented hypotheses ('{hypotheses_context}'). 
    Determine if feedback indicates approval ('proceed': true) or requires revision ('proceed': false). 
    Output JSON: {{"proceed": true/false, "reasoning": "...", "key_points": [], "suggested_changes": []}}""", # Escaped braces for JSON literal
    expected_output="JSON object evaluating the feedback.",
    agent=feedback_evaluator_agent,
    # Context dependency managed in main loop
)

# **** MODIFIED ****: Pass needed variables explicitly in description
task_assess_feasibility = Task(
    description="""Assess experimental feasibility for the provided hypotheses ('{hypotheses_context}'). 
    For each hypothesis, suggest 1-2 concrete experimental approaches using standard resources (cells, organoids, mice, assays). 
    Comment on feasibility (High/Medium/Low) and note key challenges.
    **Collaboration Hint:** If evaluating feasibility requires understanding the synthesis rationale or facts *not explicitly stated in the hypothesis text*, consider delegating a question to the `Hypothesis_Synthesizer` providing the hypothesis and asking for clarification.""",
    expected_output="Structured feasibility assessment for each hypothesis. May include delegation attempts.",
    agent=feasibility_assessor_agent,
    # Context dependency managed in main loop
)

# **** MODIFIED ****: Pass needed variables explicitly in description AND strengthened prompt
task_refine_hypotheses_feedback = Task(
    description="""**Refine hypotheses based on negative user feedback.** The user provided feedback ('{human_feedback}') indicating they did not like the previous hypotheses ('{hypotheses_context}'). 
    Feedback Evaluation: '{feedback_evaluation_context}'. 
    **Your goal is to generate a *completely NEW set* of 2-3 hypotheses.** Use the provided Knowledge Base Content ('{knowledge_base_content}') to formulate these new hypotheses, ensuring they are different from the previous set and grounded in the KB facts. Address any specific 'suggested_changes' if provided in the evaluation context.""",
    expected_output="A *new* list of 2-3 hypotheses addressing user feedback, grounded in provided KB facts.",
    agent=hypothesis_synthesizer_agent,
    # Context dependency managed in main loop
)

# **** MODIFIED ****: Pass needed variables explicitly in description
task_refine_hypotheses_feasibility = Task(
    description="""**Refine hypotheses based on feasibility assessment.** Original Hypotheses: '{hypotheses_context}'. 
    Feasibility Report: '{feasibility_context}'.
    Knowledge Base Content: '{knowledge_base_content}'
    Modify hypotheses using the provided Knowledge Base Content to improve experimental tractability based on the report. 
    Explain refinements in rationale, referencing the KB Content. Output the final revised list.""",
    expected_output="Revised list of 2-3 hypotheses optimized for testability, grounded in provided KB facts.",
    agent=hypothesis_synthesizer_agent,
    # Context dependency managed in main loop
)

# Fact Extraction Task Definition
def create_fact_extraction_tasks(pdf_full_path, pdf_filename):
    task_extract_text = Task( description=f"Extract text from '{pdf_filename}' at path: '{pdf_full_path}'.", expected_output="Extracted text.", agent=pdf_reader_agent )
    task_extract_facts = Task( description=f"Extract key facts related to early lung cancer/GGOs from text of '{pdf_filename}'.", expected_output="Bulleted list of facts.", agent=fact_summarizer_agent, context=[task_extract_text] )
    task_store_facts = Task( description=f"Store facts for '{pdf_filename}'.", expected_output="Confirmation message.", agent=knowledge_manager_agent, context=[task_extract_facts] )
    return [task_extract_text, task_extract_facts, task_store_facts]

# --- Define Crews ---
analysis_crew = Crew(
    agents=[
        knowledge_reader_agent,
        hypothesis_synthesizer_agent,
        hypothesis_presenter_agent,
        feedback_evaluator_agent,
        feasibility_assessor_agent
    ],
    tasks=[], # Added dynamically
    process=Process.sequential,
    verbose=True
)

extraction_crew = Crew(
    agents=[pdf_reader_agent, fact_summarizer_agent, knowledge_manager_agent],
    tasks=[], # Added dynamically
    process=Process.sequential,
    verbose=True
)

# --- Helper Functions ---
def parse_json_from_llm_output(text):
    # Same robust JSON parsing function as before...
    if not isinstance(text, str):
        print("Warning: Input to parse_json_from_llm_output was not a string.")
        return {"proceed": False, "reasoning": "Invalid input type for parsing.", "key_points": [], "suggested_changes": []}
    patterns = [ r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```', ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            try: return json.loads(json_str)
            except json.JSONDecodeError as e: print(f"Warn: Failed JSON block parse ({pattern}): {e}")
    try:
        start = text.find('{'); end = text.rfind('}') + 1
        if 0 <= start < end:
            json_str = text[start:end]
            if json_str.count('{') == json_str.count('}') and json_str.count('[') == json_str.count(']'):
                 try: return json.loads(json_str)
                 except json.JSONDecodeError as e: print(f"Warn: Failed raw object parse: {e}")
    except Exception as e: print(f"Warn: Error during raw object search: {e}")
    print(f"ERROR: Failed to parse JSON from LLM output: {text}")
    return {"proceed": False, "reasoning": "Failed parsing.", "key_points": [], "suggested_changes": []}

def collect_direct_feedback():
    """ Collects feedback directly from the user via console input. """
    print("\n=== Provide Your Feedback ===")
    print("(Type 'OK'/'Looks good' etc. for approval, or specific changes. Press Enter alone when done.)")
    user_input_lines = []
    while True:
        try: line = input("Your feedback: ").strip()
        except EOFError: break
        if line: user_input_lines.append(line)
        else: break
    combined = " ".join(user_input_lines).strip()
    feedback_tracker.add_feedback(combined or "OK") # Record feedback (or implicit OK)
    last = feedback_tracker.get_last_feedback() # Get the feedback just added
    # Don't reset flag here, reset it after it's used in the loop
    print(f"\n--- Feedback Received This Cycle: '{last}' ---")
    return last


# --- Main Workflow Logic ---
def main():
    global SKIP_FACT_EXTRACTION, feedback_tracker, extraction_crew, analysis_crew

    kb_file = kb_reader_tool.input_file

    # --- Step 1: Fact Extraction (Optional) ---
    if not SKIP_FACT_EXTRACTION:
        # ... (fact extraction logic remains the same) ...
        if os.path.exists(kb_file):
            print(f"Clearing existing KB file: {kb_file}")
            try: os.remove(kb_file)
            except OSError as e: print(f"Error removing KB: {e}"); sys.exit(1)
        feedback_tracker.clear()
        pdf_directory = os.getenv("PAPERS_PATH")
        if not pdf_directory:
             try: from config import PAPERS_PATH; pdf_directory = PAPERS_PATH; print("Using config.py PAPERS_PATH")
             except: pdf_directory = input("Enter path to PDF directory: ").strip()
        if not pdf_directory or not os.path.isdir(pdf_directory): print(f"Error: Invalid PDF directory: {pdf_directory}"); sys.exit(1)
        pdf_files = sorted([f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")])
        if not pdf_files: print(f"No PDF files found in {pdf_directory}."); sys.exit(1)
        print(f"--- Starting Fact Extraction for {len(pdf_files)} files ---")
        all_extraction_tasks = []
        for pdf_filename in pdf_files:
             pdf_full_path = os.path.join(pdf_directory, pdf_filename)
             tasks = create_fact_extraction_tasks(pdf_full_path, pdf_filename)
             all_extraction_tasks.extend(tasks)
        if all_extraction_tasks:
            extraction_crew.tasks = all_extraction_tasks
            print(f"--- Running Extraction Crew ({len(extraction_crew.tasks)} tasks) ---")
            try: extraction_crew.kickoff(); print("--- Fact Extraction Finished ---")
            except Exception as e: print(f"\n--- ERROR during Extraction Crew: {e} ---"); traceback.print_exc(); sys.exit(1)
        else: print("--- No extraction tasks created. ---"); sys.exit(1)

    else:
        print(f"Skipping fact extraction. Using existing KB: {kb_file}")
        if not os.path.exists(kb_file) or os.path.getsize(kb_file) == 0: print(f"Warning: KB {kb_file} missing or empty!")

    # --- Step 2: Read Knowledge Base ---
    print("\n\n--- Reading Knowledge Base ---")
    analysis_crew.tasks = [task_read_knowledge_base]
    knowledge_base_content = ""
    try:
        result = analysis_crew.kickoff()
        # Accessing raw attribute is safer if available
        kb_content_result = getattr(result, 'raw', str(result))
        if "ERROR:" in kb_content_result or "failed" in kb_content_result.lower() or "empty" in kb_content_result.lower() or "not found" in kb_content_result.lower() :
             print(f"\n--- ERROR or Empty KB: Reading knowledge base failed: {kb_content_result} ---"); sys.exit(1)
        knowledge_base_content = kb_content_result
        print("--- Knowledge Base Read Successfully ---")
    except Exception as e: print(f"\n--- ERROR during KB Reading Crew: {e} ---"); traceback.print_exc(); sys.exit(1)

    # --- Step 3: Initial Hypothesis Synthesis ---
    print("\n\n--- Starting Initial Hypothesis Synthesis ---")
    # **** MODIFIED: Define inputs dictionary with all potential keys ****
    synthesis_inputs = {
        'user_feedback': feedback_tracker.get_last_feedback(),
        'knowledge_base_content': knowledge_base_content, # Passed via context, but include just in case
        'hypotheses_context': '',
        'human_feedback': '',
        'feedback_evaluation_context': '{}',
        'feasibility_context': ''
    }
    # Assign tasks for this specific run (KB read already done, its context is available)
    analysis_crew.tasks = [task_synthesize_hypotheses]
    hypotheses_result = None
    try:
        result_obj = analysis_crew.kickoff(inputs=synthesis_inputs)
        # Use the result directly, assuming it's the final output of the last task
        hypotheses_result = getattr(result_obj, 'raw', str(result_obj))
        if not hypotheses_result or "ERROR:" in hypotheses_result or "failed" in hypotheses_result.lower():
             print(f"\n--- ERROR: Hypothesis Synthesis failed: {hypotheses_result} ---"); sys.exit(1)
        print("\n--- Initial Synthesis Finished ---")
        print("=== Generated Hypotheses ===")
        print(hypotheses_result)
    except Exception as e: print(f"\n--- ERROR during Initial Synthesis run: {e} ---"); traceback.print_exc(); sys.exit(1)

    # --- Step 4: Feedback Loop ---
    current_hypotheses = hypotheses_result
    proceed_to_feasibility = False
    max_refinement_cycles = 2

    for cycle in range(max_refinement_cycles + 1):
        print(f"\n\n--- Starting Interaction Cycle {cycle + 1} ---")

        # 4a: Present Hypotheses
        print("--- Presenting Hypotheses ---")
        analysis_crew.tasks = [task_present_hypotheses]
        # **** MODIFIED: Define inputs dictionary with all potential keys ****
        present_inputs = {
            'hypotheses_context': current_hypotheses,
            'user_feedback': feedback_tracker.get_last_feedback(),
            'human_feedback': '',
            'feedback_evaluation_context': '{}',
            'knowledge_base_content': knowledge_base_content,
            'feasibility_context': ''
        }
        try:
             # Pass the comprehensive inputs dictionary
             analysis_crew.kickoff(inputs=present_inputs)
        except Exception as e:
            print(f"Error presenting hypotheses: {e}")
            print("\n--- Fallback Hypothesis Presentation ---")
            print(current_hypotheses)
            print("\nPlease review these hypotheses. Feedback will be collected directly.")

        # 4b: Collect Feedback
        human_feedback = collect_direct_feedback()
        feedback_tracker.reset_feedback_flag() # Reset flag now that feedback is collected

        # 4c: Check for Approval
        if feedback_tracker.is_approval(human_feedback):
             print("--- Approval detected. Proceeding. ---")
             proceed_to_feasibility = True
             break

        # 4d: Evaluate Non-Approval Feedback
        if cycle >= max_refinement_cycles:
             print("--- Max revision cycles reached. Proceeding with current hypotheses. ---")
             proceed_to_feasibility = True
             break

        print("--- Evaluating Feedback ---")
        analysis_crew.tasks = [task_evaluate_feedback]
        # **** MODIFIED: Define inputs dictionary with all potential keys ****
        eval_inputs = {
            'human_feedback': human_feedback,
            'hypotheses_context': current_hypotheses,
            'user_feedback': human_feedback, # Use current feedback for user_feedback too
            'feedback_evaluation_context': '{}',
            'knowledge_base_content': knowledge_base_content,
            'feasibility_context': ''
        }
        feedback_evaluation_json = None
        try:
             result = analysis_crew.kickoff(inputs=eval_inputs)
             eval_text = getattr(result, 'raw', str(result))
             feedback_evaluation_json = parse_json_from_llm_output(eval_text)
             print(f"--- Feedback Evaluation Result: {feedback_evaluation_json} ---")
             if feedback_evaluation_json.get('proceed', False):
                  print("--- Evaluator determined feedback allows proceeding. ---")
                  proceed_to_feasibility = True
                  break
        except Exception as e:
             print(f"Error during feedback evaluation: {e}")
             feedback_evaluation_json = None

        # 4e: Refine Based on Feedback (Only if evaluation didn't say proceed)
        print("--- Revision Required. Refining Hypotheses based on Feedback ---")
        analysis_crew.tasks = [task_refine_hypotheses_feedback]
        # **** MODIFIED: Define inputs dictionary with all potential keys ****
        refine_inputs = {
             'hypotheses_context': current_hypotheses,
             'human_feedback': human_feedback,
             'feedback_evaluation_context': json.dumps(feedback_evaluation_json or {}), # Pass evaluation result as JSON string
             'knowledge_base_content': knowledge_base_content, # Pass KB content
             'user_feedback': human_feedback, # Use current feedback
             'feasibility_context': ''
        }
        try:
             result = analysis_crew.kickoff(inputs=refine_inputs)
             refined_hypotheses = getattr(result, 'raw', str(result))
             if not refined_hypotheses or "ERROR:" in refined_hypotheses or "failed" in refined_hypotheses.lower():
                  print(f"\n--- WARNING: Feedback Refinement failed: {refined_hypotheses}. Retrying loop. ---")
                  time.sleep(2); continue
             current_hypotheses = refined_hypotheses
             print("--- Hypotheses Refined ---")
        except Exception as e:
             print(f"Error during feedback refinement: {e}. Retrying loop.")
             time.sleep(2); continue

    # --- Step 5: Feasibility Assessment ---
    feasibility_report = None
    if proceed_to_feasibility:
        print("\n\n--- Starting Feasibility Assessment ---")
        analysis_crew.tasks = [task_assess_feasibility]
        # **** MODIFIED: Define inputs dictionary with all potential keys ****
        feasibility_inputs = {
            'hypotheses_context': current_hypotheses,
            'user_feedback': feedback_tracker.get_last_feedback(), # Use last feedback status
            'human_feedback': '',
            'feedback_evaluation_context': '{}',
            'knowledge_base_content': knowledge_base_content,
            'feasibility_context': ''
            }
        try:
            result = analysis_crew.kickoff(inputs=feasibility_inputs)
            feasibility_report = getattr(result, 'raw', str(result))
            if not feasibility_report or "ERROR:" in feasibility_report or "failed" in feasibility_report.lower():
                 print(f"\n--- WARNING: Feasibility Assessment failed: {feasibility_report} ---"); feasibility_report = None
            else: print("\n--- Feasibility Assessment Finished ---"); print(feasibility_report)
        except Exception as e: print(f"\n--- ERROR during Feasibility run: {e} ---"); traceback.print_exc()
    else: print("\n\n--- Workflow ended before Feasibility Assessment. ---")

    # --- Step 6: Final Refinement based on Feasibility ---
    final_hypotheses = current_hypotheses
    if feasibility_report:
        print("\n\n--- Starting Final Refinement based on Feasibility ---")
        analysis_crew.tasks = [task_refine_hypotheses_feasibility]
        # **** MODIFIED: Define inputs dictionary with all potential keys ****
        refine_feas_inputs = {
             'hypotheses_context': current_hypotheses,
             'feasibility_context': feasibility_report,
             'knowledge_base_content': knowledge_base_content,
             'user_feedback': feedback_tracker.get_last_feedback(),
             'human_feedback': '',
             'feedback_evaluation_context': '{}'
        }
        try:
            result = analysis_crew.kickoff(inputs=refine_feas_inputs)
            final_hypotheses = getattr(result, 'raw', str(result))
            if not final_hypotheses or "ERROR:" in final_hypotheses or "failed" in final_hypotheses.lower():
                 print(f"\n--- WARNING: Feasibility Refinement failed. Using pre-refinement. ---"); final_hypotheses = current_hypotheses
            else: print("\n--- Final Refinement Finished ---")
        except Exception as e: print(f"\n--- ERROR during Feasibility Refinement run: {e} ---"); final_hypotheses = current_hypotheses

    # --- Final Output ---
    print_final_output(final_hypotheses)


def print_final_output(final_hypotheses):
    """ Helper to print the final result clearly. """
    print("\n\n\n--- Workflow Complete ---")
    print("========================================")
    print("       Final Generated Hypotheses       ")
    print("========================================")
    print(final_hypotheses)
    print("========================================")

if __name__ == "__main__":
    main()