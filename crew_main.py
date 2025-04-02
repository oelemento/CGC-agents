# crew_main.py (Redesigned for Collaboration Potential)

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

# --- Configuration Flag ---
# Set to True to skip PDF reading/fact extraction and use existing knowledge_base.txt
SKIP_FACT_EXTRACTION = True
# ---

# Load environment variables
load_dotenv()

# --- Class for tracking user feedback ---
class FeedbackTracker:
    def __init__(self):
        self.feedback_history = []
    
    def add_feedback(self, feedback):
        if feedback is not None and isinstance(feedback, str):
            self.feedback_history.append(feedback)
    
    def get_all_feedback(self):
        return self.feedback_history
    
    def get_combined_feedback(self):
        return " ".join([f for f in self.feedback_history if f])
    
    def has_negative_feedback(self):
        negative_terms = ['no', 'not', "don't", 'change', 'revise', 'modify', "don't like", 'dislike']
        for fb in self.feedback_history:
            if any(term in fb.lower() for term in negative_terms):
                return True
        return False
    
    def clear(self):
        self.feedback_history = []

# Initialize global feedback tracker
feedback_tracker = FeedbackTracker()

# --- Define Custom Tools ---

class PDFExtractionTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts text content from a given local PDF file path."
    def _run(self, pdf_path: str) -> str:
        print(f"--- TOOL: Attempting PDFExtractionTool on: {pdf_path} ---")
        if not pdf_path or not isinstance(pdf_path, str): print(f"--- ERROR: Invalid pdf_path received: {pdf_path} ---"); return f"Error: Invalid PDF path provided: {pdf_path}"
        if not os.path.exists(pdf_path): print(f"--- ERROR: PDF file not found by tool at path: {pdf_path} ---"); return f"Error: PDF file not found at path: {pdf_path}"
        if not pdf_path.lower().endswith(".pdf"): print(f"--- ERROR: Invalid file type received by tool: {pdf_path} ---"); return f"Error: Invalid file type. Expected a .pdf file, got: {pdf_path}"
        try:
            text_content = [];
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file); num_pages = len(reader.pages)
                print(f"Reading {num_pages} pages from '{os.path.basename(pdf_path)}'...")
                for page_num in range(num_pages):
                    page = reader.pages[page_num]; text = page.extract_text()
                    if text: text_content.append(text)
                extracted_text = "\n".join(text_content)
                if not extracted_text.strip(): print(f"--- WARNING: No text extracted from {pdf_path} ---"); return f"Warning: No text could be extracted from {pdf_path}."
                print(f"Successfully extracted text (approx {len(extracted_text)} chars).")
                max_chars = 15000
                if len(extracted_text) > max_chars: print(f"Truncating extracted text to {max_chars} chars for downstream tasks."); return extracted_text[:max_chars]
                else: return extracted_text
        except PyPDF2.errors.PdfReadError as e: print(f"--- ERROR reading PDF {pdf_path}: {e} ---"); return f"Error reading PDF file {pdf_path}: {e}. Corrupted or encrypted?"
        except Exception as e: print(f"--- ERROR during PDF extraction {pdf_path}: {e} ---"); traceback.print_exc(); return f"An unexpected error occurred during PDF text extraction: {e}"

class KnowledgeAppendTool(BaseTool):
    name: str = "Knowledge Append Tool"
    description: str = "Appends extracted key facts/observations for a specific source file to the central knowledge base (knowledge_base.txt)."
    output_file: str = "knowledge_base.txt"
    def _run(self, source_filename: str, facts_to_store: str) -> str:
        print(f"--- TOOL: Attempting KnowledgeAppendTool for: {source_filename} ---")
        if not isinstance(source_filename, str) or not source_filename: print(f"--- ERROR: Invalid source_filename for KB Append: {source_filename} ---"); return "Error: Invalid source_filename provided."
        if not isinstance(facts_to_store, str): print(f"--- WARNING: facts_to_store is not a string for {source_filename}. Attempting conversion. ---"); facts_to_store = str(facts_to_store)
        try:
            with open(self.output_file, "a") as f:
                f.write(f"--- Facts from: {source_filename} ---\n"); f.write(f"{facts_to_store}\n"); f.write("--- End Facts ---\n\n")
            return f"Successfully appended facts for {source_filename}."
        except Exception as e: print(f"--- ERROR storing knowledge for {source_filename}: {e} ---"); traceback.print_exc(); return f"Error storing knowledge for {source_filename}: {e}"

class KnowledgeReaderTool(BaseTool):
    name: str = "Knowledge Reader Tool"
    description: str = "Reads and returns the entire accumulated content from the knowledge base (knowledge_base.txt)."
    input_file: str = "knowledge_base.txt"
    def _run(self) -> str:
        print(f"--- TOOL: Attempting KnowledgeReaderTool on: {self.input_file} ---")
        try:
            if not os.path.exists(self.input_file): print(f"--- ERROR: Knowledge base file '{self.input_file}' not found. ---"); return "ERROR: Knowledge base file not found."
            with open(self.input_file, "r") as f: content = f.read()
            if not content.strip(): print(f"--- WARNING: Knowledge base file '{self.input_file}' is empty. ---"); return "Knowledge base is empty."
            print(f"--- TOOL: Successfully read {len(content)} chars from KB. ---")
            max_kb_chars = 30000 # Increased slightly, adjust as needed
            if len(content) > max_kb_chars: print(f"--- WARNING: KB content is long, truncating to {max_kb_chars} for hypothesis agent. ---"); return content[:max_kb_chars]
            return content
        except Exception as e: print(f"--- ERROR reading knowledge base: {e} ---"); traceback.print_exc(); return f"Error reading knowledge base: {e}"

# --- Instantiate Tools ---
pdf_tool = PDFExtractionTool()
kb_append_tool = KnowledgeAppendTool()
kb_reader_tool = KnowledgeReaderTool()

# --- Configure LLM ---
# Using a potentially more capable model might help with delegation decisions
llm = ChatOpenAI(model_name="gpt-4o") # Changed to gpt-4o for potentially better reasoning/delegation

# --- Define Agents ---
# Agents retain their core roles but some now allow delegation

pdf_reader_agent = Agent(
    role='PDF Text Reading Specialist',
    goal='Accurately extract text from provided PDF files using the PDF Text Extractor tool.',
    backstory="A meticulous agent focused on executing the PDF extraction tool flawlessly.",
    tools=[pdf_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False # Delegation likely not needed for this specific tool task
)

fact_summarizer_agent = Agent(
    role='Scientific Fact Extractor & Summarizer',
    goal='Analyze provided text to identify and concisely list key scientific facts, observations, and gene mentions relevant to the research area (early lung cancer/GGOs).',
    backstory="An analytical agent skilled at distilling crucial information from dense scientific text, focusing solely on the provided material.",
    tools=[],
    llm=llm,
    verbose=False,
    allow_delegation=False # Focuses on its core task
)

knowledge_manager_agent = Agent(
    role='Information Steward',
    goal='Organize and store extracted facts accurately into the central knowledge base using the Knowledge Append Tool, associating facts with their source file.',
    backstory="A reliable agent dedicated to maintaining the integrity and organization of the accumulated research findings.",
    tools=[kb_append_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False # Tool-specific task
)

hypothesis_synthesizer_agent = Agent(
    role='Mechanistic Hypothesis Synthesizer & Refiner',
    goal='Develop novel, specific, and testable mechanistic hypotheses based on the accumulated knowledge base. Refine hypotheses based on feedback (user or feasibility assessment).',
    backstory="""A creative yet rigorous scientific thinker specializing in generating hypotheses about biological mechanisms (intrinsic or TME-related) that could explain observed phenomena in early lung cancer. 
    Proposes potential therapeutic targets/strategies derived from these hypotheses. 
    Clearly justifies hypotheses by citing supporting facts from the knowledge base. 
    Can consult other specialists (like the feasibility agent) if needed.""",
    tools=[kb_reader_tool], # Primarily reads knowledge, but might delegate analysis or writing sub-tasks
    llm=llm,
    verbose=True,
    allow_delegation=True # **** CHANGED: Allow delegation ****
)

# Simplified Human Reviewer Agent
human_reviewer_agent = Agent(
    role='Hypothesis Display Agent',
    goal='Clearly present generated or refined hypotheses to the human user for review.',
    backstory="""An interface agent focused on presenting information clearly. It displays hypotheses and indicates that feedback will be collected externally.""",
    tools=[],
    llm=llm,
    verbose=False,
    allow_delegation=False # Simple display task
)

# Feedback Evaluation Agent
feedback_evaluation_agent = Agent(
    role='Feedback Evaluation Specialist',
    goal='Interpret human feedback on hypotheses to decide if they should proceed to feasibility testing or require revision, providing clear justification.',
    backstory="""An expert in understanding nuanced scientific feedback. Evaluates whether the user's input signifies satisfaction or requires changes, analyzing tone and specific points. 
    Provides a structured evaluation (proceed/revise, reasoning, key points).""",
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False # Focused evaluation task, though could be enabled if complex interpretation needed delegation
)

experimental_feasibility_agent = Agent(
    role='Experimental Biologist & Target Validation Strategist',
    goal="Assess the experimental testability of specific, provided hypotheses, focusing on validating proposed mechanisms and targets using available resources. Suggest concrete experimental approaches and highlight potential challenges.",
    backstory="""A pragmatic experimentalist evaluating hypotheses against practical constraints. 
    Considers available resources: Cell lines, Organoids (robotics), Co-cultures, Mouse models (slow), Standard assays (PCR, WB, CRISPR, Chem screens etc.). 
    Can break down complex assessments or consult on specific techniques if needed.""",
    tools=[], # Primarily evaluates based on knowledge/context, but could delegate literature search etc. if given tools
    llm=llm,
    verbose=True,
    allow_delegation=True # **** CHANGED: Allow delegation ****
)

# --- Define Tasks ---
# Task descriptions slightly adjusted for more flexibility

task_extract_text = Task(
    description="Execute the 'PDF Text Extractor' tool for the PDF found at the path: '{pdf_path}'.",
    expected_output="A string containing the extracted text from the PDF.",
    agent=pdf_reader_agent
)

task_extract_facts = Task(
    description="Analyze the text provided in the context (from the previous task). Identify and list key facts, experimental observations, and specific gene/pathway mentions related to early lung adenocarcinoma or Ground Glass Opacities (GGOs). Present findings concisely, perhaps as a bulleted list.",
    expected_output="A structured list (e.g., bullet points) of key facts and findings extracted from the text.",
    agent=fact_summarizer_agent,
    context=[task_extract_text]
)

task_store_facts = Task(
    description="Take the extracted facts (provided in context from the previous task) and store them using the 'Knowledge Append Tool'. Ensure they are associated with the source filename: '{pdf_filename}'.",
    expected_output="A confirmation message indicating successful storage of facts for the specified source file.",
    agent=knowledge_manager_agent,
    context=[task_extract_facts]
)

task_synthesize_hypotheses = Task(
    description="""Access the accumulated knowledge using the 'Knowledge Reader Tool'. Analyze the entire collection of facts.
    **Goal:** Generate 2-3 novel, specific, testable *mechanistic* hypotheses explaining phenomena observed in the facts, particularly regarding early lung cancer/GGOs. 
    **Requirements:** Each hypothesis should involve specific *genes, pathways, or cellular interactions*. Categorize each (e.g., tumor-intrinsic, TME interaction). Suggest a potential *therapeutic target or strategy* based on the mechanism. Provide clear 'Rationale' linking the hypothesis to *specific supporting facts* from the knowledge base (cite sources if possible based on KB format).
    **Feedback Integration:** Consider user feedback provided: '{user_feedback}'. If feedback is present and suggests changes (e.g., contains 'revise', 'clarify', 'different', 'no', 'don't'), develop NEW or REVISED hypotheses addressing the feedback. Otherwise, generate initial hypotheses.
    **Collaboration Hint:** If unsure about the potential testability of a proposed mechanism, consider formulating the hypothesis in a way that highlights this uncertainty or notes the need for feasibility input.""",
    expected_output="A set of 2-3 well-defined, mechanistic hypotheses, each including category, proposed target/strategy, and explicit rationale citing supporting facts. Hypotheses should reflect user feedback if provided.",
    agent=hypothesis_synthesizer_agent,
    # Note: Context for feedback is passed in the main loop, not explicitly here, 
    # allowing feedback to be incorporated on subsequent runs.
)

task_display_hypotheses = Task(
    description="""Present the hypotheses generated in the previous step to the user. The hypotheses are:

--- GENERATED HYPOTHESES ---
{hypotheses_context}
--- END HYPOTHESES ---

Include the message: "Please review these hypotheses. Feedback will be collected directly." 
Focus solely on clear presentation.""",
    expected_output="A clear display of the generated hypotheses for the user, along with the instructional message.",
    agent=human_reviewer_agent,
    context=[task_synthesize_hypotheses] # Takes output from synthesis to display
)

task_evaluate_feedback = Task(
    description="""Analyze the human feedback provided regarding the previously generated hypotheses.
--- HYPOTHESES REVIEWED ---
{hypotheses_context}
--- END HYPOTHESES ---

--- HUMAN FEEDBACK RECEIVED ---
{human_feedback}
--- END FEEDBACK ---

**Goal:** Determine the user's intent – are they approving the hypotheses for the next step (feasibility), or do they require revisions?
**Analysis:** Consider the explicit statements, tone, and overall sentiment. Look for keywords indicating approval (e.g., 'OK', 'proceed', 'looks good') or rejection/revision requests (e.g., 'change', 'revise', 'don't like', 'clarify', 'issue with X').
**Output:** Return your analysis as a JSON object:
{{
  "proceed": true/false, // True if feedback indicates approval, False otherwise
  "reasoning": "Detailed explanation for the proceed/revise decision based on feedback analysis.",
  "key_points": ["Bulleted list of key takeaways or specific points raised in the feedback."],
  "suggested_changes": ["If proceed is false, list concrete changes suggested by the feedback or required for revision."]
}}""",
    expected_output="A JSON object containing the structured evaluation of the human feedback, including a proceed/revise decision and supporting details.",
    agent=feedback_evaluation_agent
    # Context (hypotheses and feedback) provided dynamically in the main loop
)


task_assess_feasibility = Task(
    description="""Critically evaluate the experimental feasibility of the following hypotheses intended for mechanism/target validation:
--- HYPOTHESES TO ASSESS ---
{hypotheses_input}
--- END HYPOTHESES ---

**Evaluation Criteria:** Assess based on standard laboratory resources: Cell lines, Organoids (potentially with robotics), Co-culture systems, Mouse models (noting they are slower), common molecular biology assays (PCR, Western Blot, ELISA, FACS), functional genomics tools (CRISPR screens, RNAi), and chemical screening capabilities.
**Required Output:** For each hypothesis:
1.  Briefly restate the core testable claim.
2.  Suggest 1-2 concrete experimental approaches to validate the mechanism or target.
3.  Comment on the general feasibility (e.g., High, Medium, Low) using available resources.
4.  Note key potential challenges or limitations (e.g., model relevance, assay sensitivity, time/cost).
**Collaboration Hint:** If a specific aspect requires deep expertise outside standard techniques (e.g., complex bioinformatics, specialized imaging), note this as a point for potential further consultation or delegation.""",
    expected_output="A structured feasibility assessment for each provided hypothesis, outlining proposed experiments, feasibility rating, and potential challenges.",
    agent=experimental_feasibility_agent,
    context=[task_synthesize_hypotheses] # Or potentially context from feedback loop confirming which hypotheses to assess
)

task_refine_hypotheses = Task(
    description="""Revise the initial set of hypotheses based on the provided experimental feasibility assessment.
--- ORIGINAL HYPOTHESES ---
{original_hypotheses}
--- FEASIBILITY ASSESSMENT FEEDBACK ---
{feasibility_feedback}
--- END ASSESSMENT ---

**Goal:** Modify the hypotheses to improve their experimental tractability while preserving the core scientific question, incorporating suggestions and addressing challenges raised in the feasibility assessment.
**Requirements:**
1.  Adjust the proposed mechanisms or targets slightly if needed to align with feasible experiments.
2.  Refine the wording to be more precise regarding the experimental validation path.
3.  Ensure the 'Rationale' still clearly links to the original supporting facts and briefly explains *why* the hypothesis was refined (based on feasibility).
Output the revised list of 2-3 hypotheses.""",
    expected_output="A revised list of 2-3 mechanistic hypotheses, optimized for experimental testability based on the feasibility feedback, with updated rationale.",
    agent=hypothesis_synthesizer_agent # Synthesizer refines its own work based on feedback
    # Context provided dynamically in the main loop
)


# --- Define Crews ---
# Crews remain sequential for overall workflow, but agents within can delegate

fact_extraction_crew = Crew(
    agents=[pdf_reader_agent, fact_summarizer_agent, knowledge_manager_agent],
    tasks=[task_extract_text, task_extract_facts, task_store_facts],
    process=Process.sequential,
    verbose=1
)

# Crew for initial synthesis OR refinement based on feedback
hypothesis_crew = Crew(
    agents=[hypothesis_synthesizer_agent], # This agent can delegate now
    tasks=[task_synthesize_hypotheses], # Task description encourages considering feasibility
    process=Process.sequential, # Overall process step is sequential
    verbose=True
)

display_crew = Crew(
    agents=[human_reviewer_agent],
    tasks=[task_display_hypotheses],
    process=Process.sequential,
    verbose=1
)

feedback_evaluation_crew = Crew(
    agents=[feedback_evaluation_agent],
    tasks=[task_evaluate_feedback],
    process=Process.sequential,
    verbose=True
)

feasibility_assessment_crew = Crew(
    agents=[experimental_feasibility_agent], # This agent can delegate now
    tasks=[task_assess_feasibility], # Task description allows noting need for consultation
    process=Process.sequential,
    verbose=True
)

# Refinement crew uses the synthesizer again, potentially informed by delegation during feasibility
refinement_crew = Crew(
    agents=[hypothesis_synthesizer_agent], # Synthesizer refines, can delegate
    tasks=[task_refine_hypotheses],
    process=Process.sequential,
    verbose=True
)

# --- Helper Functions ---
# (Keep existing helper functions: parse_json_from_llm_output, collect_direct_feedback)
def parse_json_from_llm_output(text):
    """Parse JSON from LLM output, handling cases where the JSON might be embedded in other text"""
    try:
        # First try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Look for JSON within the text
        try:
            # Look for JSON content between curly braces
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Try more lenient approaches
            try:
                # Look for content inside code blocks
                if '```json' in text and '```' in text:
                    json_block = text.split('```json')[1].split('```')[0].strip()
                    return json.loads(json_block)
                elif '```' in text:
                    json_block = text.split('```')[1].split('```')[0].strip()
                    return json.loads(json_block)
            except (json.JSONDecodeError, ValueError, IndexError):
                pass
    
    # If parsing fails, return a default structure with clear error message
    print("WARNING: Failed to parse JSON from LLM output. Using default response.")
    return {
        "proceed": False,
        "reasoning": "Failed to parse evaluation response properly. Defaulting to not proceeding.",
        "key_points": ["Error parsing feedback evaluation"],
        "suggested_changes": ["Please provide clearer feedback on what to change"]
    }

def collect_direct_feedback(hypotheses_text):
    """
    Collect feedback directly from the user instead of through an agent
    """
    print("\n=== Please Review These Hypotheses ===")
    print(hypotheses_text)
    print("\n=== Provide Your Feedback ===")
    print("Do you approve these hypotheses to proceed to feasibility assessment?")
    print("Type 'OK' or similar approval, or provide specific feedback on what needs to change.")
    print("Press Enter without text when you have completed giving feedback.")
    
    # Collect feedback until user hits Enter without text
    user_feedback = []
    while True:
        feedback = input("\nYour feedback: ").strip()
        if feedback:
            user_feedback.append(feedback)
            feedback_tracker.add_feedback(feedback) # Track feedback
            print("Feedback recorded. Press Enter without text when you're done.")
        else:
            break
    
    if user_feedback:
        combined_feedback = " ".join(user_feedback)
        print(f"\n=== Your combined feedback: '{combined_feedback}' ===")
        return combined_feedback
    else:
        # Check if there was previous negative feedback before treating empty as approval
        if feedback_tracker.has_negative_feedback():
             print("\n=== Empty input after previous negative feedback. Treating as signal to re-evaluate/revise based on prior feedback. ===")
             # Return the history so the evaluation agent knows context
             return feedback_tracker.get_combined_feedback() 
        else:
            print("\n=== No feedback provided, treating as approval ===")
            # Explicitly return an approval signal if desired, or empty string
            return "OK" # Or return "" if the evaluation task handles empty string as approval

# --- Main Workflow ---
def main():
    global SKIP_FACT_EXTRACTION # Ensure global scope
    
    kb_file = kb_append_tool.output_file
    if not SKIP_FACT_EXTRACTION:
        if os.path.exists(kb_file): 
            print(f"Clearing existing knowledge base file: {kb_file}")
            os.remove(kb_file)
            # Clear feedback tracker if restarting from scratch
            feedback_tracker.clear() 
    else:
        print(f"Attempting to skip fact extraction. Checking for existing KB: {kb_file}")
        if not os.path.exists(kb_file) or os.path.getsize(kb_file) == 0:
            print(f"WARNING: Cannot skip extraction. Knowledge base file '{kb_file}' not found or is empty.")
            print("Forcing fact extraction phase.")
            SKIP_FACT_EXTRACTION = False
            feedback_tracker.clear() # Clear feedback if forcing extraction

    try:
        # --- Determine PDF Directory ---
        pdf_directory = os.getenv("PAPERS_PATH")
        if not pdf_directory: 
            try:
                # Attempt to import from config.py if env var not set
                # Ensure config.py is in the Python path or same directory
                from config import PAPERS_PATH
                pdf_directory = PAPERS_PATH
                print(f"Using PAPERS_PATH from config.py: {pdf_directory}")
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not get PAPERS_PATH from environment or config.py ({e}).")
                pdf_directory = input("Please enter the full path to the directory containing PDF files: ")

        if not pdf_directory or not os.path.isdir(pdf_directory):
            print(f"Error: PDF directory path '{pdf_directory}' is not valid or not found.")
            sys.exit(1)
        
        print(f"Looking for PDF files in: {pdf_directory}")
        pdf_files = sorted([f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")])
        
        # --- Conditional Fact Extraction ---
        if not pdf_files and not SKIP_FACT_EXTRACTION:
             print(f"Error: No PDF files found in '{pdf_directory}'. Cannot perform fact extraction.")
             # Decide if you want to exit or try to proceed with existing KB if SKIP is allowed later
             sys.exit(1) 
        elif not SKIP_FACT_EXTRACTION:
            print(f"Found {len(pdf_files)} PDF files:")
            [print(f"- {f}") for f in pdf_files]
            print("\n--- Starting Fact Extraction Phase ---")
            for pdf_filename in pdf_files:
                print(f"\n--- Processing: {pdf_filename} ---")
                pdf_full_path = os.path.join(pdf_directory, pdf_filename)
                crew_inputs = {'pdf_path': pdf_full_path, 'pdf_filename': pdf_filename}
                try:
                    fact_extraction_crew.kickoff(inputs=crew_inputs)
                    print(f"--- Fact Extraction Finished for: {pdf_filename} ---")
                except Exception as e:
                    print(f"ERROR during fact extraction kickoff for {pdf_filename}: {e}")
                    traceback.print_exc()
                    print(f"--- Attempting to continue with next file ---")
        elif SKIP_FACT_EXTRACTION:
             print("\n--- Skipping Fact Extraction Phase (Using existing knowledge_base.txt) ---")
        else:
             print("\n--- No PDFs found, but skipping extraction. Attempting to proceed with existing KB. ---")


        # --- Hypothesis Loop (Allows for Revision) ---
        current_hypotheses_text = None
        proceed_to_feasibility = False
        max_revision_cycles = 3 # Limit revision cycles
        cycle = 0

        while cycle < max_revision_cycles:
            cycle += 1
            print(f"\n\n--- Starting Hypothesis Cycle {cycle} ---")

            # --- Step 2 / N: Hypothesis Synthesis / Refinement ---
            # Use accumulated feedback for synthesis task
            feedback_for_synthesis = feedback_tracker.get_combined_feedback()
            print(f"--- Providing feedback to synthesizer: '{feedback_for_synthesis[:100]}...' ---")
            synthesis_inputs = {'user_feedback': feedback_for_synthesis} 
            
            try:
                print(f"--- Kicking off Hypothesis Crew (Cycle {cycle}) ---")
                hypotheses_result = hypothesis_crew.kickoff(inputs=synthesis_inputs)
                
                if hypotheses_result and hasattr(hypotheses_result, 'raw') and hypotheses_result.raw.strip():
                    current_hypotheses_text = hypotheses_result.raw
                    print("\n--- Hypothesis Crew Finished ---")
                    # Display hypotheses using the dedicated crew
                    display_inputs = {'hypotheses_context': current_hypotheses_text}
                    try:
                         print("--- Displaying Hypotheses ---")
                         display_crew.kickoff(inputs=display_inputs)
                    except Exception as e:
                         print(f"Error displaying hypotheses: {e}")
                         # Fallback: print directly if display crew fails
                         print("\n=== Generated Hypotheses (Fallback Display) ===")
                         print(current_hypotheses_text)

                else:
                    print("\n--- WARNING: Hypothesis Crew did not produce valid output this cycle. Cannot proceed. ---")
                    current_hypotheses_text = None
                    break # Exit loop if synthesis fails

            except Exception as e:
                print(f"Error during hypothesis synthesis/refinement (Cycle {cycle}): {e}")
                traceback.print_exc()
                current_hypotheses_text = None
                break # Exit loop on error

            # --- Step 3 / N: Collect Direct Feedback ---
            if current_hypotheses_text:
                print("\n--- Starting Direct Feedback Collection ---")
                human_feedback = collect_direct_feedback(current_hypotheses_text)
                print(f"\n--- Direct Feedback Received (Cycle {cycle}) ---\n{human_feedback if human_feedback else 'Empty/Approval'}")
            else:
                 print("\n--- Skipping Feedback: No hypotheses generated this cycle. ---")
                 break # Exit loop if no hypotheses

            # --- Step 4 / N: Feedback Evaluation ---
            if human_feedback is not None: # Can be empty string "" or "OK" for approval now
                print("\n--- Starting Feedback Evaluation Phase ---")
                # Handle simple approval case directly first
                if human_feedback.strip().upper() == "OK":
                     print("--- Direct 'OK' approval received. Proceeding to feasibility. ---")
                     proceed_to_feasibility = True
                     break # Exit loop, proceed to feasibility

                # Otherwise, evaluate complex feedback
                evaluation_inputs = {
                    'hypotheses_context': current_hypotheses_text,
                    'human_feedback': human_feedback # Pass the potentially combined feedback
                }
                try:
                    evaluation_result = feedback_evaluation_crew.kickoff(inputs=evaluation_inputs)
                    evaluation_text = evaluation_result.raw if hasattr(evaluation_result, 'raw') else str(evaluation_result)
                    feedback_decision = parse_json_from_llm_output(evaluation_text)
                    
                    print("\n--- Feedback Evaluation Crew Finished ---")
                    print("\n=== Feedback Evaluation Result ===")
                    print(f"Decision: {'Proceed' if feedback_decision.get('proceed', False) else 'Revise'}") # Safer get
                    print(f"Reasoning: {feedback_decision.get('reasoning', 'N/A')}")

                    if feedback_decision.get('proceed', False):
                        print("\n--- Proceeding to Feasibility Assessment based on Feedback Evaluation. ---")
                        proceed_to_feasibility = True
                        break # Exit loop, proceed to feasibility
                    else:
                        print("\n--- Revision requested based on Feedback Evaluation. ---")
                        changes = feedback_decision.get('suggested_changes', ['No specific changes suggested, revise generally.'])
                        print(f"Suggested Changes: {'; '.join(changes)}")
                        # Feedback is already in tracker, loop will continue for revision
                        print("--- Will attempt revision in the next cycle. ---")
                        # Make sure we don't infinite loop if feedback doesn't change
                        if cycle >= max_revision_cycles:
                             print(f"--- Max revision cycles ({max_revision_cycles}) reached. Exiting loop. ---")
                             break

                except Exception as e:
                    print(f"Error during feedback evaluation kickoff: {e}")
                    traceback.print_exc()
                    print("--- Error evaluating feedback. Assuming revision needed to be safe. ---")
                    # Optional: Ask user again? For now, just loop for revision.
                    if cycle >= max_revision_cycles:
                         print(f"--- Max revision cycles ({max_revision_cycles}) reached during error. Exiting loop. ---")
                         break
            else:
                # Should not happen if collect_direct_feedback works correctly
                print("--- Internal Warning: human_feedback was None after collection. Assuming revision needed. ---")
                if cycle >= max_revision_cycles:
                     print(f"--- Max revision cycles ({max_revision_cycles}) reached. Exiting loop. ---")
                     break
            
            # End of while loop cycle


        # --- Post-Loop Steps ---
        if proceed_to_feasibility and current_hypotheses_text:
            # --- Step 5: Feasibility Assessment ---
            print("\n\n--- Starting Feasibility Assessment Phase ---")
            feasibility_inputs = {'hypotheses_input': current_hypotheses_text}
            feasibility_assessment_output = None
            try:
                feasibility_assessment_result = feasibility_assessment_crew.kickoff(inputs=feasibility_inputs)
                if feasibility_assessment_result and hasattr(feasibility_assessment_result, 'raw'):
                    feasibility_assessment_output = feasibility_assessment_result.raw
                    print("\n--- Feasibility Crew Finished ---")
                    print("\n=== Feasibility Assessment ===")
                    print(feasibility_assessment_output)
                else:
                     print("--- WARNING: Feasibility assessment did not produce valid output. ---")

            except Exception as e:
                print(f"Error during feasibility assessment kickoff: {e}")
                traceback.print_exc()

            # --- Step 6: Hypothesis Refinement (Optional Post-Feasibility) ---
            # Decide if you ALWAYS want refinement after feasibility, or only if needed
            # For now, let's assume we always run it if feasibility assessment was done
            if feasibility_assessment_output:
                print("\n\n--- Starting Post-Feasibility Hypothesis Refinement Phase ---")
                refinement_inputs = {
                    'original_hypotheses': current_hypotheses_text,
                    'feasibility_feedback': feasibility_assessment_output
                }
                try:
                    refined_hypotheses_result = refinement_crew.kickoff(inputs=refinement_inputs)
                    print("\n--- Refinement Crew Finished ---")
                    print("\n=== Final Refined Hypotheses (Post-Feasibility) ===")
                    print(refined_hypotheses_result.raw if hasattr(refined_hypotheses_result, 'raw') else refined_hypotheses_result)
                except Exception as e:
                    print(f"Error during post-feasibility hypothesis refinement kickoff: {e}")
                    traceback.print_exc()
            else:
                 print("\n--- Skipping Post-Feasibility Refinement: Feasibility assessment output missing. ---")

        elif not proceed_to_feasibility:
             print("\n\n--- Workflow ended without proceeding to feasibility assessment. ---")
        elif not current_hypotheses_text:
             print("\n\n--- Workflow ended because no hypotheses were successfully generated. ---")

    except FileNotFoundError as e:
        print(f"ERROR: A required file or directory was not found: {e}")
        traceback.print_exc()
    except ImportError as e:
         print(f"ERROR: Failed to import necessary modules. Is config.py available? Details: {e}")
         traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred in the main workflow: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Workflow Execution Finished ---")


if __name__ == "__main__":
    main()