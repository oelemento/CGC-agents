# crew_main.py (Includes Hypothesis Refinement based on Feasibility)

import os
import traceback
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import PyPDF2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Use desired model

# Load environment variables
load_dotenv()

# --- Define Custom Tools ---

# Tool 1: PDF Text Extraction
class PDFExtractionTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = "Extracts text content from a given local PDF file path."
    def _run(self, pdf_path: str) -> str:
        print(f"--- TOOL: Attempting PDFExtractionTool on: {pdf_path} ---")
        # ...(rest of unchanged PDFExtractionTool code)...
        if not os.path.exists(pdf_path): print(f"--- ERROR: PDF file not found by tool at path: {pdf_path} ---"); return f"Error: PDF file not found at path: {pdf_path}"
        if not pdf_path.lower().endswith(".pdf"): print(f"--- ERROR: Invalid file type received by tool: {pdf_path} ---"); return f"Error: Invalid file type. Expected a .pdf file, got: {pdf_path}"
        try:
            text_content = []
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file); num_pages = len(reader.pages)
                print(f"Reading {num_pages} pages...")
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


# Tool 2: Knowledge Base Append
class KnowledgeAppendTool(BaseTool):
    name: str = "Knowledge Append Tool"
    description: str = "Appends extracted key facts/observations for a specific source file to the central knowledge base (knowledge_base.txt)."
    output_file: str = "knowledge_base.txt"
    def _run(self, source_filename: str, facts_to_store: str) -> str:
        print(f"--- TOOL: Attempting KnowledgeAppendTool for: {source_filename} ---")
        # ...(rest of unchanged KnowledgeAppendTool code)...
        try:
            with open(self.output_file, "a") as f:
                f.write(f"--- Facts from: {source_filename} ---\n"); f.write(f"{facts_to_store}\n"); f.write("--- End Facts ---\n\n")
            return f"Successfully appended facts for {source_filename}."
        except Exception as e: print(f"--- ERROR storing knowledge for {source_filename}: {e} ---"); traceback.print_exc(); return f"Error storing knowledge for {source_filename}: {e}"

# Tool 3: Knowledge Base Reader
class KnowledgeReaderTool(BaseTool):
    name: str = "Knowledge Reader Tool"
    description: str = "Reads and returns the entire accumulated content from the knowledge base (knowledge_base.txt)."
    input_file: str = "knowledge_base.txt"
    def _run(self) -> str:
        print(f"--- TOOL: Attempting KnowledgeReaderTool on: {self.input_file} ---")
        # ...(rest of unchanged KnowledgeReaderTool code)...
        try:
            if not os.path.exists(self.input_file): return "ERROR: Knowledge base file not found."
            with open(self.input_file, "r") as f: content = f.read()
            print(f"--- TOOL: Successfully read {len(content)} chars from KB. ---")
            max_kb_chars = 30000
            if len(content) > max_kb_chars: print(f"--- WARNING: KB content is long, truncating to {max_kb_chars} for hypothesis agent. ---"); return content[:max_kb_chars]
            return content if content else "Knowledge base is empty."
        except Exception as e: print(f"--- ERROR reading knowledge base: {e} ---"); traceback.print_exc(); return f"Error reading knowledge base: {e}"

# --- Instantiate Tools ---
pdf_tool = PDFExtractionTool()
kb_append_tool = KnowledgeAppendTool()
kb_reader_tool = KnowledgeReaderTool()

# --- Configure LLM ---
llm = ChatOpenAI(model_name="gpt-4-turbo") # Using Turbo

# --- Define Agents ---
pdf_reader_agent = Agent(role='PDF Text Reading Specialist', goal='Use the PDF Text Extractor tool to accurately read the content of a given PDF file path.', backstory="You execute the PDF extraction tool.", tools=[pdf_tool], llm=llm, verbose=False, allow_delegation=False)
fact_summarizer_agent = Agent(role='Scientific Fact Extractor & Summarizer', goal='Analyze provided text and accurately extract/list key facts relevant to early lung cancer progression.', backstory="You identify concrete facts based *only* on the text provided.", tools=[], llm=llm, verbose=False, allow_delegation=False)
knowledge_manager_agent = Agent(role='Information Steward', goal='Accurately store extracted facts using the Knowledge Append Tool.', backstory="You organize research findings.", tools=[kb_append_tool], llm=llm, verbose=False, allow_delegation=False)
# Modify hypothesis synthesizer slightly to also handle refinement
hypothesis_synthesizer_agent = Agent(
    role='Hypothesis Synthesizer & Refiner', # Updated role
    goal='Analyze accumulated facts to generate novel hypotheses OR refine existing hypotheses based on feasibility feedback.', # Updated goal
    backstory="You are an expert AI researcher skilled at integrating findings to propose new research directions or improve existing ones based on experimental considerations.",
    tools=[kb_reader_tool], # Keeps reader tool for initial synthesis if needed
    llm=llm,
    verbose=True, # Keep verbose
    allow_delegation=False
)
experimental_feasibility_agent = Agent( role='Experimental Biologist & Feasibility Assessor', goal="Evaluate proposed hypotheses for experimental testability using standard techniques and specified available lab resources.", backstory=("You are an experienced lab scientist. Available resources include: Standard cell lines, Lung tumor organoids (robotics screening possible), Immunocompetent tumor/T-cell co-cultures, Mouse models (slow), Standard molecular/cell assays. Assess feasibility, suggest approaches, note challenges."), tools=[], llm=llm, verbose=True, allow_delegation=False)

# --- Define Tasks ---
# Tasks for Per-Paper Fact Extraction Crew (Unchanged)
task_extract_text = Task(description="Use 'PDF Text Extractor' for PDF at path: '{pdf_path}'.", expected_output="Extracted text string.", agent=pdf_reader_agent)
task_extract_facts = Task(description="Analyze text context from previous task. Extract key facts/observations/genes related to early lung cancer/GGOs. List concisely.", expected_output="Bulleted list of key facts.", agent=fact_summarizer_agent, context=[task_extract_text])
task_store_facts = Task(description="Store extracted facts (from previous task) using 'Knowledge Append Tool'. Use source filename: '{pdf_filename}'.", expected_output="Confirmation message.", agent=knowledge_manager_agent, context=[task_extract_facts])

# Task for Initial Hypothesis Synthesis Crew (Unchanged)
task_synthesize_hypotheses = Task(description=("Use 'Knowledge Reader Tool' to get all accumulated facts. Analyze the entire collection. Identify cross-paper patterns/connections. Generate 2-3 novel, specific, testable hypotheses about early lung cancer progression mechanisms (AAH to AIS/MIA). Provide rationale grounded in facts."), expected_output="2-3 clear hypotheses with rationale.", agent=hypothesis_synthesizer_agent)

# Task for Feasibility Assessment Crew (Unchanged Task, input comes from previous crew)
task_assess_feasibility = Task(description=("Carefully review the following list of generated hypotheses:\n--- START HYPOTHESES ---\n{hypotheses_input}\n--- END HYPOTHESES ---\n\nFor each specific hypothesis listed above, evaluate its experimental feasibility using the available lab resources (cell lines, organoids w/ robotics, co-cultures, slow mouse models, standard assays - as detailed in your backstory). Suggest 1-2 potential experimental approaches per hypothesis. Comment briefly on feasibility and potential challenges."), expected_output=("An assessment for each input hypothesis, outlining potential experimental approaches using available resources, comments on feasibility, and challenges."), agent=experimental_feasibility_agent)

# Task for Hypothesis Refinement Crew (NEW)
task_refine_hypotheses = Task(
    description=(
        "You are given a set of initial hypotheses and feedback on their experimental feasibility.\n\n"
        "--- INITIAL HYPOTHESES ---\n"
        "{original_hypotheses}\n"
        "--- END INITIAL HYPOTHESES ---\n\n"
        "--- FEASIBILITY ASSESSMENT ---\n"
        "{feasibility_feedback}\n"
        "--- END FEASIBILITY ASSESSMENT ---\n\n"
        "Your task is to refine the initial hypotheses based on the feasibility feedback. "
        "Focus on making them more experimentally tractable or specific using the available resources mentioned in the feedback (cell lines, organoids, co-cultures, mouse models). "
        "If a hypothesis seems fundamentally infeasible according to the feedback, note that. "
        "Output the refined list of 2-3 hypotheses, prioritizing those that are now more feasible or better defined experimentally."
    ),
    expected_output=(
        "A revised list of 2-3 hypotheses, modified based on the feasibility assessment to improve testability, "
        "along with brief notes on why changes were made or why a hypothesis remains challenging."
    ),
    agent=hypothesis_synthesizer_agent # Re-use the synthesizer agent for refinement
)


# --- Define Crews ---
fact_extraction_crew = Crew(agents=[pdf_reader_agent, fact_summarizer_agent, knowledge_manager_agent], tasks=[task_extract_text, task_extract_facts, task_store_facts], process=Process.sequential, verbose=1)
hypothesis_crew = Crew(agents=[hypothesis_synthesizer_agent], tasks=[task_synthesize_hypotheses], process=Process.sequential, verbose=True)
feasibility_assessment_crew = Crew(agents=[experimental_feasibility_agent], tasks=[task_assess_feasibility], process=Process.sequential, verbose=True)
# Crew 4: Hypothesis Refinement (NEW)
refinement_crew = Crew(
    agents=[hypothesis_synthesizer_agent], # Reuse the synthesizer agent
    tasks=[task_refine_hypotheses],
    process=Process.sequential,
    verbose=True # See the refined output
)

# --- Run Workflow ---
kb_file = kb_append_tool.output_file
if os.path.exists(kb_file): print(f"Clearing existing knowledge base file: {kb_file}"); os.remove(kb_file)
try:
    pdf_directory = os.getenv("PAPERS_PATH");
    if not pdf_directory: from config import PAPERS_PATH; pdf_directory = PAPERS_PATH
    if not pdf_directory or not os.path.isdir(pdf_directory): raise FileNotFoundError
except (ImportError, KeyError, FileNotFoundError, TypeError): print("Warning: Could not get PAPERS_PATH."); pdf_directory = input("Enter path to PDF directory: ")

print(f"Looking for PDF files in: {pdf_directory}")
if not os.path.isdir(pdf_directory): print(f"Error: Path '{pdf_directory}' is not valid.")
else:
    pdf_files = sorted([f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")])
    if not pdf_files: print(f"No PDF files found.")
    else:
        print(f"Found {len(pdf_files)} PDF files:"); [print(f"- {f}") for f in pdf_files]
        # --- Loop 1: Fact Extraction ---
        print("\n--- Starting Fact Extraction Phase ---")
        for pdf_filename in pdf_files:
            print(f"\n--- Processing for Facts: {pdf_filename} ---")
            pdf_full_path = os.path.join(pdf_directory, pdf_filename)
            crew_inputs = {'pdf_path': pdf_full_path, 'pdf_filename': pdf_filename }
            try: fact_extraction_crew.kickoff(inputs=crew_inputs); print(f"--- Fact Extraction Finished for: {pdf_filename} ---")
            except Exception as e: print(f"Error during fact extraction for {pdf_filename}: {e}"); traceback.print_exc()

        # --- Step 2: Hypothesis Synthesis ---
        print("\n\n--- Starting Hypothesis Synthesis Phase ---")
        final_hypotheses_output = None
        try:
            final_hypotheses_output = hypothesis_crew.kickoff()
            print("\n--- Hypothesis Crew Finished ---"); print("\n=== Generated Initial Hypotheses ==="); print(final_hypotheses_output)
        except Exception as e: print(f"Error during hypothesis synthesis: {e}"); traceback.print_exc()

        # --- Step 3: Feasibility Assessment ---
        feasibility_assessment_output = None
        if final_hypotheses_output:
            print("\n\n--- Starting Feasibility Assessment Phase ---")
            hypotheses_text = final_hypotheses_output.raw if hasattr(final_hypotheses_output, 'raw') else str(final_hypotheses_output)
            feasibility_inputs = {'hypotheses_input': hypotheses_text}
            try:
                feasibility_assessment_output = feasibility_assessment_crew.kickoff(inputs=feasibility_inputs)
                print("\n--- Feasibility Crew Finished ---"); print("\n=== Feasibility Assessment ==="); print(feasibility_assessment_output)
            except Exception as e: print(f"Error during feasibility assessment: {e}"); traceback.print_exc()
        else: print("\n--- Skipping Feasibility Assessment: No initial hypotheses were generated. ---")

        # --- Step 4: Hypothesis Refinement ---
        if final_hypotheses_output and feasibility_assessment_output:
             print("\n\n--- Starting Hypothesis Refinement Phase ---")
             # Extract raw text for both inputs
             original_hypotheses_text = final_hypotheses_output.raw if hasattr(final_hypotheses_output, 'raw') else str(final_hypotheses_output)
             feasibility_feedback_text = feasibility_assessment_output.raw if hasattr(feasibility_assessment_output, 'raw') else str(feasibility_assessment_output)

             refinement_inputs = {
                 'original_hypotheses': original_hypotheses_text,
                 'feasibility_feedback': feasibility_feedback_text
             }
             try:
                 refined_hypotheses = refinement_crew.kickoff(inputs=refinement_inputs)
                 print("\n--- Refinement Crew Finished ---"); print("\n=== Final Refined Hypotheses ==="); print(refined_hypotheses)
             except Exception as e: print(f"Error during hypothesis refinement: {e}"); traceback.print_exc()
        else:
            print("\n--- Skipping Hypothesis Refinement: Missing initial hypotheses or feasibility assessment. ---")


print("\n--- Full Workflow Completed ---")