# config.py
import autogen
import os
from dotenv import load_dotenv

# Load environment variables (optional, recommended for API keys)
load_dotenv()

# Configuration for the LLMs
config_list = [
    {
        "model": "gpt-4-turbo", # Specify desired model
        "api_key": os.getenv("OPENAI_API_KEY"),
        #"model": "claude-3-sonnet-20240229", # Specify Anthropic model
        #"api_key": os.getenv("ANTHROPIC_API_KEY"), # Use Anthropic key env variable

    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 42,  # Use None for disable caching
    "temperature": 0.1, # Lower temperature for more deterministic behavior initially
    # "timeout": 120, # Optional: Add timeout
}

# --- Data Paths ---
# !! IMPORTANT: Update these paths to point to your actual data files !!
BASE_DATA_PATH = "/Users/ole2001/PROGRAMS/CGC-HCI-data/expression" # Example base path, adjust as needed

# Gene Expression Data (Update filenames)
GENE_COUNTS_PATH = os.path.join(BASE_DATA_PATH, "GGO.merged_COUNTS_nodups_zeros_removed_013024.txt") # Assumed filename for raw counts
GENE_FPKM_PATH = os.path.join(BASE_DATA_PATH, "GGO.merged_FPKMs_nodups_zeros_removed_013024.txt")     # Assumed filename for FPKM/normalized data
METADATA_PATH = os.path.join(BASE_DATA_PATH, "metadata_Skim.txt") # Assumed filename for metadata

# Other paths
DEP_MAP_PATH = os.path.join(BASE_DATA_PATH, "depmap_data.csv") # Placeholder
LITERATURE_DB_PATH = "path/to/literature_vector_db" # For RAG/LlamaIndex
PAPERS_PATH = "/Users/ole2001/PROGRAMS/CGC-HCI-data/papers/summaries" # Directory for PDF papers

print("Configuration loaded.")
print(f"Metadata Path: {METADATA_PATH}")
print(f"Counts Path: {GENE_COUNTS_PATH}")
# print(f"LLM Config: {llm_config}") # Uncomment to verify config
