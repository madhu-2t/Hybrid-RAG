# config.py
import os
from pathlib import Path
from dotenv import load_dotenv  # <--- Import this

# Load environment variables from .env file
load_dotenv()  # <--- Add this line immediately
# Base Paths (Calculated relative to this config file)
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"

# File Paths
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
STATE_FILE = DATA_DIR / "processed_state.json"

# Index Paths
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"
BM25_INDEX_PATH = BASE_DIR / "bm25_retriever.pkl"

# Model Config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Default to 2.5-flash as 1.5 is deprecated
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Retrieval Config
TOP_K_RETRIEVE = 30
TOP_K_RETURN = 5