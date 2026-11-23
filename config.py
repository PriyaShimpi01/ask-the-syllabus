"""Central configuration for the Ask-the-Syllabus Bot.

Every tunable value lives here so the code, the evaluation and the report
always agree on the same settings.
"""
import os
from pathlib import Path

try:  # python-dotenv is optional (tests run without it)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# ---- folders -------------------------------------------------------------
PDF_DIR = BASE_DIR / "data" / "pdfs"            # put the course PDFs here
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # chunks.jsonl + ingestion report
INDEX_DIR = BASE_DIR / "faiss_index"             # saved vector store (index.faiss + index.pkl)
RESULTS_DIR = BASE_DIR / "results"               # evaluation outputs
TESTSET_PATH = BASE_DIR / "tests" / "test_set.json"

# ---- chunking ------------------------------------------------------------
CHUNK_SIZE = 1000        # characters per chunk (about one slide / one concept)
CHUNK_OVERLAP = 150      # characters shared between neighbouring chunks
MIN_CHUNK_CHARS = 40     # drop title-only / empty chunks such as "NLP Phases"

# ---- embeddings ----------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, pretrained, not fine-tuned

# ---- retrieval -----------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# Optional hard gate on FAISS L2 distance (smaller = closer). None = disabled.
# evaluate.py prints the distance distribution so you can judge whether a gate is useful.
_md = os.getenv("MAX_DISTANCE", "").strip()
MAX_DISTANCE = float(_md) if _md else None

# ---- LLM -----------------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TEMPERATURE = 0
