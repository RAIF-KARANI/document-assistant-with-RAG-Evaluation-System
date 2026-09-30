from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "documents"

# Chunking (used by src/ingest.py, Stage 2) - size/overlap in CHARACTERS,
# not tokens. ~500 chars is roughly 120-130 tokens for English text.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval (used by src/retrieve.py) - how many chunks come back per query.
TOP_K = 4
# Used only when a single document is scoped (app.py's document picker, and
# eval questions with a "scope_source" field) - safe to retrieve more since
# there's no risk of diluting across other documents. 12 wasn't always
# enough (a real query's answer-bearing chunk ranked 16th within one
# document - see CONCEPTS.md), so raised to 20.
SCOPED_TOP_K = 20

# Models (local, no API key required)
# Embedding model (used by src/ingest.py Stage 3, and src/retrieve.py -
# must be the same model in both places). Runs locally via
# sentence-transformers, downloaded once and cached by Hugging Face.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Chat model (used by src/generate.py) - served locally by Ollama, install
# from https://ollama.com and run `ollama pull llama3.2` before first use.
CHAT_MODEL = "llama3.2"
