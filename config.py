import os

# Load .env file if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# === EXPERIMENT CONFIG ===
NUM_QUERIES = 200                    # Number of questions from NQ
TOP_K = 3                            # Number of retrieved chunks
CHUNK_SIZE = 500                     # Characters per chunk
CHUNK_OVERLAP = 50                   # Character overlap between chunks
SELFCHECK_SAMPLES = 5                # Stochastic samples for SelfCheckGPT
TEMPERATURE_DETERMINISTIC = 0.0      # For main answer
TEMPERATURE_STOCHASTIC = 0.7         # For SelfCheckGPT samples
CHECKPOINT_INTERVAL = 10             # Save checkpoint every N questions

# === MODEL CONFIG ===
MODEL_NAME = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# === PATHS ===
CHROMA_DB_PATH = "./data/chroma_db"
RESULTS_PATH = "./results"
FILTERED_NQ_FILE = "./data/filtered_nq.json"
RAW_RESULTS_FILE = "./results/raw_results.json"
CHECKPOINT_FILE = "./results/checkpoint.json"
SUMMARY_FILE = "./results/summary.csv"

# === EVIDENCE CONDITIONS ===
CONDITIONS = ["full", "partial", "none"]

# === API CONFIG ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_SLEEP = 0.5                      # Seconds between API calls
API_MAX_BACKOFF = 60                 # Max backoff seconds

# === PROMPTS ===
PROMPTS_DIR = "./prompts"
