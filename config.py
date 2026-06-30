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
NUM_QUERIES = 200                    # Number of questions from SQuAD
TOP_K = 3                            # Number of retrieved chunks
CHUNK_SIZE = 500                     # Characters per chunk
CHUNK_OVERLAP = 50                   # Character overlap between chunks
SELFCHECK_SAMPLES = 5                # Stochastic samples for SelfCheckGPT
TEMPERATURE_DETERMINISTIC = 0.0      # For main answer
TEMPERATURE_STOCHASTIC = 0.7         # For SelfCheckGPT samples
CHECKPOINT_INTERVAL = 1              # Save checkpoint every N questions (after each full question)

# === MODEL CONFIG ===
MODEL_NAME = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# === PATHS ===
CHROMA_DB_PATH = "./data/chroma_db"
RESULTS_PATH = "./results"
FILTERED_SQUAD_FILE = "./data/filtered_squad.json"
FILTERED_NQ_FILE = FILTERED_SQUAD_FILE  # backward-compat alias
RAW_RESULTS_FILE = "./results/raw_results.json"
CHECKPOINT_FILE = "./results/checkpoint.json"
SUMMARY_FILE = "./results/summary.csv"

# === FACTORIAL DESIGN ===
PROMPT_TYPES = ["constrained", "unconstrained"]
CONDITIONS = ["full", "partial", "none"]

# === API CONFIG ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_SLEEP = 0.5                      # Seconds between API calls
API_MAX_BACKOFF = 60                 # Max backoff seconds

# === PROMPTS ===
PROMPTS_DIR = "./prompts"

# ===========================================================================
# CROSS-MODEL EXTENSION (added 2026-06-29)
# The generator model is the single varying factor. Switch a generator by
# changing GENERATOR_PROVIDER + GENERATOR_MODEL only. The original Haiku
# pipeline above is untouched; MODEL_NAME stays for backward-compat / audit.
# Version strings are PINNED (no -latest aliases). Pricing verified 2026-06-29.
# ===========================================================================

# --- Generator selection (single switch) ---
# provider ∈ {"anthropic", "openai", "google"}
GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER", "anthropic")

# Registry: provider -> pinned model id + USD price per 1M tokens (in, out).
GENERATOR_REGISTRY = {
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",   # existing anchor; NOT regenerated
        "price_in": 1.00, "price_out": 5.00,     # Haiku 4.5 list price
    },
    "openai": {
        "model": "gpt-4.1-mini-2025-04-14",      # pinned snapshot
        "price_in": 0.40, "price_out": 1.60,
    },
    "google": {
        "model": "gemini-2.5-flash",             # stable GA id (not a -preview- snapshot)
        "price_in": 0.30, "price_out": 2.50,     # output price includes thinking tokens
    },
}
GENERATOR_MODEL = GENERATOR_REGISTRY[GENERATOR_PROVIDER]["model"]

# Google 2.5-flash is a hybrid-reasoning model with thinking ON by default.
# We disable thinking for the GENERATOR so it behaves as a plain instruct model
# comparable to Haiku, keeps output deterministic-ish, and avoids paying for
# thinking tokens. Documented design choice (recorded in run manifest).
GOOGLE_THINKING_BUDGET = 0

# --- External judge (3rd family, non-thinking, temp 0, pinned) ---
# Decoupled from the generator. Identical for ALL generators so judge-based
# cross-model comparisons are valid (removes generator==judge self-enhancement).
# The judge is ALWAYS reached via an OpenAI-compatible endpoint, so EITHER a
# direct first-party Qwen key (Alibaba DashScope/Model Studio) OR an OpenRouter
# key works — just set the matching preset below. Direct DashScope is preferred
# for reproducibility (single fixed first-party backend, canonical weights).
#
# Preset A — DIRECT Qwen (Alibaba DashScope, intl/Singapore endpoint):
#   JUDGE_PROVIDER=dashscope
#   JUDGE_MODEL=qwen2.5-72b-instruct
#   JUDGE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
#   key in .env as DASHSCOPE_API_KEY=...   (exact id/price verify before run)
#
# Preset B — OpenRouter reseller (pin one backend for reproducibility):
#   JUDGE_PROVIDER=openrouter
#   JUDGE_MODEL=qwen/qwen-2.5-72b-instruct
#   JUDGE_BASE_URL=https://openrouter.ai/api/v1
#   key in .env as OPENROUTER_API_KEY=...  ; set JUDGE_PROVIDER_PIN to one provider
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "openrouter")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen-2.5-72b-instruct")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
JUDGE_PRICE_IN = float(os.environ.get("JUDGE_PRICE_IN", "0.36"))   # USD / 1M tokens (verify per provider)
JUDGE_PRICE_OUT = float(os.environ.get("JUDGE_PRICE_OUT", "0.40"))
JUDGE_TEMPERATURE = 0.0
# OpenRouter-only: pin a single backend (no silent fallback to a different/
# quantized provider). Ignored for direct DashScope (already a fixed backend).
JUDGE_PROVIDER_PIN = os.environ.get("JUDGE_PROVIDER_PIN", "")
# Judge API key: whichever of these is set (checked in order).
JUDGE_API_KEY = (
    os.environ.get("JUDGE_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("DASHSCOPE_API_KEY")
)

# --- Additional API keys (set in .env before the paid run) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
