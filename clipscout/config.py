"""Every tunable number and model name lives here, nowhere else.

Model names change often. Any of them can be overridden from the .env file, e.g.
CLIPSCOUT_GEMINI_MODELS=gemini-3.8-flash,gemini-3.5-flash
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project folder into environment variables


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()] or default


def get_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to the .env file in the project folder.")
    return value


# ---------------- models ----------------
# Gemini watches the video. Tried in order: if one is not available on your key, the next is used.
GEMINI_MODELS = _env_list("CLIPSCOUT_GEMINI_MODELS", ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"])
# Gemini also turns text into vectors (multilingual, so Hinglish works; no local model to load).
EMBED_MODEL = os.getenv("CLIPSCOUT_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768

# Groq runs the agent's text steps. gpt-oss is a reasoning model; "low" effort keeps it fast.
GROQ_MODEL = os.getenv("CLIPSCOUT_GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_REASONING_EFFORT = os.getenv("CLIPSCOUT_GROQ_REASONING", "low")

# ---------------- indexing ----------------
SEGMENT_SEC = 100   # video goes to Gemini in 5-minute pieces
MOMENT_SEC = 10     # Gemini describes roughly every 10 seconds
INDEX_WORKERS = 3   # pieces watched in parallel (keep low on the free tier)

# ---------------- search ----------------
FETCH_K = 20          # candidates from EACH retriever before fusion
TOP_K = 5             # clips returned to the user
RRF_K = 60            # standard Reciprocal Rank Fusion constant
MERGE_GAP_SEC = 5.0   # moments closer than this may join one clip...
MERGE_MIN_RATIO = 0.6 # ...if their score is at least 60% of the clip's best moment
MAX_CLIP_SEC = 60.0   # and the clip stays under a minute
VERIFY_TOP_N = 2      # clips Gemini double-checks against the real frames (in parallel)
VERIFY_PAD_SEC = 3.0  # extra seconds around a clip when verifying
MAX_RETRIES = 1       # query rewrites the agent may try when nothing is found

# ---------------- storage ----------------
DATA_DIR = os.getenv("CLIPSCOUT_DATA_DIR", "data")
LIBRARY_DIR = os.path.join(DATA_DIR, "library")  # one JSON per video = the "index once" cache
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
COLLECTION = "clipscout_moments_gemini768"       # name includes the embedding, so a model change starts fresh
