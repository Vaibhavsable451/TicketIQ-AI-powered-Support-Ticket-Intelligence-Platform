"""Central configuration: paths, seeds, and API keys (loaded from .env)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
KB_DIR = ROOT / "data" / "kb"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
LOG_DIR = REPORTS_DIR / "logs"
for _d in (DATA_RAW, DATA_PROCESSED, KB_DIR, MODELS_DIR, FIG_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_CLASS = "category"
TARGET_REG = "resolution_hours"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ticket-kb")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
EMBED_DIM = 384
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ticketiq")
