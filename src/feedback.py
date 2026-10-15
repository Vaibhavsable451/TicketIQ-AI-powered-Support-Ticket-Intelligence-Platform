"""Feature 6: human feedback loop - thumbs up/down saved as JSONL (use later to retrain or evaluate)."""
import json
from datetime import datetime, timezone
import pandas as pd
from src.config import LOG_DIR

FILE = LOG_DIR / "feedback.jsonl"


def save_feedback(ticket_safe: str, category: str, answer: str, helpful: bool, comment: str = ""):
    rec = dict(ts=datetime.now(timezone.utc).isoformat(), ticket=ticket_safe, category=category,
               answer=answer, helpful=helpful, comment=comment)
    with open(FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_feedback() -> pd.DataFrame:
    return pd.read_json(FILE, lines=True) if FILE.exists() else pd.DataFrame()