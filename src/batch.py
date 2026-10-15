"""Feature 4: fast batch scoring of many tickets (no LLM) - used by Streamlit and the API."""
from functools import lru_cache
import joblib
import pandas as pd
from src.config import MODELS_DIR
from src.data.preprocess import prepare_frame
from src.utils.sentiment import score_text


@lru_cache(maxsize=1)
def _models():
    pri = MODELS_DIR / "priority.joblib"
    return (joblib.load(MODELS_DIR / "classifier.joblib"), joblib.load(MODELS_DIR / "regressor.joblib"),
            joblib.load(pri) if pri.exists() else None)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    clf, reg, pri = _models()
    X = df.copy().reset_index(drop=True)
    for c, d in (("tenure_months", 12), ("prior_tickets", 0)):
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(d)
    X = prepare_frame(X)
    out = df.copy().reset_index(drop=True)
    out["category"] = clf.predict(X)
    out["confidence"] = clf.predict_proba(X).max(axis=1).round(3)
    out["priority"] = pri.predict(X) if pri is not None else "n/a"
    out["eta_hours"] = reg.predict(X).round(1)
    out["sentiment"] = [score_text(t)[1] for t in X["text"]]
    out["needs_review"] = out["confidence"] < 0.55
    return out