"""Feature 8: find similar past tickets (TF-IDF cosine similarity) to spot duplicates and recurring issues."""
from functools import lru_cache
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DATA_PROCESSED, MODELS_DIR
from src.data.preprocess import clean_text


@lru_cache(maxsize=1)
def _index():
    prep, path = MODELS_DIR / "preprocessor.joblib", DATA_PROCESSED / "train.csv"
    if not (prep.exists() and path.exists()):
        return None
    tv = joblib.load(prep).named_transformers_["tfidf"]
    df = pd.read_csv(path)
    return tv, tv.transform(df["text_clean"].fillna("")), df


def similar_tickets(text: str, k: int = 3, min_sim: float = 0.6):
    idx = _index()
    if idx is None:
        return []
    tv, M, df = idx
    sims = cosine_similarity(tv.transform([clean_text(text)]), M)[0]
    out, seen = [], set()
    for i in sims.argsort()[::-1]:
        if sims[i] < min_sim or len(out) >= k:
            break
        if df.loc[i, "text"] in seen:  # skip exact repeats
            continue
        seen.add(df.loc[i, "text"])
        out.append({"text": df.loc[i, "text"], "category": df.loc[i, "category"], "similarity": round(float(sims[i]), 2)})
    return out