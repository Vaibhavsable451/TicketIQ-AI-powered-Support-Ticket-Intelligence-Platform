"""Feature 5: explain a category prediction with the top contributing words (Logistic Regression coefficients)."""
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from src.config import MODELS_DIR
from src.data.preprocess import prepare_frame


@lru_cache(maxsize=1)
def _explainer():
    p = MODELS_DIR / "explainer.joblib"
    return joblib.load(p) if p.exists() else None


def top_terms(text: str, category: str, k: int = 5):
    pipe = _explainer()
    if pipe is None:
        return []
    try:
        prep, lr = pipe.named_steps["prep"], pipe.named_steps["clf"]
        X = prep.transform(prepare_frame(pd.DataFrame([{"text": text}])))
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        contrib = X[0] * lr.coef_[list(lr.classes_).index(category)]
        names = prep.get_feature_names_out()
        idx = [i for i in np.argsort(contrib)[::-1] if names[i].startswith("tfidf__") and contrib[i] > 0][:k]
        return [(names[i].split("__", 1)[1], round(float(contrib[i]), 3)) for i in idx]
    except Exception:  # noqa: BLE001
        return []