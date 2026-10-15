"""Feature engineering: TF-IDF text features + scaled numerics + one-hot categoricals."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["n_chars", "n_words", "n_exclaim", "n_urgent", "tenure_months", "prior_tickets"]
CAT_COLS = ["channel"]


def build_preprocessor(max_features: int = 1500) -> ColumnTransformer:
    return ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True), "text_clean"),
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
