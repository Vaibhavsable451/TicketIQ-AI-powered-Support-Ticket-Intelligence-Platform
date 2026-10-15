"""Cleaning, NLP normalisation, engineered columns, and train/test split."""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SEED, TARGET_CLASS, DATA_PROCESSED
from src.data.collect import URGENT

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = _URL.sub(" ", str(t).lower())
    t = _NON.sub(" ", t)
    return _WS.sub(" ", t).strip()


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Used for BOTH training and serving so features are identical."""
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].map(clean_text)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text_clean"].str.split().str.len()
    df["n_exclaim"] = df["text"].str.count("!")
    df["n_urgent"] = df["text_clean"].apply(lambda s: sum(w in s for w in URGENT))
    for c, default in (("tenure_months", 12), ("prior_tickets", 0), ("channel", "email")):
        if c not in df:
            df[c] = default
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna(subset=["text", TARGET_CLASS])
    df = df[df["text"].str.len() > 5]
    if "resolution_hours" in df:  # winsorise extreme targets
        df = df.assign(resolution_hours=df["resolution_hours"].clip(upper=df["resolution_hours"].quantile(.99)))
    return prepare_frame(df).reset_index(drop=True)


def split(df: pd.DataFrame, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET_CLASS], random_state=SEED)
    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test.to_csv(DATA_PROCESSED / "test.csv", index=False)
    return train.reset_index(drop=True), test.reset_index(drop=True)
