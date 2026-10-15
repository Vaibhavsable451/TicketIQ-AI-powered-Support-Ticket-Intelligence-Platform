from src.data.collect import generate_synthetic
from src.data.preprocess import clean_text, preprocess
from src.features.engineering import build_preprocessor


def test_clean_text():
    assert clean_text("Visit http://x.com NOW!!!  Please") == "visit now please"


def test_features_shape():
    df = preprocess(generate_synthetic(300))
    X = build_preprocessor(200).fit_transform(df)
    assert X.shape[0] == len(df) and X.shape[1] > 10
