"""End-to-end training pipeline:  python -m src.pipeline [--tune] [--skip-deep] [--finetune]"""
import argparse, json, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from src.config import MODELS_DIR, REPORTS_DIR, TARGET_CLASS, TARGET_REG
from src.data import collect, preprocess
from src.features.engineering import build_preprocessor
from src.models import classical, evaluate


def run(n=4000, source="synthetic", tune=False, skip_deep=False, finetune=False, force=False):
    collect.write_knowledge_base()
    raw = collect.load_or_generate(n, source, force)
    df = preprocess.preprocess(raw)
    train, test = preprocess.split(df)
    print(f"data: {len(df)} rows  train={len(train)} test={len(test)}")
    evaluate.plot_eda(df)

    prep = build_preprocessor().fit(train)
    Xtr, Xte = prep.transform(train), prep.transform(test)
    joblib.dump(prep, MODELS_DIR / "preprocessor.joblib")

    clf_res, clf_models = classical.train_classifiers(Xtr, train[TARGET_CLASS], Xte, test[TARGET_CLASS], tune=tune)
    best = clf_res.iloc[0]["model"]
    joblib.dump(Pipeline([("prep", prep), ("clf", clf_models[best])]), MODELS_DIR / "classifier.joblib")
    # Feature 5: interpretable model used to explain predictions
    joblib.dump(Pipeline([("prep", prep), ("clf", clf_models["LogisticRegression"])]), MODELS_DIR / "explainer.joblib")
    classes = sorted(train[TARGET_CLASS].unique())
    evaluate.plot_confusion(test[TARGET_CLASS], clf_models[best].predict(Xte), classes)
    evaluate.plot_top_terms(prep, clf_models["LogisticRegression"])

    # Feature 1: priority prediction model
    pri = LogisticRegression(max_iter=1000).fit(Xtr, train["priority"])
    ppred = pri.predict(Xte)
    priority_metrics = dict(model="Priority LogReg", accuracy=accuracy_score(test["priority"], ppred),
                            f1_macro=f1_score(test["priority"], ppred, average="macro"))
    joblib.dump(Pipeline([("prep", prep), ("clf", pri)]), MODELS_DIR / "priority.joblib")
    print(f"[priority] acc={priority_metrics['accuracy']:.3f} f1={priority_metrics['f1_macro']:.3f}")

    reg_res, reg_models = classical.train_regressors(Xtr, train[TARGET_REG], Xte, test[TARGET_REG])
    rbest = reg_res.iloc[0]["model"]
    joblib.dump(Pipeline([("prep", prep), ("reg", reg_models[rbest])]), MODELS_DIR / "regressor.joblib")
    evaluate.plot_regression(test[TARGET_REG], reg_models[rbest].predict(Xte))

    extra = []
    if not skip_deep:
        from src.models import deep
        m, hist = deep.train_mlp(Xtr, train[TARGET_CLASS], Xte, test[TARGET_CLASS], classes)
        extra.append(m); evaluate.plot_history(hist)
    if finetune:
        from src.models import finetune as ft
        extra.append(ft.finetune_distilbert(train, test, sample=2000))
    if extra:
        clf_res = pd.concat([clf_res, pd.DataFrame(extra)], ignore_index=True).sort_values("f1_macro", ascending=False)

    evaluate.plot_leaderboard(clf_res)
    evaluate.plot_leaderboard(reg_res, "rmse", "reg_leaderboard.png", "Regression RMSE (lower is better)")
    out = {"classification": clf_res.to_dict("records"), "regression": reg_res.to_dict("records"),
           "priority": priority_metrics, "best_classifier": best, "best_regressor": rbest}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\nBest classifier:", best, "| best regressor:", rbest)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--source", choices=["synthetic", "hf"], default="synthetic")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate raw data")
    a = ap.parse_args()
    run(a.n, a.source, a.tune, a.skip_deep, a.finetune, a.force)