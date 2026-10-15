"""Classical ML: classification (LR, KNN, SVM, DT, RF, Boosting) and regression (Linear, Ridge, RF, Boosting)."""
import time
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.config import SEED


def classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="cosine", algorithm="brute"),
        "SVM (RBF)": SVC(kernel="rbf", C=3.0, probability=True, random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, random_state=SEED),
    }


def regressors():
    log = lambda m: TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)
    return {
        "LinearRegression": log(LinearRegression()),
        "Ridge": log(Ridge(alpha=1.0)),
        "RandomForest": log(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)),
        "GradientBoosting": log(GradientBoostingRegressor(n_estimators=200, random_state=SEED)),
    }


def tune_random_forest(X, y, n_iter=8):
    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=SEED),
        {"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20, 40], "min_samples_leaf": [1, 2, 4],
         "max_features": ["sqrt", "log2"]},
        n_iter=n_iter, cv=3, scoring="f1_macro", random_state=SEED, n_jobs=1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_classifiers(Xtr, ytr, Xte, yte, tune=False):
    models, rows = classifiers(), []
    if tune:
        best, params = tune_random_forest(Xtr, ytr)
        models["RandomForest (tuned)"] = best
        print("Best RF params:", params)
    fitted = {}
    for name, m in models.items():
        t = time.time()
        m.fit(Xtr, ytr)
        dt = time.time() - t
        pred = m.predict(Xte)
        rows.append(dict(model=name, accuracy=accuracy_score(yte, pred),
                         f1_macro=f1_score(yte, pred, average="macro"), train_sec=round(dt, 2)))
        fitted[name] = m
        print(f"[clf] {name:22s} acc={rows[-1]['accuracy']:.3f} f1={rows[-1]['f1_macro']:.3f} ({dt:.1f}s)")
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True), fitted


def train_regressors(Xtr, ytr, Xte, yte):
    rows, fitted = [], {}
    for name, m in regressors().items():
        t = time.time()
        m.fit(Xtr, ytr)
        dt = time.time() - t
        pred = m.predict(Xte)
        rows.append(dict(model=name, mae=mean_absolute_error(yte, pred),
                         rmse=float(np.sqrt(mean_squared_error(yte, pred))), r2=r2_score(yte, pred),
                         train_sec=round(dt, 2)))
        fitted[name] = m
        print(f"[reg] {name:22s} MAE={rows[-1]['mae']:.2f} R2={rows[-1]['r2']:.3f}")
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), fitted
