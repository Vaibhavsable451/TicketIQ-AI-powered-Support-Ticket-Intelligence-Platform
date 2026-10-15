"""Matplotlib / Seaborn evaluation plots saved to reports/figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import FIG_DIR

sns.set_theme(style="whitegrid")


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=130); plt.close(fig)


def plot_eda(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=df, x="category", order=df.category.value_counts().index, ax=ax[0]); ax[0].set_title("Tickets per category")
    sns.boxplot(data=df, x="category", y="resolution_hours", ax=ax[1]); ax[1].set_title("Resolution hours by category")
    sns.histplot(df.n_words, bins=25, ax=ax[2]); ax[2].set_title("Words per ticket")
    _save(fig, "eda.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[["n_chars", "n_words", "n_exclaim", "n_urgent", "tenure_months", "prior_tickets", "resolution_hours"]].corr(),
                annot=True, fmt=".2f", cmap="vlag", ax=ax); ax.set_title("Feature correlations")
    _save(fig, "correlations.png")


def plot_leaderboard(df, metric="f1_macro", name="clf_leaderboard.png", title="Classification leaderboard"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, y="model", x=metric, ax=ax, color="#4C72B0")
    ax.set_title(title)
    if metric in ("f1_macro", "accuracy"):
        ax.set_xlim(max(0, df[metric].min() - .1), 1.0)
    _save(fig, name)


def plot_confusion(y_true, y_pred, classes, name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalised confusion matrix")
    _save(fig, name)


def plot_top_terms(prep, logreg, k=8, name="top_terms.png"):
    names = np.array(prep.get_feature_names_out())
    fig, axes = plt.subplots(1, len(logreg.classes_), figsize=(4 * len(logreg.classes_), 4))
    for ax, c, w in zip(np.atleast_1d(axes), logreg.classes_, logreg.coef_):
        idx = np.argsort(w)[-k:]
        ax.barh([n.split("__")[-1] for n in names[idx]], w[idx], color="#55A868"); ax.set_title(c)
    _save(fig, name)


def plot_regression(y_true, y_pred, name="regression_fit.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=.4)
    lim = [0, max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual hours"); ax.set_ylabel("Predicted hours"); ax.set_title("Resolution time")
    _save(fig, name)


def plot_history(hist, name="mlp_training.png"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(hist["val_acc"]); ax[1].set_title("Validation accuracy")
    _save(fig, name)
