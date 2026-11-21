"""Metrics, plots, and model interpretation helpers."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def prediction_scores(pipeline, texts):
    """Return continuous positive-class scores when supported."""

    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(texts)
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return None


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Calculate the common metrics used in the report."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics["roc_auc"] = (
        roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    )
    return metrics


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a report-ready confusion matrix."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Negative", "Positive"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(display.figure_)


def top_linear_svm_terms(pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract the strongest negative and positive Linear SVM terms."""

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("Feature weights are available only for linear models.")

    feature_names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    negative_indices = weights.argsort()[:top_n]
    positive_indices = weights.argsort()[-top_n:][::-1]
    return pd.DataFrame(
        {
            "negative_term": feature_names[negative_indices],
            "negative_weight": weights[negative_indices],
            "positive_term": feature_names[positive_indices],
            "positive_weight": weights[positive_indices],
        }
    )


def save_roc_curves(scores_by_model, y_true, output_path: str | Path, title: str) -> None:
    """Save one report-ready ROC panel comparing all models on the same labels."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.0), layout="constrained")
    for name, scores in scores_by_model.items():
        false_positive, true_positive, _ = roc_curve(y_true, scores)
        ax.plot(
            false_positive,
            true_positive,
            linewidth=1.8,
            label=f"{name} (AUC = {auc(false_positive, true_positive):.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def top_nb_terms(pipeline, top_n: int = 20) -> pd.DataFrame:
    """Rank terms by Multinomial NB log odds of positive versus negative class."""

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_log_prob_"):
        raise ValueError("Log-odds interpretation requires MultinomialNB.")
    log_odds = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    names = vectorizer.get_feature_names_out()
    negative_indices = log_odds.argsort()[:top_n]
    positive_indices = log_odds.argsort()[-top_n:][::-1]
    return pd.DataFrame(
        {
            "negative_term": names[negative_indices],
            "negative_log_odds": log_odds[negative_indices],
            "positive_term": names[positive_indices],
            "positive_log_odds": log_odds[positive_indices],
        }
    )


def plot_alpha_cv(cv_results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot training-only NB smoothing results with English axis labels."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = cv_results.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(5.6, 3.5), layout="constrained")
    ax.errorbar(
        ordered["alpha"],
        ordered["f1_mean"],
        yerr=ordered["f1_std"],
        fmt="o-",
        capsize=4,
        color="#9a3866",
    )
    ax.set(
        xlabel="Multinomial NB alpha",
        ylabel="Training-fold F1 (positive class)",
        title="Training-only five-fold smoothing comparison",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, output_path: str | Path) -> None:
    """Plot final test metrics for all models with English labels."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "f1", "roc_auc"]
    labels = ["Accuracy", "F1", "ROC-AUC"]
    x = np.arange(len(comparison))
    width = 0.24
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    for offset, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        bars = ax.bar(
            x + (offset - 1) * width,
            comparison[metric],
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set(
        xticks=x,
        xticklabels=comparison["model"].str.upper(),
        ylabel="Score",
        title="Final test-set model comparison",
        ylim=(0, 1.12),
    )
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_NEGATIONS = {
    "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither",
    "nor", "cannot", "can't", "cant", "don't", "dont", "didn't", "didnt",
    "doesn't", "doesnt", "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent",
    "weren't", "werent", "won't", "wont", "wouldn't", "wouldnt", "couldn't",
    "couldnt", "shouldn't", "shouldnt", "haven't", "havent", "hasn't", "hasnt",
    "hadn't", "hadnt",
}
_CONTRASTS = {"but", "however", "although", "though", "yet", "still", "except", "unfortunately"}
_ERROR_CATEGORIES = (
    "mixed sentiment terms",
    "negation or contraction",
    "contrast or qualification",
    "short review",
    "other",
)
_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "delicious", "love", "loved", "nice",
    "perfect", "wonderful", "best", "favorite", "friendly", "fresh", "recommend",
    "recommended", "tasty", "happy", "superb", "awesome", "pleasant", "clean",
}
_NEGATIVE_WORDS = {
    "bad", "awful", "terrible", "worst", "disgusting", "hate", "hated", "poor",
    "rude", "slow", "cold", "bland", "dirty", "disappointing", "disappointed",
    "never", "waste", "overpriced", "dry", "gross", "horrible",
}


def build_error_frame(test_frame: pd.DataFrame, predictions, scores, model_name: str) -> pd.DataFrame:
    """Describe every test prediction for transparent, non-causal error analysis.

    The returned frame includes the original review text and must stay local.
    Commit only aggregates produced by :func:`summarize_errors`.
    """

    frame = test_frame.loc[:, ["review_id", "Review", "Liked"]].copy().reset_index(drop=True)
    frame["model"] = model_name
    frame["predicted_label"] = np.asarray(predictions, dtype=int)
    frame["score"] = np.asarray(scores, dtype=float)
    frame["true_label"] = frame["Liked"].astype(int)
    frame["is_correct"] = frame["true_label"].eq(frame["predicted_label"])
    boundary = prediction_boundary_for_scores(frame["score"])
    frame["margin"] = np.abs(frame["score"] - boundary)
    if frame["margin"].std(ddof=0) > 0:
        threshold = frame["margin"].quantile(0.75)
        frame["high_confidence_error"] = (~frame["is_correct"]) & (frame["margin"] >= threshold)
    else:
        frame["high_confidence_error"] = False

    tokens = frame["Review"].map(lambda text: _TOKEN.findall(str(text).casefold().replace("’", "'")))
    frame["word_count"] = tokens.map(len)
    frame["has_negation"] = tokens.map(lambda words: bool(_NEGATIONS.intersection(words)))
    frame["has_contrast"] = tokens.map(lambda words: bool(_CONTRASTS.intersection(words)))
    frame["mixed_sentiment_terms"] = tokens.map(
        lambda words: bool(_POSITIVE_WORDS.intersection(words) and _NEGATIVE_WORDS.intersection(words))
    )

    # One descriptive primary category per error; priority is fixed in advance.
    def category(row) -> str:
        if row["mixed_sentiment_terms"]:
            return "mixed sentiment terms"
        if row["has_negation"]:
            return "negation or contraction"
        if row["has_contrast"]:
            return "contrast or qualification"
        if row["word_count"] <= 6:
            return "short review"
        return "other"

    frame["error_category"] = frame.apply(category, axis=1)
    frame["error_type"] = np.select(
        [
            (~frame["is_correct"]) & frame["true_label"].eq(0),
            (~frame["is_correct"]) & frame["true_label"].eq(1),
        ],
        ["false positive", "false negative"],
        default="correct",
    )
    return frame


def prediction_boundary_for_scores(scores) -> float:
    """Infer the neutral score for a model from its positive-class score range."""

    values = np.asarray(scores, dtype=float)
    return 0.5 if values.min() >= 0 and values.max() <= 1 else 0.0


def summarize_errors(error_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error counts without retaining any review text."""

    incorrect = error_frame.loc[~error_frame["is_correct"]]
    row = {
        "model": error_frame["model"].iloc[0],
        "reviews": len(error_frame),
        "correct": int(error_frame["is_correct"].sum()),
        "errors": len(incorrect),
        "false_positive": int(incorrect["error_type"].eq("false positive").sum()),
        "false_negative": int(incorrect["error_type"].eq("false negative").sum()),
        "high_confidence_errors": int(incorrect["high_confidence_error"].sum()),
        "errors_with_negation": int(incorrect["has_negation"].sum()),
        "errors_with_contrast": int(incorrect["has_contrast"].sum()),
        "mixed_sentiment_errors": int(incorrect["mixed_sentiment_terms"].sum()),
        "median_words_errors": float(incorrect["word_count"].median()) if len(incorrect) else float("nan"),
        "median_words_correct": float(
            error_frame.loc[error_frame["is_correct"], "word_count"].median()
        ),
    }
    for name in _ERROR_CATEGORIES:
        row[f"category_{name}"] = int(incorrect["error_category"].eq(name).sum())
    return pd.DataFrame([row])


def plot_error_categories(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot descriptive error categories from an aggregated summary table."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = summary.iloc[0]
    categories = {
        key.replace("category_", ""): value
        for key, value in row.items()
        if key.startswith("category_") and value > 0
    }
    names = list(categories)
    values = [categories[name] for name in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    ax.barh(names[::-1], values[::-1], color="#8c4a21")
    ax.set(
        xlabel="Number of misclassified reviews",
        title=f"Descriptive error categories: {row['model'].upper()}",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
