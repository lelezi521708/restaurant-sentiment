"""Metrics, plots, and model interpretation helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
