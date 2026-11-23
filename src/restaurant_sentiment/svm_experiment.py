"""Member 4: training-only Linear SVM tuning and locked evaluation."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import PROJECT_ROOT, RANDOM_STATE
from .evaluation import (
    classification_metrics,
    prediction_scores,
    save_confusion_matrix,
    top_linear_svm_terms,
)
from .models import svm_parameter_grid
from .pipeline import build_pipeline
from .preprocessing import load_fixed_partitions

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def tune_linear_svm(train: pd.DataFrame, cv_folds: int = 5) -> GridSearchCV:
    """Select C using stratified training folds and the full text pipeline."""

    if cv_folds < 2 or train["Liked"].value_counts().min() < cv_folds:
        raise ValueError("Each training class needs at least cv_folds reviews.")
    splitter = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        estimator=build_pipeline("svm"),
        param_grid=svm_parameter_grid(),
        scoring={metric: metric for metric in METRICS},
        cv=splitter,
        n_jobs=-1,
        refit="f1",
        return_train_score=False,
        error_score="raise",
    )
    return search.fit(train["Review"], train["Liked"])


def summarize_svm_cv(search: GridSearchCV) -> pd.DataFrame:
    """Return one report-ready row per candidate C value."""

    results = search.cv_results_
    rows = []
    for index, c_value in enumerate(results["param_classifier__C"]):
        row = {
            "C": float(c_value),
            "rank_f1": int(results["rank_test_f1"][index]),
            "fit_time_mean_seconds": float(results["mean_fit_time"][index]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(results[f"mean_test_{metric}"][index])
            row[f"{metric}_std"] = float(results[f"std_test_{metric}"][index])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("C").reset_index(drop=True)


def plot_svm_cv(cv_results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot positive-class F1 against C with fold variability."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.errorbar(
        cv_results["C"],
        cv_results["f1_mean"],
        yerr=cv_results["f1_std"],
        marker="o",
        capsize=4,
        color="#1f77b4",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Linear SVM regularization parameter C")
    axis.set_ylabel("Mean validation F1 (positive class)")
    axis.set_title("Training-only Five-fold Selection of Linear SVM C")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_locked_svm(model, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the locked model once and return metrics and safe predictions."""

    predictions = np.asarray(model.predict(test["Review"]), dtype=int)
    scores = np.asarray(prediction_scores(model, test["Review"]), dtype=float)
    metrics = classification_metrics(test["Liked"], predictions, scores)
    metrics_frame = pd.DataFrame([{"model": "svm", **metrics}])
    predictions_frame = pd.DataFrame(
        {
            "review_id": test["review_id"].to_numpy(),
            "true_label": test["Liked"].to_numpy(dtype=int),
            "predicted_label": predictions,
            "decision_score": scores,
            "correct": predictions == test["Liked"].to_numpy(dtype=int),
        }
    )
    return metrics_frame, predictions_frame


def run_svm_experiment(
    output_root: str | Path = PROJECT_ROOT, cv_folds: int = 5
) -> dict:
    """Tune C on training folds, then perform one locked test evaluation."""

    started = time.perf_counter()
    root = Path(output_root)
    processed = root / "data" / "processed"
    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    model_dir = root / "models" / "svm"
    for directory in (tables, figures, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train, test = load_fixed_partitions(processed)
    search = tune_linear_svm(train, cv_folds=cv_folds)
    cv_results = summarize_svm_cv(search)
    best_c = float(search.best_params_["classifier__C"])
    locked_model = search.best_estimator_

    cv_results.to_csv(tables / "svm_cv_results.csv", index=False)
    plot_svm_cv(cv_results, figures / "svm_c_cv_f1.png")

    test_metrics, predictions = evaluate_locked_svm(locked_model, test)
    test_metrics.to_csv(tables / "svm_test_metrics.csv", index=False)
    predictions.to_csv(tables / "svm_test_predictions.csv", index=False)
    top_linear_svm_terms(locked_model).to_csv(
        tables / "svm_top_terms.csv", index=False
    )
    save_confusion_matrix(
        test["Liked"],
        predictions["predicted_label"],
        figures / "svm_test_confusion_matrix.png",
        "Confusion Matrix: Locked Linear SVM on Test Set",
    )
    joblib.dump(locked_model, model_dir / "linear_svm_pipeline.joblib")

    metadata = {
        "random_state": RANDOM_STATE,
        "cv_folds": cv_folds,
        "cv_splitter": "StratifiedKFold(shuffle=True, random_state=8017)",
        "selection_metric": "positive-class F1",
        "svm_c_grid": svm_parameter_grid()["classifier__C"],
        "selected_c": best_c,
        "selected_cv_f1": float(search.best_score_),
        "ngram_range": [1, 1],
        "train_rows": len(train),
        "test_rows": len(test),
        "test_set_used_only_after_c_locked": True,
        "test_metrics": {
            key: float(test_metrics.iloc[0][key]) for key in METRICS
        },
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {
            name: hashlib.sha256((processed / name).read_bytes()).hexdigest()
            for name in ("train.csv", "test.csv", "split_manifest.csv")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (tables / "svm_experiment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_svm_experiment(args.output_root, args.cv_folds), indent=2))


if __name__ == "__main__":
    main()
