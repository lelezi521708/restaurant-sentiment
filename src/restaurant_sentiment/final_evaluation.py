"""Member 5: training-only NB tuning and a single-pass final evaluation."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import PROJECT_ROOT, RANDOM_STATE
from .evaluation import (
    build_error_frame,
    classification_metrics,
    plot_alpha_cv,
    plot_error_categories,
    plot_model_comparison,
    prediction_scores,
    save_confusion_matrix,
    save_roc_curves,
    summarize_errors,
    top_nb_terms,
    top_linear_svm_terms,
)
from .models import nb_parameter_grid
from .pipeline import build_pipeline
from .preprocessing import load_fixed_partitions

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
DISPLAY_NAMES = {"dummy": "Dummy", "nb": "Multinomial NB", "svm": "Linear SVM"}


def tune_nb(train: pd.DataFrame, cv_folds: int = 5):
    """Select NB smoothing on training folds only, refitting TF-IDF per fold."""

    if cv_folds < 2 or train["Liked"].value_counts().min() < cv_folds:
        raise ValueError("Each training class needs at least cv_folds reviews.")
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_pipeline("nb"),
        param_grid=nb_parameter_grid(),
        scoring={metric: metric for metric in METRICS},
        cv=splitter,
        n_jobs=-1,
        refit="f1",
        error_score="raise",
    )
    return search.fit(train["Review"], train["Liked"])


def summarize_nb_cv(search: GridSearchCV) -> pd.DataFrame:
    """Convert NB cross-validation results to a compact report-ready table."""

    results = search.cv_results_
    rows = []
    for index, alpha in enumerate(results["param_classifier__alpha"]):
        row = {"alpha": float(alpha)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(results[f"mean_test_{metric}"][index])
            row[f"{metric}_std"] = float(results[f"std_test_{metric}"][index])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def fit_final_models(train: pd.DataFrame, nb_alpha: float, svm_c: float) -> dict:
    """Fit the three locked models on the full training partition."""

    models = {
        "dummy": build_pipeline("dummy"),
        "nb": build_pipeline("nb").set_params(classifier__alpha=nb_alpha),
        "svm": build_pipeline("svm").set_params(classifier__C=svm_c),
    }
    for pipeline in models.values():
        pipeline.fit(train["Review"], train["Liked"])
    return models


def evaluate_once(models: dict, test: pd.DataFrame):
    """Evaluate every locked model on the same untouched test partition once."""

    rows = []
    scores_by_model = {}
    predictions_by_model = {}
    for name, pipeline in models.items():
        predictions = np.asarray(pipeline.predict(test["Review"]), dtype=int)
        scores = np.asarray(prediction_scores(pipeline, test["Review"]), dtype=float)
        if len(predictions) != len(test) or len(scores) != len(test):
            raise ValueError(f"Model '{name}' returned an unexpected number of rows.")
        scores_by_model[DISPLAY_NAMES[name]] = scores
        predictions_by_model[name] = predictions
        metrics = classification_metrics(test["Liked"], predictions, scores)
        rows.append({"model": name, **{metric: metrics[metric] for metric in METRICS}})
    return pd.DataFrame(rows), scores_by_model, predictions_by_model


def run_final_evaluation(
    output_root: str | Path = PROJECT_ROOT,
    svm_c: float = 0.1,
    nb_alpha: float | None = None,
    cv_folds: int = 5,
) -> dict:
    """Run training-only NB tuning, then evaluate all locked models on the test set.

    The test partition is used only after NB alpha and SVM C are fixed. Review
    text is written only to git-ignored ``data/processed/`` files; committed
    tables and figures contain aggregate statistics only.
    """

    root = Path(output_root)
    processed = root / "data" / "processed"
    tables, figures = root / "results" / "tables", root / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    train, test = load_fixed_partitions(processed)
    search = tune_nb(train, cv_folds)
    nb_cv = summarize_nb_cv(search)
    selected_alpha = float(search.best_params_["classifier__alpha"])
    final_alpha = selected_alpha if nb_alpha is None else float(nb_alpha)

    nb_cv.to_csv(tables / "final_nb_cv.csv", index=False)
    plot_alpha_cv(nb_cv, figures / "final_nb_alpha_cv.png")

    models = fit_final_models(train, final_alpha, svm_c)
    comparison, scores_by_model, predictions_by_model = evaluate_once(models, test)
    comparison.to_csv(tables / "final_model_comparison.csv", index=False)

    for name, predictions in predictions_by_model.items():
        save_confusion_matrix(
            test["Liked"],
            predictions,
            figures / f"final_confusion_matrix_{name}.png",
            f"Confusion Matrix: {DISPLAY_NAMES[name]} on Test Set",
        )
    save_roc_curves(
        scores_by_model,
        test["Liked"],
        figures / "final_roc_curves.png",
        "ROC Curves on the Fixed Test Set",
    )
    plot_model_comparison(comparison, figures / "final_model_comparison.png")

    top_nb_terms(models["nb"]).to_csv(tables / "final_nb_top_terms.csv", index=False)
    top_linear_svm_terms(models["svm"]).to_csv(
        tables / "final_svm_top_terms.csv", index=False
    )

    error_frames = {}
    summaries = []
    for name, predictions in predictions_by_model.items():
        frame = build_error_frame(test, predictions, scores_by_model[DISPLAY_NAMES[name]], name)
        error_frames[name] = frame
        summaries.append(summarize_errors(frame))
        frame.to_csv(processed / f"final_errors_{name}.csv", index=False)
    error_summary = pd.concat(summaries, ignore_index=True)
    error_summary.to_csv(tables / "final_error_summary.csv", index=False)

    finalists = comparison.loc[comparison["model"].ne("dummy")]
    best_name = finalists.sort_values(["f1", "model"], ascending=[False, True]).iloc[0]["model"]
    plot_error_categories(
        summarize_errors(error_frames[best_name]),
        figures / "final_error_categories.png",
    )

    nb_model_dir = root / "models" / "nb"
    nb_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["nb"], nb_model_dir / "final_nb_pipeline.joblib")

    predictions_long = pd.concat(
        [
            error_frames[name][["review_id", "true_label", "predicted_label", "score", "error_type"]]
            .assign(model=name)
            for name in predictions_by_model
        ],
        ignore_index=True,
    )
    predictions_long.to_csv(processed / "final_predictions.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "cv_folds": cv_folds,
        "cv_std_ddof": 0,
        "nb_alpha_grid": nb_parameter_grid()["classifier__alpha"],
        "nb_alpha_selected_by_training_f1": selected_alpha,
        "nb_alpha_used_in_final_fit": final_alpha,
        "nb_alpha_overridden": nb_alpha is not None,
        "svm_c_used_in_final_fit": svm_c,
        "svm_c_source": (
            f"Locked value supplied to unified evaluation; Member 4 selected C={svm_c:g} "
            "using training-fold cross-validation"
        ),
        "train_rows": len(train),
        "test_rows": len(test),
        "best_final_model_by_f1": best_name,
        "test_metrics_computed": True,
        "test_set_used_only_after_parameters_locked": True,
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
        "committed_outputs": [
            "results/tables/final_nb_cv.csv",
            "results/tables/final_model_comparison.csv",
            "results/tables/final_nb_top_terms.csv",
            "results/tables/final_svm_top_terms.csv",
            "results/tables/final_error_summary.csv",
            "results/tables/final_evaluation.json",
            "results/figures/final_nb_alpha_cv.png",
            "results/figures/final_model_comparison.png",
            "results/figures/final_roc_curves.png",
            "results/figures/final_confusion_matrix_dummy.png",
            "results/figures/final_confusion_matrix_nb.png",
            "results/figures/final_confusion_matrix_svm.png",
            "results/figures/final_error_categories.png",
        ],
        "local_only_outputs": [
            "data/processed/final_errors_dummy.csv",
            "data/processed/final_errors_nb.csv",
            "data/processed/final_errors_svm.csv",
            "data/processed/final_predictions.csv",
        ],
    }
    (tables / "final_evaluation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--svm-c", type=float, default=0.1, help="Locked Linear SVM C value.")
    parser.add_argument("--nb-alpha", type=float, default=None, help="Override the NB alpha selected on training folds.")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    metadata = run_final_evaluation(args.output_root, args.svm_c, args.nb_alpha, args.cv_folds)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
